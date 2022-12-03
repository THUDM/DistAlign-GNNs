import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import GATLayer
from .base_model import BaseModel
from cogdl.utils import (
    EdgeSoftmax,
    MultiHeadSpMM,
    get_activation,
    get_norm_layer,
    check_fused_gat,
    fused_gat_op,
    spmm,
)


class GATLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self, in_feats, out_feats, nhead=1, alpha=0.2, attn_drop=0.5, activation=None, residual=False, norm=None
    ):
        super(GATLayer, self).__init__()
        self.in_features = in_feats
        self.out_features = out_feats
        self.alpha = alpha
        self.nhead = nhead

        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats * nhead))

        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_feats)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_feats)))

        self.edge_softmax = EdgeSoftmax()
        self.mhspmm = MultiHeadSpMM()

        self.dropout = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.act = None if activation is None else get_activation(activation)
        self.norm = None if norm is None else get_norm_layer(norm, out_feats * nhead)

        if residual:
            self.residual = nn.Linear(in_feats, out_feats * nhead)
        else:
            self.register_buffer("residual", None)
        self.reset_parameters()

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.a_l)
        reset(self.a_r)
        reset(self.W)

    def forward(self, graph, x):
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_features)
        h[torch.isnan(h)] = 0.0

        row, col = graph.edge_index
        # Self-attention on the nodes - Shared attention mechanism
        h_l = (self.a_l * h).sum(dim=-1)
        h_r = (self.a_r * h).sum(dim=-1)

        # edge_attention: E * H
        edge_attention = self.leakyrelu(h_l[row] + h_r[col])
        edge_attention = self.edge_softmax(graph, edge_attention)
        edge_attention = self.dropout(edge_attention)

        out = self.mhspmm(graph, edge_attention, h)

        if self.residual:
            res = self.residual(x)
            out += res
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out
    
    def forward_without_prop(self, graph, x):
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_features)
        h[torch.isnan(h)] = 0.0

        out = spmm(graph, h)

        if self.residual:
            res = self.residual(x)
            out += res
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class GAT(BaseModel):
    r"""The GAT model from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    Args:
        num_features (int) : Number of input features.
        num_classes (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
        alpha (float) : Coefficient of leaky_relu.
        nheads (int) : Number of attention heads.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden_size-size", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.6)
        parser.add_argument("--attn-drop", type=float, default=0.5)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--last-nhead", type=int, default=1)
        parser.add_argument("--norm", type=str, default=None)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.attn_drop,
            args.alpha,
            args.nhead,
            args.residual,
            args.last_nhead,
            args.norm,
        )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout,
        attn_drop,
        alpha,
        nhead,
        residual,
        last_nhead,
        norm=None,
    ):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        self.attentions.append(
            GATLayer(in_feats, hidden_size, nhead=nhead, attn_drop=attn_drop, alpha=alpha, residual=residual, norm=norm)
        )
        for i in range(num_layers - 2):
            self.attentions.append(
                GATLayer(
                    hidden_size * nhead,
                    hidden_size,
                    nhead=nhead,
                    attn_drop=attn_drop,
                    alpha=alpha,
                    residual=residual,
                    norm=norm,
                )
            )
        self.attentions.append(
            GATLayer(
                hidden_size * nhead,
                out_feats,
                attn_drop=attn_drop,
                alpha=alpha,
                nhead=last_nhead,
                residual=False,
            )
        )
        self.num_layers = num_layers
        self.last_nhead = last_nhead
        self.residual = residual

    def embed(self, graph):
        x = graph.x
        for i, layer in enumerate(self.attentions):
            x = layer(graph, x)
            if i != self.num_layers - 1:
                x = F.elu(x)
        return x
    
    def embed_without_prop(self, graph, i_layer):
        x = graph.x
        for i in range(i_layer):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.attentions[i].forward_without_prop(graph, x)
            if i != i_layer - 1:
                x = F.elu(x)
        return x

    def forward(self, graph):
        x = graph.x
        for i, layer in enumerate(self.attentions):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(graph, x)
            if i != self.num_layers - 1:
                x = F.elu(x)
        return x

    def predict(self, graph):
        return self.forward(graph)
