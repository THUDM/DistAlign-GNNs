import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .gat import GAT
from .mlp import MLP
from cogdl.utils import split_dataset_general, spmm
import copy

class GATX(BaseModel):
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
        gat_first,
        activation="relu",
        norm=None,
        act_first=False,
        bias=True,
    ):
        """Sparse version of GAT."""
        super(GATX, self).__init__()
        self.gat_first = gat_first
        if self.gat_first:
            self.gat = GAT(in_feats,
                       hidden_size,
                       hidden_size,
                       num_layers,
                       dropout,
                       attn_drop,
                       alpha,
                       nhead,
                       residual,
                       last_nhead,
                       norm)
            self.mlp = MLP(hidden_size,
                out_feats,
                hidden_size,
                num_layers,
                dropout,
                activation,
                norm,
                act_first,
                bias)
        else:
            self.gat = GAT(hidden_size,
                       hidden_size,
                       out_feats,
                       num_layers,
                       dropout,
                       attn_drop,
                       alpha,
                       nhead,
                       residual,
                       last_nhead,
                       norm)
            self.mlp = MLP(in_feats,
                hidden_size,
                hidden_size,
                num_layers,
                dropout,
                activation,
                norm,
                act_first,
                bias)

    def forward(self, graph):
        if self.gat_first:
            x = self.gat(graph)
            graph_tmp = self.mlp(x)
        else:
            x = self.mlp(graph)
            graph_tmp = copy.copy(graph)
            graph_tmp.x = x
            graph_tmp = self.gat(graph_tmp)
        return graph_tmp

    def embed(self, graph):
        if self.gat_first:
            x = self.gat.embed(graph)
            graph_tmp = self.mlp.embed(x)
        else:
            x = self.mlp.embed(graph)
            graph_tmp = copy.copy(graph)
            graph_tmp.x = x
            graph_tmp = self.gat.embed(graph_tmp)
        return graph_tmp