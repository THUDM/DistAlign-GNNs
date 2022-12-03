import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from cogdl.layers import MLP
from cogdl.layers import GINLayer
from cogdl.utils import split_dataset_general, spmm


class GINLayer(nn.Module):
    r"""Graph Isomorphism Network layer from paper `"How Powerful are Graph
    Neural Networks?" <https://arxiv.org/pdf/1810.00826.pdf>`__.

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{sum}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Parameters
    ----------
    apply_func : callable layer function)
        layer or function applied to update node feature
    eps : float32, optional
        Initial `\epsilon` value.
    train_eps : bool, optional
        If True, `\epsilon` will be a learnable parameter.
    """

    def __init__(self, apply_func=None, eps=0, train_eps=True):
        super(GINLayer, self).__init__()
        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([eps]))
        self.apply_func = apply_func

    def forward(self, graph, x):
        out = (1 + self.eps) * x + spmm(graph, x)
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out
    
    def forward_without_prop(self, graph, x):
        x = (2 + self.eps) * x
        if self.apply_func is not None:
            x = self.apply_func(x)
        return x

class GIN(BaseModel):
    r"""Graph Isomorphism Network from paper `"How Powerful are Graph
    Neural Networks?" <https://arxiv.org/pdf/1810.00826.pdf>`__.

    Args:
        num_layers : int
            Number of GIN layers
        in_feats : int
            Size of each input sample
        out_feats : int
            Size of each output sample
        hidden_dim : int
            Size of each hidden layer dimension
        num_mlp_layers : int
            Number of MLP layers
        eps : float32, optional
            Initial `\epsilon` value, default: ``0``
        pooling : str, optional
            Aggregator type to use, default:　``sum``
        train_eps : bool, optional
            If True, `\epsilon` will be a learnable parameter, default: ``True``
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--epsilon", type=float, default=0.0)
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--num-layers", type=int, default=3)
        parser.add_argument("--num-mlp-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--train-epsilon", dest="train_epsilon", action="store_false")
        parser.add_argument("--pooling", type=str, default="sum")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_layers,
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_mlp_layers,
            args.epsilon,
            args.pooling,
            args.train_epsilon,
            args.dropout,
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        return split_dataset_general(dataset, args)

    def __init__(
        self,
        num_layers,
        in_feats,
        out_feats,
        hidden_dim,
        num_mlp_layers,
        eps=0,
        pooling="sum",
        train_eps=False,
        dropout=0.5,
    ):
        super(GIN, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            if i == 0:
                mlp = MLP(in_feats, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            self.gin_layers.append(GINLayer(mlp, eps, train_eps))
            self.batch_norm.append(nn.BatchNorm1d(hidden_dim))

        self.linear_prediction = nn.ModuleList()
        for i in range(self.num_layers + 1):
            if i == 0:
                self.linear_prediction.append(nn.Linear(in_feats, out_feats))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, out_feats))
        self.dropout = nn.Dropout(dropout)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        batch.sym_norm()
        h = batch.x
        device = h.device

        for i in range(self.num_layers):
            h = self.gin_layers[i](batch, h)
            h = self.batch_norm[i](h)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.linear_prediction[self.num_layers](h)
        return h
    
    def embed(self, batch):
        return self.forward(batch)
    
    def embed_without_prop(self, batch, i_layer):
        batch.sym_norm()
        h = batch.x
        device = h.device

        for i in range(i_layer):
            h = self.gin_layers[i].forward_without_prop(batch, h)
            h = self.batch_norm[i](h)
            h = F.relu(h)
        h = self.linear_prediction[self.num_layers](h)
        return h
    
    def embed_layer(self, batch, i_layer):
        batch.sym_norm()
        h = batch.x
        device = h.device

        h = self.gin_layers[i_layer](batch, h)
        h = self.batch_norm[i_layer](h)
        h = F.relu(h)
        if i_layer == self.num_layers - 1:
            h = self.linear_prediction[self.num_layers](h)
        return h
    
    def embed_without_prop_layer(self, batch, i_layer):
        batch.sym_norm()
        h = batch.x
        device = h.device

        h = self.gin_layers[i_layer].forward_without_prop(batch, h)
        h = self.batch_norm[i_layer](h)
        h = F.relu(h)
        if i_layer == self.num_layers - 1:
            h = self.linear_prediction[self.num_layers](h)
        return h
