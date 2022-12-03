import torch.nn as nn
import torch.nn.functional as F
from cogdl.utils import spmm
from .base_model import BaseModel
from cogdl.utils import get_activation


class SGC(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--num-layers", type=int, default=1)
        parser.add_argument("--hidden-size", type=int, default=16)
        parser.add_argument("--order", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--activation", type=str, default="relu")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(in_feats=args.num_features,
                   out_feats=args.num_classes,
                   order=args.order,
                   num_layers=args.num_layers,
                   hidden_size=args.hidden_size,
                   dropout=args.dropout,
                   activation=args.activation,
                   norm=args.norm)

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_layers,
                 hidden_size,
                 dropout=0.0,
                 activation="relu",
                 norm=None,
                 order=2,
                 remove_self_loop=False,
                 bias=True):
        super(SGC, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = get_activation(activation)
        self.order = order
        self.dropout = dropout
        self.norm = norm
        self.remove_self_loop = remove_self_loop
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.mlp = nn.ModuleList(
            [nn.Linear(shapes[layer], shapes[layer + 1], bias=bias) for layer in range(num_layers)]
        )
        if norm is not None and num_layers > 1:
            if norm == "layernorm":
                self.norm_list = nn.ModuleList(nn.LayerNorm(x) for x in shapes[1:-1])
            elif norm == "batchnorm":
                self.norm_list = nn.ModuleList(nn.BatchNorm1d(x) for x in shapes[1:-1])
            else:
                raise NotImplementedError(f"{norm} is not implemented in CogDL.")
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            layer.reset_parameters()
        if hasattr(self, "norm_list"):
            for n in self.norm_list:
                n.reset_parameters()

    def embed(self, graph):
        graph.sym_norm()
        x = graph.x
        for _ in range(self.order):
            x = spmm(graph, x)
        for i, fc in enumerate(self.mlp[:-1]):
            x = fc(x)
            if self.norm:
                x = self.norm_list[i](x)
            x = self.activation(x)
        x = self.mlp[-1](x)

        return x

    def embed_without_prop(self, graph, i_layer):
        x = graph.x
        for i, fc in enumerate(self.mlp[:i_layer]):
            x = fc(x)
            if i < len(self.mlp) - 1:
                if self.norm:
                    x = self.norm_list[i](x)
                x = self.activation(x)

        return x
    
    def embed_layer(self, graph, i_layer):
        graph.sym_norm()
        x = graph.x
        if i_layer == 0:
            for _ in range(self.order):
                x = spmm(graph, x)
        x = self.mlp[i_layer](x)
        if i_layer < len(self.mlp):
            if self.norm:
                x = self.norm_list[i_layer](x)
            x = self.activation(x)

        return x

    def embed_without_prop_layer(self, graph, i_layer):
        x = graph.x
        x = self.mlp[i_layer](x)
        if i_layer < len(self.mlp):
            if self.norm:
                x = self.norm_list[i_layer](x)
            x = self.activation(x)

        return x

    def forward_with_prop(self, graph, i_layer_prop, n_prop):
        if self.remove_self_loop:
            graph.remove_self_loops()
        graph.sym_norm()
        x = graph.x
        if i_layer_prop == 0:
            for _ in range(n_prop):
                x = spmm(graph, x)
            for i, fc in enumerate(self.mlp[:-1]):
                x = fc(x)
                if self.norm:
                    x = self.norm_list[i](x)
                x = self.activation(x)
            x = self.mlp[-1](x)
        elif i_layer_prop >= len(self.mlp) or i_layer_prop < 0:
            for i, fc in enumerate(self.mlp[:-1]):
                x = fc(x)
                if self.norm:
                    x = self.norm_list[i](x)
                x = self.activation(x)
            x = self.mlp[-1](x)
            for _ in range(n_prop):
                x = spmm(graph, x)
        else:
            for i, fc in enumerate(self.mlp[:-1]):
                x = fc(x)
                if self.norm:
                    x = self.norm_list[i](x)
                x = self.activation(x)
                if i_layer_prop - 1 == i:
                    for _ in range(n_prop):
                        x = spmm(graph, x)
            x = self.mlp[-1](x)

        return x

    def forward(self, graph):
        if self.remove_self_loop:
            graph.remove_self_loops()
        graph.sym_norm()
        x = graph.x
        for _ in range(self.order):
            x = spmm(graph, x)
        for i, fc in enumerate(self.mlp[:-1]):
            x = fc(x)
            if self.norm:
                x = self.norm_list[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp[-1](x)

        return x
