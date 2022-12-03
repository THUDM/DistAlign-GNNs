import torch.nn as nn
import torch.nn.functional as F
from cogdl.data import Graph
from cogdl.utils import spmm
from .base_model import BaseModel
from cogdl.utils import get_activation


class MLP(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden_size-size", type=int, default=16)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--activation", type=str, default="relu")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.activation,
            args.norm,
            args.act_first if hasattr(args, "act_first") else False,
        )

    def __init__(
        self,
            in_feats,
            out_feats,
            hidden_size,
            num_layers,
            dropout=0.0,
            activation="relu",
            norm=None,
            act_first=False,
            bias=True,
    ):
        super(MLP, self).__init__()
        self.norm = norm
        self.activation = get_activation(activation)
        self.act_first = act_first
        self.dropout = dropout
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

    def embed(self, x):
        if isinstance(x, Graph):
            x = x.x
        for i, fc in enumerate(self.mlp[:-1]):
            x = fc(x)
            if self.act_first:
                x = self.activation(x)
            if self.norm:
                x = self.norm_list[i](x)
            if not self.act_first:
                x = self.activation(x)
        x = self.mlp[-1](x)

        return x
    
    def embed_layer(self, x, i_layer):
        if isinstance(x, Graph):
            x = x.x
        x = self.mlp[i_layer](x)
        if i_layer < len(self.mlp):
            if self.act_first:
                x = self.activation(x)
            if self.norm:
                x = self.norm_list[i_layer](x)
            if not self.act_first:
                x = self.activation(x)
        return x
    
    def embed_without_prop_layer(self, x, i_layer):
        return self.embed_layer(x, i_layer)

    def forward_with_prop(self, x, graph, i_layer_prop, n_prop):
        graph.sym_norm()
        if i_layer_prop == 0:
            for _ in range(n_prop):
                x = spmm(graph, x)
            for i, fc in enumerate(self.mlp):
                x = fc(x)
                if i < len(self.mlp) - 1:
                    if self.act_first:
                        x = self.activation(x)
                    if self.norm:
                        x = self.norm_list[i](x)
                    if not self.act_first:
                        x = self.activation(x)
        elif i_layer_prop >= len(self.mlp):
            for i, fc in enumerate(self.mlp):
                x = fc(x)
                if i < len(self.mlp) - 1:
                    if self.act_first:
                        x = self.activation(x)
                    if self.norm:
                        x = self.norm_list[i](x)
                    if not self.act_first:
                        x = self.activation(x)
            for _ in range(n_prop):
                x = spmm(graph, x)
        else:
            for i, fc in enumerate(self.mlp):
                x = fc(x)
                if i < len(self.mlp) - 1:
                    if self.act_first:
                        x = self.activation(x)
                    if self.norm:
                        x = self.norm_list[i](x)
                    if not self.act_first:
                        x = self.activation(x)
                if i_layer_prop - 1 == i:
                    for _ in range(n_prop):
                        x = spmm(graph, x)

        return x

    def forward(self, x):
        if isinstance(x, Graph):
            x = x.x
        for i, fc in enumerate(self.mlp[:-1]):
            x = fc(x)
            if self.act_first:
                x = self.activation(x)
            if self.norm:
                x = self.norm_list[i](x)
            if not self.act_first:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp[-1](x)

        return x
    
    def embed_without_prop(self, x, i_layer):
        if isinstance(x, Graph):
            x = x.x
        for i in range(i_layer):
            x = self.mlp[i](x)
            if i < len(self.mlp) - 1:
                if self.act_first:
                    x = self.activation(x)
                if self.norm:
                    x = self.norm_list[i](x)
                if not self.act_first:
                    x = self.activation(x)
        return x
