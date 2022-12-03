import torch.nn as nn
import torch.nn.functional as F
from cogdl.data import Graph
from cogdl.utils import spmm
from .base_model import BaseModel
from cogdl.utils import get_activation
from cogdl.models.nn.correct_smooth import CorrectSmooth
from .mlp import MLP

class MLP_CAS(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=16)
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
            args.correct_alpha,
            args.smooth_alpha,
            args.num_correct_prop,
            args.num_smooth_prop,
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
            correct_alpha=0.9,
            smooth_alpha=0.9,
            num_correct_prop=30,
            num_smooth_prop=30,
            correct_norm="row",
            smooth_norm="col",
            dropout=0.0,
            activation="relu",
            norm=None,
            act_first=False,
            bias=True
    ):
        super(MLP_CAS, self).__init__()
        self.mlp = MLP(in_feats,
            out_feats,
            hidden_size,
            num_layers,
            dropout,
            activation,
            norm,
            act_first,
            bias)
        self.cas = CorrectSmooth(correct_alpha=correct_alpha,
                                 smooth_alpha=smooth_alpha,
                                 num_correct_prop=num_correct_prop,
                                 num_smooth_prop=num_smooth_prop,
                                 correct_norm=correct_norm,
                                 smooth_norm=smooth_norm)
    def reset_parameters(self):
        self.mlp.reset_parameters()
        
    def forward(self, x):
        return self.mlp.forward(x)
    
    def embed(self, graph):
        return self.mlp.embed(graph)
    
    def embed_layer(self, graph, i_layer):
        return self.mlp.embed_layer(graph, i_layer)
    
    def embed_without_prop_layer(self, x, i_layer):
        return self.mlp.embed_without_prop_layer(x, i_layer)
    
    def embed_without_prop(self, graph, i_layer):
        return self.mlp.embed_without_prop(graph, i_layer)
