import os

import torch
import torch.nn as nn
from .base_model import BaseModel
from .mlp import MLP
from cogdl.utils import spmm, dropout_adj, to_undirected


def get_adj(graph, remove_diag=False):
    if remove_diag:
        graph.remove_self_loops()
    else:
        graph.add_remaining_self_loops()
    return graph


def multi_hop_sgc(graph, x, nhop):
    results = [x]
    for _ in range(nhop):
        x = spmm(graph, x)
        results.append(x)
    return results

def multi_hop_sgc_without_prop(x, nhop):
    results = [x for _ in range(nhop + 1)]
    return results


class GAMLP(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        MLP.add_args(parser)
        parser.add_argument("--dropedge-rate", type=float, default=0.2)
        parser.add_argument("--directed", action="store_true")
        parser.add_argument("--nhop", type=int, default=3)
        parser.add_argument("--adj-norm", type=str, default=["sym"], nargs="+")
        parser.add_argument("--remove-diag", action="store_true")
        parser.add_argument("--diffusion", type=str, default="ppr")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.nhop,
            args.num_layers,
            args.dropout,
        )

    def __init__(self,
                 in_feats,
                 hidden_size,
                 out_feats,
                 nhop,
                 n_layers,
                 dropout,
                 adj_norm,
                 diffusion="sgc"):
        super(GAMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.project = MLP((nhop + 1) * in_feats, out_feats, hidden_size, n_layers, dropout)
        self.nhop = nhop
        self.adj_norm = adj_norm
        self.diffusion = diffusion
        self.remove_diag = False
        self.cache_x = None

    def _preprocessing(self, graph, x):
        device = x.device
        graph.to("cpu")
        graph.add_remaining_self_loops()
        graph.sym_norm()
        graph.eval()
        edge_index = graph.edge_index
        graph = get_adj(graph, remove_diag=self.remove_diag)

        for norm in self.adj_norm:
            with graph.local_graph():
                graph.edge_index = edge_index
                graph.normalize(norm)
                results = multi_hop_sgc(graph, graph.x, self.nhop)

        graph.to(device)
        out = [r.to(device) for r in results]

        return out
    
    def _preprocessing_no_prop(self, graph, x):
        device = x.device
        for norm in self.adj_norm:
            results = multi_hop_sgc_without_prop(graph.x, self.nhop)

        graph.to(device)
        out = [r.to(device) for r in results]

        return out

    def preprocessing(self, graph, x):
#         print("Preprocessing...")
        if graph.is_inductive():
            graph.train()
            x_train = self._preprocessing(graph, x)
            graph.eval()
            x_all = self._preprocessing(graph, x)
            train_nid = graph.train_nid
            for i in range(len(x_all)):
                x_all[i][train_nid] = x_train[i][train_nid]
        else:
            x_all = self._preprocessing(graph, x)

#         print("Preprocessing Done...")
        return x_all
    
    def preprocessing_no_prop(self, graph, x):
#         print("Preprocessing...")
        if graph.is_inductive():
            graph.train()
            x_train = self._preprocessing_no_prop(graph, x)
            graph.eval()
            x_all = self._preprocessing_no_prop(graph, x)
            train_nid = graph.train_nid
            for i in range(len(x_all)):
                x_all[i][train_nid] = x_train[i][train_nid]
        else:
            x_all = self._preprocessing_no_prop(graph, x)

#         print("Preprocessing Done...")
        return x_all

    def forward(self, graph):
        hidden = []
        if self.cache_x is None:
            x = graph.x.contiguous()
            self.cache_x = self.preprocessing(graph, x)
        x = self.cache_x
        for feat in x:
            hidden.append(feat)
        x = torch.cat(hidden, dim=-1)
        out = self.project(self.dropout(self.prelu(x)))

        return out
    
    def embed(self, graph):
        hidden = []
        if self.cache_x is None:
            x = graph.x.contiguous()
            self.cache_x = self.preprocessing(graph, x)
        x = self.cache_x
        for feat in x:
            hidden.append(feat)
        x = torch.cat(hidden, dim=-1)
        out = self.project.embed(self.prelu(x))

        return out
    
    def embed_without_prop(self, graph, i_layer):
        hidden = []
        x = graph.x.contiguous()
        x = self.preprocessing_no_prop(graph, x)
        for feat in x:
            hidden.append(feat)
        x = torch.cat(hidden, dim=-1)
        out = self.project.embed(self.prelu(x))

        return out
