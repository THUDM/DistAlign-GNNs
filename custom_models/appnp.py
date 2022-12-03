import torch
import torch.nn.functional as F

from .base_model import BaseModel
from cogdl.utils import spmm
from .mlp import MLP


class APPNP(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--propagation-type", type=str, default="appnp")
        parser.add_argument("--alpha", type=float, default=0.1)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--num-iterations", type=int, default=10)  # only for appnp
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.alpha,
            args.num_iterations,
        )

    def __init__(self, nfeat, nhid, nclass, num_layers, dropout, alpha, niter):
        super(APPNP, self).__init__()
        # GCN as a prediction and then apply the personalized page rank on the results
        self.nn = MLP(nfeat, nclass, nhid, num_layers, dropout)

        self.alpha = alpha
        self.niter = niter
        self.dropout = dropout

    def get_ready_format(input, edge_index, edge_attr=None):
        if isinstance(edge_index, tuple):
            edge_index = torch.stack(edge_index)
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1]).float().to(input.device)
        adj = torch.sparse_coo_tensor(
            edge_index,
            edge_attr,
            (input.shape[0], input.shape[0]),
        ).to(input.device)
        return adj
    
    def forward(self, graph):
        x = graph.x
        graph.sym_norm()
        # get prediction
        x = F.dropout(x, p=self.dropout, training=self.training)
        local_preds = self.nn.forward(x)

        # apply personalized pagerank, appnp
        preds = local_preds
        with graph.local_graph():
            graph.edge_weight = F.dropout(graph.edge_weight, p=self.dropout, training=self.training)
            graph.set_symmetric()
            for _ in range(self.niter):
                new_features = spmm(graph, preds)
                preds = (1 - self.alpha) * new_features + self.alpha * local_preds
            final_preds = preds
        return final_preds
    
    def embed(self, graph):
        x = graph.x
        graph.sym_norm()
        # get prediction
        local_preds = self.nn.embed(x)

        # apply personalized pagerank, appnp
        preds = local_preds
        with graph.local_graph():
            graph.set_symmetric()
            for _ in range(self.niter):
                new_features = spmm(graph, preds)
                preds = (1 - self.alpha) * new_features + self.alpha * local_preds
            final_preds = preds
        return final_preds
    
    def embed_without_prop(self, graph, i_layer):
        x = graph.x
        graph.sym_norm()
        # get prediction
        local_preds = self.nn.embed(x)
        return local_preds
    
    def embed_layer(self, graph, i_layer):
        x = graph.x
        graph.sym_norm()
        local_preds = self.nn.embed_layer(graph, i_layer)
        
        preds = local_preds
        if i_layer == len(self.nn.mlp) - 1:
            with graph.local_graph():
                graph.set_symmetric()
                for _ in range(self.niter):
                    new_features = spmm(graph, preds)
                    preds = (1 - self.alpha) * new_features + self.alpha * local_preds
                final_preds = preds
            return final_preds
        else:
            return preds
    
    def embed_without_prop_layer(self, graph, i_layer):
        return self.nn.embed_without_prop_layer(graph, i_layer)
