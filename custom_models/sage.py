import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import block_diag

from cogdl.utils import split_dataset_general, spmm, get_activation, get_norm_layer

from .base_model import BaseModel


class MeanAggregator(object):
    def __call__(self, graph, x):
        graph.row_norm()
        x = spmm(graph, x)
        return x

class SAGELayer(nn.Module):
    def __init__(
        self, in_feats, out_feats, normalize=False, dropout=0.0, norm=None, activation=None, residual=False
    ):
        super(SAGELayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(2 * in_feats, out_feats)
        self.normalize = normalize
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.aggr = MeanAggregator()
        self.act = get_activation(activation, inplace=True) if activation is not None else None
        self.norm = get_norm_layer(norm, out_feats) if norm is not None else None
        self.residual = nn.Linear(in_features=in_feats, out_features=out_feats) if residual else None

    def forward(self, graph, x):
        out = self.aggr(graph, x)
        out = torch.cat([x, out], dim=-1)
        out = self.fc(out)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        if self.residual:
            out = out + self.residual(x)
        if self.dropout is not None:
            out = self.dropout(out)

        return out
    
    def embed(self, graph, x):
        out = self.aggr(graph, x)
        out = torch.cat([x, out], dim=-1)
        out = self.fc(out)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        if self.residual:
            out = out + self.residual(x)

        return out
    
    def embed_no_prop(self, graph, x):
        out = x
        out = torch.cat([x, out], dim=-1)
        out = self.fc(out)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        if self.residual:
            out = out + self.residual(x)

        return out
    

class GraphSAGE(BaseModel):
    def __init__(
        self, in_feats, hidden_dim, out_feats, num_layers, dropout=0.5, normalize=False, use_bn=False
    ):
        super(GraphSAGE, self).__init__()
        self.convlist = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        if num_layers == 1:
            self.convlist.append(SAGELayer(in_feats, out_feats, normalize))
        else:
            self.convlist.append(SAGELayer(in_feats, hidden_dim, normalize))
            if use_bn:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.convlist.append(SAGELayer(hidden_dim, hidden_dim, normalize))
                if use_bn:
                    self.bn_list.append(nn.BatchNorm1d(hidden_dim))
            self.convlist.append(SAGELayer(hidden_dim, out_feats, normalize))

    def forward(self, graph):
        h = graph.x
        for i in range(self.num_layers - 1):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.convlist[i](graph, h)
            if self.use_bn:
                h = self.bn_list[i](h)
        return self.convlist[self.num_layers - 1](graph, h)
    
    def embed(self, graph):
        h = graph.x
        for i in range(self.num_layers - 1):
            h = self.convlist[i].embed(graph, h)
            if self.use_bn:
                h = self.bn_list[i](h)
        return self.convlist[self.num_layers - 1].embed(graph, h)
    
    def embed_without_prop(self, graph, i_layer):
        h = graph.x
        for i in range(self.num_layers - 1):
            h = self.convlist[i].embed_no_prop(graph, h)
            if self.use_bn:
                h = self.bn_list[i](h)
        return self.convlist[self.num_layers - 1].embed_no_prop(graph, h)
    
    def embed_layer(self, graph, i_layer):
        h = graph.x
        if i_layer != self.num_layers - 1:
            h = self.convlist[i_layer].embed(graph, h)
            if self.use_bn:
                h = self.bn_list[i_layer](h)
        else:
            h = self.convlist[i_layer].embed(graph, h)
        return h
    
    def embed_without_prop_layer(self, graph, i_layer):
        h = graph.x
        if i_layer != self.num_layers - 1:
            h = self.convlist[i_layer].embed_no_prop(graph, h)
            if self.use_bn:
                h = self.bn_list[i_layer](h)
        else:
            h = self.convlist[i_layer].embed_no_prop(graph, h)
        return h