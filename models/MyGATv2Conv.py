import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import (
    softmax,
)

class MyGATv2Conv(GATv2Conv):
    def __init__(self, *args, **kwargs):
        edge_in_dim = kwargs["edge_in_dim"]
        kwargs.pop("edge_in_dim")
        self.out_channels = kwargs["out_channels"]
        self.heads = kwargs["heads"]
        super().__init__(*args, **kwargs)
        self.edge_upsampler = torch.nn.Linear(edge_in_dim, self.out_channels*self.heads)
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        return super().forward(x, edge_index, edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr,
                index: Tensor, ptr,
                size_i):
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # This line is the only addition to the original code!
        upsampled_edge_attr = self.edge_upsampler(edge_attr)
        upsampled_edge_attr = upsampled_edge_attr.reshape(-1, self.heads, self.out_channels)
        o_j = x_j + upsampled_edge_attr
        return o_j * alpha.unsqueeze(-1)

class MyGATv2Conv2(GATv2Conv):
    def __init__(self, *args, **kwargs):
        self.edge_in_dim = kwargs["edge_in_dim"]
        kwargs.pop("edge_in_dim")
        self.out_channels = kwargs["out_channels"]
        self.heads = kwargs["heads"]
        kwargs["edge_dim"] = self.edge_in_dim
        super().__init__(*args, **kwargs)
        self.edge_upsampler = torch.nn.Linear(self.edge_in_dim, self.out_channels*self.heads)
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        return super().forward(x, edge_index, edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr,
                index: Tensor, ptr,
                size_i):
        edge_attr2 = edge_attr[:, self.edge_in_dim:]
        edge_attr = edge_attr[:, :self.edge_in_dim]
        x = x_i + x_j
        if edge_attr2 is not None:
            if edge_attr2.dim() == 1:
                edge_attr2 = edge_attr2.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr2 = self.lin_edge(edge_attr2)
            edge_attr2 = edge_attr2.view(-1, self.heads, self.out_channels)
            x = x + edge_attr2
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # This line is the only addition to the original code!
        upsampled_edge_attr = self.edge_upsampler(edge_attr)
        upsampled_edge_attr = upsampled_edge_attr.reshape(-1, self.heads, self.out_channels)
        o_j = x_j + upsampled_edge_attr
        return o_j * alpha.unsqueeze(-1)
