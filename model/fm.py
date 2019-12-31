import torch.nn.functional as F
from base import BaseModel
import torch as torch
import torch.nn as nn


class FM(BaseModel):

    def __init__(self, field_dims=None, embed_dim=None):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(field_dims, embed_dim), requires_grad=True)
        self.lin = nn.Linear(field_dims, 1)

    def forward(self, x):
        out_lin = self.lin(x)
        out_1 = torch.matmul(x, self.V).pow(2).sum(dim=1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(dim=1, keepdim=True)  # S_2
        # $\frac{1}{2}\sum_{k=1}^{K}[(\sum_{i=1}^{n}v_{ik}x_i)^2-\sum_{i=1}^{n}v_{ik}^2x_i^2]$
        out_inter = 0.5 * (out_1 - out_2)
        out = out_inter + out_lin

        return out