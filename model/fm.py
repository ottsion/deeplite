import torch.nn.functional as F
from base import BaseModel
import torch as torch
import torch.nn as nn

from model.layers import *


class FM(BaseModel):

    def __init__(self, field_dims=None, embed_dim=None):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = self.linear(x) + self.fm(self.embedding(x))
        return x.squeeze(1)
