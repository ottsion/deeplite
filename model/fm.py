import torch.nn.functional as F
from base import BaseModel
import torch as torch
import torch.nn as nn


class FM(BaseModel):

    def __init__(self, field_dims=None, embed_dim=None):
        super().__init__()


    def forward(self, x):
        x = x.transpose(1, 2)

        output = self.fm_layer(x)
        return output.squeeze(1)
