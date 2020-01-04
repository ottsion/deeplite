import torch
from torch import nn
import torch.nn.functional as F
from model.layers import *


class DeepAndCrossNetworkModel(nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_hidden_dims, num_layers, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.mlp_input_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.mlp_input_dim, mlp_hidden_dims, dropout, output_layer=False)
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.linear = FeaturesLinear(mlp_hidden_dims[-1]+embed_dim)


    def forward(self, x):
        embed_x = self.embedding(x)
        mlp_x = self.mlp(embed_x)
        cross_x = self.cn(embed_x)
        stack_x = torch.cat([cross_x, mlp_x], dim=1)
        return F.sigmoid(self.linear(stack_x).squeeze(1))
