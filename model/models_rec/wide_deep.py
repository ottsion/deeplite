from torch import nn
from model.layers import *


class WeedAndDeepModel(nn.Module):
    """
        理论上此处输入两部分特征，用于先行部分的特征和用于神经网络的特征，
        此处简单的将两部分的特征合并为一个特征，并且所有特征离散化处理
    """

    def __init__(self, field_dims, embed_dim, mlp_hidden_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.mlp_input_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.mlp_input_dim, mlp_hidden_dims, dropout)

    def forward(self, x):
        embed_x = self.embedding(x)
        return self.linear(x) + self.mlp(embed_x.view(-1, self.mlp_input_dim))

