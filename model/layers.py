import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeaturesLinear(nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super(FeaturesLinear, self).__init__()
        print("field_dims: ", field_dims)
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)), requires_grad=True)
        # 按照所给定的轴参数返回元素的梯形累计和，axis=0，按照行累加。axis=1，按照列累加。axis不给定具体值，就把numpy数组当成一个一维数组。
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        print("x: ", x.shape)
        print("self.offsets: ", self.offsets)
        print("==: ", x.new_tensor(self.offsets))
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        print("x: ", x.shape)
        print(x)
        x = self.fc(x)
        return torch.sum(x, dim=1)+self.bias


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum=True):
        super(FactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
             $\frac{1}{2}\sum_{k=1}^{K}[(\sum_{i=1}^{n}v_{ik}x_i)^2-\sum_{i=1}^{n}v_{ik}^2x_i^2]$

        :param x: float tensor of size (batch_size, num_fields, embed_dim)
        :return:
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.arrat(0, *np.cumsum(field_dims)[:-1], dtype=np.long)
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields-1):
            for j in range(i+1, self.num_fields):
                ix.append(xs[j][:, j] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix