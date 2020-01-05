import torch as torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class FeaturesLinear(nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super(FeaturesLinear, self).__init__()
        print("field_dims: ", field_dims)
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        # accumulation add function to sparse the categories like:[1,3,4,7]==>[1,4,8,15]
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
          to change the category Serial number to ordered number
          like we got x = [2, 4] means category_1's id is 2, and category_2's id is 4
          assume field_dims like [3, 8], category_1 has 3 ids, category_2 has 8 ids. ==> offsets=[0, 3]
          x = [0 + 2, 4 + 3] ==> [2, 7]
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1)+self.bias


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


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CrossNetwork(nn.Module):

    def __init__(self, embed_dim, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.w = nn.ModuleList([
            nn.Linear(embed_dim, 1, bias=False) for _ in range(self.num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros((embed_dim,)), requires_grad=True) for _ in range(self.num_layers)
        ])

    def forward(self, x):
        """
        $y=x_0*x^'*w + b + x$
        $x_0$ means the origin x
        :param x:  (batch_size, embed_dim)
        """
        x0 = x
        for index in range(self.num_layers):
            x = x0 * self.w[index](x) + self.b[index](x) + x
        return x


class CompressedInteractionNetwork(nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        pred_dim, fc_input_dim = input_dim, 0
        for cross_layer_size in cross_layer_sizes:
            self.conv_layers.append(
                nn.Conv1d(input_dim*pred_dim, cross_layer_size, 1, stride=1, dilation=1, bias=True)
            )
            if self.split_half:
                cross_layer_size //= 2
            pred_dim = cross_layer_size
            fc_input_dim += pred_dim
        self.fc = nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        类似于RNN，不过这里上下层参数不同，同级处理最后合并各个结果
        :param x: (batch_size, num_fields, embed_dim)   128*1205*256
        :return:
        """
        xs = list()
        # x0:(num_field_dim, embed_dim, 1)
        x0, h = x.unsqueeze(2), x
        for index in range(self.num_layers):
            # x ==> (num_field_dim, embed_dim, 1) * (num_field_dim, 1, embed_dim)
            # x ==> (num_field_dim, embed_dim, embed_dim)
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim*fin_dim, embed_dim)
            x = F.relu(self.conv_layers[index](x))
            if self.split_half and index != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1]//2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))

