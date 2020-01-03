from model.layers import *


class DeepFM(nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(sum(field_dims))
        self.fm = FactorizationMachine(reduce_sum=True)
        self.mul_input_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.mul_input_dim, mlp_dims, dropout)

    def forward(self, x):
        embed_x = self.embedding(x)
        return self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.mul_input_dim))
