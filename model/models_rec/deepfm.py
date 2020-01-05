from model.layers import *


class DeepFM(nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.mul_input_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.mul_input_dim, mlp_dims, dropout)

    def forward(self, x):
        x = x.type(torch.LongTensor)
        embed_x = self.embedding(x)
        combine_x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.mul_input_dim))
        return  torch.sigmoid(combine_x.squeeze(1))