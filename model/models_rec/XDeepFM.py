from model.layers import *


class XDeepFM(nn.Module):

    def __init__(self, field_dims, embed_dim, cross_layer_sizes, mlp_dims, dropout, split_half):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.embed_input_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_input_dim, mlp_dims, dropout)

    def forward(self, x):
        linear_x = self.linear(x)
        embed_x = self.embedding(x)
        cin_x = self.cin(embed_x)
        mlp_x = self.mlp(embed_x.view(-1, self.embed_input_dim))
        stack_x = linear_x + cin_x + mlp_x
        return F.sigmoid(stack_x.squeeze(1))
