import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from base import BaseModel


class FastText(BaseModel):
    def __init__(self, vocab_size, gramn2_vocab_size, embed_dim, hidden_dims, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding2 = nn.Embedding(gramn2_vocab_size, embed_dim)
        layers = list()
        input_dim = embed_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        word_list = x[:, 0, :]
        gram_list = x[:, 1, :]
        embed_x1 = self.embedding(word_list)
        embed_x2 = self.embedding2(gram_list)
        x = torch.cat((embed_x1, embed_x2), dim=-1)
        x = x.mean(dim=1, keepdim=True)
        x = self.fc(x)
        return F.log_softmax(x.squeeze(1), dim=1)
