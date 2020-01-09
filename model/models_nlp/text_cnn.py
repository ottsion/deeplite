import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from base import BaseModel


class TextCNN(BaseModel):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, dropout, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_filters * len(filter_sizes), n_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x[:, 0, :].unsqueeze(1)  # only choose word level
        x = self.embedding(x)   # bath_size, words_length, embed_dim
        list_x = [self.conv_and_pool(x, conv) for conv in self.convs]
        x = torch.cat(list_x, dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

