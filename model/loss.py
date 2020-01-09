import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target, size_average=True)


def bce_with_log_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target, size_average=True)
