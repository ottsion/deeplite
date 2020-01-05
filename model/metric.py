import torch
import torch.nn as nn

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy_sigmod(output, target):
    with torch.no_grad():
        pred = output.ge(0.5)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def loss_all(output, target, loss_fn=nn.MSELoss()):
    loss_all = 0.0
    loss = loss_fn(output, target)
    loss_all += loss.item()
    return loss_all

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
