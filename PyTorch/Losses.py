from torch import nn
import torch
from torch.autograd import Function
import sys
eps = 1e-6

def unsupervised_explanation_loss(pred, beta):
    return beta*torch.mean(torch.abs(pred)) + (1-beta)*(torch.mean(torch.abs(pred[: -1] - pred[1:])) + torch.mean(torch.abs(pred[:, : -1] - pred[:, 1:])))

def weaklysupervised_explanation_loss(pred, target):
    return torch.abs(torch.div(torch.sum(torch.mul(1-target, pred))+eps, torch.sum(1-target)+eps))

def hybrid_explanation_loss(pred, target, beta1, beta2):
    beta3 = 1.0-beta1-beta2
    total = beta1+beta2+beta3
    l = ((beta1/total)*torch.mean(torch.abs(pred)) + 
    (beta2/total)*(torch.mean(torch.abs(pred[: -1] - pred[1:])) + torch.mean(torch.abs(pred[:, : -1] - pred[:, 1:]))) +
    (beta3/total)*torch.abs(torch.div(torch.sum(torch.mul(1-target, pred)+eps), torch.sum(1-target)+eps)))
    return l

def batch_unsupervised_explanation_loss(preds, beta, reduction='mean'):
    loss = 0.0
    for idx, pred in enumerate(preds):
        loss += unsupervised_explanation_loss(pred.squeeze(), beta)

    if(reduction == 'mean'):
        return loss / len(preds)
    elif(reduction == 'sum'):
        return loss
    else:
        print('Invalid reduction value.')
        sys.exit(-1)


def batch_weaklysupervised_explanation_loss(preds, targets, reduction='mean'):
    loss = 0.0
    for idx, pred in enumerate(preds):
        loss += weaklysupervised_explanation_loss(pred.squeeze(), targets[idx][0])

    if(reduction == 'mean'):
        return loss / len(preds)
    elif(reduction == 'sum'):
        return loss
    else:
        print('Invalid reduction value.')
        sys.exit(-1)

def batch_hybrid_explanation_loss(preds, targets, beta1, beta2, reduction='mean'):
    loss = 0.0
    for idx, pred in enumerate(preds):
        loss += hybrid_explanation_loss(pred.squeeze(), targets[idx][0], beta1, beta2)

    if(reduction == 'mean'):
        return loss / len(preds)
    elif(reduction == 'sum'):
        return loss
    else:
        print('Invalid reduction value.')
        sys.exit(-1)

