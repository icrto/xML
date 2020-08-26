import torch
import sys

eps = 1e-6  # additive constant to avoid divisions by 0


def unsupervised_explanation_loss(pred, beta):
    """unsupervised_explanation_loss calculates the unsupervised explanation loss for a single prediction

    Arguments:
        pred {tensor CHW} -- predicted explanation
        beta {float} -- beta hyperparameter (weighs each loss element)

    Returns:
        float -- loss value for a single prediction
    """
    return beta * torch.mean(torch.abs(pred)) + (1 - beta) * (
        torch.mean(torch.abs(pred[:-1] - pred[1:]))
        + torch.mean(torch.abs(pred[:, :-1] - pred[:, 1:]))
    )


def hybrid_explanation_loss(pred, target, beta, gamma):
    """hybrid_explanation_loss calculates the hybrid explanation loss for a single prediction

    Arguments:
        pred {tensor CHW} -- predicted explanation
        target {tensor CHW} -- ground truth mask
        beta {float} -- beta hyperparameter (weighs each unsupervised loss element)
        gamma {float} -- gamma hyperparameter (weighs the weakly supervised loss element)

    Returns:
        float -- loss value for a single prediction
    """
    l1 = torch.mean(torch.abs(pred))
    tv = torch.mean(torch.abs(pred[:-1] - pred[1:])) + torch.mean(
        torch.abs(pred[:, :-1] - pred[:, 1:])
    )
    weakly = torch.abs(
        torch.div(torch.sum(torch.mul(1 - target, pred)), torch.sum(1 - target) + eps)
    )
    l = beta * l1 + (1 - beta) * tv + gamma * weakly
    return l


def batch_unsupervised_explanation_loss(preds, beta, reduction="mean"):
    """batch_unsupervised_explanation_loss calculates the unsupervised explanation loss for one mini-batch

    Arguments:
        preds {tensor CHW} -- predicted explanations
        beta {float} -- beta hyperparameter (weighs each unsupervised loss element)

    Keyword Arguments:
        reduction {str} -- reduction to apply (either mean or sum of the mini-batch individual losses) (default: {'mean'})

    Returns:
        float -- loss value for the whole mini-batch either summed or averaged (see reduction)
    """
    loss = 0.0
    for pred in preds:
        loss += unsupervised_explanation_loss(pred.squeeze(), beta)

    if reduction == "mean":
        return loss / len(preds)
    elif reduction == "sum":
        return loss
    else:
        print("Invalid reduction value.")
        sys.exit(-1)


def batch_hybrid_explanation_loss(preds, targets, beta, gamma, reduction="mean"):
    """batch_hybrid_explanation_loss calculates the hybrid explanation loss for one mini-batch

    Arguments:
        preds {tensor CHW} -- predicted explanations
        targets {tensor CHW} -- ground truth masks
        beta {float} -- beta hyperparameter (weighs each unsupervised loss element)
        gamma {float} -- gamma hyperparameter (weighs the weakly supervised loss element)

    Keyword Arguments:
        reduction {str} -- reduction to apply (either mean or sum of the mini-batch individual losses) (default: {'mean'})

    Returns:
        float -- loss value for the whole mini-batch either summed or averaged (see reduction)
    """
    loss = 0.0
    for idx, pred in enumerate(preds):
        loss += hybrid_explanation_loss(pred.squeeze(), targets[idx][0], beta, gamma)

    if reduction == "mean":
        return loss / len(preds)
    elif reduction == "sum":
        return loss
    else:
        print("Invalid reduction value.")
        sys.exit(-1)
