"""Long-tail objectives for Phase D (module M2).

plain CE / weighted CE / focal (gamma 2) / class-balanced (effective number,
beta 0.9999) / weighted sampler. The first four are losses, the fifth changes
the sampler and keeps plain CE.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_counts(labels, num_classes):
    return np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)


def inverse_frequency_weights(labels, num_classes):
    """w_c = N / (K * n_c), the usual balanced-CE weighting."""
    n = class_counts(labels, num_classes)
    n = np.maximum(n, 1.0)
    return torch.tensor(n.sum() / (num_classes * n), dtype=torch.float32)


def class_balanced_weights(labels, num_classes, beta=0.9999):
    """Cui et al. 2019 effective number of samples; normalised to mean 1."""
    n = np.maximum(class_counts(labels, num_classes), 1.0)
    eff = (1.0 - np.power(beta, n)) / (1.0 - beta)
    w = 1.0 / eff
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[target]
        return loss.mean()


def build_criterion(kind, labels, num_classes, device, label_smoothing=0.0):
    kind = (kind or "ce").lower()
    ls = float(label_smoothing or 0.0)
    if kind in ("ce", "plain_ce", "sampler"):
        return nn.CrossEntropyLoss(label_smoothing=ls), None
    if kind == "weighted_ce":
        w = inverse_frequency_weights(labels, num_classes).to(device)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls), w.tolist()
    if kind == "focal":
        return FocalLoss(gamma=2.0), None
    if kind in ("cb", "class_balanced"):
        w = class_balanced_weights(labels, num_classes).to(device)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls), w.tolist()
    raise ValueError(f"unknown loss {kind!r}")


def weighted_sampler(labels, num_classes):
    from torch.utils.data import WeightedRandomSampler
    n = np.maximum(class_counts(labels, num_classes), 1.0)
    per_sample = (1.0 / n)[np.asarray(labels)]
    return WeightedRandomSampler(weights=torch.tensor(per_sample, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)
