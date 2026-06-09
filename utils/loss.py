import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FocalDiceLoss",
]


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(2, logits.ndim))
        inter = (probs * targets).sum(dims)
        den = probs.sum(dims) + targets.sum(dims)
        dice_score = (2.0 * inter + self.eps) / (den + self.eps)
        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        logpt = -self.bce_criterion(logits, targets)
        pt = torch.exp(logpt)
        focal_term = (1 - pt).pow(self.gamma)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * focal_term * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalDiceLoss(nn.Module):
    def __init__(
            self,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            dice_eps: float = 1e-6,
            focal_weight: float = 0.25,
            dice_weight: float = 0.75,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(eps=dice_eps)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        return total_loss
