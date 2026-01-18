import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FocalDiceLoss",
]


class DiceLoss(nn.Module):
    """基础 Dice Loss，用于计算损失值"""

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
    """基础 Focal Loss"""

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
    """
    选定Dice Loss 和 Focal Loss 的权重
    """

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


# ---------- 快速自测 ----------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, C, H, W = 4, 1, 128, 128

    criterion = FocalDiceLoss()  # 使用默认参数，权重都固定为0.5

    # 场景1：训练初期，模型预测很差 (随机噪声)
    print("--- 场景1: 训练初期，预测效果差 ---")
    logits_early = torch.randn(N, C, H, W, device=device)
    target = (torch.rand(N, C, H, W, device=device) > 0.8).float()  # 模拟不均衡标签
    loss_early = criterion(logits_early, target)

    print(f"总损失: {loss_early.item():.4f}\n")

    # 场景2：训练后期，模型预测较好
    print("--- 场景2: 训练后期，预测效果好 ---")
    # 模拟一个比较好的预测，logits 大部分与 target 一致
    logits_late = torch.where(target == 1, 2.0, -2.0) + torch.randn(N, C, H, W, device=device) * 0.5
    loss_late = criterion(logits_late, target)

    print(f"总损失: {loss_late.item():.4f}")