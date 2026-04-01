import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=256):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input: (B, C, H, W) logits
            target: (B, H, W) or (B, 1, H, W) labels
        """
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-2:] != target.shape[-2:]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

        return F.cross_entropy(input=input, target=target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, smooth=1e-5, ignore_index=256, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input: (B, C, H, W) logits
            target: (B, H, W) or (B, 1, H, W) labels
        Returns:
            loss: scalar if reduction='mean'/'sum', else (B,) tensor
        """
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-2:] != target.shape[-2:]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

        prob = F.softmax(input, dim=1)
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        target_one_hot = F.one_hot(target_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        mask = valid_mask.unsqueeze(1).float()

        if self.num_classes == 2:
            # 二分类：只处理前景类
            prob_fg = prob[:, 1:2, :, :]  # (B, 1, H, W)
            target_fg = target_one_hot[:, 1:2, :, :]

            # 应用mask
            prob_fg = prob_fg * mask
            target_fg = target_fg * mask

            # 计算每个样本的Dice
            intersection = (prob_fg * target_fg).view(prob_fg.size(0), -1).sum(dim=1)
            union = prob_fg.view(prob_fg.size(0), -1).sum(dim=1) + \
                    target_fg.view(target_fg.size(0), -1).sum(dim=1)

            # 处理union为0的情况
            dice = torch.ones_like(intersection)
            valid = union > 0
            dice[valid] = (2. * intersection[valid] + self.smooth) / (union[valid] + self.smooth)
            loss = 1 - dice

        else:
            # 多分类：计算所有类别的Dice
            prob_masked = prob * mask
            target_masked = target_one_hot * mask

            # 计算每个样本、每个类别的intersection和union
            intersection = (prob_masked * target_masked).view(
                prob_masked.size(0), prob_masked.size(1), -1).sum(dim=2)
            union = prob_masked.view(prob_masked.size(0), prob_masked.size(1), -1).sum(dim=2) + \
                    target_masked.view(target_masked.size(0), target_masked.size(1), -1).sum(dim=2)

            # 计算每个类别的Dice
            dice = torch.ones_like(intersection)
            valid = union > 0
            dice[valid] = (2. * intersection[valid] + self.smooth) / (union[valid] + self.smooth)

            # 对每个样本，对所有类别取平均
            loss = 1 - dice.mean(dim=1)

        # 应用reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # reduction='none'时返回每个样本的损失

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for binary and multi-class segmentation
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, num_classes=2, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=256):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        Args:
            input: (B, C, H, W) logits
            target: (B, H, W) or (B, 1, H, W) labels
        Returns:
            loss: scalar if reduction='mean'/'sum', else (B,) tensor
        """
        # 维度处理
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        target = target.long()

        if input.shape[-2:] != target.shape[-2:]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)
        ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if self.num_classes == 2:
                alpha_t = torch.ones_like(target, dtype=torch.float)
                alpha_t[target == 1] = self.alpha
                alpha_t[target == 0] = 1 - self.alpha
                if self.ignore_index is not None:
                    alpha_t[target == self.ignore_index] = 0
                focal_loss = alpha_t * focal_loss
            else:
                if isinstance(self.alpha, (int, float)):
                    # 标量alpha：背景为1-alpha，前景为alpha
                    alpha_t = torch.ones_like(target, dtype=torch.float) * (1 - self.alpha)
                    for c in range(1, self.num_classes):
                        alpha_t[target == c] = self.alpha
                elif isinstance(self.alpha, (list, tuple, np.ndarray)):
                    # 列表alpha：每个类别单独设置
                    alpha_t = torch.zeros_like(target, dtype=torch.float)
                    for c in range(self.num_classes):
                        alpha_t[target == c] = self.alpha[c]
                else:
                    raise ValueError(f"Unsupported alpha type: {type(self.alpha)}")

                if self.ignore_index is not None:
                    alpha_t[target == self.ignore_index] = 0
                focal_loss = alpha_t * focal_loss

        # 计算每个样本的损失
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            # 每个样本的损失和
            batch_loss = (focal_loss * mask).view(focal_loss.size(0), -1).sum(dim=1)
            # 每个样本的有效像素数
            valid_pixels = mask.view(mask.size(0), -1).sum(dim=1)
            # 避免除以0
            batch_loss = batch_loss / (valid_pixels + 1e-8)
        else:
            batch_loss = focal_loss.view(focal_loss.size(0), -1).mean(dim=1)

        # 应用reduction
        if self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
        else:  # 'none'
            loss = batch_loss

        return loss

class JaccardLoss(nn.Module):
    def __init__(self, num_classes=2, smooth=1e-5, ignore_index=256, reduction='mean'):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input: (B, C, H, W) logits
            target: (B, H, W) or (B, 1, H, W) labels
        Returns:
            loss: scalar if reduction='mean'/'sum', else (B,) tensor
        """
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-2:] != target.shape[-2:]:
            input = F.interpolate(input, size=target.shape[-2:], mode='bilinear', align_corners=True)

        prob = F.softmax(input, dim=1)
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        # 将 target 转换为 one-hot 编码
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        mask = valid_mask.unsqueeze(1).float()

        intersection = (prob * target_one_hot * mask).sum(dim=[2, 3])
        union = (prob * mask).sum(dim=[2, 3]) + (target_one_hot * mask).sum(dim=[2, 3])

        jaccard = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - jaccard

        # 对每个类别取平均损失
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class CombinedLoss(nn.Module):
    def __init__(self, loss_type=None, loss_weight=None,
                 num_classes=2, smooth=1e-5, focal_alpha=0.25, focal_gamma=2.0,
                 ignore_index=256, reduction='mean'):
        super(CombinedLoss, self).__init__()
        if loss_type is None:
            loss_type = ['ce', 'dice', 'focal', 'iou']
        if loss_weight is None:
            loss_weight = [1.0, 0.5, 0.5, 0.5]
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.reduction = reduction

        assert len(self.loss_type) == len(self.loss_weight), \
            "The lengths of loss_type and loss_weight must be consistent."

        self.loss_map = nn.ModuleDict({
            'ce': CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction),
            'dice': DiceLoss(num_classes=num_classes, smooth=smooth,
                             ignore_index=ignore_index, reduction=reduction),
            'focal': FocalLoss(num_classes=num_classes, alpha=focal_alpha,
                               gamma=focal_gamma, reduction=reduction,
                               ignore_index=ignore_index),
            'iou': JaccardLoss(num_classes=num_classes, smooth=smooth,
                               ignore_index=ignore_index, reduction=reduction)
        })

    def forward(self, input, target):
        total_loss = 0.0

        for loss_name, weight in zip(self.loss_type, self.loss_weight):
            if loss_name not in self.loss_map:
                raise ValueError(f"Unsupported loss type: {loss_name}")
            total_loss = total_loss + weight * self.loss_map[loss_name](input, target)

        return total_loss



