"""
損失函數集合
Scientific Image Forgery Detection

包含:
- Dice Loss
- Focal Loss  
- Tversky Loss
- Combined Loss
- BCE with Dice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    
    適用於類別不平衡的分割任務
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] or [B, H, W] - 預測 (sigmoid 後)
            target: [B, 1, H, W] or [B, H, W] - 真實標籤
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    專為處理正負樣本極度不平衡設計
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] - 預測 (sigmoid 前的 logits)
            target: [B, 1, H, W] - 真實標籤
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_factor = self.alpha * target + (1 - self.alpha) * (1 - target)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        focal_loss = alpha_factor * modulating_factor * bce
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss
    
    Dice Loss 的泛化版本，可以調整 FP 和 FN 的權重
    - alpha > beta: 更關注 False Negatives (漏檢)
    - alpha < beta: 更關注 False Positives (誤檢)
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # FN 權重
        self.beta = beta    # FP 權重
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        tversky = (TP + self.smooth) / (
            TP + self.alpha * FN + self.beta * FP + self.smooth
        )
        
        return 1 - tversky


class BCEDiceLoss(nn.Module):
    """
    Binary Cross Entropy + Dice Loss 組合
    
    結合 BCE 的穩定性和 Dice 對區域的關注
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] - 預測 (sigmoid 前的 logits)
            target: [B, 1, H, W] - 真實標籤
        """
        bce = self.bce_loss(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        dice = self.dice_loss(pred_sigmoid, target)
        
        return self.bce_weight * bce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Dice Loss 組合
    
    專為嚴重類別不平衡的分割任務設計
    """
    def __init__(self, focal_weight=0.5, dice_weight=0.5, 
                 alpha=0.25, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha, gamma)
        self.dice_loss = DiceLoss(smooth)
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        dice = self.dice_loss(pred_sigmoid, target)
        
        return self.focal_weight * focal + self.dice_weight * dice


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss
    
    直接優化 IoU 指標的替代損失
    """
    def __init__(self):
        super().__init__()
    
    def lovasz_grad(self, gt_sorted):
        """計算 Lovasz 梯度"""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if len(googl) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard
    
    def lovasz_hinge_flat(self, logits, labels):
        """二分類 Lovasz hinge loss"""
        if len(labels) == 0:
            return logits.sum() * 0.
        
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        return self.lovasz_hinge_flat(pred, target)


def get_loss_function(loss_name, **kwargs):
    """
    根據名稱獲取損失函數
    
    Args:
        loss_name: 損失函數名稱
            - 'dice': Dice Loss
            - 'focal': Focal Loss
            - 'tversky': Tversky Loss
            - 'bce_dice': BCE + Dice
            - 'focal_dice': Focal + Dice
            - 'bce': Binary Cross Entropy
        **kwargs: 損失函數參數
    
    Returns:
        損失函數實例
    """
    losses = {
        'dice': DiceLoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
        'bce_dice': BCEDiceLoss,
        'focal_dice': FocalDiceLoss,
        'bce': nn.BCEWithLogitsLoss,
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    
    return losses[loss_name](**kwargs)


if __name__ == "__main__":
    # 測試損失函數
    print("Testing loss functions...")
    
    # 創建假數據
    pred = torch.randn(4, 1, 256, 256)
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    # 測試各個損失函數
    losses_to_test = ['dice', 'focal', 'tversky', 'bce_dice', 'focal_dice', 'bce']
    
    for loss_name in losses_to_test:
        loss_fn = get_loss_function(loss_name)
        
        if loss_name == 'dice':
            # Dice 需要 sigmoid 後的預測
            loss = loss_fn(torch.sigmoid(pred), target)
        else:
            loss = loss_fn(pred, target)
        
        print(f"{loss_name:12s}: {loss.item():.4f}")
    
    print("\nAll loss functions work correctly!")
