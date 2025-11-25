"""
輔助工具函數
Scientific Image Forgery Detection
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import cv2


def set_seed(seed=42):
    """設置隨機種子確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """獲取可用設備"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class AverageMeter:
    """計算和存儲平均值和當前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(pred, target, threshold=0.5):
    """
    計算分割指標
    
    Args:
        pred: 預測 (sigmoid 後, 0-1)
        target: 真實標籤 (0 或 1)
        threshold: 二值化閾值
    
    Returns:
        dict: 包含各種指標
    """
    pred_binary = (pred > threshold).float()
    
    # 展平
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # 計算 TP, FP, FN, TN
    TP = (pred_flat * target_flat).sum().item()
    FP = (pred_flat * (1 - target_flat)).sum().item()
    FN = ((1 - pred_flat) * target_flat).sum().item()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    # 計算指標
    eps = 1e-7
    
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    # IoU (Jaccard Index)
    iou = TP / (TP + FP + FN + eps)
    
    # Dice Coefficient
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    
    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    }


def batch_metrics(pred, target, threshold=0.5):
    """
    計算批次的平均指標
    
    Args:
        pred: [B, 1, H, W] 預測
        target: [B, 1, H, W] 標籤
        threshold: 二值化閾值
    
    Returns:
        dict: 批次平均指標
    """
    batch_size = pred.shape[0]
    metrics_sum = {
        'precision': 0, 'recall': 0, 'f1': 0,
        'iou': 0, 'dice': 0, 'accuracy': 0
    }
    
    for i in range(batch_size):
        m = calculate_metrics(pred[i], target[i], threshold)
        for k in metrics_sum.keys():
            metrics_sum[k] += m[k]
    
    return {k: v / batch_size for k, v in metrics_sum.items()}


def save_checkpoint(state, filepath):
    """保存檢查點"""
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """加載檢查點"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_score = checkpoint.get('best_score', 0)
    
    print(f"Checkpoint loaded: {filepath} (epoch {epoch})")
    return epoch, best_score


class EarlyStopping:
    """早停機制"""
    def __init__(self, patience=7, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


def visualize_predictions(images, masks, predictions, num_samples=4, 
                         save_path=None, denormalize=True):
    """
    視覺化預測結果
    
    Args:
        images: [B, C, H, W] 輸入圖像
        masks: [B, 1, H, W] 真實遮罩
        predictions: [B, 1, H, W] 預測遮罩
        num_samples: 顯示樣本數
        save_path: 保存路徑
        denormalize: 是否反正規化圖像
    """
    # ImageNet 標準化參數
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # 獲取當前樣本
        img = images[i].cpu()
        mask = masks[i].cpu().squeeze()
        pred = predictions[i].cpu().squeeze()
        
        # 反正規化圖像
        if denormalize:
            img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        mask = mask.numpy()
        pred = (pred > 0.5).float().numpy()
        
        # 創建差異圖
        diff = np.zeros((*mask.shape, 3))
        diff[..., 0] = (pred > 0) & (mask == 0)  # FP: 紅色
        diff[..., 1] = (pred > 0) & (mask > 0)   # TP: 綠色
        diff[..., 2] = (pred == 0) & (mask > 0)  # FN: 藍色
        
        # 顯示
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(diff)
        axes[i, 3].set_title("Diff (R:FP, G:TP, B:FN)")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    繪製訓練歷史
    
    Args:
        history: dict with keys like 'train_loss', 'val_loss', 'val_dice', etc.
        save_path: 保存路徑
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics
    if 'val_dice' in history:
        axes[1].plot(history['val_dice'], label='Val Dice', marker='o')
    if 'val_iou' in history:
        axes[1].plot(history['val_iou'], label='Val IoU', marker='o')
    if 'val_f1' in history:
        axes[1].plot(history['val_f1'], label='Val F1', marker='o')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.close()


def rle_encode(mask):
    """
    Run-length encoding for masks
    
    Args:
        mask: binary mask [H, W] 
    
    Returns:
        RLE string
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(rle_string, shape):
    """
    Decode RLE string to mask
    
    Args:
        rle_string: RLE encoded string
        shape: (H, W) of the mask
    
    Returns:
        binary mask [H, W]
    """
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros(shape, dtype=np.uint8)
    
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    
    return mask.reshape(shape)


def resize_mask(mask, target_size):
    """
    調整遮罩大小
    
    Args:
        mask: numpy array [H, W]
        target_size: (H, W)
    
    Returns:
        調整後的遮罩
    """
    return cv2.resize(mask.astype(np.float32), 
                     (target_size[1], target_size[0]), 
                     interpolation=cv2.INTER_NEAREST)


if __name__ == "__main__":
    # 測試函數
    print("Testing utility functions...")
    
    # 測試指標計算
    pred = torch.rand(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    metrics = batch_metrics(pred, target)
    print("\nBatch metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 測試 RLE
    mask = np.random.randint(0, 2, (256, 256)).astype(np.uint8)
    rle = rle_encode(mask)
    decoded = rle_decode(rle, (256, 256))
    print(f"\nRLE test - Original == Decoded: {np.allclose(mask, decoded)}")
    
    print("\nAll utility functions work correctly!")
