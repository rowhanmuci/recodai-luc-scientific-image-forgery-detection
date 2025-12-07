"""
改進版訓練腳本 - ResNet34-UNet with SCSE
參考組員的成功架構，目標提升到 0.28+ 分數

主要改進：
1. ResNet34-UNet 架構 + SCSE 注意力機制
2. Weighted BCE + Dice Loss
3. Self Copy-Move 資料增強
4. 更多 epochs (30-50)
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms, models
from PIL import Image
import cv2
from tqdm import tqdm
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet-UNet with SCSE')
    parser.add_argument('--data_root', type=str, default='/home/muci/forgery_detection/data')
    parser.add_argument('--output_dir', type=str, default='./outputs_improved')
    parser.add_argument('--backbone', type=str, default='resnet34', choices=['resnet34', 'resnet50', 'resnet101'],
                        help='Backbone architecture')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pos_weight', type=float, default=5.0, help='Positive class weight for BCE')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_copy_move_aug', action='store_true', help='Use self copy-move augmentation')
    parser.add_argument('--include_authentic', action='store_true', help='Include authentic images in training')
    parser.add_argument('--include_supplemental', action='store_true', help='Include supplemental images in training')
    parser.add_argument('--authentic_ratio', type=float, default=0.3, help='Ratio of authentic images (0.3 = 30%)')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# SCSE 注意力模組
# ============================================================
class SCSEModule(nn.Module):
    """Spatial and Channel Squeeze & Excitation"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel SE
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        # Spatial SE
        self.sse = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        cse = self.cse(x) * x
        sse = self.sse(x) * x
        return cse + sse

# ============================================================
# ResNet-UNet with SCSE (支援 ResNet34/50/101)
# ============================================================
class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_scse=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvRelu(in_channels + skip_channels, out_channels)
        self.conv2 = ConvRelu(out_channels, out_channels)
        self.use_scse = use_scse
        if use_scse:
            self.scse = SCSEModule(out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # 處理尺寸不匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.use_scse:
            x = self.scse(x)
        
        return x

class ResNetUNet(nn.Module):
    """ResNet-UNet with SCSE attention, supports ResNet34/50/101"""
    
    # 不同 backbone 的通道數
    CHANNELS = {
        'resnet34': [64, 64, 128, 256, 512],
        'resnet50': [64, 256, 512, 1024, 2048],
        'resnet101': [64, 256, 512, 1024, 2048],
    }
    
    def __init__(self, backbone='resnet34', out_channels=1, pretrained=True):
        super().__init__()
        
        self.backbone_name = backbone
        channels = self.CHANNELS[backbone]
        
        # 載入預訓練 backbone
        if backbone == 'resnet34':
            resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        
        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # channels[0], stride 2
        
        self.pool = resnet.maxpool  # stride 2
        self.encoder1 = resnet.layer1  # channels[1]
        self.encoder2 = resnet.layer2  # channels[2]
        self.encoder3 = resnet.layer3  # channels[3]
        self.encoder4 = resnet.layer4  # channels[4]
        
        # Decoder with SCSE
        self.decoder4 = DecoderBlock(channels[4], channels[3], 256)
        self.decoder3 = DecoderBlock(256, channels[2], 128)
        self.decoder2 = DecoderBlock(128, channels[1], 64)
        self.decoder1 = DecoderBlock(64, channels[0], 64)
        
        # Final
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)     # [B, 64, H/2, W/2]
        e0_pool = self.pool(e0)   # [B, 64, H/4, W/4]
        e1 = self.encoder1(e0_pool)  # [B, C1, H/4, W/4]
        e2 = self.encoder2(e1)    # [B, C2, H/8, W/8]
        e3 = self.encoder3(e2)    # [B, C3, H/16, W/16]
        e4 = self.encoder4(e3)    # [B, C4, H/32, W/32]
        
        # Decoder
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)
        
        # Final
        out = self.final_upsample(d1)
        out = self.final_conv(out)
        
        return out

# ============================================================
# Loss Functions
# ============================================================
class WeightedBCEDiceLoss(nn.Module):
    """Weighted BCE + Dice Loss"""
    def __init__(self, pos_weight=5.0, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        
        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=torch.tensor([self.pos_weight]).to(pred.device)
        )
        
        # Dice Loss
        smooth = 1.0
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return self.bce_weight * bce + self.dice_weight * dice

# ============================================================
# Dataset
# ============================================================
class ForgeryDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=512, augment=True, use_copy_move=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment
        self.use_copy_move = use_copy_move
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def self_copy_move_aug(self, image, mask):
        """Self Copy-Move Augmentation"""
        h, w = image.shape[:2]
        
        # 隨機選擇區域大小
        region_h = random.randint(h // 8, h // 4)
        region_w = random.randint(w // 8, w // 4)
        
        # 隨機選擇源區域
        src_y = random.randint(0, h - region_h)
        src_x = random.randint(0, w - region_w)
        
        # 隨機選擇目標區域
        dst_y = random.randint(0, h - region_h)
        dst_x = random.randint(0, w - region_w)
        
        # 複製區域
        region = image[src_y:src_y+region_h, src_x:src_x+region_w].copy()
        
        # 隨機變換
        if random.random() > 0.5:
            region = cv2.flip(region, 1)  # 水平翻轉
        if random.random() > 0.5:
            region = cv2.flip(region, 0)  # 垂直翻轉
        
        # 貼上
        image[dst_y:dst_y+region_h, dst_x:dst_x+region_w] = region
        
        # 更新 mask
        new_mask = mask.copy()
        new_mask[dst_y:dst_y+region_h, dst_x:dst_x+region_w] = 1
        
        return image, new_mask
    
    def __getitem__(self, idx):
        # 載入圖像
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 載入 mask
        mask_path = self.mask_paths[idx]
        if mask_path is not None and Path(mask_path).exists():
            mask = np.load(str(mask_path))
            if mask.ndim == 3:
                mask = mask.max(axis=0)
            mask = (mask > 0).astype(np.float32)
        else:
            # authentic 圖像：mask 全為 0
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # 資料增強
        if self.augment:
            # Self Copy-Move（10% 機率）
            if self.use_copy_move and random.random() < 0.1:
                image, mask = self.self_copy_move_aug(image, mask)
            
            # 基本增強
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            
            # 顏色增強
            if random.random() > 0.5:
                # 亮度
                factor = random.uniform(0.8, 1.2)
                image = np.clip(image * factor, 0, 255).astype(np.uint8)
            
            # JPEG 壓縮模擬
            if random.random() > 0.7:
                quality = random.randint(70, 95)
                _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 轉換為 tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask

# ============================================================
# 評估指標
# ============================================================
def compute_dice(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    smooth = 1.0
    intersection = (pred_binary * target).sum()
    return (2. * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)

def compute_iou(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    smooth = 1.0
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# ============================================================
# 訓練函數
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device, accumulation_steps):
    model.train()
    total_loss = 0
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc='Training')
    
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': total_loss / (i + 1)})
    
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    for images, masks in tqdm(loader, desc='Validation'):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        pred = torch.sigmoid(outputs)
        dice = compute_dice(pred, masks)
        iou = compute_iou(pred, masks)
        
        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
    
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

# ============================================================
# 主函數
# ============================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 收集資料
    data_root = Path(args.data_root)
    forged_dir = data_root / 'train_images' / 'forged'
    authentic_dir = data_root / 'train_images' / 'authentic'
    mask_dir = data_root / 'train_masks'
    
    # 補充資料目錄（重要！）
    supp_img_dir = data_root / 'supplemental_images'
    supp_mask_dir = data_root / 'supplemental_masks'
    
    image_paths = []
    mask_paths = []
    
    # 1. 加入偽造圖像（有 mask）
    if forged_dir.exists():
        for img_path in forged_dir.glob('*'):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                mask_path = mask_dir / f"{img_path.stem}.npy"
                if mask_path.exists():
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
    
    n_forged = len(image_paths)
    print(f"Found {n_forged} forged images with masks")
    
    # 2. 加入補充資料（關鍵！）
    n_supp = 0
    if supp_img_dir.exists() and args.include_supplemental:
        for img_path in supp_img_dir.glob('*'):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                mask_path = supp_mask_dir / f"{img_path.stem}.npy"
                if mask_path.exists():
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    n_supp += 1
        print(f"Added {n_supp} supplemental images with masks")
    
    n_total_forged = len(image_paths)
    
    # 3. 加入真實圖像（mask 為 None，會生成全 0）
    if authentic_dir.exists() and args.include_authentic:
        authentic_images = list(authentic_dir.glob('*'))
        authentic_images = [p for p in authentic_images if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        # 根據 authentic_ratio 控制數量
        n_authentic = int(n_total_forged * args.authentic_ratio / (1 - args.authentic_ratio))
        n_authentic = min(n_authentic, len(authentic_images))
        
        random.shuffle(authentic_images)
        authentic_images = authentic_images[:n_authentic]
        
        for img_path in authentic_images:
            image_paths.append(img_path)
            mask_paths.append(None)  # authentic 沒有 mask
        
        print(f"Added {n_authentic} authentic images (mask=0)")
        print(f"Ratio: forged={n_forged}/{len(image_paths)} ({n_forged/len(image_paths)*100:.1f}%), "
              f"authentic={n_authentic}/{len(image_paths)} ({n_authentic/len(image_paths)*100:.1f}%)")
    
    print(f"Total training images: {len(image_paths)}")
    
    # 分割訓練/驗證
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    val_size = int(len(indices) * args.val_ratio)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_images = [image_paths[i] for i in train_indices]
    train_masks = [mask_paths[i] for i in train_indices]
    val_images = [image_paths[i] for i in val_indices]
    val_masks = [mask_paths[i] for i in val_indices]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # 建立 Dataset
    train_dataset = ForgeryDataset(
        train_images, train_masks,
        image_size=args.image_size,
        augment=True,
        use_copy_move=args.use_copy_move_aug
    )
    val_dataset = ForgeryDataset(
        val_images, val_masks,
        image_size=args.image_size,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    # 建立模型
    model = ResNetUNet(backbone=args.backbone, out_channels=1, pretrained=True).to(device)
    print(f"Model: {args.backbone}-UNet with SCSE")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss 和優化器
    criterion = WeightedBCEDiceLoss(pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')
    
    # 訓練
    best_dice = 0
    history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.accumulation_steps
        )
        
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou
        })
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'backbone': args.backbone,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"✅ Best model saved! Dice: {best_dice:.4f}")
        
        # 保存最新模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'backbone': args.backbone,
            'args': vars(args)
        }, os.path.join(args.output_dir, 'last_model.pth'))
    
    # 保存訓練歷史
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, 'history.csv'), index=False)
    print(f"\n✅ Training complete! Best Dice: {best_dice:.4f}")

if __name__ == '__main__':
    main()