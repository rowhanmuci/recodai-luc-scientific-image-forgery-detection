"""
訓練腳本
Scientific Image Forgery Detection

使用方法:
    python train.py --model unet --encoder resnet50 --epochs 50
    python train.py --model unetpp --encoder efficientnet-b4 --loss focal_dice
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import segmentation_models_pytorch as smp

from dataset import (
    ForgeryDataset, 
    get_train_transforms, 
    get_val_transforms,
    create_dataloaders
)
from losses import get_loss_function
from utils import (
    set_seed, get_device, AverageMeter, 
    batch_metrics, save_checkpoint, load_checkpoint,
    EarlyStopping, visualize_predictions, plot_training_history
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Forgery Detection Model')
    
    # 資料路徑
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='資料目錄路徑')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='輸出目錄')
    
    # 模型配置
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'unetpp', 'deeplabv3', 'deeplabv3p', 'fpn', 'pspnet', 'manet'],
                       help='分割模型架構')
    parser.add_argument('--encoder', type=str, default='resnet50',
                       help='Encoder backbone (resnet50, efficientnet-b4, etc.)')
    parser.add_argument('--pretrained', type=str, default='imagenet',
                       help='預訓練權重')
    
    # 訓練配置
    parser.add_argument('--epochs', type=int, default=50,
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='學習率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='權重衰減')
    parser.add_argument('--image_size', type=int, default=512,
                       help='輸入圖像大小')
    
    # 損失函數
    parser.add_argument('--loss', type=str, default='bce_dice',
                       choices=['dice', 'focal', 'tversky', 'bce_dice', 'focal_dice', 'bce'],
                       help='損失函數')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='驗證集比例')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    parser.add_argument('--use_supplemental', action='store_true',
                       help='是否使用補充資料')
    parser.add_argument('--resume', type=str, default=None,
                       help='從檢查點恢復訓練')
    
    return parser.parse_args()


def get_model(model_name, encoder_name, pretrained='imagenet', in_channels=3, classes=1):
    """
    獲取分割模型
    
    可用模型:
    - unet: U-Net
    - unetpp: U-Net++
    - deeplabv3: DeepLabV3
    - deeplabv3p: DeepLabV3+
    - fpn: Feature Pyramid Network
    - pspnet: Pyramid Scene Parsing Network
    - manet: Multi-scale Attention Net
    """
    model_dict = {
        'unet': smp.Unet,
        'unetpp': smp.UnetPlusPlus,
        'deeplabv3': smp.DeepLabV3,
        'deeplabv3p': smp.DeepLabV3Plus,
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
        'manet': smp.MAnet,
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model_dict[model_name](
        encoder_name=encoder_name,
        encoder_weights=pretrained,
        in_channels=in_channels,
        classes=classes,
    )
    
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """訓練一個 epoch"""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(loader, desc='Training', leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # 混合精度訓練
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    return losses.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    """驗證"""
    model.eval()
    losses = AverageMeter()
    all_metrics = {
        'precision': [], 'recall': [], 'f1': [],
        'iou': [], 'dice': [], 'accuracy': []
    }
    
    pbar = tqdm(loader, desc='Validation', leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        losses.update(loss.item(), images.size(0))
        
        # 計算指標
        preds = torch.sigmoid(outputs)
        metrics = batch_metrics(preds, masks)
        
        for k, v in metrics.items():
            all_metrics[k].append(v)
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'dice': f'{np.mean(all_metrics["dice"]):.4f}'
        })
    
    # 計算平均指標
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    avg_metrics['loss'] = losses.avg
    
    return avg_metrics


def main():
    args = parse_args()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 設備
    device = get_device()
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 資料路徑 - 支援新的目錄結構 (train_images/forged/)
    data_dir = Path(args.data_dir)
    
    # 檢查目錄結構
    if (data_dir / 'train_images' / 'forged').exists():
        # 新結構：只用 forged 圖像訓練分割模型
        train_image_dir = data_dir / 'train_images' / 'forged'
        print("📂 Using forged images only for segmentation training")
    else:
        # 舊結構
        train_image_dir = data_dir / 'train_images'
    
    train_mask_dir = data_dir / 'train_masks'
    
    # 如果使用補充資料，合併路徑
    if args.use_supplemental:
        supp_image_dir = data_dir / 'supplemental_images'
        supp_mask_dir = data_dir / 'supplemental_masks'
        print("Using supplemental data for training")
    
    # 創建 DataLoader
    print("\n📦 Creating DataLoaders...")
    image_size = (args.image_size, args.image_size)
    
    train_loader, val_loader = create_dataloaders(
        train_image_dir=train_image_dir,
        train_mask_dir=train_mask_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split
    )
    
    # 創建模型
    print(f"\n🔧 Creating model: {args.model} with {args.encoder} encoder...")
    model = get_model(
        model_name=args.model,
        encoder_name=args.encoder,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 損失函數
    print(f"\n📉 Loss function: {args.loss}")
    criterion = get_loss_function(args.loss)
    
    # 優化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 學習率調度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # 混合精度訓練
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 早停
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # 訓練歷史
    history = {
        'train_loss': [], 'val_loss': [],
        'val_dice': [], 'val_iou': [], 'val_f1': []
    }
    
    # 從檢查點恢復
    start_epoch = 0
    best_score = 0
    
    if args.resume:
        start_epoch, best_score = load_checkpoint(args.resume, model, optimizer)
    
    # 開始訓練
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # 訓練
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # 驗證
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 更新學習率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 記錄歷史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_f1'].append(val_metrics['f1'])
        
        # 打印結果
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f} | "
              f"IoU: {val_metrics['iou']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # 保存最佳模型
        current_score = val_metrics['dice']
        if current_score > best_score:
            best_score = current_score
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'args': vars(args)
            }, output_dir / 'best_model.pth')
            print(f"✅ New best model! Dice: {best_score:.4f}")
        
        # 定期保存檢查點
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'args': vars(args)
            }, output_dir / f'checkpoint_epoch{epoch+1}.pth')
        
        # 早停檢查
        if early_stopping(current_score):
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break
    
    # 保存最終模型
    save_checkpoint({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_score': best_score,
        'args': vars(args)
    }, output_dir / 'final_model.pth')
    
    # 繪製訓練歷史
    plot_training_history(history, output_dir / 'training_history.png')
    
    print("\n" + "=" * 60)
    print(f"🎉 Training complete!")
    print(f"   Best Dice Score: {best_score:.4f}")
    print(f"   Models saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()