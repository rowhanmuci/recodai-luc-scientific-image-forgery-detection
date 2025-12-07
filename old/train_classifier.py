"""
分類訓練腳本
Scientific Image Forgery Detection

任務: 二分類 (authentic vs forged)

使用方法:
    python train_classifier.py --model efficientnet_b0 --epochs 30
    python train_classifier.py --model resnet50 --epochs 50 --mixup
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import timm

from dataset_classification import (
    ForgeryClassificationDataset,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms
)
from utils import (
    set_seed, get_device, AverageMeter,
    save_checkpoint, load_checkpoint, EarlyStopping,
    plot_training_history
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Forgery Classifier')
    
    # 資料路徑
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    # 模型配置
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                       help='Model name from timm (efficientnet_b0, resnet50, convnext_tiny, etc.)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # 訓練配置
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=384)
    
    # 增強技巧
    parser.add_argument('--mixup', action='store_true', help='Use MixUp augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--resume', type=str, default=None)
    
    return parser.parse_args()


class ForgeryClassifier(nn.Module):
    """
    偽造檢測分類器
    
    使用 timm 預訓練模型作為 backbone
    """
    def __init__(self, model_name='efficientnet_b0', num_classes=2, 
                 pretrained=True, dropout=0.3):
        super().__init__()
        
        # 載入預訓練模型
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分類頭
            global_pool='avg'
        )
        
        # 獲取特徵維度
        self.num_features = self.backbone.num_features
        
        # 自定義分類頭
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x):
        """獲取特徵向量 (用於分析)"""
        return self.backbone(x)


def mixup_data(x, y, alpha=0.2):
    """MixUp 資料增強"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp 損失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, 
                    scaler=None, use_mixup=False, mixup_alpha=0.2):
    """訓練一個 epoch"""
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # MixUp
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
        
        # 前向傳播
        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
        # 統計
        losses.update(loss.item(), images.size(0))
        
        if not use_mixup:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{100.*correct/max(total,1):.1f}%' if total > 0 else 'N/A'
        })
    
    return losses.avg, correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """驗證"""
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc='Validation', leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        losses.update(loss.item(), images.size(0))
        
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # P(forged)
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    # 計算詳細指標
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = correct / total
    
    # 計算每個類別的指標
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': losses.avg,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }
    
    return metrics


def main():
    args = parse_args()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 設備
    device = get_device()
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建 DataLoader
    print("\n📦 Creating DataLoaders...")
    image_size = (args.image_size, args.image_size)
    
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed
    )
    
    # 創建模型
    print(f"\n🔧 Creating model: {args.model}...")
    model = ForgeryClassifier(
        model_name=args.model,
        num_classes=2,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 損失函數 (帶 label smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # 優化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 學習率調度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 早停
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # 訓練歷史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_f1': [], 'val_auc': []
    }
    
    # 從檢查點恢復
    start_epoch = 0
    best_score = 0
    
    if args.resume:
        start_epoch, best_score = load_checkpoint(args.resume, model, optimizer)
    
    # 開始訓練
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    if args.mixup:
        print(f"   Using MixUp with alpha={args.mixup_alpha}")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # 訓練
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_mixup=args.mixup, mixup_alpha=args.mixup_alpha
        )
        
        # 更新學習率
        if not isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # 驗證
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 記錄歷史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # 打印結果
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.1f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']*100:.1f}% | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f}")
        print(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")
        
        # 保存最佳模型
        current_score = val_metrics['f1']  # 使用 F1 作為主要指標
        if current_score > best_score:
            best_score = current_score
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'args': vars(args)
            }, output_dir / 'best_classifier.pth')
            print(f"✅ New best model! F1: {best_score:.4f}")
        
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
    }, output_dir / 'final_classifier.pth')
    
    # 繪製訓練歷史
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    
    axes[2].plot(history['val_f1'], label='F1')
    axes[2].plot(history['val_auc'], label='AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Validation Metrics')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    plt.close()
    
    print("\n" + "=" * 60)
    print(f"🎉 Training complete!")
    print(f"   Best F1 Score: {best_score:.4f}")
    print(f"   Models saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()