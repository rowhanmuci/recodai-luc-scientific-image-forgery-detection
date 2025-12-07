"""
分類推理腳本
Scientific Image Forgery Detection

生成提交檔案 (case_id, annotation)

使用方法:
    python inference_classifier.py --checkpoint outputs/best_classifier.pth
    python inference_classifier.py --checkpoint outputs/best_classifier.pth --tta
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import timm

from dataset_classification import (
    ForgeryTestDataset, 
    get_val_transforms, 
    get_tta_transforms
)
from train_classifier import ForgeryClassifier
from utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Forgery Classification')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--submission_name', type=str, default='submission.csv')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for forged prediction')
    parser.add_argument('--tta', action='store_true',
                       help='Use Test Time Augmentation')
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """載入模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    
    model_name = args.get('model', 'efficientnet_b0')
    dropout = args.get('dropout', 0.3)
    
    print(f"Loading {model_name}...")
    
    model = ForgeryClassifier(
        model_name=model_name,
        num_classes=2,
        pretrained=False,
        dropout=dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    best_score = checkpoint.get('best_score', 0)
    print(f"Model loaded. Best validation F1: {best_score:.4f}")
    
    return model, args


def predict_batch(model, images, device):
    """批次預測"""
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = F.softmax(outputs, dim=1)
    return probs.cpu().numpy()


def predict_with_tta(model, image_np, tta_transforms, device):
    """TTA 預測"""
    probs_list = []
    
    for transform in tta_transforms:
        augmented = transform(image=image_np)
        img_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
        
        probs_list.append(probs.cpu().numpy())
    
    # 平均
    avg_probs = np.mean(probs_list, axis=0)
    return avg_probs


def main():
    args = parse_args()
    
    device = get_device()
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入模型
    print("\n🔧 Loading model...")
    model, model_args = load_model(args.checkpoint, device)
    
    # 資料路徑
    data_dir = Path(args.data_dir)
    test_dir = data_dir / 'test_images'
    sample_sub_path = data_dir / 'sample_submission.csv'
    
    # 讀取 sample submission 了解格式
    if sample_sub_path.exists():
        sample_df = pd.read_csv(sample_sub_path)
        print(f"\nSample submission format:")
        print(sample_df.head())
        id_col = sample_df.columns[0]  # case_id
        annotation_col = sample_df.columns[1]  # annotation
    else:
        id_col = 'case_id'
        annotation_col = 'annotation'
    
    # 準備轉換
    image_size = (args.image_size, args.image_size)
    transform = get_val_transforms(image_size)
    tta_transforms = get_tta_transforms(image_size) if args.tta else None
    
    # 收集測試圖像
    print(f"\n📊 Collecting test images from {test_dir}...")
    
    test_images = []
    
    # 檢查測試目錄結構
    if test_dir.exists():
        # 可能有子目錄
        for item in test_dir.iterdir():
            if item.is_dir():
                for f in item.iterdir():
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                        test_images.append(f)
            elif item.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                test_images.append(item)
    
    test_images = sorted(test_images)
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("❌ No test images found!")
        return
    
    # 推理
    print(f"\n🚀 Running inference {'with TTA' if args.tta else ''}...")
    
    results = []
    
    for img_path in tqdm(test_images, desc='Inference'):
        # 讀取圖像
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.array(Image.open(img_path).convert('RGB'))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 預測
        if args.tta:
            probs = predict_with_tta(model, image, tta_transforms, device)
        else:
            augmented = transform(image=image)
            img_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
        
        # 獲取預測
        prob_forged = probs[0, 1]  # P(forged)
        prediction = 'forged' if prob_forged > args.threshold else 'authentic'
        
        # 提取 case_id (通常是文件名的數字部分)
        case_id = img_path.stem
        # 嘗試提取數字 ID
        try:
            case_id = int(''.join(filter(str.isdigit, case_id)))
        except:
            pass
        
        results.append({
            id_col: case_id,
            annotation_col: prediction,
            'prob_forged': prob_forged  # 保存概率供分析
        })
    
    # 創建提交 DataFrame
    submission_df = pd.DataFrame(results)
    
    # 保存完整結果 (包含概率)
    full_results_path = output_dir / 'predictions_with_probs.csv'
    submission_df.to_csv(full_results_path, index=False)
    print(f"\n📊 Full predictions saved to: {full_results_path}")
    
    # 創建提交檔案 (只有 case_id 和 annotation)
    submission_df = submission_df[[id_col, annotation_col]]
    submission_path = output_dir / args.submission_name
    submission_df.to_csv(submission_path, index=False)
    
    print(f"✅ Submission saved to: {submission_path}")
    print(f"   Total predictions: {len(submission_df)}")
    
    # 統計
    n_authentic = (submission_df[annotation_col] == 'authentic').sum()
    n_forged = (submission_df[annotation_col] == 'forged').sum()
    print(f"\n📈 Prediction statistics:")
    print(f"   Authentic: {n_authentic} ({n_authentic/len(submission_df)*100:.1f}%)")
    print(f"   Forged: {n_forged} ({n_forged/len(submission_df)*100:.1f}%)")
    
    # 顯示前幾個結果
    print("\nFirst 10 predictions:")
    print(submission_df.head(10))
    
    # 繪製概率分佈
    import matplotlib.pyplot as plt
    
    full_df = pd.read_csv(full_results_path)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(full_df['prob_forged'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=args.threshold, color='r', linestyle='--', label=f'Threshold={args.threshold}')
    plt.xlabel('P(Forged)')
    plt.ylabel('Count')
    plt.title('Distribution of Forged Probabilities')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.pie([n_authentic, n_forged], 
           labels=['Authentic', 'Forged'],
           autopct='%1.1f%%',
           colors=['lightgreen', 'salmon'])
    plt.title('Prediction Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distribution.png', dpi=150)
    plt.close()
    
    print(f"\n📊 Distribution plot saved to: {output_dir / 'prediction_distribution.png'}")


if __name__ == '__main__':
    main()
