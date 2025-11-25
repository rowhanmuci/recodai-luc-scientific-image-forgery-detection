"""
推理和提交生成腳本
Scientific Image Forgery Detection

使用方法:
    python inference.py --checkpoint outputs/best_model.pth
    python inference.py --checkpoint outputs/best_model.pth --tta
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
import segmentation_models_pytorch as smp

from dataset import ForgeryDataset, get_val_transforms, get_tta_transforms
from utils import get_device, rle_encode


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Forgery Detection')
    
    # 路徑
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型檢查點路徑')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='資料目錄')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='輸出目錄')
    parser.add_argument('--submission_name', type=str, default='submission.csv',
                       help='提交檔案名稱')
    
    # 推理配置
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--image_size', type=int, default=512,
                       help='輸入圖像大小')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='二值化閾值')
    parser.add_argument('--tta', action='store_true',
                       help='使用 Test Time Augmentation')
    parser.add_argument('--save_masks', action='store_true',
                       help='保存預測遮罩圖像')
    
    return parser.parse_args()


def get_model_from_checkpoint(checkpoint_path, device):
    """從檢查點加載模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    
    # 獲取模型配置
    model_name = args.get('model', 'unet')
    encoder_name = args.get('encoder', 'resnet50')
    
    print(f"Loading {model_name} with {encoder_name} encoder...")
    
    # 創建模型
    model_dict = {
        'unet': smp.Unet,
        'unetpp': smp.UnetPlusPlus,
        'deeplabv3': smp.DeepLabV3,
        'deeplabv3p': smp.DeepLabV3Plus,
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
        'manet': smp.MAnet,
    }
    
    model = model_dict[model_name](
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    # 加載權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    best_score = checkpoint.get('best_score', 0)
    print(f"Model loaded. Best validation score: {best_score:.4f}")
    
    return model, args


def inference_single(model, image, transform, device):
    """單張圖像推理"""
    # 應用轉換
    augmented = transform(image=image)
    img_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output)
    
    return pred.squeeze().cpu().numpy()


def inference_with_tta(model, image, tta_transforms, device):
    """帶 TTA 的推理"""
    predictions = []
    
    for transform in tta_transforms:
        augmented = transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
        
        predictions.append(pred)
    
    # 反轉增強 (需要對預測結果進行相應的反變換)
    # 這裡簡化處理，只做平均
    # predictions[1] 是水平翻轉，需要再翻轉回來
    predictions[1] = np.fliplr(predictions[1])
    # predictions[2] 是垂直翻轉
    predictions[2] = np.flipud(predictions[2])
    # predictions[3] 是轉置
    predictions[3] = predictions[3].T
    
    # 平均
    final_pred = np.mean(predictions, axis=0)
    
    return final_pred


def resize_prediction(pred, original_size):
    """將預測調整回原始大小"""
    return cv2.resize(pred, (original_size[1], original_size[0]), 
                     interpolation=cv2.INTER_LINEAR)


def main():
    args = parse_args()
    
    # 設備
    device = get_device()
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_masks:
        mask_dir = output_dir / 'predicted_masks'
        mask_dir.mkdir(exist_ok=True)
    
    # 加載模型
    print("\n🔧 Loading model...")
    model, model_args = get_model_from_checkpoint(args.checkpoint, device)
    
    # 資料路徑
    data_dir = Path(args.data_dir)
    test_image_dir = data_dir / 'test_images'
    sample_sub_path = data_dir / 'sample_submission.csv'
    
    # 檢查測試資料
    if not test_image_dir.exists():
        print(f"❌ Test directory not found: {test_image_dir}")
        return
    
    # 獲取所有測試圖像
    test_images = sorted([
        f for f in test_image_dir.iterdir()
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    ])
    print(f"\n📊 Found {len(test_images)} test images")
    
    # 準備轉換
    image_size = (args.image_size, args.image_size)
    transform = get_val_transforms(image_size)
    tta_transforms = get_tta_transforms(image_size) if args.tta else None
    
    # 讀取提交範例以了解格式
    if sample_sub_path.exists():
        sample_df = pd.read_csv(sample_sub_path)
        print(f"\nSample submission format:")
        print(sample_df.head())
        submission_columns = sample_df.columns.tolist()
    else:
        print("\n⚠️ Sample submission not found, using default format")
        submission_columns = ['image_id', 'rle_mask']
    
    # 推理
    print(f"\n🚀 Running inference {'with TTA' if args.tta else ''}...")
    results = []
    
    for img_path in tqdm(test_images, desc='Inference'):
        # 讀取圖像
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (H, W)
        
        # 推理
        if args.tta:
            pred = inference_with_tta(model, image, tta_transforms, device)
        else:
            pred = inference_single(model, image, transform, device)
        
        # 調整回原始大小
        pred_resized = resize_prediction(pred, original_size)
        
        # 二值化
        pred_binary = (pred_resized > args.threshold).astype(np.uint8)
        
        # RLE 編碼
        rle = rle_encode(pred_binary)
        
        # 記錄結果
        results.append({
            'image_id': img_path.stem,
            'rle_mask': rle if rle else ''
        })
        
        # 保存遮罩圖像
        if args.save_masks:
            mask_save_path = mask_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(mask_save_path), pred_binary * 255)
    
    # 創建提交檔案
    submission_df = pd.DataFrame(results)
    
    # 確保列名匹配 (可能需要根據實際比賽調整)
    if 'image_id' not in submission_columns:
        # 嘗試找到正確的列名
        id_col = [c for c in submission_columns if 'id' in c.lower()]
        mask_col = [c for c in submission_columns if 'mask' in c.lower() or 'rle' in c.lower()]
        
        if id_col and mask_col:
            submission_df.columns = [id_col[0], mask_col[0]]
    
    # 保存提交檔案
    submission_path = output_dir / args.submission_name
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n✅ Submission saved to: {submission_path}")
    print(f"   Total predictions: {len(submission_df)}")
    print(f"   Non-empty masks: {(submission_df.iloc[:, 1] != '').sum()}")
    
    # 顯示前幾個結果
    print("\nFirst 5 predictions:")
    print(submission_df.head())
    
    # 統計
    mask_sizes = []
    for rle in submission_df.iloc[:, 1]:
        if rle:
            # 從 RLE 計算遮罩大小
            values = [int(x) for x in rle.split()[1::2]]
            mask_sizes.append(sum(values))
        else:
            mask_sizes.append(0)
    
    print(f"\nMask statistics:")
    print(f"   Mean size: {np.mean(mask_sizes):.0f} pixels")
    print(f"   Max size: {max(mask_sizes)} pixels")
    print(f"   Zero masks: {mask_sizes.count(0)}")


if __name__ == '__main__':
    main()
