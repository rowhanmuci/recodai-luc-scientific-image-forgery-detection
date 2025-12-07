"""
資料探索與分析腳本
Scientific Image Forgery Detection - EDA

資料結構:
- train_images/
  - authentic/  (真實圖像)
  - forged/     (偽造圖像)
- train_masks/  (偽造區域遮罩，僅對應 forged 圖像)
- test_images/  (測試圖像)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置路徑 - 請根據你的資料位置修改
# ============================================================
DATA_DIR = Path("./data")  # 修改為你的資料目錄

TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TRAIN_AUTHENTIC_DIR = TRAIN_IMAGES_DIR / "authentic"
TRAIN_FORGED_DIR = TRAIN_IMAGES_DIR / "forged"
TRAIN_MASKS_DIR = DATA_DIR / "train_masks"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
SUPP_IMAGES_DIR = DATA_DIR / "supplemental_images"
SUPP_MASKS_DIR = DATA_DIR / "supplemental_masks"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"


def count_files(directory, recursive=False):
    """計算目錄中的文件數量"""
    if not directory.exists():
        return 0
    
    if recursive:
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len([f for f in files if not f.startswith('.')])
        return count
    else:
        return len([f for f in directory.iterdir() if f.is_file()])


def get_image_stats(image_dir, sample_size=100):
    """獲取圖像統計資訊"""
    if not image_dir.exists():
        return None
    
    files = list(image_dir.glob("*"))[:sample_size]
    stats = {
        'widths': [],
        'heights': [],
        'channels': [],
        'extensions': defaultdict(int),
        'file_sizes': []
    }
    
    for f in files:
        try:
            img = Image.open(f)
            w, h = img.size
            stats['widths'].append(w)
            stats['heights'].append(h)
            stats['channels'].append(len(img.getbands()))
            stats['extensions'][f.suffix.lower()] += 1
            stats['file_sizes'].append(f.stat().st_size / 1024)  # KB
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    return stats


def analyze_masks(mask_dir, sample_size=100):
    """分析遮罩統計資訊 (支援 .npy 和圖像格式)"""
    if not mask_dir.exists():
        return None
    
    files = list(mask_dir.glob("*"))[:sample_size]
    stats = {
        'forgery_ratios': [],
        'num_components': [],
        'has_forgery': 0,
        'no_forgery': 0,
        'shapes': []
    }
    
    for f in files:
        try:
            # 根據副檔名選擇讀取方式
            if f.suffix.lower() == '.npy':
                mask = np.load(str(f))
            else:
                mask = np.array(Image.open(f).convert('L'))
            
            # 記錄原始 shape
            stats['shapes'].append(mask.shape)
            
            # 處理 (N, H, W) 格式 - 合併所有通道
            if mask.ndim == 3:
                # 取所有通道的最大值 (任何通道有偽造就算偽造)
                mask = mask.max(axis=0)
            
            # 二值化 (mask 值是 0 或 1)
            if mask.max() <= 1:
                binary_mask = (mask > 0).astype(np.uint8)
            else:
                binary_mask = (mask > 127).astype(np.uint8)
            
            # 計算偽造區域比例
            forgery_ratio = binary_mask.sum() / binary_mask.size
            stats['forgery_ratios'].append(forgery_ratio)
            
            if forgery_ratio > 0:
                stats['has_forgery'] += 1
            else:
                stats['no_forgery'] += 1
                
        except Exception as e:
            print(f"Error reading mask {f}: {e}")
    
    return stats


def visualize_samples(image_dir, mask_dir, num_samples=6, save_path=None):
    """視覺化樣本圖像和遮罩"""
    if not image_dir.exists() or not mask_dir.exists():
        print("Directory not found!")
        return
    
    image_files = sorted(list(image_dir.glob("*")))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for idx, img_path in enumerate(image_files[:num_samples]):
        # 嘗試找到對應的遮罩
        mask_name = img_path.stem  # 不含副檔名
        possible_masks = list(mask_dir.glob(f"{mask_name}.*"))
        
        if not possible_masks:
            print(f"No mask found for {img_path.name}")
            continue
            
        mask_path = possible_masks[0]
        
        # 讀取圖像和遮罩
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # 創建覆蓋圖
        overlay = img.copy()
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay]*3, axis=-1)
        overlay[mask > 127] = [255, 0, 0]  # 紅色標記偽造區域
        
        # 顯示
        axes[idx, 0].imshow(img, cmap='gray' if len(img.shape)==2 else None)
        axes[idx, 0].set_title(f"Image: {img_path.name}")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title(f"Mask (forgery ratio: {(mask>127).mean():.2%})")
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Overlay")
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def visualize_samples_new_structure(forged_dir, mask_dir, authentic_dir=None, 
                                    num_samples=4, save_path=None):
    """
    視覺化新結構的樣本 (authentic/forged 分開的目錄)
    支援 .npy 格式的 mask，包括 (N, H, W) 格式
    
    顯示：
    - Forged 圖像 + 對應 Mask + Overlay
    - Authentic 圖像 (如果有)
    """
    # 收集 forged 圖像
    forged_files = sorted(list(forged_dir.glob("*")))[:num_samples]
    
    # 收集 authentic 圖像
    authentic_files = []
    if authentic_dir and authentic_dir.exists():
        authentic_files = sorted(list(authentic_dir.glob("*")))[:num_samples]
    
    # 計算需要的行數
    n_forged = len(forged_files)
    n_authentic = min(len(authentic_files), 2)  # 最多顯示 2 個 authentic
    n_rows = n_forged + (1 if n_authentic > 0 else 0)
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 顯示 Forged 圖像
    for idx, img_path in enumerate(forged_files):
        # 嘗試找到對應的遮罩 (支援多種格式)
        mask_name = img_path.stem
        possible_masks = []
        for ext in ['.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            mask_path = mask_dir / f"{mask_name}{ext}"
            if mask_path.exists():
                possible_masks.append(mask_path)
                break
        
        # 讀取圖像
        img = np.array(Image.open(img_path))
        
        # 讀取遮罩 (如果存在)
        if possible_masks:
            mask_path = possible_masks[0]
            if mask_path.suffix.lower() == '.npy':
                mask = np.load(str(mask_path))
                
                # 處理 (N, H, W) 格式 - 合併所有通道
                if mask.ndim == 3:
                    mask = mask.max(axis=0)  # 取最大值合併
                
            else:
                mask = np.array(Image.open(mask_path).convert('L'))
            
            # 調整 mask 大小以匹配圖像
            img_h, img_w = img.shape[:2]
            mask_h, mask_w = mask.shape[:2]
            
            if (mask_h, mask_w) != (img_h, img_w):
                from PIL import Image as PILImage
                # 正規化到 0-255 後再 resize
                mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
                mask = np.array(PILImage.fromarray(mask_uint8).resize(
                    (img_w, img_h), PILImage.NEAREST
                ))
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            print(f"⚠️ No mask found for {img_path.name}")
        
        # 正規化 mask 到 0-255
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        
        # 創建覆蓋圖
        overlay = img.copy()
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay]*3, axis=-1)
        elif overlay.shape[2] == 4:  # RGBA
            overlay = overlay[:, :, :3]
        
        # 紅色半透明覆蓋 (閾值 > 0 因為值是 0 或 255)
        mask_binary = mask > 0
        overlay_float = overlay.astype(float)
        overlay_float[mask_binary] = overlay_float[mask_binary] * 0.5 + np.array([255, 0, 0]) * 0.5
        overlay = overlay_float.astype(np.uint8)
        
        # 顯示
        axes[idx, 0].imshow(img, cmap='gray' if len(img.shape)==2 else None)
        axes[idx, 0].set_title(f"FORGED: {img_path.name[:30]}...")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask, cmap='hot')
        forgery_pct = (mask > 0).mean() * 100
        axes[idx, 1].set_title(f"Mask ({forgery_pct:.1f}% forged)")
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Overlay (red = forged region)")
        axes[idx, 2].axis('off')
    
    # 顯示 Authentic 圖像 (最後一行)
    if n_authentic > 0:
        row_idx = n_forged
        for col_idx, img_path in enumerate(authentic_files[:min(3, n_authentic)]):
            img = np.array(Image.open(img_path))
            axes[row_idx, col_idx].imshow(img, cmap='gray' if len(img.shape)==2 else None)
            axes[row_idx, col_idx].set_title(f"AUTHENTIC: {img_path.name[:25]}...")
            axes[row_idx, col_idx].axis('off')
        
        # 如果不足3張，隱藏空的subplot
        for col_idx in range(n_authentic, 3):
            axes[row_idx, col_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    plt.show()


def main():
    print("=" * 60)
    print("Scientific Image Forgery Detection - EDA")
    print("=" * 60)
    
    # 1. 文件數量統計
    print("\n📁 Dataset Statistics:")
    print("-" * 40)
    
    # 檢查 train_images 結構
    authentic_count = count_files(TRAIN_AUTHENTIC_DIR)
    forged_count = count_files(TRAIN_FORGED_DIR)
    
    print(f"  📂 Train Images:")
    if authentic_count > 0 or forged_count > 0:
        print(f"     ✅ authentic/: {authentic_count} files")
        print(f"     ✅ forged/: {forged_count} files")
        print(f"     📊 Total: {authentic_count + forged_count} files")
        print(f"     📊 Forged ratio: {forged_count/(authentic_count+forged_count)*100:.1f}%")
    else:
        # 嘗試直接在 train_images 下找文件
        direct_count = count_files(TRAIN_IMAGES_DIR)
        if direct_count > 0:
            print(f"     ✅ {direct_count} files (flat structure)")
        else:
            print(f"     ❌ No images found")
    
    # 其他目錄
    datasets = {
        "Train Masks": TRAIN_MASKS_DIR,
        "Test Images": TEST_IMAGES_DIR,
        "Supplemental Images": SUPP_IMAGES_DIR,
        "Supplemental Masks": SUPP_MASKS_DIR,
    }
    
    for name, path in datasets.items():
        count = count_files(path, recursive=True)
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {name}: {count} files")
    
    # 2. 樣本提交檔案
    print("\n📋 Sample Submission:")
    print("-" * 40)
    if SAMPLE_SUB_PATH.exists():
        df = pd.read_csv(SAMPLE_SUB_PATH)
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        print(f"\n  Head:")
        print(df.head(10))
        
        # 分析提交格式
        if 'annotation' in df.columns:
            print(f"\n  📊 This is a CLASSIFICATION task!")
            print(f"     Predict: 'authentic' or 'forged'")
            unique_annotations = df['annotation'].unique()
            print(f"     Unique values: {unique_annotations}")
    else:
        print("  ❌ sample_submission.csv not found")
    
    # 3. 圖像統計 (優先檢查 forged 目錄)
    print("\n🖼️ Image Statistics:")
    print("-" * 40)
    
    # Forged images
    if TRAIN_FORGED_DIR.exists() and count_files(TRAIN_FORGED_DIR) > 0:
        print("  [Forged Images]")
        img_stats = get_image_stats(TRAIN_FORGED_DIR)
        if img_stats and img_stats['widths']:
            print(f"    Width:  min={min(img_stats['widths'])}, max={max(img_stats['widths'])}, "
                  f"mean={np.mean(img_stats['widths']):.0f}")
            print(f"    Height: min={min(img_stats['heights'])}, max={max(img_stats['heights'])}, "
                  f"mean={np.mean(img_stats['heights']):.0f}")
            print(f"    Channels: {set(img_stats['channels'])}")
            print(f"    Extensions: {dict(img_stats['extensions'])}")
    
    # Authentic images
    if TRAIN_AUTHENTIC_DIR.exists() and count_files(TRAIN_AUTHENTIC_DIR) > 0:
        print("\n  [Authentic Images]")
        img_stats = get_image_stats(TRAIN_AUTHENTIC_DIR)
        if img_stats and img_stats['widths']:
            print(f"    Width:  min={min(img_stats['widths'])}, max={max(img_stats['widths'])}, "
                  f"mean={np.mean(img_stats['widths']):.0f}")
            print(f"    Height: min={min(img_stats['heights'])}, max={max(img_stats['heights'])}, "
                  f"mean={np.mean(img_stats['heights']):.0f}")
    
    # 4. 遮罩統計
    print("\n🎭 Mask Statistics (Train):")
    print("-" * 40)
    mask_stats = analyze_masks(TRAIN_MASKS_DIR)
    if mask_stats and mask_stats['forgery_ratios']:
        ratios = mask_stats['forgery_ratios']
        print(f"  Forgery ratio: min={min(ratios):.4f}, max={max(ratios):.4f}, "
              f"mean={np.mean(ratios):.4f}")
        print(f"  Images with forgery region: {mask_stats['has_forgery']}")
        print(f"  Images without forgery region: {mask_stats['no_forgery']}")
        
        # 繪製分佈圖
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(ratios, bins=50, edgecolor='black', alpha=0.7, color='salmon')
        plt.xlabel('Forgery Ratio (% of image)')
        plt.ylabel('Count')
        plt.title('Distribution of Forgery Region Sizes')
        
        plt.subplot(1, 3, 2)
        plt.bar(['Has Forgery', 'No Forgery'], 
                [mask_stats['has_forgery'], mask_stats['no_forgery']],
                color=['salmon', 'lightblue'])
        plt.title('Masks with/without Forgery')
        plt.ylabel('Count')
        
        plt.subplot(1, 3, 3)
        if authentic_count > 0 and forged_count > 0:
            plt.pie([authentic_count, forged_count], 
                   labels=['Authentic', 'Forged'],
                   autopct='%1.1f%%',
                   colors=['lightgreen', 'salmon'])
            plt.title('Train Set Distribution')
        
        plt.tight_layout()
        plt.savefig('forgery_distribution.png', dpi=150)
        print("\n  📊 Saved: forgery_distribution.png")
        plt.show()
    
    # 5. 視覺化樣本 (使用新結構)
    print("\n🔍 Visualizing Samples...")
    print("-" * 40)
    
    if TRAIN_FORGED_DIR.exists() and TRAIN_MASKS_DIR.exists():
        visualize_samples_new_structure(
            forged_dir=TRAIN_FORGED_DIR,
            mask_dir=TRAIN_MASKS_DIR,
            authentic_dir=TRAIN_AUTHENTIC_DIR,
            num_samples=4,
            save_path='sample_visualization.png'
        )
    
    # 6. 任務總結
    print("\n" + "=" * 60)
    print("📝 Task Summary:")
    print("=" * 60)
    print("""
    This competition has TWO tasks:
    
    1. CLASSIFICATION: Predict if image is 'authentic' or 'forged'
       - Output: case_id, annotation (authentic/forged)
    
    2. SEGMENTATION: For forged images, locate the manipulated regions
       - Masks provided in train_masks/ for forged images
       - This helps understand WHERE the forgery is
    
    Recommended Approach:
    - Train a classifier (ResNet, EfficientNet) for detection
    - Optionally train a segmentation model (U-Net) to help classification
    - Use segmentation features as auxiliary input to classifier
    """)
    print("=" * 60)
    print("EDA Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()