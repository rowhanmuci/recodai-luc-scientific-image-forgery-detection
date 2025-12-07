"""
檢查 Mask 檔案的內容和檔名匹配
"""

import numpy as np
from pathlib import Path
import os

DATA_DIR = Path("./data")
MASK_DIR = DATA_DIR / "train_masks"
FORGED_DIR = DATA_DIR / "train_images" / "forged"

print("=" * 60)
print("Mask File Analysis")
print("=" * 60)

# 1. 列出 mask 檔案
mask_files = sorted(list(MASK_DIR.glob("*")))[:20]
print(f"\n📁 Sample mask files ({len(list(MASK_DIR.glob('*')))} total):")
for f in mask_files[:10]:
    print(f"  {f.name}")

# 2. 列出 forged 圖像檔案
forged_files = sorted(list(FORGED_DIR.glob("*")))[:20]
print(f"\n📁 Sample forged image files ({len(list(FORGED_DIR.glob('*')))} total):")
for f in forged_files[:10]:
    print(f"  {f.name}")

# 3. 檢查檔名匹配
print("\n🔍 Checking filename matching...")
mask_stems = set(f.stem for f in MASK_DIR.glob("*"))
forged_stems = set(f.stem for f in FORGED_DIR.glob("*"))

matched = mask_stems & forged_stems
only_in_masks = mask_stems - forged_stems
only_in_forged = forged_stems - mask_stems

print(f"  Matched: {len(matched)}")
print(f"  Only in masks: {len(only_in_masks)}")
print(f"  Only in forged: {len(only_in_forged)}")

if only_in_forged:
    print(f"\n  ⚠️ Forged images without masks (first 10):")
    for name in sorted(only_in_forged)[:10]:
        print(f"    {name}")

# 4. 檢查 mask 內容
print("\n📊 Analyzing mask contents...")

non_zero_masks = 0
zero_masks = 0
mask_stats = []

for f in list(MASK_DIR.glob("*.npy"))[:100]:
    try:
        mask = np.load(str(f))
        
        # 檢查 mask 的統計資訊
        mask_max = mask.max()
        mask_min = mask.min()
        mask_mean = mask.mean()
        non_zero_ratio = (mask != 0).mean()
        
        if non_zero_ratio > 0:
            non_zero_masks += 1
            mask_stats.append({
                'name': f.name,
                'shape': mask.shape,
                'dtype': mask.dtype,
                'min': mask_min,
                'max': mask_max,
                'mean': mask_mean,
                'non_zero_ratio': non_zero_ratio
            })
        else:
            zero_masks += 1
            
    except Exception as e:
        print(f"  Error loading {f.name}: {e}")

print(f"\n  Non-zero masks: {non_zero_masks}")
print(f"  All-zero masks: {zero_masks}")

if mask_stats:
    print(f"\n  📋 Non-zero mask details (first 10):")
    for stat in mask_stats[:10]:
        print(f"    {stat['name']}: shape={stat['shape']}, dtype={stat['dtype']}, "
              f"range=[{stat['min']:.3f}, {stat['max']:.3f}], non_zero={stat['non_zero_ratio']:.2%}")

# 5. 檢查一個具體的 mask
print("\n🔬 Detailed analysis of first non-zero mask:")
if mask_stats:
    first_mask_name = mask_stats[0]['name']
    mask = np.load(str(MASK_DIR / first_mask_name))
    print(f"  File: {first_mask_name}")
    print(f"  Shape: {mask.shape}")
    print(f"  Dtype: {mask.dtype}")
    print(f"  Min: {mask.min()}, Max: {mask.max()}")
    print(f"  Unique values: {np.unique(mask)[:20]}...")  # 前20個唯一值
    print(f"  Non-zero pixels: {(mask != 0).sum()} / {mask.size} = {(mask != 0).mean():.2%}")
else:
    # 檢查任意一個 mask
    sample_mask_file = list(MASK_DIR.glob("*.npy"))[0]
    mask = np.load(str(sample_mask_file))
    print(f"  File: {sample_mask_file.name}")
    print(f"  Shape: {mask.shape}")
    print(f"  Dtype: {mask.dtype}")
    print(f"  Min: {mask.min()}, Max: {mask.max()}")
    print(f"  Unique values: {np.unique(mask)}")

print("\n" + "=" * 60)
