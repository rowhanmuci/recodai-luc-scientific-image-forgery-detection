"""
自定義 Dataset 類
Scientific Image Forgery Detection
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class ForgeryDataset(Dataset):
    """
    科學圖像偽造檢測資料集
    
    Args:
        image_dir: 圖像目錄路徑
        mask_dir: 遮罩目錄路徑 (訓練時需要，推理時可為 None)
        transform: 資料增強轉換
        image_size: 統一的圖像大小 (H, W)
        is_train: 是否為訓練模式
    """
    
    def __init__(
        self, 
        image_dir, 
        mask_dir=None, 
        transform=None,
        image_size=(512, 512),
        is_train=True
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.image_size = image_size
        self.is_train = is_train
        
        # 獲取所有圖像文件
        self.image_files = sorted([
            f for f in self.image_dir.iterdir() 
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        ])
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def _find_mask(self, image_stem):
        """根據圖像名稱找到對應的遮罩文件 (支援 .npy 和圖像格式)"""
        if self.mask_dir is None:
            return None
        
        # 優先查找 .npy 格式
        npy_path = self.mask_dir / f"{image_stem}.npy"
        if npy_path.exists():
            return npy_path
            
        # 嘗試不同的圖像副檔名
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            mask_path = self.mask_dir / f"{image_stem}{ext}"
            if mask_path.exists():
                return mask_path
        return None
    
    def __getitem__(self, idx):
        # 讀取圖像
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 讀取遮罩 (如果存在)
        mask = None
        if self.mask_dir:
            mask_path = self._find_mask(img_path.stem)
            if mask_path:
                # 根據副檔名選擇讀取方式
                if mask_path.suffix.lower() == '.npy':
                    mask = np.load(str(mask_path))
                    
                    # 處理 (N, H, W) 格式 - 合併所有通道
                    if mask.ndim == 3:
                        mask = mask.max(axis=0)  # 取最大值合併
                        
                else:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                # 二值化遮罩 (0 或 1)
                if mask.max() <= 1:
                    mask = (mask > 0).astype(np.float32)
                else:
                    mask = (mask > 127).astype(np.float32)
                
                # 確保 mask 大小與圖像匹配
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # 如果沒有遮罩，創建全零遮罩
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # 應用資料增強
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # 基本預處理：調整大小和正規化
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # 轉換為張量
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()
        
        # 確保 mask 有正確的維度 [H, W] -> [1, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'image_id': img_path.stem
        }


def get_train_transforms(image_size=(512, 512)):
    """獲取訓練資料增強"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # 幾何變換
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # 輕度仿射變換
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        
        # 色彩增強
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=1
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1
            ),
        ], p=0.5),
        
        # 模糊和噪聲 (模擬圖像處理後的偽造)
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1),
            A.GaussNoise(var_limit=(10, 30), p=1),
        ], p=0.3),
        
        # JPEG 壓縮 (常見的偽造痕跡)
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
        
        # 正規化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size=(512, 512)):
    """獲取驗證/測試資料增強"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_tta_transforms(image_size=(512, 512)):
    """獲取 Test Time Augmentation 變換列表"""
    base_transform = [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]
    
    return [
        A.Compose(base_transform),  # 原圖
        A.Compose([A.HorizontalFlip(p=1.0)] + base_transform),
        A.Compose([A.VerticalFlip(p=1.0)] + base_transform),
        A.Compose([A.Transpose(p=1.0)] + base_transform),
    ]


def create_dataloaders(
    train_image_dir,
    train_mask_dir,
    val_image_dir=None,
    val_mask_dir=None,
    image_size=(512, 512),
    batch_size=8,
    num_workers=4,
    val_split=0.2
):
    """
    創建訓練和驗證 DataLoader
    
    如果沒有單獨的驗證集，會自動從訓練集分割
    """
    from sklearn.model_selection import train_test_split
    
    # 創建完整的訓練數據集
    full_dataset = ForgeryDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=None,
        image_size=image_size,
        is_train=True
    )
    
    if val_image_dir and val_mask_dir:
        # 使用提供的驗證集
        train_dataset = ForgeryDataset(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            transform=get_train_transforms(image_size),
            image_size=image_size,
            is_train=True
        )
        val_dataset = ForgeryDataset(
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            transform=get_val_transforms(image_size),
            image_size=image_size,
            is_train=False
        )
    else:
        # 從訓練集分割
        indices = list(range(len(full_dataset)))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=val_split, 
            random_state=42
        )
        
        train_dataset = torch.utils.data.Subset(
            ForgeryDataset(
                image_dir=train_image_dir,
                mask_dir=train_mask_dir,
                transform=get_train_transforms(image_size),
                image_size=image_size,
                is_train=True
            ),
            train_idx
        )
        
        val_dataset = torch.utils.data.Subset(
            ForgeryDataset(
                image_dir=train_image_dir,
                mask_dir=train_mask_dir,
                transform=get_val_transforms(image_size),
                image_size=image_size,
                is_train=False
            ),
            val_idx
        )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 測試 Dataset
    print("Testing ForgeryDataset...")
    
    # 修改為你的資料路徑
    DATA_DIR = Path("./data")
    
    dataset = ForgeryDataset(
        image_dir=DATA_DIR / "train_images",
        mask_dir=DATA_DIR / "train_masks",
        transform=get_train_transforms((512, 512)),
        image_size=(512, 512)
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Image ID: {sample['image_id']}")
        print(f"Image dtype: {sample['image'].dtype}")
        print(f"Mask dtype: {sample['mask'].dtype}")
        print(f"Mask unique values: {torch.unique(sample['mask'])}")
    else:
        print("No images found. Please check the data directory.")