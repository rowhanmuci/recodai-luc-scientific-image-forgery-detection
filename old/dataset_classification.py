"""
分類任務 Dataset
Scientific Image Forgery Detection

任務: 二分類 (authentic vs forged)
資料結構:
- train_images/
  - authentic/
  - forged/
- train_masks/ (optional, for forged images)
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


class ForgeryClassificationDataset(Dataset):
    """
    科學圖像偽造檢測 - 分類資料集
    
    Args:
        authentic_dir: 真實圖像目錄
        forged_dir: 偽造圖像目錄
        mask_dir: 遮罩目錄 (可選，用於輔助訓練)
        transform: 資料增強
        image_size: 圖像大小
        use_mask_channel: 是否將 mask 作為額外輸入通道
    """
    
    def __init__(
        self,
        authentic_dir=None,
        forged_dir=None,
        mask_dir=None,
        transform=None,
        image_size=(512, 512),
        use_mask_channel=False,
        image_list=None,  # 可以直接傳入圖像列表 [(path, label), ...]
    ):
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.image_size = image_size
        self.use_mask_channel = use_mask_channel
        
        self.samples = []  # [(image_path, label), ...]
        
        if image_list is not None:
            self.samples = image_list
        else:
            # 收集 authentic 圖像 (label = 0)
            if authentic_dir and Path(authentic_dir).exists():
                for f in Path(authentic_dir).iterdir():
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                        self.samples.append((f, 0))
            
            # 收集 forged 圖像 (label = 1)
            if forged_dir and Path(forged_dir).exists():
                for f in Path(forged_dir).iterdir():
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                        self.samples.append((f, 1))
        
        # 統計
        n_authentic = sum(1 for _, label in self.samples if label == 0)
        n_forged = sum(1 for _, label in self.samples if label == 1)
        print(f"Dataset: {len(self.samples)} samples (Authentic: {n_authentic}, Forged: {n_forged})")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_mask(self, image_stem):
        """載入對應的 mask (如果存在)，支援 .npy 和圖像格式"""
        if self.mask_dir is None:
            return None
        
        # 優先查找 .npy 格式
        npy_path = self.mask_dir / f"{image_stem}.npy"
        if npy_path.exists():
            mask = np.load(str(npy_path))
            
            # 處理 (N, H, W) 格式 - 合併所有通道
            if mask.ndim == 3:
                mask = mask.max(axis=0)  # 取最大值合併
            
            return mask
        
        # 其次查找圖像格式
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            mask_path = self.mask_dir / f"{image_stem}{ext}"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                return mask
        return None
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 讀取圖像
        image = cv2.imread(str(img_path))
        if image is None:
            # 嘗試用 PIL 讀取
            image = np.array(Image.open(img_path).convert('RGB'))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 載入 mask (如果需要)
        mask = None
        if self.use_mask_channel and label == 1:  # 只有 forged 圖像有 mask
            mask = self._load_mask(img_path.stem)
        
        # 應用轉換
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transform(image=image)
                image = augmented['image']
        else:
            # 基本預處理
            image = cv2.resize(image, self.image_size)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # 如果使用 mask channel，將其合併到圖像
        if self.use_mask_channel:
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).float().unsqueeze(0)
                else:
                    mask = mask.float().unsqueeze(0)
            else:
                # 沒有 mask 時用全零
                mask = torch.zeros(1, image.shape[1], image.shape[2])
            
            image = torch.cat([image, mask], dim=0)  # [4, H, W]
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': img_path.stem
        }


class ForgeryTestDataset(Dataset):
    """
    測試資料集 (無標籤)
    """
    def __init__(self, test_dir, transform=None, image_size=(512, 512)):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.image_size = image_size
        
        # 收集所有測試圖像 (可能在子目錄中)
        self.image_files = []
        
        # 檢查是否有子目錄
        subdirs = [d for d in self.test_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # 有子目錄，遞歸搜索
            for subdir in subdirs:
                for f in subdir.iterdir():
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                        self.image_files.append(f)
        else:
            # 直接在目錄下
            for f in self.test_dir.iterdir():
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                    self.image_files.append(f)
        
        self.image_files = sorted(self.image_files)
        print(f"Test dataset: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.array(Image.open(img_path).convert('RGB'))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_size = image.shape[:2]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = cv2.resize(image, self.image_size)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'image': image,
            'image_id': img_path.stem,
            'original_size': original_size
        }


def get_train_transforms(image_size=(512, 512)):
    """訓練資料增強"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # 幾何變換
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # 仿射變換
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        
        # 色彩增強
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
        ], p=0.5),
        
        # 噪聲和模糊
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1),
            A.GaussNoise(var_limit=(10, 50), p=1),
            A.ISONoise(p=1),
        ], p=0.3),
        
        # JPEG 壓縮 (常見於偽造圖像)
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        
        # 高級增強
        A.OneOf([
            A.CLAHE(clip_limit=2, p=1),
            A.Sharpen(p=1),
            A.Emboss(p=1),
        ], p=0.2),
        
        # Cutout / CoarseDropout
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, p=0.2
        ),
        
        # 正規化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size=(512, 512)):
    """驗證/測試資料增強"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_tta_transforms(image_size=(512, 512)):
    """Test Time Augmentation"""
    base = [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    return [
        A.Compose(base),  # 原圖
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.Transpose(p=1.0)] + base),
    ]


def create_dataloaders(
    data_dir,
    image_size=(512, 512),
    batch_size=16,
    num_workers=4,
    val_split=0.2,
    seed=42
):
    """
    創建訓練和驗證 DataLoader
    
    Args:
        data_dir: 資料目錄 (包含 train_images/authentic, train_images/forged)
        image_size: 圖像大小
        batch_size: 批次大小
        num_workers: DataLoader workers
        val_split: 驗證集比例
        seed: 隨機種子
    
    Returns:
        train_loader, val_loader
    """
    from sklearn.model_selection import train_test_split
    
    data_dir = Path(data_dir)
    authentic_dir = data_dir / 'train_images' / 'authentic'
    forged_dir = data_dir / 'train_images' / 'forged'
    mask_dir = data_dir / 'train_masks'
    
    # 收集所有樣本
    all_samples = []
    
    if authentic_dir.exists():
        for f in authentic_dir.iterdir():
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                all_samples.append((f, 0))
    
    if forged_dir.exists():
        for f in forged_dir.iterdir():
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                all_samples.append((f, 1))
    
    # 分層抽樣
    images, labels = zip(*all_samples)
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        images, labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed
    )
    
    train_samples = list(zip(train_imgs, train_labels))
    val_samples = list(zip(val_imgs, val_labels))
    
    # 創建 Dataset
    train_dataset = ForgeryClassificationDataset(
        image_list=train_samples,
        mask_dir=mask_dir,
        transform=get_train_transforms(image_size),
        image_size=image_size,
        use_mask_channel=False
    )
    
    val_dataset = ForgeryClassificationDataset(
        image_list=val_samples,
        mask_dir=mask_dir,
        transform=get_val_transforms(image_size),
        image_size=image_size,
        use_mask_channel=False
    )
    
    # 創建 DataLoader
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
    # 測試
    print("Testing ForgeryClassificationDataset...")
    
    DATA_DIR = Path("./data")
    
    train_loader, val_loader = create_dataloaders(
        data_dir=DATA_DIR,
        image_size=(384, 384),
        batch_size=8,
        val_split=0.2
    )
    
    # 測試一個 batch
    batch = next(iter(train_loader))
    print(f"\nBatch info:")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Labels: {batch['label']}")
    print(f"  Label distribution: {batch['label'].sum().item()}/{len(batch['label'])} forged")
