# 🔬 Scientific Image Forgery Detection - Kaggle Competition

> **競賽連結**: [RecoDAI LUC Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)
> 
> **目標**: 偵測科學論文圖像中的偽造區域，輸出 RLE 編碼的分割遮罩

---

## 📊 目前最佳成績

| 模型 | Public Score | 備註 |
|------|--------------|------|
| **ResNet34-UNet + SCSE (v2)** | **0.293** ⭐ | 目前最佳 |
| ResNet34-UNet + SCSE (v2) | 0.281 | MIN_FORGERY_RATIO=0.05 |
| ResNet34-UNet + SCSE + Authentic (v3) | 0.220 | 加入 authentic 訓練反而變差 |
| ResNet50-UNet + SCSE (v4) | 0.209 | 嚴重過擬合 |
| 原始版本 | 0.122 | 第一次成功提交 |
| 組員模型 (ResNet50-UNet) | 0.303 | 參考對象，使用 supplemental 資料 |

---

## 🏗️ 專案結構

```
recodai-luc-scientific-image-forgery-detection/
├── data/
│   ├── train_images/
│   │   ├── forged/          # 偽造圖像
│   │   └── authentic/       # 真實圖像
│   ├── train_masks/         # .npy 格式的 mask
│   ├── test_images/         # 測試圖像
│   ├── supplemental_images/ # 補充訓練圖像 ⚠️ 重要！
│   ├── supplemental_masks/  # 補充 mask
│   └── sample_submission.csv
├── outputs_improved/        # 訓練輸出
│   ├── best_model.pth
│   └── history.csv
├── train_improved.py        # 訓練腳本
├── kaggle_resnet_universal.py  # 推理腳本
└── README.md
```

---

## 🔧 環境設置

### 本地環境 (Windows)
```powershell
conda activate ml
pip install torch torchvision numpy pandas opencv-python pillow tqdm
```

### Kaggle Notebook
- 啟用 GPU (Settings → Accelerator → GPU T4 x2)
- 無需額外安裝套件

---

## 📝 RLE 格式說明

### ⚠️ 重要發現
Kaggle 要求的 RLE 格式與常見格式不同！

| 格式類型 | 範例 |
|---------|------|
| ❌ 錯誤格式 | `424960 5 426401 14 ...` |
| ✅ 正確格式 | `[424960, 5, 426401, 14];[123, 4]` |

### 正確的 RLE 編碼函數
```python
import json

def _rle_encode_single(mask):
    """單個連通區域的 RLE 編碼"""
    pixels = mask.T.flatten()  # Fortran order (column-major)
    dots = np.where(pixels == 1)[0]
    if len(dots) == 0:
        return []
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend([b + 1, 0])  # 1-indexed
        run_lengths[-1] += 1
        prev = b
    return [int(x) for x in run_lengths]

def rle_encode(masks):
    """多個連通區域的 RLE 編碼，用分號分隔"""
    if not masks:
        return "authentic"
    encodings = []
    for mask in masks:
        encoded_list = _rle_encode_single(mask)
        if encoded_list:
            encodings.append(json.dumps(encoded_list))  # JSON array 格式
    if not encodings:
        return "authentic"
    return ';'.join(encodings)  # 多個區域用分號分隔
```

---

## 🏋️ 訓練

### 基本訓練指令
```powershell
python train_improved.py `
    --data_root "D:\NSYSU\recodai-luc-scientific-image-forgery-detection\data" `
    --backbone resnet34 `
    --epochs 30 `
    --batch_size 8 `
    --output_dir "./outputs_improved"
```

### 加入補充資料（推薦！）
```powershell
python train_improved.py `
    --data_root "D:\NSYSU\recodai-luc-scientific-image-forgery-detection\data" `
    --backbone resnet34 `
    --epochs 15 `
    --batch_size 8 `
    --include_supplemental `
    --output_dir "./outputs_resnet34_supp"
```

### 所有參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--data_root` | - | 資料根目錄 |
| `--backbone` | resnet34 | 骨幹網路 (resnet34/50/101) |
| `--epochs` | 30 | 訓練輪數 |
| `--batch_size` | 8 | 批次大小 |
| `--lr` | 1e-4 | 學習率 |
| `--weight_decay` | 1e-4 | 權重衰減 |
| `--pos_weight` | 5.0 | BCE 正樣本權重 |
| `--image_size` | 512 | 輸入圖像大小 |
| `--include_supplemental` | False | 是否包含補充資料 |
| `--include_authentic` | False | 是否包含真實圖像 |
| `--authentic_ratio` | 0.3 | 真實圖像比例 |
| `--use_copy_move_aug` | False | 是否使用 Copy-Move 增強 |

---

## 🧪 模型架構

### ResNet-UNet + SCSE

```
輸入圖像 (3, 512, 512)
    │
    ▼
┌─────────────────────────────────┐
│  Encoder (ResNet34/50 Backbone) │
│  ├─ encoder0: Conv+BN+ReLU      │ → skip0 (64)
│  ├─ encoder1: Layer1            │ → skip1 (64/256)
│  ├─ encoder2: Layer2            │ → skip2 (128/512)
│  ├─ encoder3: Layer3            │ → skip3 (256/1024)
│  └─ encoder4: Layer4            │ → bottleneck (512/2048)
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Decoder with SCSE Attention    │
│  ├─ decoder4 + SCSE             │ ← skip3
│  ├─ decoder3 + SCSE             │ ← skip2
│  ├─ decoder2 + SCSE             │ ← skip1
│  └─ decoder1 + SCSE             │ ← skip0
└─────────────────────────────────┘
    │
    ▼
輸出 Mask (1, 512, 512)
```

### SCSE (Spatial and Channel Squeeze & Excitation)

```python
class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel SE: 全局池化 → FC → Sigmoid
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        # Spatial SE: 1x1 Conv → Sigmoid
        self.sse = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.cse(x) * x + self.sse(x) * x
```

---

## 📈 訓練結果分析

### ResNet34 v2（最佳模型）

| Epoch | Train Loss | Val Dice | Val IoU |
|-------|------------|----------|---------|
| 1 | 1.381 | 0.417 | 0.269 |
| 10 | 0.791 | 0.577 | 0.419 |
| 20 | 0.536 | 0.601 | 0.442 |
| **24** | 0.464 | **0.610** ⭐ | **0.450** ⭐ |
| 30 | 0.434 | 0.595 | 0.435 |

### ResNet50 v4（過擬合問題）

| Epoch | Train Loss | Val Loss | Val Dice |
|-------|------------|----------|----------|
| 1 | 1.334 | 1.249 | 0.416 |
| 19 | 0.600 | 0.912 | **0.577** |
| 30 | 0.454 | **1.079** ↑ | 0.568 |

**問題**: Val Loss 持續上升 = 嚴重過擬合

**原因**: 
- ResNet50 參數量大，資料量不足
- 沒有使用 supplemental 資料

---

## 🎯 推理與提交

### 最佳閾值配置 (v2 模型)
```python
MASK_HIGH_THRESHOLD = 0.5   # Mask 高閾值
MASK_LOW_THRESHOLD = 0.3    # Mask 低閾值
MIN_OBJECT_SIZE = 100       # 最小連通區域大小
MIN_FORGERY_RATIO = 0.1     # 最小偽造區域比例（低於此視為 authentic）
```

### Kaggle 提交步驟

1. **上傳模型到 Kaggle Dataset**
   - 創建 Dataset: `muciforgery-detection-models`
   - 上傳 `best_model.pth`（重命名為 `best_model_v2.pth` 等）

2. **創建 Notebook**
   - 添加 Dataset 作為 Input
   - 複製 `kaggle_resnet_universal.py` 內容
   - 修改 `MODEL_PATH` 為正確的檔名
   - 啟用 GPU
   - 運行並提交

3. **提交結果格式**
   ```csv
   case_id,annotation
   1,authentic
   2,"[123, 4, 567, 8]"
   3,"[100, 50];[200, 30]"
   ```

---

## 🔬 實驗記錄

### 閾值調整實驗 (v2 模型)

| MIN_FORGERY_RATIO | Public Score |
|-------------------|--------------|
| **0.10** | **0.293** ⭐ |
| 0.05 | 0.281 |
| 0.08 | 待測試 |
| 0.12 | 待測試 |

### 模型比較

| 模型 | Val Dice | Public Score | 狀態 |
|------|----------|--------------|------|
| ResNet34-UNet + SCSE | 0.610 | 0.293 | ✅ 最佳 |
| ResNet50-UNet + SCSE | 0.577 | 0.209 | ❌ 過擬合 |
| + Authentic 訓練 | ~0.58 | 0.220 | ❌ 效果差 |

### ResNet50 過擬合診斷

測試圖像的預測機率分析：
```
Case 45:
  mean_prob: 0.0100  ← 極低！
  max_prob: 0.1957   ← 遠低於閾值 0.5
  > 0.5 pixels: 0    ← 沒有任何像素超過閾值
```

**結論**: ResNet50 模型預測機率極低，導致所有圖像都被判為 authentic

---

## 🔑 關鍵發現

### 1. Supplemental 資料的重要性
組員使用了 `supplemental_images` 和 `supplemental_masks`，這是分數差距的關鍵！

```python
# 組員的資料載入
search_dirs = [Config.TRAIN_IMG_DIR, Config.SUPP_IMG_DIR]  # 包含補充資料
```

### 2. 訓練輪數
- 組員: 15 epochs
- 我們: 30 epochs → 可能過擬合

### 3. Authentic 訓練的陷阱
加入 authentic 圖像訓練反而讓分數下降：
- 模型學到「預測全 0 是安全的」
- 對偽造區域的敏感度下降

---

## 🚀 待嘗試的改進

### 短期（不需重訓練）
- [ ] 調整閾值：`MIN_FORGERY_RATIO` = 0.08, 0.12
- [ ] 調整閾值：`MASK_HIGH_THRESHOLD` = 0.45, 0.55

### 中期（需重訓練）⚠️ 優先！
- [ ] **加入 supplemental 資料訓練**
- [ ] 減少 epochs 到 15（避免過擬合）
- [ ] 嘗試更強的資料增強 (Affine, Blur, Compression)

### 長期
- [ ] Ensemble 多個模型
- [ ] 使用 EfficientNet 作為 backbone
- [ ] 嘗試 Focal Loss

---

## 📁 檔案說明

| 檔案 | 說明 |
|------|------|
| `train_improved.py` | 主要訓練腳本，支援 ResNet34/50/101 |
| `kaggle_resnet_universal.py` | Kaggle 推理腳本（通用版） |
| `kaggle_resnet34unet_v2.py` | v2 專用推理腳本 |
| `kaggle_tta.py` | TTA 增強版推理（舊版，需 smp） |
| `kaggle_final_v3.py` | 最早成功提交的版本 |

---

## 🐛 已解決的問題

### 1. RLE 格式錯誤 (Submission Scoring Error)
- **症狀**: 提交後顯示 Scoring Error
- **原因**: 使用空格分隔而非 JSON array
- **解決**: 改用 `json.dumps()` 和分號分隔多個區域

### 2. ResNet50 預測全為 authentic
- **症狀**: 所有圖像都輸出 "authentic"
- **原因**: 模型過擬合，預測機率極低 (max=0.19)
- **解決**: 使用 ResNet34 或加入更多資料

### 3. 訓練找不到資料
- **症狀**: `Found 0 forged images`
- **原因**: mask 目錄名稱錯誤（`masks` vs `train_masks`）
- **解決**: 修改路徑為 `train_masks`

### 4. Kaggle Notebook GPU 未啟用
- **症狀**: Device 顯示 CPU，推理極慢
- **原因**: 未在 Settings 中啟用 GPU
- **解決**: Settings → Accelerator → GPU T4 x2

---

## 📊 組員程式碼分析

組員達到 0.303 分的關鍵差異：

| 項目 | 我們的版本 | 組員版本 |
|------|-----------|---------|
| 資料來源 | 只用 train_images/forged | **train + supplemental** |
| Epochs | 30 | **15** |
| Backbone | ResNet34 | **ResNet50** |
| SCSE | 有 | 無 |
| 增強 | 基本 | **Affine, Blur, Compression** |

**結論**: supplemental 資料 + 較少 epochs 可能是關鍵

---

## 📚 參考資料

- [SCSE 論文: Concurrent Spatial and Channel Squeeze & Excitation](https://arxiv.org/abs/1803.02579)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Kaggle Competition Discussion](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection/discussion)

---

## 👥 團隊

- **訓練與實驗**: Muci (M143040043)
- **課程**: CSE544 Computer Vision and Deep Learning, NCKU

---

## 📅 更新日誌

| 日期 | 更新內容 |
|------|---------|
| 2024-12-07 | 新增 supplemental 資料支援 |
| 2024-12-07 | 分析 ResNet50 過擬合問題 |
| 2024-12-06 | 達到 0.293 最佳分數 (v2) |
| 2024-12-06 | 修復 RLE 格式，首次成功提交 |
| 2024-12-05 | 開始競賽 |

---

*最後更新: 2024-12-07*
