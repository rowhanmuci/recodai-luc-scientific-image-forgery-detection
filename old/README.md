# 🔬 Scientific Image Forgery Detection

Kaggle 競賽：[Recod.ai/LUC - Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)

## 📋 比賽概述

| 項目 | 說明 |
|------|------|
| **目標** | 檢測生物醫學研究圖像是否經過偽造 (Copy-Move Forgery) |
| **任務類型** | 二分類 (Binary Classification) |
| **預測目標** | `authentic` (真實) 或 `forged` (偽造) |
| **獎金** | $55,000 |
| **截止日期** | January 8, 2026 |

## 📁 資料結構

```
data/
├── train_images/
│   ├── authentic/          # 真實圖像 (2,377 張)
│   └── forged/             # 偽造圖像 (2,751 張)
├── train_masks/            # 偽造區域遮罩 (.npy 格式，對應 forged 圖像)
├── test_images/            # 測試圖像
├── supplemental_images/    # 補充圖像 (48 張)
├── supplemental_masks/     # 補充遮罩 (48 張)
└── sample_submission.csv   # 提交範例
```

### 資料特點

| 特徵 | 說明 |
|------|------|
| 總訓練樣本 | 5,128 張 |
| 類別比例 | Authentic 46.4% / Forged 53.6% (接近平衡) |
| 圖像類型 | SEM 顯微鏡、螢光顯微鏡、Western Blot、統計圖等 |
| 圖像大小 | 不固定 (113×64 ~ 3888×3888) |
| Mask 格式 | `.npy`，Shape: `(N, H, W)`，值: 0 或 1 |
| 偽造區域大小 | 約 0.2% ~ 7% 的圖像面積 |

## 📤 提交格式

這是一個 **Code Competition**，需要在 Kaggle Notebook 中提交。

```csv
case_id,annotation
1,authentic
2,"[123 4 200 10]"
```

| 預測結果 | annotation 格式 |
|----------|-----------------|
| 真實圖像 | `authentic` |
| 偽造圖像 | RLE 編碼的 mask，如 `"[start1 length1 start2 length2 ...]"` |

**重要**：這是一個 **分類 + 分割** 的組合任務！
- 先判斷圖像是否為偽造
- 如果是偽造，還需要提供偽造區域的 RLE 編碼 mask

## 🚀 完整流程

### 方案 A：分類 + 分割（推薦，更高分數）

```bash
# 1. 訓練分類器
python train_classifier.py --model efficientnet_b3 --image_size 384 --batch_size 8 --epochs 50 --mixup

# 2. 訓練分割器
python train.py --model unet --encoder efficientnet-b3 --image_size 384 --epochs 50

# 3. 上傳模型到 Kaggle Dataset，然後在 Notebook 中運行 kaggle_submission.py
```

### 方案 B：僅分割（簡單方案）

```bash
# 1. 訓練分割器（預測偽造區域，如果沒有偽造區域則為 authentic）
python train.py --model unet --encoder efficientnet-b3 --image_size 512 --epochs 50

# 2. 上傳模型到 Kaggle，使用分割結果生成提交
```

---

## 🏃 快速開始（本地訓練）

### 1. 環境設置

```bash
pip install torch torchvision timm albumentations opencv-python pandas numpy tqdm matplotlib scikit-learn
```

### 2. 資料探索 (EDA)

```bash
python eda.py
```

輸出：
- `forgery_distribution.png` - 資料分佈圖
- `sample_visualization.png` - 樣本視覺化（含偽造區域標註）

### 3. 檢查 Mask 格式

```bash
python check_masks.py
```

### 4. 訓練模型

**快速測試 (驗證 pipeline)：**
```bash
python train_classifier.py --model efficientnet_b0 --image_size 384 --batch_size 16 --epochs 5
```

**正式訓練 (推薦)：**
```bash
python train_classifier.py --model efficientnet_b3 --image_size 384 --batch_size 8 --epochs 50 --mixup
```

**高性能配置：**
```bash
python train_classifier.py --model efficientnet_b4 --image_size 512 --batch_size 4 --epochs 80 --mixup --label_smoothing 0.1
```

### 5. 生成提交檔案

```bash
python inference_classifier.py --checkpoint outputs/best_classifier.pth --tta
```

輸出：
- `outputs/submission.csv` - 提交檔案
- `outputs/predictions_with_probs.csv` - 含機率的完整預測
- `outputs/prediction_distribution.png` - 預測分佈圖

## 📂 檔案說明

### 核心檔案

| 檔案 | 說明 |
|------|------|
| `eda.py` | 資料探索與視覺化 |
| `check_masks.py` | 檢查 Mask 檔案格式與內容 |
| `kaggle_submission.py` | **Kaggle Notebook 提交腳本** ⭐ |

### 分類任務

| 檔案 | 說明 |
|------|------|
| `dataset_classification.py` | 分類任務 Dataset |
| `train_classifier.py` | 分類模型訓練腳本 |
| `inference_classifier.py` | 分類推理（本地測試用） |

### 分割任務

| 檔案 | 說明 |
|------|------|
| `dataset.py` | 分割任務 Dataset |
| `train.py` | 分割模型訓練腳本 |
| `inference.py` | 分割推理腳本 |
| `losses.py` | 損失函數 (Dice, Focal, Tversky 等) |

### 輔助檔案

| 檔案 | 說明 |
|------|------|
| `utils.py` | 工具函數 |
| `advanced_config.py` | 進階訓練配置與技巧 |

## ⚙️ 訓練參數說明

```bash
python train_classifier.py [OPTIONS]
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | efficientnet_b0 | 模型架構 (timm 支援的模型) |
| `--image_size` | 384 | 輸入圖像大小 |
| `--batch_size` | 16 | 批次大小 |
| `--epochs` | 30 | 訓練輪數 |
| `--lr` | 1e-4 | 學習率 |
| `--mixup` | False | 使用 MixUp 資料增強 |
| `--label_smoothing` | 0.1 | Label Smoothing |
| `--patience` | 7 | Early Stopping 耐心值 |
| `--val_split` | 0.2 | 驗證集比例 |

## 🏆 推薦模型

| 模型 | 參數量 | 速度 | 性能 | 推薦場景 |
|------|--------|------|------|----------|
| efficientnet_b0 | 5.3M | ⚡⚡⚡ | ⭐⭐ | 快速實驗 |
| efficientnet_b3 | 12M | ⚡⚡ | ⭐⭐⭐ | **推薦** |
| efficientnet_b4 | 19M | ⚡ | ⭐⭐⭐⭐ | 追求高分 |
| convnext_tiny | 28M | ⚡⚡ | ⭐⭐⭐ | 現代架構 |
| resnet50 | 25M | ⚡⚡ | ⭐⭐⭐ | 經典穩定 |

## 📈 預期結果

| 配置 | 訓練時間 | 預估 F1 | 預估 AUC |
|------|----------|---------|----------|
| efficientnet_b0, 5 epochs | 5-10 分鐘 | ~0.65 | ~0.55 |
| efficientnet_b3, 50 epochs | 1-2 小時 | ~0.85-0.90 | ~0.90 |
| efficientnet_b4 + TTA, 80 epochs | 2-3 小時 | ~0.90+ | ~0.93+ |

## 💡 提升分數的技巧

### 資料增強
- ✅ MixUp (`--mixup`)
- ✅ 水平/垂直翻轉
- ✅ 旋轉、縮放
- ✅ 色彩抖動
- ✅ JPEG 壓縮模擬

### 訓練技巧
- ✅ Label Smoothing (`--label_smoothing 0.1`)
- ✅ Cosine Annealing / OneCycleLR
- ✅ Early Stopping
- ✅ Mixed Precision Training (自動啟用)

### 推理技巧
- ✅ Test Time Augmentation (`--tta`)
- 🔲 模型集成 (多模型投票)
- 🔲 閾值調整 (`--threshold`)

### 進階方法
- 🔲 使用 Mask 資訊輔助分類
- 🔲 多任務學習 (分類 + 分割)
- 🔲 K-Fold 交叉驗證
- 🔲 Pseudo Labeling

## ⚠️ 常見問題

### GPU 記憶體不足
```bash
# 減小 batch_size 和 image_size
python train_classifier.py --model efficientnet_b0 --image_size 256 --batch_size 4 --epochs 50
```

### 訓練太慢
```bash
# 使用較小的模型和圖像大小
python train_classifier.py --model efficientnet_b0 --image_size 256 --epochs 30
```

### 過擬合 (Train Loss 低但 Val Loss 高)
```bash
# 增加正則化
python train_classifier.py --model efficientnet_b3 --mixup --label_smoothing 0.2 --dropout 0.4
```

## 📚 參考資料

- [Competition Page](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)
- [timm 模型列表](https://huggingface.co/docs/timm/index)
- [Albumentations 文檔](https://albumentations.ai/docs/)
- [Copy-Move Forgery Detection 論文](https://arxiv.org/abs/2109.08503)

## 🎯 Kaggle 提交步驟

### Step 1: 本地訓練模型
```bash
# 訓練分類器
python train_classifier.py --model efficientnet_b3 --epochs 50 --mixup

# 訓練分割器
python train.py --model unet --encoder efficientnet-b3 --epochs 50
```

### Step 2: 上傳模型到 Kaggle
1. 在 Kaggle 創建一個新的 Dataset
2. 上傳 `outputs/best_classifier.pth` 和 `outputs/best_model.pth`

### Step 3: 創建 Kaggle Notebook
1. 新建 Notebook，加入你的模型 Dataset
2. 複製 `kaggle_submission.py` 的內容
3. 修改模型路徑指向你上傳的 Dataset
4. 運行並提交

### 注意事項
- CPU Notebook: ≤ 4 小時
- GPU Notebook: ≤ 4 小時
- **必須關閉網路存取**
- 輸出檔案必須命名為 `submission.csv`

## 📝 License

This project is for educational and competition purposes.

---

**Good luck! 🍀**