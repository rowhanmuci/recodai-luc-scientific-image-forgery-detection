#!/bin/bash
# 快速開始腳本
# Scientific Image Forgery Detection

echo "=========================================="
echo "Scientific Image Forgery Detection Setup"
echo "=========================================="

# 1. 安裝依賴
echo ""
echo "📦 Installing dependencies..."
pip install torch torchvision --break-system-packages -q
pip install segmentation-models-pytorch albumentations opencv-python pandas numpy tqdm matplotlib scikit-learn --break-system-packages -q

echo "✅ Dependencies installed!"

# 2. 檢查資料結構
echo ""
echo "📁 Checking data structure..."

DATA_DIR="./data"

if [ -d "$DATA_DIR/train_images" ]; then
    echo "  ✅ train_images found: $(ls $DATA_DIR/train_images | wc -l) files"
else
    echo "  ❌ train_images not found"
fi

if [ -d "$DATA_DIR/train_masks" ]; then
    echo "  ✅ train_masks found: $(ls $DATA_DIR/train_masks | wc -l) files"
else
    echo "  ❌ train_masks not found"
fi

if [ -d "$DATA_DIR/test_images" ]; then
    echo "  ✅ test_images found: $(ls $DATA_DIR/test_images | wc -l) files"
else
    echo "  ❌ test_images not found"
fi

if [ -d "$DATA_DIR/supplemental_images" ]; then
    echo "  ✅ supplemental_images found: $(ls $DATA_DIR/supplemental_images | wc -l) files"
else
    echo "  ⚠️  supplemental_images not found (optional)"
fi

# 3. 創建輸出目錄
echo ""
echo "📂 Creating output directories..."
mkdir -p outputs
mkdir -p logs

echo "✅ Directories created!"

# 4. 運行 EDA
echo ""
echo "🔍 Running EDA..."
python eda.py

echo ""
echo "=========================================="
echo "Setup complete! Next steps:"
echo "=========================================="
echo ""
echo "1. Train a model:"
echo "   python train.py --model unet --encoder resnet50 --epochs 50"
echo ""
echo "2. Or train with better encoder:"
echo "   python train.py --model unetpp --encoder efficientnet-b4 --epochs 50"
echo ""
echo "3. Generate submission:"
echo "   python inference.py --checkpoint outputs/best_model.pth"
echo ""
echo "4. With TTA for better results:"
echo "   python inference.py --checkpoint outputs/best_model.pth --tta"
echo ""
