#!/bin/bash
# Train DeepSeek-R1 on Cloud GPU (Vast.ai, RunPod, AWS)
# This script prepares and uploads training artifacts to a cloud GPU instance

set -e

echo "🚀 DeepSeek-R1 Cloud Training Setup"
echo "======================================"

# Configuration
TRAINING_DIR="$HOME/AI"
PACKAGE_NAME="shaggoth-training-$(date +%Y%m%d-%H%M%S).tar.gz"
DATASETS_DIR="$TRAINING_DIR/datasets"
CHECKPOINT_DIR="$TRAINING_DIR/shaggoth-trained"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
error() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
    exit 1
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if [ ! -d "$DATASETS_DIR" ]; then
    error "Datasets directory not found: $DATASETS_DIR"
fi

if [ ! -f "$TRAINING_DIR/app.py" ]; then
    error "Training app not found: $TRAINING_DIR/app.py"
fi

if [ ! -f "$TRAINING_DIR/requirements.txt" ]; then
    error "Requirements file not found: $TRAINING_DIR/requirements.txt"
fi

success "Prerequisites OK"

# Package training files
echo ""
echo "📦 Packaging training files..."

cd "$TRAINING_DIR"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
PACKAGE_DIR="$TEMP_DIR/shaggoth-training"
mkdir -p "$PACKAGE_DIR"

# Copy files
cp -r datasets/ "$PACKAGE_DIR/"
cp app.py requirements.txt "$PACKAGE_DIR/"
cp -r DeepSeek-R1/ "$PACKAGE_DIR/" 2>/dev/null || warn "DeepSeek-R1 model not found locally (will need to download on cloud instance)"

# Create training script
cat > "$PACKAGE_DIR/train.sh" << 'EOF'
#!/bin/bash
# Auto-training script for cloud GPU

set -e

echo "🎮 Setting up training environment..."

# Install dependencies
pip install -r requirements.txt

# Check GPU
python3 << 'PYTHON'
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ No GPU detected!")
    exit(1)
PYTHON

# Download model if not present
if [ ! -d "DeepSeek-R1" ]; then
    echo "📥 Downloading DeepSeek-R1 model..."
    echo "⚠️  This is a 300GB+ download and will take time!"

    # Install Hugging Face CLI
    pip install -U huggingface_hub

    # Download model
    huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir DeepSeek-R1
fi

echo ""
echo "🚀 Starting training interface..."
echo "Access at: http://localhost:7860"
echo ""

# Start training
python3 app.py
EOF

chmod +x "$PACKAGE_DIR/train.sh"

# Create README
cat > "$PACKAGE_DIR/README.txt" << EOF
Shaggoth AI - DeepSeek-R1 Training Package
==========================================

This package contains everything needed to train DeepSeek-R1 on your Shaggoth knowledge base.

Quick Start:
-----------
1. Extract this archive:
   tar -xzf $PACKAGE_NAME

2. Run training:
   cd shaggoth-training
   bash train.sh

3. Access interface:
   http://localhost:7860 (or your instance IP)

4. After training, download checkpoint:
   scp -r shaggoth-trained/ your-local-machine:~/

Requirements:
------------
- GPU: 48GB+ VRAM (A100/H100 recommended)
- RAM: 64GB+
- Storage: 500GB+
- CUDA 11.8+

Cloud Providers:
---------------
- Vast.ai: https://vast.ai (A100 80GB: ~$1.50/hr)
- RunPod: https://runpod.io (A100 80GB: ~$2.89/hr)
- AWS p4d.24xlarge: ~$32/hr

Datasets Included:
-----------------
- shaggoth_general.jsonl: 1000 examples
- shaggoth_reasoning.jsonl: 548 examples
- shaggoth_mixed.jsonl: 405 examples

Training Steps:
--------------
1. Model Setup → Load with 8-bit quantization
2. LoRA Setup → Default settings (rank=8, alpha=16)
3. Dataset → Upload shaggoth_mixed.jsonl
4. Training → Start with default parameters
5. Wait 2-4 hours for completion
6. Test model with sample prompts
7. Download checkpoint

Estimated Time:
--------------
- Model download: 30-60 minutes
- Training: 2-4 hours
- Total: 3-5 hours

Estimated Cost:
--------------
- Vast.ai A100: $6-8 total
- RunPod A100: $10-15 total
- AWS: $64-128 total

Support:
-------
See HARDWARE_REQUIREMENTS.md and DEEPSEEK_INTEGRATION.md for details
EOF

# Create package
echo "Creating archive: $PACKAGE_NAME"
tar -czf "$TRAINING_DIR/$PACKAGE_NAME" -C "$TEMP_DIR" shaggoth-training

# Cleanup
rm -rf "$TEMP_DIR"

success "Package created: $TRAINING_DIR/$PACKAGE_NAME"

# Show instructions
echo ""
echo "======================================"
echo "📤 NEXT STEPS"
echo "======================================"
echo ""
echo "1️⃣  Choose a Cloud GPU Provider:"
echo "   - Vast.ai (cheapest): https://vast.ai"
echo "   - RunPod (easiest): https://runpod.io"
echo "   - AWS EC2: p3.2xlarge or p4d.24xlarge"
echo ""
echo "2️⃣  Launch GPU Instance:"
echo "   - Select A100 40GB or 80GB"
echo "   - Choose PyTorch template (CUDA 11.8+)"
echo "   - Note the instance IP address"
echo ""
echo "3️⃣  Upload Package:"
echo "   scp $PACKAGE_NAME user@instance-ip:/workspace/"
echo ""
echo "4️⃣  SSH and Extract:"
echo "   ssh user@instance-ip"
echo "   cd /workspace"
echo "   tar -xzf $PACKAGE_NAME"
echo ""
echo "5️⃣  Start Training:"
echo "   cd shaggoth-training"
echo "   bash train.sh"
echo ""
echo "6️⃣  Access Interface:"
echo "   http://instance-ip:7860"
echo ""
echo "7️⃣  After Training, Download Checkpoint:"
echo "   scp -r user@instance-ip:/workspace/shaggoth-training/shaggoth-trained ."
echo ""
echo "======================================"
echo ""
echo "Package ready: $TRAINING_DIR/$PACKAGE_NAME"
echo "Package size: $(du -h "$TRAINING_DIR/$PACKAGE_NAME" | cut -f1)"
echo ""
