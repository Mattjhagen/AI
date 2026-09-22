#!/bin/bash
# Shaggoth AI Trainer - Quick Start Script

echo "🚀 Starting Shaggoth AI Trainer..."
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from ~/AI/"
    exit 1
fi

# Check if DeepSeek-R1 model exists
if [ ! -d "DeepSeek-R1" ]; then
    echo "⚠️  Warning: DeepSeek-R1 directory not found"
    echo "   Please download the model from:"
    echo "   https://huggingface.co/deepseek-ai/DeepSeek-R1"
    echo ""
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python3 -c "import gradio" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "✅ Dependencies already installed"
fi

# Check GPU availability
echo ""
echo "🎮 Checking GPU availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  No GPU detected - training will be very slow on CPU")
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Shaggoth AI Trainer - DeepSeek-R1"
echo "  Access at: http://localhost:7860"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the trainer
python3 app.py
