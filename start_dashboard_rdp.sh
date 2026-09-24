#!/bin/bash
# Shaggoth AI Training Dashboard Launcher for RDP Sessions

cd /home/matt/AI

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies if needed
if ! python3 -c "import textual" 2>/dev/null; then
    echo "Installing required packages..."
    pip install textual psutil torch transformers
fi

# Launch the dashboard
echo "Starting Shaggoth AI Command Center..."
python3 shaggoth_tty.py
