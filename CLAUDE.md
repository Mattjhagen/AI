# Shaggoth AI Training System

## Project Overview

This is a terminal-based training interface for the DeepSeek-R1 671B parameter model using LoRA/QLoRA fine-tuning. The project includes both a Gradio web interface and a TUI (terminal user interface) for monitoring training progress.

## Key Files

### Main Applications
- **`shaggoth_tty.py`** - Terminal UI (TUI) for monitoring training with live resource graphs
- **`app.py`** - Gradio web interface for model setup, dataset upload, and training control
- **`app_demo.py`** - Demo version of the Gradio interface

### Training & Monitoring
- **`start_trainer.sh`** - Shell script to launch training process
- **`monitor_gpu.py`** - GPU monitoring script (currently CPU-only on R410)
- **`generate_dataset.py`** - Script to create training datasets in JSONL format

### Data & Configuration
- **`sample_dataset.jsonl`** - Example training data format
- **`datasets/`** - Directory for training datasets
- **`deploy/`** - Deployment configurations
- **`.env`** - Environment variables (contains Hugging Face API key)

### Documentation
- **`README.md`** - Full project documentation
- **`HARDWARE_REQUIREMENTS.md`** - System requirements and specs
- **`QUICKSTART.txt`** - Quick start guide
- **`PROJECT_COMPLETE.md`** - Project completion notes
- **`TEST_WITHOUT_TRAINING.md`** - Testing instructions
- **`VASTAI_SETUP.md`** - Setup guide for vast.ai GPU rental

## Current Status

### Hardware Context
- **R410**: 16GB RAM, no NVIDIA GPU - where project was developed
- **R510**: 40GB RAM, no NVIDIA GPU - target deployment server for CPU training
- **Model**: DeepSeek-R1 (671B parameters, 37B activated MoE)

### Recent Changes
- Project transferred from R410 to R510 via rsync (excluding venv and tar.gz files)
- TUI interface (`shaggoth_tty.py`) shows "No CUDA GPU detected" and "Model path not found"
- Virtual environment needs to be recreated on R510

## Setup Instructions

### On R510 (Target Server)

1. **Create virtual environment**
   ```bash
   cd ~/AI
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download DeepSeek-R1 model** (if not already present)
   ```bash
   # Model should be in ./DeepSeek-R1/ directory
   # Use Hugging Face CLI or download manually
   ```

4. **Run the TUI interface**
   ```bash
   python3 shaggoth_tty.py
   ```

5. **Or run the web interface**
   ```bash
   python3 app.py
   # Access at http://localhost:7860
   ```

## Training Data Format

Training data must be in JSONL format with instruction-input-output structure:

```jsonl
{"instruction": "Your task", "input": "Optional input", "output": "Expected output"}
```

## Architecture

### LoRA Fine-Tuning
- Reduces trainable parameters from 671B → ~8M (0.001%)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 8-bit quantization for memory efficiency
- Output: Lightweight adapter weights (~50MB)

### CPU Training Considerations
- Training on CPU is 100-1000x slower than GPU
- Requires 40GB+ RAM (R510 meets this requirement)
- Batch size should be kept small (1-2)
- Consider using vast.ai or similar for GPU-accelerated training

## Known Issues

1. **Model Path**: TUI shows "Model path not found" - need to verify DeepSeek-R1 model location
2. **No GPU**: Both R410 and R510 lack NVIDIA GPUs, limiting to CPU training
3. **Virtual Environment**: Needs to be recreated on R510 after transfer

## Next Steps

1. Set up virtual environment on R510
2. Verify/download DeepSeek-R1 model weights
3. Test TUI interface on R510 with 40GB RAM
4. Create or import training datasets
5. Configure training parameters for CPU execution
6. Run initial fine-tuning experiments

## Environment Variables

The `.env` file should contain:
```bash
HUGGING_FACE_API=your_hf_api_key_here
```

## Resources

- DeepSeek-R1 Paper: https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf
- DeepSeek-R1 GitHub: https://github.com/deepseek-ai/DeepSeek-R1
- Hugging Face Model: https://huggingface.co/deepseek-ai/DeepSeek-R1
- LoRA Paper: https://arxiv.org/abs/2106.09685

## Development Notes

- Primary development was on R410
- Target deployment is R510 for better RAM capacity
- Consider GPU cloud services (vast.ai, Lambda Labs) for production training
- TUI provides real-time monitoring of CPU/RAM/GPU during training
- Web interface allows full control over model loading, dataset upload, and training parameters
