# Shaggoth AI Trainer - DeepSeek-R1

Fine-tuning interface for DeepSeek-R1 model using LoRA/QLoRA parameter-efficient training.

## Model Specs

- **Model**: DeepSeek-R1
- **Parameters**: 671B total (37B activated MoE)
- **Architecture**: Based on DeepSeek-V3-Base
- **Capabilities**: Chain-of-thought reasoning, self-verification, reflection
- **Performance**: Comparable to OpenAI o1 on math/code/reasoning benchmarks

## Features

### 1. Model Setup
- Load DeepSeek-R1 with 8-bit or 4-bit quantization
- Automatic device mapping for multi-GPU systems
- LoRA configuration (rank, alpha, dropout)
- View trainable vs total parameter counts

### 2. Dataset Management
- Upload training data in JSONL or JSON format
- Expected format: `{"instruction": "...", "input": "...", "output": "..."}`
- Live preview of uploaded data
- Automatic tokenization and preprocessing

### 3. Training Pipeline
- Configurable training parameters:
  - Epochs, batch size, learning rate
  - Save and evaluation intervals
  - Gradient accumulation
- Real-time training logs
- Automatic checkpoint management
- Mixed precision training (bf16)

### 4. Model Testing
- Test trained model with custom prompts
- Adjustable generation parameters:
  - Max length, temperature, top_p
- Live inference from trained checkpoints

## Installation

```bash
cd ~/AI

# Install dependencies
pip install -r requirements.txt

# Make sure you have CUDA installed for GPU acceleration
```

## Usage

### Start the Interface

```bash
python3 app.py
```

Access at: http://localhost:7860

### Training Workflow

1. **Load Model**
   - Navigate to "Model Setup" tab
   - Keep default path: `./DeepSeek-R1`
   - Enable 8-bit quantization (recommended)
   - Click "Load Model"

2. **Setup LoRA**
   - Configure LoRA rank (8-64, default: 8)
   - Set alpha (8-128, default: 16)
   - Set dropout (0-0.2, default: 0.05)
   - Click "Setup LoRA"

3. **Upload Dataset**
   - Go to "Dataset" tab
   - Upload JSONL file with training data
   - Preview first 5 examples
   - Click "Load Dataset"

4. **Configure Training**
   - Switch to "Training" tab
   - Set output directory
   - Configure epochs, batch size, learning rate
   - Adjust save/eval intervals
   - Click "Start Training"

5. **Test Model**
   - Go to "Test Model" tab
   - Enter a test prompt
   - Adjust generation parameters
   - Click "Generate"

## Training Data Format

Your training data should be in JSONL format:

```jsonl
{"instruction": "Explain quantum entanglement", "input": "", "output": "Quantum entanglement is..."}
{"instruction": "Write Python code", "input": "Create a function to sort a list", "output": "def sort_list(items):\n    return sorted(items)"}
{"instruction": "Solve this math problem", "input": "What is 15% of 200?", "output": "15% of 200 is 30. Calculation: 200 × 0.15 = 30"}
```

## LoRA Fine-Tuning

This trainer uses LoRA (Low-Rank Adaptation) to efficiently fine-tune the massive DeepSeek-R1 model:

- **Trainable Parameters**: Reduces from 671B → ~8M (0.001%)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Memory Efficiency**: 8-bit quantization reduces memory requirements
- **Output**: Lightweight adapter weights (~50MB) instead of full model

## Hardware Requirements

### Minimum (8-bit quantization)
- GPU: 48GB+ VRAM (RTX 6000 Ada, A100, H100)
- RAM: 64GB
- Storage: 200GB+ for model and checkpoints

### Recommended
- GPU: 80GB+ VRAM (A100 80GB, H100)
- RAM: 128GB
- Storage: 500GB SSD

### CPU Training (Not Recommended)
- Possible but extremely slow
- Requires 128GB+ RAM
- Training time: 100-1000x slower than GPU

## Output Structure

After training, you'll find:

```
./shaggoth-trained/
├── adapter_config.json       # LoRA configuration
├── adapter_model.bin          # Trained adapter weights
├── training_args.bin          # Training arguments
├── checkpoint-100/            # Intermediate checkpoints
├── checkpoint-200/
└── ...
```

## Next Steps

1. Create training datasets from Shaggoth knowledge base
2. Run initial fine-tuning experiments
3. Integrate trained model into Shaggoth API
4. Monitor GPU memory usage during training
5. Deploy best checkpoint to production

## Environment Variables

Create a `.env` file with:

```bash
HUGGING_FACE_API=your_hf_api_key_here
```

## Troubleshooting

### Out of Memory
- Enable 4-bit quantization instead of 8-bit
- Reduce batch size
- Reduce LoRA rank
- Enable gradient checkpointing

### Model Loading Errors
- Verify all 163 safetensors files downloaded
- Check CUDA compatibility
- Ensure transformers/torch versions match

### Training Slow
- Use GPU instead of CPU
- Increase batch size (if memory allows)
- Reduce sequence length
- Use gradient accumulation

## Resources

- [DeepSeek-R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
- [Hugging Face Model](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## License

MIT - See DeepSeek-R1 LICENSE file for model license
