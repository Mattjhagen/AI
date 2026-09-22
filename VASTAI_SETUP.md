# Vast.ai Setup Guide - DeepSeek-R1 Training

## Step 1: Create Account & Add Credit

1. Go to: **https://vast.ai**
2. Click "Sign Up" (top right)
3. Create account with email
4. Verify email
5. Go to **Account → Billing**
6. Click "Add Credit"
7. Add **$10** (minimum, enough for 6+ training runs)

## Step 2: Package Training Files

Run this on your R510:

```bash
cd ~/AI/deploy
bash train_on_cloud.sh
```

This creates: `shaggoth-training-YYYYMMDD-HHMMSS.tar.gz` (~20MB)

## Step 3: Find GPU Instance

1. Go to: **https://vast.ai/console/create/**
2. Configure search:
   - **GPU:** A100 80GB (or A100 40GB)
   - **RAM:** 64GB minimum
   - **Storage:** 500GB minimum
   - **vCPUs:** 4+ recommended
   
3. **Sort by:** "DPH ($/hr)" - cheapest first
4. Look for: $1.00-$2.00/hr instances
5. Check:
   - ✅ "Available" status
   - ✅ "PyTorch" or "CUDA 11.8+" image
   - ✅ Good reliability score (>98%)

## Step 4: Launch Instance

### Recommended Template:
- **Image:** `pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel`
- **Or:** `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel`

### Launch Options:
1. Click "Rent" on your chosen instance
2. **Image/Template:**
   - Select: "pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel"
   - Or paste: `pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel`
3. **On-start script:** (optional, leave blank for now)
4. **Disk Space:** 500 GB minimum
5. Click "Rent"

Wait 1-2 minutes for instance to start.

## Step 5: Connect to Instance

Once instance shows "Running":

1. Click "Connect" button
2. Copy the SSH command, looks like:
   ```bash
   ssh -p 12345 root@ssh6.vast.ai -L 8080:localhost:8080
   ```

3. Run it from your R510:
   ```bash
   ssh -p 12345 root@ssh6.vast.ai
   ```

First time will ask: "Are you sure (yes/no)?" → type **yes**

## Step 6: Upload Training Package

From your R510 (in a new terminal):

```bash
# Find your training package
cd ~/AI
ls -lh shaggoth-training-*.tar.gz

# Upload to Vast.ai (replace PORT and FILE with actual values)
scp -P 12345 shaggoth-training-*.tar.gz root@ssh6.vast.ai:/workspace/
```

**Note:** Use `-P` (capital P) for port with scp!

## Step 7: Extract and Prepare

Back in your SSH session on Vast.ai:

```bash
# Go to workspace
cd /workspace

# Extract package
tar -xzf shaggoth-training-*.tar.gz
cd shaggoth-training

# Check files
ls -lh

# Install dependencies
pip install -r requirements.txt
```

## Step 8: Download Model (30-60 minutes)

If DeepSeek-R1 isn't included in your package:

```bash
# Install Hugging Face CLI
pip install -U huggingface_hub

# Download model (this is BIG - 300GB+)
echo "⚠️  This will take 30-60 minutes!"
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir DeepSeek-R1
```

**Alternative:** If model download fails or is too slow, you can:
1. Use a smaller model for testing first
2. Choose an instance that has DeepSeek pre-cached

## Step 9: Start Training Interface

```bash
# Start Gradio interface
python3 app.py
```

You'll see:
```
🚀 Starting Shaggoth AI Trainer...
📊 Access the interface at: http://localhost:7860
Running on local URL:  http://0.0.0.0:7860
```

## Step 10: Access Interface via SSH Tunnel

From your R510, create SSH tunnel:

```bash
# Replace PORT with your instance port
ssh -p 12345 -L 7860:localhost:7860 root@ssh6.vast.ai
```

Now open in your browser: **http://localhost:7860**

## Step 11: Train Model

In the Gradio interface:

### Model Setup Tab:
1. Path: `./DeepSeek-R1`
2. Check "Use 8-bit Quantization"
3. Click **"Load Model"** (takes 5-10 minutes)
4. LoRA Rank: 8, Alpha: 16, Dropout: 0.05
5. Click **"Setup LoRA"**

### Dataset Tab:
1. Click "Upload Dataset"
2. Select `datasets/shaggoth_mixed.jsonl`
3. Click **"Load Dataset"**
4. Verify preview shows 405 examples

### Training Tab:
1. Output Directory: `./shaggoth-trained`
2. Epochs: 3
3. Batch Size: 4
4. Learning Rate: 2e-4
5. Save Steps: 100
6. Eval Steps: 100
7. Click **"Start Training"**

### Monitor Training:
- Watch "Training Status" field
- Check "Refresh Logs" periodically
- Training takes **2-4 hours**

## Step 12: Download Trained Model

After training completes:

From your R510:

```bash
# Download checkpoint (replace PORT with your instance port)
scp -P 12345 -r root@ssh6.vast.ai:/workspace/shaggoth-training/shaggoth-trained ~/AI/

# Verify
ls -lh ~/AI/shaggoth-trained/
```

Should see:
- `adapter_config.json`
- `adapter_model.bin`
- `training_args.bin`
- `checkpoint-*/` directories

## Step 13: Stop Instance (IMPORTANT!)

**Don't forget to stop your instance when done!**

1. Go to: **https://vast.ai/console/instances/**
2. Click "Destroy" on your instance
3. Confirm

**Or via CLI:**
```bash
# From Vast.ai instance
exit  # Exit SSH

# Then destroy via web interface
```

## Cost Tracking

### During Training:
- Check: https://vast.ai/console/instances/
- Shows: Current cost, $/hr rate, elapsed time

### Example Costs:
- A100 80GB @ $1.50/hr × 4 hours = **$6.00**
- A100 40GB @ $1.20/hr × 5 hours = **$6.00**
- Model download time: ~$1.00 (if 30-40 min)
- **Total:** ~$6-8 for complete training

## Troubleshooting

### "Permission denied (publickey)"
```bash
# Generate SSH key if needed
ssh-keygen -t rsa -b 4096
# Add key to Vast.ai account settings
```

### "CUDA out of memory"
In training interface:
- Enable 4-bit quantization (instead of 8-bit)
- Reduce batch size to 2 or 1
- Reduce LoRA rank to 4

### "Model download stuck"
```bash
# Resume download
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir DeepSeek-R1 --resume-download
```

### "Training crashes"
```bash
# Check GPU
nvidia-smi

# Check logs
tail -50 trainer.log

# Restart with lower memory settings
```

### "Can't access Gradio interface"
```bash
# Make sure SSH tunnel is active
ssh -p 12345 -L 7860:localhost:7860 root@ssh6.vast.ai

# Then: http://localhost:7860
```

## Alternative: CLI Training (No Gradio)

If Gradio has issues, train directly:

```bash
cd /workspace/shaggoth-training

# Create simple training script
cat > train_direct.py << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("./DeepSeek-R1", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "./DeepSeek-R1",
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)

# Setup LoRA
print("Setting up LoRA...")
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files="datasets/shaggoth_mixed.jsonl", split="train")

# Train
print("Starting training...")
training_args = TrainingArguments(
    output_dir="./shaggoth-trained",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    bf16=True,
    save_steps=100,
    logging_steps=10
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
trainer.save_model("./shaggoth-trained")

print("✅ Training complete!")
EOF

# Run training
python3 train_direct.py
```

## Next Steps After Training

1. **Download checkpoint** to R510
2. **Deploy to EC2:**
   ```bash
   cd ~/AI/deploy
   bash deploy_to_ec2.sh ~/AI/shaggoth-trained
   ```
3. **Test inference** from mobile apps
4. **Monitor performance** in production

## Quick Reference

**Vast.ai Console:** https://vast.ai/console/instances/  
**Training Package:** `~/AI/shaggoth-training-*.tar.gz`  
**Checkpoint Output:** `/workspace/shaggoth-training/shaggoth-trained/`  
**Estimated Time:** 4-5 hours total (1hr setup, 3-4hr training)  
**Estimated Cost:** $6-8 total  

## Support

- Vast.ai Discord: https://discord.gg/vast
- DeepSeek Docs: https://github.com/deepseek-ai/DeepSeek-R1
- Your guides: `~/AI/HARDWARE_REQUIREMENTS.md`, `~/AI/deploy/README.md`
