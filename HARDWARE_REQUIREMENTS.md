# Hardware Requirements & Training Options

## Current R510 Hardware

**CPU:** Intel Xeon E5620 (2.40GHz, 2010-era)  
**RAM:** 24GB  
**GPU:** Matrox G200eW (server management only, NO CUDA)  
**Status:** ❌ **CANNOT train DeepSeek-R1**

## DeepSeek-R1 Requirements

**Minimum (8-bit quantization):**
- GPU: 48GB+ VRAM (RTX 6000 Ada, A100, H100)
- RAM: 64GB
- Storage: 200GB+

**Your R510:** Has no training-capable GPU

## Solutions for Training

### Option 1: Cloud GPU Training (RECOMMENDED)
Use cloud GPU providers for training, then deploy the trained model:

**A. AWS EC2 (you already use this)**
```bash
# Launch p3.2xlarge (V100 16GB) - $3.06/hr
# Or p4d.24xlarge (A100 40GB x8) - ~$32/hr
aws ec2 run-instances --instance-type p3.2xlarge ...
```

**B. Vast.ai (Cheapest)**
- A100 40GB: $0.80-1.50/hr
- A100 80GB: $1.20-2.00/hr
- Website: https://vast.ai

**C. RunPod (Easy to use)**
- A100 40GB: $1.89/hr
- A100 80GB: $2.89/hr
- Website: https://runpod.io

**D. Lambda Labs**
- A100 40GB: $1.10/hr
- Website: https://lambdalabs.com

### Option 2: Use Smaller Models on R510 CPU
Train smaller models that work without GPU:

```bash
# Instead of DeepSeek-R1, use:
- TinyGPT (you already have this!)
- Qwen2.5 0.5B-7B (via Ollama)
- LLaMA 3.2 1B-3B
- Gemma 2 2B
```

### Option 3: Remote Training Setup
Keep R510 as dev environment, train remotely:

1. **Dev on R510:** Test code, datasets, inference
2. **Train on Cloud:** Use cloud GPU for fine-tuning
3. **Deploy:** Bring trained model back to R510 or EC2

### Option 4: GPU Upgrade ($$$$)
Add GPU to R510:
- Requires PCIe slot, power supply upgrade
- Cost: $500-5000 depending on GPU
- **Not recommended** - cloud GPUs more flexible

## Recommended Workflow

### For Immediate Training:

```bash
# 1. Package everything
cd ~/AI
tar -czf shaggoth-training.tar.gz datasets/ app.py requirements.txt

# 2. Launch Vast.ai A100 instance (cheapest)
# - Go to https://vast.ai
# - Select A100 40GB/80GB instance
# - Upload your files

# 3. SSH in and run training
ssh root@<vast-instance-ip>
cd /workspace
tar -xzf shaggoth-training.tar.gz
pip install -r requirements.txt
python3 app.py  # Or run training script directly
```

### For Production Setup:

```bash
# Use your existing AWS EC2 infrastructure
# Launch GPU instance only when training:

# 1. Start GPU instance
aws ec2 start-instances --instance-ids i-xxxxx

# 2. Train model
ssh aws-gpu
cd ~/AI && python3 app.py

# 3. Save checkpoint to S3
aws s3 cp ./shaggoth-trained/ s3://shaggoth-models/ --recursive

# 4. Stop GPU instance (save $$)
aws ec2 stop-instances --instance-ids i-xxxxx

# 5. Load model on your t3.small API server
# (inference is cheap, training is expensive)
```

## What You CAN Do on R510

✅ **Dataset preparation** (already done!)  
✅ **Model testing/inference** (with Ollama qwen2.5-coder:7b)  
✅ **API development** (Shaggoth API integration)  
✅ **Small model training** (TinyGPT, your existing setup)  
✅ **Development & testing**  

❌ **Cannot:** Train DeepSeek-R1 or any 100B+ model  

## Cost Comparison

**Training DeepSeek-R1 (estimated):**
- 1 epoch on 1000 examples: ~2-4 hours
- Vast.ai A100 80GB: $1.50/hr × 4hr = **$6**
- AWS p4d.24xlarge: $32/hr × 2hr = **$64**

**Monthly if constantly training:**
- $6 per training run × 10 runs = **$60/month**

Much cheaper than buying a GPU!

## Next Steps

**Quick Start (Training Today):**
1. Sign up for Vast.ai (5 minutes)
2. Rent A100 instance ($1.50/hr)
3. Upload datasets + training code
4. Train model
5. Download checkpoint
6. Deploy to your EC2 API

**Or continue with Tasks 4 & 5:**
- Skip actual training for now
- Integrate DeepSeek-R1 API endpoint (mock/placeholder)
- Set up deployment pipeline
- Train later when you have GPU access
