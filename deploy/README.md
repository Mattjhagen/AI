# DeepSeek-R1 Deployment Guide

Complete deployment pipeline from training to production.

## Overview

```
┌─────────────┐      ┌──────────────┐      ┌──────────┐      ┌─────────────┐
│   R510      │      │  Cloud GPU   │      │    S3    │      │  EC2 Prod   │
│ (Dev/Test)  │─────►│  (Training)  │─────►│ (Backup) │─────►│ (Inference) │
│             │      │              │      │          │      │             │
│ • Datasets  │      │ • A100 80GB  │      │ • Model  │      │ • t3.small  │
│ • Interface │      │ • 4hr train  │      │ • $0.23  │      │ • API       │
│ • Testing   │      │ • $6 cost    │      │   /month │      │ • Mobile    │
└─────────────┘      └──────────────┘      └──────────┘      └─────────────┘
```

## Quick Start (Full Pipeline)

### 1. Prepare Training Package (R510)

```bash
cd ~/AI/deploy
bash train_on_cloud.sh
```

This creates `shaggoth-training-YYYYMMDD-HHMMSS.tar.gz` with:
- Training datasets (1000+ examples)
- Training interface
- All dependencies

### 2. Launch Cloud GPU (Vast.ai Recommended)

#### Option A: Vast.ai (Cheapest - $1.50/hr)

1. Go to https://vast.ai
2. Create account
3. Add $10 credit
4. Search for "A100 80GB"
5. Sort by price (lowest first)
6. Select instance with:
   - GPU: A100 80GB
   - RAM: 64GB+
   - Storage: 500GB+
   - Image: PyTorch 2.0+ with CUDA 11.8+
7. Launch instance
8. Copy SSH command

#### Option B: RunPod ($2.89/hr)

1. Go to https://runpod.io
2. Create account
3. GPU Pods → Rent
4. Select A100 80GB
5. Choose PyTorch template
6. Deploy

#### Option C: AWS EC2 (Most Expensive)

```bash
# Launch p3.2xlarge (V100 16GB) - $3.06/hr
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type p3.2xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxx
```

### 3. Upload and Train

```bash
# Upload package
scp shaggoth-training-*.tar.gz root@<gpu-ip>:/workspace/

# SSH in
ssh root@<gpu-ip>

# Extract and train
cd /workspace
tar -xzf shaggoth-training-*.tar.gz
cd shaggoth-training
bash train.sh

# Access interface at http://<gpu-ip>:7860
```

### 4. Training Steps (Gradio Interface)

1. **Model Setup Tab:**
   - Path: `./DeepSeek-R1`
   - Quantization: 8-bit ✓
   - Click "Load Model" (takes 5-10 min)
   - Click "Setup LoRA"

2. **Dataset Tab:**
   - Upload `datasets/shaggoth_mixed.jsonl`
   - Click "Load Dataset"
   - Preview should show 405 examples

3. **Training Tab:**
   - Output dir: `./shaggoth-trained`
   - Epochs: 3
   - Batch size: 4
   - Learning rate: 2e-4
   - Click "Start Training"

4. **Wait:**
   - Training: 2-4 hours
   - Monitor GPU usage
   - Checkpoints saved every 100 steps

5. **Test Tab:**
   - Test with sample prompts
   - Verify model responses
   - Compare to pre-training baseline

### 5. Download Checkpoint

```bash
# From R510 or your local machine
scp -r root@<gpu-ip>:/workspace/shaggoth-training/shaggoth-trained ~/AI/

# Verify
ls -lh ~/AI/shaggoth-trained/
# Should see:
# - adapter_config.json
# - adapter_model.bin
# - training_args.bin
# - checkpoint-*/
```

### 6. Deploy to EC2 Production

```bash
cd ~/AI/deploy
bash deploy_to_ec2.sh ~/AI/shaggoth-trained
```

Follow prompts to:
1. Upload to S3 (backup)
2. Deploy to EC2
3. Update configuration
4. Restart service
5. Verify deployment

### 7. Verify Production

```bash
# Check status
curl http://your-ec2-ip:8420/deepseek/status | jq

# Test inference
curl -X POST http://your-ec2-ip:8420/deepseek/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Shaggoth AI?"}' | jq
```

## Cost Breakdown

### One-Time Training Costs

| Provider | GPU | $/hr | Time | Total |
|----------|-----|------|------|-------|
| Vast.ai  | A100 80GB | $1.50 | 4h | **$6** |
| RunPod   | A100 80GB | $2.89 | 4h | $11.56 |
| AWS p3.2xlarge | V100 16GB | $3.06 | 6h | $18.36 |
| AWS p4d.24xlarge | A100 40GB | $32.00 | 2h | $64.00 |

**Recommended:** Vast.ai A100 80GB = **$6 total**

### Monthly Ongoing Costs

| Service | Cost | Notes |
|---------|------|-------|
| S3 storage | $0.23/mo | 10GB checkpoint backup |
| EC2 t3.small | ~$15/mo | Already running |
| CloudWatch | ~$2/mo | Optional monitoring |
| **Total** | **$17.23/mo** | No inference costs! |

### Alternative: API-Only (No Training)

| Service | Cost | Notes |
|---------|------|-------|
| DeepSeek API | ~$0.14/1M tokens | Pay per use |
| Typical usage | ~$5-20/mo | Depends on traffic |

## File Structure After Deployment

### R510 (Development)
```
~/AI/
├── datasets/                    # Training data
│   ├── shaggoth_general.jsonl
│   ├── shaggoth_reasoning.jsonl
│   └── shaggoth_mixed.jsonl
├── shaggoth-trained/            # Downloaded checkpoint
│   ├── adapter_config.json
│   └── adapter_model.bin
├── app.py                       # Training interface
├── deploy/                      # This directory
│   ├── train_on_cloud.sh
│   ├── deploy_to_ec2.sh
│   └── README.md
└── HARDWARE_REQUIREMENTS.md
```

### S3 Backup
```
s3://shaggoth-models/
├── deepseek-r1-20260912/        # Dated backup
│   ├── adapter_config.json
│   └── adapter_model.bin
└── deepseek-r1-latest/          # Latest version
    ├── adapter_config.json
    └── adapter_model.bin
```

### EC2 Production
```
/opt/shaggoth/
├── models/
│   └── deepseek-r1/             # Deployed checkpoint
│       ├── adapter_config.json
│       └── adapter_model.bin
└── Shaggoth-a1/                 # Main application
    └── shaggoth/
        ├── models/
        │   ├── deepseek.py      # Model adapter
        │   └── deepseek_endpoint.py
        ├── server.py            # API server
        └── DEEPSEEK_INTEGRATION.md
```

## Troubleshooting

### Training Issues

**"CUDA out of memory"**
- Enable 4-bit quantization instead of 8-bit
- Reduce batch size to 2 or 1
- Reduce LoRA rank to 4
- Use gradient accumulation

**"Model download too slow"**
- Choose instance with faster network
- Download overnight
- Use cached instance with model pre-loaded

**"Training stalled"**
- Check GPU utilization: `nvidia-smi`
- Check logs: `tail -f trainer.log`
- Reduce dataset size for testing

### Deployment Issues

**"SSH connection refused"**
- Check security group rules (port 22)
- Verify EC2 instance is running
- Check key pair permissions: `chmod 400 key.pem`

**"S3 access denied"**
- Configure AWS credentials: `aws configure`
- Check IAM permissions
- Verify bucket exists

**"Service won't start"**
```bash
# Check logs
ssh aws "sudo journalctl -u shaggoth -n 50"

# Check environment
ssh aws "cat /etc/environment | grep DEEPSEEK"

# Manual start for debugging
ssh aws "cd ~/Shaggoth-a1 && python3 -m shaggoth serve --port 8420"
```

**"/deepseek/status returns mock mode"**
- Verify checkpoint deployed: `ssh aws "ls -la /opt/shaggoth/models/deepseek-r1"`
- Check environment variable: `ssh aws "echo $DEEPSEEK_CHECKPOINT"`
- Restart service: `ssh aws "sudo systemctl restart shaggoth"`

## Monitoring

### GPU Usage During Training

```bash
# On training instance
watch -n 1 nvidia-smi

# Or use the monitoring script
python3 monitor_gpu.py --log gpu.log
```

### Production Inference

```bash
# Test latency
time curl -X POST http://ec2-ip:8420/deepseek/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'

# Monitor logs
ssh aws "sudo tail -f /var/log/shaggoth.log"

# Check system resources
ssh aws "htop"
```

### CloudWatch Setup (Optional)

```bash
# Install CloudWatch agent on EC2
ssh aws << 'EOF'
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
EOF
```

## Updating the Model

To retrain with new data:

1. Generate new datasets on R510
2. Package and upload to cloud GPU
3. Train new checkpoint
4. Deploy with new date: `deploy_to_ec2.sh ~/AI/shaggoth-trained-v2`
5. Old checkpoints remain in S3 for rollback

## Rollback

```bash
# List available versions
aws s3 ls s3://shaggoth-models/

# Deploy older version
ssh aws "aws s3 sync s3://shaggoth-models/deepseek-r1-20260911/ /opt/shaggoth/models/deepseek-r1/"
ssh aws "sudo systemctl restart shaggoth"
```

## Support

- See `DEEPSEEK_INTEGRATION.md` for API details
- See `HARDWARE_REQUIREMENTS.md` for GPU options
- Check training logs: `~/AI/trainer.log`
- Check deployment logs: `~/AI/deploy/deployment.log`

## Maintenance Schedule

- **Weekly:** Check S3 costs
- **Monthly:** Review inference latency
- **Quarterly:** Retrain with new knowledge
- **As needed:** Update for model improvements
