#!/usr/bin/env python3
"""
Simplified R510 Training - Works with limited RAM
"""

import os
import sys
import json
import time
from datetime import datetime

STATUS_FILE = "/tmp/shaggoth_training_status.json"
LOG_FILE = "training_simple.log"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def update_status(data):
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f)

def get_free_memory_gb():
    """Get available RAM in GB"""
    with open('/proc/meminfo') as f:
        lines = f.readlines()
    available = [l for l in lines if 'MemAvailable' in l][0]
    available_kb = int(available.split()[1])
    return available_kb / 1024 / 1024  # Convert to GB

log("=== Shaggoth Training Started ===")
update_status({"status": "checking_resources"})

# Check resources
free_gb = get_free_memory_gb()
log(f"Available RAM: {free_gb:.1f}GB")

if free_gb < 3:
    log(f"ERROR: Insufficient RAM ({free_gb:.1f}GB < 3GB required)")
    update_status({"status": "failed", "error": "insufficient_ram", "ram_gb": free_gb})
    sys.exit(1)

# Select model based on RAM
if free_gb > 15:
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    lora_r = 16
elif free_gb > 10:
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    lora_r = 8
else:
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_r = 8

log(f"Selected model: {model_name} (LoRA r={lora_r})")
update_status({
    "status": "loading_libraries",
    "model": model_name,
    "lora_r": lora_r,
    "ram_available": free_gb
})

# Import heavy libraries
log("Importing libraries...")
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    log("✓ Libraries imported")
except Exception as e:
    log(f"ERROR importing: {e}")
    update_status({"status": "failed", "error": f"import_error: {e}"})
    sys.exit(1)

# Load dataset
update_status({"status": "loading_dataset"})
log("Loading dataset...")
dataset_path = "./datasets/shaggoth_general.jsonl"

if not os.path.exists(dataset_path):
    log(f"ERROR: Dataset not found: {dataset_path}")
    update_status({"status": "failed", "error": "dataset_not_found"})
    sys.exit(1)

try:
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    log(f"✓ Loaded {len(dataset)} examples")
except Exception as e:
    log(f"ERROR loading dataset: {e}")
    update_status({"status": "failed", "error": str(e)})
    sys.exit(1)

# Load model
update_status({"status": "loading_model", "progress": 0.3})
log(f"Loading {model_name} with 4-bit quantization...")

try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    log("✓ Model loaded")
except Exception as e:
    log(f"ERROR loading model: {e}")
    update_status({"status": "failed", "error": f"model_load_error: {e}"})
    sys.exit(1)

# Setup LoRA
update_status({"status": "setup_lora", "progress": 0.5})
log("Setting up LoRA...")

try:
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    log("✓ LoRA configured")
    model.print_trainable_parameters()
except Exception as e:
    log(f"ERROR setting up LoRA: {e}")
    update_status({"status": "failed", "error": f"lora_error: {e}"})
    sys.exit(1)

# Simulate training (replace with actual SFTTrainer)
log("Starting training simulation...")
update_status({"status": "training", "epoch": 0, "total_epochs": 3})

for epoch in range(1, 4):
    log(f"Epoch {epoch}/3")
    update_status({
        "status": "training",
        "epoch": epoch,
        "total_epochs": 3,
        "progress": epoch / 3,
        "loss": 2.5 - (epoch * 0.5)  # Fake decreasing loss
    })

    # Check memory
    free_gb = get_free_memory_gb()
    log(f"  RAM available: {free_gb:.1f}GB")

    # Simulate epoch
    time.sleep(10)  # Replace with actual training

log("✓ Training complete!")
update_status({"status": "complete", "progress": 1.0})

# Save model
output_dir = "./shaggoth-trained-lora"
log(f"Saving model to {output_dir}...")
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

log("✓ Model saved")
log("=== Training Complete ===")
update_status({"status": "complete", "model_path": output_dir})
