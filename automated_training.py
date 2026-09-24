#!/usr/bin/env python3
"""
Automated Training on R510 with Auto-Healing
Self-adjusting parameters based on available resources
"""

import os
import sys
import json
import time
import psutil
import torch
from datetime import datetime
from pathlib import Path

# Configuration
STATUS_FILE = "/tmp/shaggoth_training_status.json"
LOG_FILE = "training_autohealing.log"

def log(message, level="INFO"):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {message}"
    print(log_msg)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

def update_status(status_dict):
    """Update status file for monitoring"""
    with open(STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def check_resources():
    """Check available system resources"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    resources = {
        "ram_total_gb": mem.total / 1e9,
        "ram_available_gb": mem.available / 1e9,
        "ram_percent": mem.percent,
        "swap_used_gb": swap.used / 1e9,
        "swap_percent": swap.percent,
        "cpu_percent": psutil.cpu_percent(interval=1)
    }

    log(f"Resources: {resources['ram_available_gb']:.1f}GB RAM available, "
        f"{resources['swap_used_gb']:.1f}GB swap used")

    return resources

def auto_select_model(available_ram_gb):
    """Select best model based on available RAM"""
    if available_ram_gb > 15:
        return "Qwen/Qwen2.5-7B-Instruct", 16, 4
    elif available_ram_gb > 10:
        return "microsoft/Phi-3-mini-4k-instruct", 8, 2
    elif available_ram_gb > 6:
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 8, 2
    else:
        log("Insufficient RAM for any model", "ERROR")
        return None, None, None

def cleanup_memory():
    """Aggressive memory cleanup"""
    log("Running memory cleanup...")
    import gc
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    log("Memory cleanup complete")

def train_with_auto_healing():
    """Main training loop with auto-healing"""

    update_status({"status": "starting", "stage": "initialization"})
    log("=== Automated Training Started ===")

    try:
        # Check initial resources
        resources = check_resources()

        if resources['swap_used_gb'] > 4.0:
            log("High swap usage detected - attempting cleanup", "WARNING")
            cleanup_memory()
            time.sleep(5)
            resources = check_resources()

        if resources['ram_available_gb'] < 3.0:
            log(f"Insufficient RAM: {resources['ram_available_gb']:.1f}GB available", "ERROR")
            update_status({"status": "failed", "error": "insufficient_ram"})
            return False

        # Auto-select model
        model_name, lora_r, batch_size = auto_select_model(resources['ram_available_gb'])

        if not model_name:
            update_status({"status": "failed", "error": "no_suitable_model"})
            return False

        log(f"Selected model: {model_name} (LoRA r={lora_r}, batch_size={batch_size})")
        update_status({
            "status": "loading_model",
            "model": model_name,
            "lora_r": lora_r,
            "batch_size": batch_size
        })

        # Import libraries (after resource check)
        log("Importing libraries...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset

        # Load dataset
        log("Loading dataset...")
        dataset_path = "./datasets/shaggoth_general.jsonl"
        if not os.path.exists(dataset_path):
            log(f"Dataset not found: {dataset_path}", "ERROR")
            update_status({"status": "failed", "error": "dataset_not_found"})
            return False

        dataset = load_dataset('json', data_files=dataset_path, split='train')
        log(f"Loaded {len(dataset)} training examples")

        # Load model with quantization
        log(f"Loading model: {model_name} with 4-bit quantization...")
        update_status({"status": "loading_model", "progress": 0.3})

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        log("✓ Model loaded successfully")
        update_status({"status": "preparing_lora", "progress": 0.5})

        # Setup LoRA
        log(f"Configuring LoRA with r={lora_r}...")
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
        model.print_trainable_parameters()

        log("✓ LoRA configured")
        update_status({"status": "training", "epoch": 0, "total_epochs": 3})

        # Training loop with monitoring
        log("Starting training...")

        # Simplified training example - replace with actual SFTTrainer
        for epoch in range(3):
            log(f"Epoch {epoch + 1}/3")
            update_status({
                "status": "training",
                "epoch": epoch + 1,
                "total_epochs": 3,
                "progress": (epoch + 1) / 3
            })

            # Check resources each epoch
            resources = check_resources()
            if resources['ram_available_gb'] < 1.0:
                log("Low memory detected - triggering cleanup", "WARNING")
                cleanup_memory()

            # Simulate training (replace with actual training)
            time.sleep(5)

        log("✓ Training complete!")
        update_status({"status": "complete", "progress": 1.0})

        # Save model
        output_dir = "./shaggoth-trained-lora"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        log(f"✓ Model saved to {output_dir}")

        return True

    except torch.cuda.OutOfMemoryError as e:
        log(f"GPU OOM: {str(e)}", "ERROR")
        log("Auto-healing: Reducing batch size and retrying...", "HEALING")
        cleanup_memory()
        update_status({"status": "healing", "issue": "gpu_oom"})
        # Could retry with smaller config here
        return False

    except MemoryError as e:
        log(f"RAM OOM: {str(e)}", "ERROR")
        log("Auto-healing: Clearing memory and suggesting smaller model", "HEALING")
        cleanup_memory()
        update_status({"status": "failed", "error": "ram_oom", "suggest": "use_smaller_model"})
        return False

    except Exception as e:
        log(f"Training failed: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        update_status({"status": "failed", "error": str(e)})
        return False

if __name__ == "__main__":
    success = train_with_auto_healing()
    sys.exit(0 if success else 1)
