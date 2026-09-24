#!/usr/bin/env python3
"""
Quick test of Phi-3-mini (3.8B) - smaller, more stable for 23GB RAM
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import psutil

def print_memory():
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB (Available: {mem.available/1e9:.1f}GB)")

print("=" * 60)
print("Phi-3-Mini (3.8B) Test on R510")
print("Smaller model - should work reliably on 23GB RAM")
print("=" * 60)
print(f"\nInitial Memory:")
print_memory()

# Load Phi-3-mini with 4-bit quantization
print("\n[1/2] Loading Phi-3-mini with 4-bit quantization...")
load_start = time.time()

try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model_name = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    load_time = time.time() - load_start
    print(f"✓ Model loaded in {load_time:.1f} seconds")
    print(f"\nMemory after loading:")
    print_memory()

except Exception as e:
    print(f"✗ FAILED to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test inference
print("\n[2/2] Testing inference...")
test_prompt = "Write a Python function to calculate fibonacci numbers."

try:
    inference_start = time.time()

    messages = [{"role": "user", "content": test_prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

    response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
    inference_time = time.time() - inference_start

    print(f"✓ Generated response in {inference_time:.1f} seconds")
    print(f"\nResponse:\n{response}")
    print(f"\nMemory after inference:")
    print_memory()

except Exception as e:
    print(f"✗ FAILED during inference: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Model: Phi-3-mini-4k-instruct (3.8B parameters)")
print(f"Load time:   {load_time:.1f}s")
print(f"Inference:   {inference_time:.1f}s")
print(f"\n✓ Phi-3-mini works on R510!")
print(f"\nThis model is suitable for Shaggoth training.")
print(f"LoRA fine-tuning should use ~5-7GB RAM total.")
