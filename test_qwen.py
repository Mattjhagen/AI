#!/usr/bin/env python3
"""
Quick test of Qwen2.5-7B-Instruct on R510 hardware
Tests loading speed, inference time, and memory usage
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import psutil

def print_memory():
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB ({mem.percent}%)")

print("=" * 60)
print("Qwen2.5-7B-Instruct Test on R510")
print("=" * 60)
print(f"\nInitial Memory:")
print_memory()

# Test 1: Load model with 4-bit quantization
print("\n[1/3] Loading Qwen2.5-7B with 4-bit quantization...")
load_start = time.time()

try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    load_time = time.time() - load_start
    print(f"✓ Model loaded in {load_time:.1f} seconds")
    print(f"\nMemory after loading:")
    print_memory()

except Exception as e:
    print(f"✗ FAILED to load model: {e}")
    exit(1)

# Test 2: Simple inference
print("\n[2/3] Testing inference speed...")
test_prompt = "Explain what Shaggoth AI is in one sentence."

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": test_prompt}
]

try:
    inference_start = time.time()

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    inference_time = time.time() - inference_start

    print(f"✓ Generated response in {inference_time:.1f} seconds")
    print(f"\nResponse: {response}")
    print(f"\nMemory after inference:")
    print_memory()

except Exception as e:
    print(f"✗ FAILED during inference: {e}")
    exit(1)

# Test 3: Longer generation
print("\n[3/3] Testing longer generation (200 tokens)...")
long_prompt = "Write a Python function to calculate fibonacci numbers."

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": long_prompt}
]

try:
    long_start = time.time()

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    long_time = time.time() - long_start

    print(f"✓ Generated 200 tokens in {long_time:.1f} seconds")
    print(f"  Tokens/second: {200/long_time:.1f}")
    print(f"\nResponse preview: {response[:200]}...")

except Exception as e:
    print(f"✗ FAILED during long generation: {e}")
    exit(1)

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Model load time:      {load_time:.1f}s")
print(f"Short inference:      {inference_time:.1f}s (50 tokens)")
print(f"Long inference:       {long_time:.1f}s (200 tokens)")
print(f"Generation speed:     {200/long_time:.1f} tokens/sec")
print(f"\nFinal memory:")
print_memory()
print("\n✓ All tests passed - Qwen2.5-7B is working on R510!")
