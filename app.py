import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import os
from datetime import datetime
import pandas as pd

# Global variables
model = None
tokenizer = None
training_state = {"status": "idle", "progress": 0, "logs": []}

def load_model(model_path="./DeepSeek-R1", use_8bit=True, use_4bit=False):
    """Load DeepSeek-R1 model with quantization for efficient training"""
    global model, tokenizer

    try:
        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model with quantization...")

        # Load with quantization for memory efficiency
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }

        if use_8bit:
            load_kwargs["load_in_8bit"] = True
        elif use_4bit:
            load_kwargs["load_in_4bit"] = True

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded successfully!")
        return "✅ Model loaded successfully!"

    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
        return error_msg

def prepare_dataset(data_file, data_format="jsonl"):
    """Prepare training dataset from file"""
    try:
        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading dataset...")

        if data_format == "jsonl":
            with open(data_file.name, 'r') as f:
                data = [json.loads(line) for line in f]
        elif data_format == "json":
            with open(data_file.name, 'r') as f:
                data = json.load(f)
        else:
            return None, "❌ Unsupported format"

        # Expected format: [{"instruction": "...", "input": "...", "output": "..."}]
        dataset = Dataset.from_list(data)

        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loaded: {len(data)} examples")

        # Show preview
        preview_df = pd.DataFrame(data[:5])
        return dataset, f"✅ Loaded {len(data)} examples", preview_df

    except Exception as e:
        error_msg = f"❌ Error loading dataset: {str(e)}"
        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
        return None, error_msg, None

def tokenize_function(examples, tokenizer):
    """Tokenize examples for training"""
    prompts = []
    for instruction, input_text, output in zip(
        examples.get("instruction", [""] * len(examples["output"])),
        examples.get("input", [""] * len(examples["output"])),
        examples["output"]
    ):
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
        else:
            prompt = f"Instruction: {instruction}\nResponse: {output}"
        prompts.append(prompt)

    return tokenizer(prompts, truncation=True, padding="max_length", max_length=2048)

def setup_lora_training(
    rank=8,
    alpha=16,
    dropout=0.05,
    target_modules=None
):
    """Setup LoRA configuration for parameter-efficient fine-tuning"""
    global model

    try:
        if model is None:
            return "❌ Please load model first!"

        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Setting up LoRA...")

        # Default target modules for DeepSeek
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

        info = f"✅ LoRA setup complete!\nTrainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)\nTotal params: {all_params:,}"
        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {info}")

        return info

    except Exception as e:
        error_msg = f"❌ Error setting up LoRA: {str(e)}"
        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
        return error_msg

def start_training(
    dataset,
    output_dir="./shaggoth-trained",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    save_steps=100,
    eval_steps=100
):
    """Start the training process"""
    global model, tokenizer

    try:
        if model is None or tokenizer is None:
            return "❌ Please load model first!"

        if dataset is None:
            return "❌ Please load dataset first!"

        training_state["status"] = "training"
        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )

        # Split into train/eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=10,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            warmup_steps=100,
            optim="adamw_torch",
            report_to="none",
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
        )

        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Training started!")

        # Train
        trainer.train()

        # Save final model
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        training_state["status"] = "completed"
        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed!")

        return f"✅ Training completed! Model saved to {output_dir}"

    except Exception as e:
        training_state["status"] = "error"
        error_msg = f"❌ Training error: {str(e)}"
        training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
        return error_msg

def test_model(prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Test the model with a prompt"""
    global model, tokenizer

    try:
        if model is None or tokenizer is None:
            return "❌ Please load model first!"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_training_logs():
    """Get training logs"""
    return "\n".join(training_state["logs"][-50:])  # Last 50 logs

# Gradio Interface
with gr.Blocks(title="Shaggoth AI Trainer - DeepSeek-R1", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Shaggoth AI Trainer - DeepSeek-R1")
    gr.Markdown("Train DeepSeek-R1 model with your custom data using LoRA fine-tuning")

    with gr.Tab("📥 Model Setup"):
        gr.Markdown("### Load Model")
        with gr.Row():
            model_path = gr.Textbox(value="./DeepSeek-R1", label="Model Path")
            use_8bit = gr.Checkbox(value=True, label="Use 8-bit Quantization")
            use_4bit = gr.Checkbox(value=False, label="Use 4-bit Quantization")

        load_btn = gr.Button("Load Model", variant="primary")
        load_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### LoRA Configuration")
        with gr.Row():
            lora_rank = gr.Slider(4, 64, value=8, step=4, label="LoRA Rank")
            lora_alpha = gr.Slider(8, 128, value=16, step=8, label="LoRA Alpha")
            lora_dropout = gr.Slider(0, 0.2, value=0.05, step=0.01, label="LoRA Dropout")

        setup_lora_btn = gr.Button("Setup LoRA", variant="primary")
        lora_status = gr.Textbox(label="LoRA Status", interactive=False)

    with gr.Tab("📚 Dataset"):
        gr.Markdown("### Upload Training Data")
        gr.Markdown("Expected format: JSONL with `instruction`, `input` (optional), and `output` fields")

        data_file = gr.File(label="Upload Dataset", file_types=[".jsonl", ".json"])
        data_format = gr.Radio(["jsonl", "json"], value="jsonl", label="Data Format")
        load_data_btn = gr.Button("Load Dataset", variant="primary")
        data_status = gr.Textbox(label="Status", interactive=False)
        data_preview = gr.Dataframe(label="Data Preview (first 5 examples)")

        # Store dataset in state
        dataset_state = gr.State(None)

    with gr.Tab("🚀 Training"):
        gr.Markdown("### Training Configuration")

        with gr.Row():
            output_dir = gr.Textbox(value="./shaggoth-trained", label="Output Directory")
            num_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")

        with gr.Row():
            batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
            learning_rate = gr.Number(value=2e-4, label="Learning Rate")

        with gr.Row():
            save_steps = gr.Slider(50, 500, value=100, step=50, label="Save Steps")
            eval_steps = gr.Slider(50, 500, value=100, step=50, label="Eval Steps")

        train_btn = gr.Button("Start Training", variant="primary", size="lg")
        training_status = gr.Textbox(label="Training Status", interactive=False)

        gr.Markdown("### Training Logs")
        logs_display = gr.Textbox(label="Logs", lines=15, interactive=False)
        refresh_logs_btn = gr.Button("Refresh Logs")

    with gr.Tab("🧪 Test Model"):
        gr.Markdown("### Test Your Trained Model")

        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Enter your test prompt here...",
            lines=3
        )

        with gr.Row():
            max_length = gr.Slider(128, 2048, value=512, step=128, label="Max Length")
            temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")

        test_btn = gr.Button("Generate", variant="primary")
        output_display = gr.Textbox(label="Model Response", lines=10)

    # Event handlers
    load_btn.click(
        fn=lambda p, q8, q4: load_model(p, q8, q4),
        inputs=[model_path, use_8bit, use_4bit],
        outputs=[load_status]
    )

    setup_lora_btn.click(
        fn=lambda r, a, d: setup_lora_training(r, a, d),
        inputs=[lora_rank, lora_alpha, lora_dropout],
        outputs=[lora_status]
    )

    def load_dataset_wrapper(file, fmt):
        ds, status, preview = prepare_dataset(file, fmt)
        return ds, status, preview

    load_data_btn.click(
        fn=load_dataset_wrapper,
        inputs=[data_file, data_format],
        outputs=[dataset_state, data_status, data_preview]
    )

    train_btn.click(
        fn=lambda ds, out, ep, bs, lr, ss, es: start_training(ds, out, ep, bs, lr, ss, es),
        inputs=[dataset_state, output_dir, num_epochs, batch_size, learning_rate, save_steps, eval_steps],
        outputs=[training_status]
    )

    refresh_logs_btn.click(
        fn=get_training_logs,
        outputs=[logs_display]
    )

    test_btn.click(
        fn=lambda p, ml, t, tp: test_model(p, ml, t, tp),
        inputs=[prompt_input, max_length, temperature, top_p],
        outputs=[output_display]
    )

if __name__ == "__main__":
    print("🚀 Starting Shaggoth AI Trainer...")
    print("📊 Access the interface at: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
