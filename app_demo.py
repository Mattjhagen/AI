import gradio as gr
import json
from pathlib import Path
from datetime import datetime

# Demo mode - no GPU required
print("="*60)
print("🎮 SHAGGOTH AI TRAINER - DEMO MODE")
print("="*60)
print("⚠️  Running in DEMO mode (no GPU detected)")
print("This interface shows the training workflow without actual model loading.")
print(f"Access at: http://192.168.0.169:7860")
print("="*60)

training_state = {"status": "idle", "logs": []}

def demo_load_model():
    return "✅ [DEMO] Model load simulated (DeepSeek-R1 requires GPU)"

def demo_setup_lora(rank, alpha, dropout):
    return f"✅ [DEMO] LoRA configured - Rank: {rank}, Alpha: {alpha}, Dropout: {dropout}"

def demo_load_dataset(file, fmt):
    if file is None:
        return None, "❌ No file uploaded", None

    try:
        import pandas as pd
        with open(file.name, 'r') as f:
            data = [json.loads(line) for line in f]
        preview_df = pd.DataFrame(data[:5])
        return data, f"✅ [DEMO] Loaded {len(data)} examples", preview_df
    except Exception as e:
        return None, f"❌ Error: {e}", None

def demo_train(dataset, output_dir, epochs, batch_size, lr, save_steps, eval_steps):
    if dataset is None:
        return "❌ Load a dataset first"

    msg = f"""✅ [DEMO] Training simulated

📊 Configuration:
- Dataset: {len(dataset)} examples
- Output: {output_dir}
- Epochs: {epochs}
- Batch size: {batch_size}
- Learning rate: {lr}

⚠️  GPU Required:
To actually train DeepSeek-R1, you need:
- GPU: 48GB+ VRAM (A100/H100)
- Cloud options: Vast.ai, RunPod, AWS

See HARDWARE_REQUIREMENTS.md for details."""

    return msg

def demo_test(prompt, max_len, temp, top_p):
    return f"""[DEMO RESPONSE]

This is a simulated response. In production, DeepSeek-R1 would generate:
- Chain-of-thought reasoning
- Self-verification steps
- Detailed analytical response

Your prompt: "{prompt}"

To use the real model:
1. Train on cloud GPU (Vast.ai recommended)
2. Deploy checkpoint to your EC2
3. Use Shaggoth API for inference"""

def get_logs():
    return f"""[{datetime.now().strftime('%H:%M:%S')}] Demo mode active
[{datetime.now().strftime('%H:%M:%S')}] No GPU detected (Matrox G200eW)
[{datetime.now().strftime('%H:%M:%S')}] DeepSeek-R1 requires NVIDIA GPU with 48GB+ VRAM
[{datetime.now().strftime('%H:%M:%S')}] See HARDWARE_REQUIREMENTS.md for training options
[{datetime.now().strftime('%H:%M:%S')}] Interface functional - ready for cloud GPU setup"""

# Gradio Interface
with gr.Blocks(title="Shaggoth AI Trainer (Demo)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Shaggoth AI Trainer - DEMO MODE")
    gr.Markdown("⚠️ **No GPU detected.** This is a demo interface. See [HARDWARE_REQUIREMENTS.md](file:///home/matt/AI/HARDWARE_REQUIREMENTS.md) for training options.")

    with gr.Tab("📥 Model Setup"):
        gr.Markdown("### Demo Mode - GPU Required for Actual Training")
        load_btn = gr.Button("Simulate Model Load", variant="primary")
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
        gr.Markdown("**Available datasets in ~/AI/datasets/:**\n- shaggoth_general.jsonl (1000 examples)\n- shaggoth_reasoning.jsonl (548 examples)\n- shaggoth_mixed.jsonl (405 examples)")

        data_file = gr.File(label="Upload Dataset", file_types=[".jsonl", ".json"])
        data_format = gr.Radio(["jsonl", "json"], value="jsonl", label="Data Format")
        load_data_btn = gr.Button("Load Dataset", variant="primary")
        data_status = gr.Textbox(label="Status", interactive=False)
        data_preview = gr.Dataframe(label="Data Preview")

        dataset_state = gr.State(None)

    with gr.Tab("🚀 Training"):
        gr.Markdown("### Training Configuration (Demo)")

        with gr.Row():
            output_dir = gr.Textbox(value="./shaggoth-trained", label="Output Directory")
            num_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")

        with gr.Row():
            batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
            learning_rate = gr.Number(value=2e-4, label="Learning Rate")

        with gr.Row():
            save_steps = gr.Slider(50, 500, value=100, step=50, label="Save Steps")
            eval_steps = gr.Slider(50, 500, value=100, step=50, label="Eval Steps")

        train_btn = gr.Button("Simulate Training", variant="primary", size="lg")
        training_status = gr.Textbox(label="Training Status", interactive=False, lines=10)

        gr.Markdown("### System Info")
        logs_display = gr.Textbox(label="Logs", lines=8, interactive=False)
        refresh_logs_btn = gr.Button("Refresh Logs")

    with gr.Tab("📖 Guide"):
        gr.Markdown("""
        ## Training DeepSeek-R1 on Your Hardware

        ### Current System (R510)
        - **CPU:** Intel Xeon E5620
        - **RAM:** 24GB
        - **GPU:** ❌ None (Matrox server GPU only)
        - **Status:** Cannot train DeepSeek-R1

        ### Options to Train

        #### 1. Cloud GPU (Recommended)
        **Vast.ai** (Cheapest):
        - A100 80GB: $1.50/hr
        - Training cost: ~$6 per run
        - Sign up: https://vast.ai

        **RunPod**:
        - A100 80GB: $2.89/hr
        - Easy interface: https://runpod.io

        **AWS EC2** (You already use AWS):
        - p3.2xlarge (V100 16GB): $3.06/hr
        - p4d.24xlarge (A100 40GB): $32/hr

        #### 2. What's Ready
        ✅ **Training datasets created:**
        - 1000 general knowledge examples
        - 548 reasoning examples
        - 5 Shaggoth-specific examples
        - Files in `~/AI/datasets/`

        ✅ **Training interface built:**
        - This Gradio interface
        - LoRA fine-tuning setup
        - GPU monitoring script

        ✅ **API integration ready:**
        - Can add DeepSeek endpoint to Shaggoth
        - Deployment scripts prepared

        #### 3. Next Steps
        1. **Continue without training:** Complete API integration (mock endpoint)
        2. **Train on cloud:** Set up Vast.ai/RunPod, upload datasets, train
        3. **Deploy:** Bring checkpoint back to EC2 production

        ### Quick Cloud Training
        ```bash
        # Package for cloud
        cd ~/AI
        tar -czf training.tar.gz datasets/ app.py requirements.txt DeepSeek-R1/

        # Upload to cloud GPU instance
        scp training.tar.gz user@gpu-instance:/workspace/

        # Train
        ssh user@gpu-instance
        cd /workspace && tar -xzf training.tar.gz
        python3 app.py
        # Access at http://<gpu-ip>:7860
        ```

        ### Using Existing Models
        You can also use:
        - **Ollama** on R510: qwen2.5-coder:7b (already installed)
        - **Claude API**: For production inference
        - **Gemini/Cloudflare Workers AI**: Free tier options
        """)

    # Event handlers
    load_btn.click(fn=demo_load_model, outputs=[load_status])
    setup_lora_btn.click(
        fn=demo_setup_lora,
        inputs=[lora_rank, lora_alpha, lora_dropout],
        outputs=[lora_status]
    )
    load_data_btn.click(
        fn=demo_load_dataset,
        inputs=[data_file, data_format],
        outputs=[dataset_state, data_status, data_preview]
    )
    train_btn.click(
        fn=demo_train,
        inputs=[dataset_state, output_dir, num_epochs, batch_size, learning_rate, save_steps, eval_steps],
        outputs=[training_status]
    )
    refresh_logs_btn.click(fn=get_logs, outputs=[logs_display])

if __name__ == "__main__":
    print("\n🚀 Launching Shaggoth AI Trainer (Demo Mode)...")
    print("📊 Access at: http://192.168.0.169:7860")
    print("📖 Press Ctrl+C to stop\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
