#!/usr/bin/env python3
"""
Real-time GPU memory monitoring for DeepSeek-R1 training.
Tracks VRAM usage, temperature, and utilization during fine-tuning.
"""

import subprocess
import time
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class GPUMonitor:
    def __init__(self, log_file=None, interval=2.0):
        self.log_file = log_file
        self.interval = interval
        self.start_time = time.time()
        self.peak_memory = 0
        self.logs = []

        if log_file:
            self.log_path = Path(log_file)
            self.log_path.parent.mkdir(exist_ok=True, parents=True)

    def get_nvidia_smi_stats(self):
        """Get GPU stats using nvidia-smi"""
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                check=True
            )

            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_used_mb': float(parts[2]),
                        'memory_total_mb': float(parts[3]),
                        'temperature_c': float(parts[4]),
                        'utilization_percent': float(parts[5]),
                        'power_draw_w': float(parts[6]) if parts[6] != 'N/A' else 0.0
                    })

            return gpus

        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

    def get_torch_stats(self):
        """Get GPU stats using PyTorch"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
            reserved = torch.cuda.memory_reserved(i) / 1024**2  # MB
            total = props.total_memory / 1024**2  # MB

            gpus.append({
                'index': i,
                'name': props.name,
                'memory_allocated_mb': allocated,
                'memory_reserved_mb': reserved,
                'memory_total_mb': total,
            })

        return gpus

    def format_memory(self, mb):
        """Format memory in human-readable format"""
        if mb < 1024:
            return f"{mb:.1f} MB"
        else:
            return f"{mb/1024:.2f} GB"

    def display_stats(self, nvidia_gpus, torch_gpus):
        """Display GPU statistics"""
        elapsed = time.time() - self.start_time

        # Clear screen (ANSI escape code)
        print("\033[2J\033[H", end='')

        print("=" * 80)
        print(f"🎮 GPU MONITORING - DeepSeek-R1 Training")
        print(f"Runtime: {elapsed/60:.1f} minutes | Peak VRAM: {self.format_memory(self.peak_memory)}")
        print("=" * 80)

        if nvidia_gpus:
            for gpu in nvidia_gpus:
                print(f"\n📊 GPU {gpu['index']}: {gpu['name']}")
                print(f"   Memory: {self.format_memory(gpu['memory_used_mb'])} / {self.format_memory(gpu['memory_total_mb'])}")

                usage_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
                bar_width = 40
                filled = int(bar_width * usage_percent / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"   [{bar}] {usage_percent:.1f}%")

                print(f"   Temperature: {gpu['temperature_c']:.0f}°C")
                print(f"   Utilization: {gpu['utilization_percent']:.0f}%")
                print(f"   Power Draw: {gpu['power_draw_w']:.1f}W")

                # Track peak memory
                if gpu['memory_used_mb'] > self.peak_memory:
                    self.peak_memory = gpu['memory_used_mb']

        elif torch_gpus:
            for gpu in torch_gpus:
                print(f"\n📊 GPU {gpu['index']}: {gpu['name']}")
                print(f"   Allocated: {self.format_memory(gpu['memory_allocated_mb'])}")
                print(f"   Reserved: {self.format_memory(gpu['memory_reserved_mb'])}")
                print(f"   Total: {self.format_memory(gpu['memory_total_mb'])}")

                usage_percent = (gpu['memory_reserved_mb'] / gpu['memory_total_mb']) * 100
                bar_width = 40
                filled = int(bar_width * usage_percent / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"   [{bar}] {usage_percent:.1f}%")

                if gpu['memory_reserved_mb'] > self.peak_memory:
                    self.peak_memory = gpu['memory_reserved_mb']

        else:
            print("\n⚠️  No GPU detected or monitoring unavailable")

        print("\n" + "=" * 80)
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)

    def log_stats(self, nvidia_gpus, torch_gpus):
        """Log statistics to file"""
        if not self.log_file:
            return

        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'elapsed_seconds': time.time() - self.start_time,
            'nvidia_gpus': nvidia_gpus,
            'torch_gpus': torch_gpus,
            'peak_memory_mb': self.peak_memory
        }

        self.logs.append(log_entry)

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def run(self):
        """Main monitoring loop"""
        print("🚀 Starting GPU monitor...")
        print(f"Update interval: {self.interval}s")

        if self.log_file:
            print(f"Logging to: {self.log_file}")

        print("\nInitializing...")
        time.sleep(2)

        try:
            while True:
                nvidia_gpus = self.get_nvidia_smi_stats()
                torch_gpus = self.get_torch_stats() if not nvidia_gpus else []

                self.display_stats(nvidia_gpus, torch_gpus)
                self.log_stats(nvidia_gpus, torch_gpus)

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped by user")
            self.print_summary()

    def print_summary(self):
        """Print monitoring summary"""
        if not self.logs:
            return

        print("\n" + "=" * 80)
        print("📈 MONITORING SUMMARY")
        print("=" * 80)
        print(f"Total runtime: {(time.time() - self.start_time)/60:.1f} minutes")
        print(f"Peak VRAM usage: {self.format_memory(self.peak_memory)}")
        print(f"Total measurements: {len(self.logs)}")

        if self.log_file:
            print(f"Full log saved to: {self.log_file}")

        print("=" * 80)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Monitor GPU during DeepSeek-R1 training')
    parser.add_argument('--interval', type=float, default=2.0, help='Update interval in seconds')
    parser.add_argument('--log', type=str, help='Log file path (optional)')
    parser.add_argument('--check', action='store_true', help='Check GPU availability and exit')

    args = parser.parse_args()

    if args.check:
        print("🔍 Checking GPU availability...\n")

        # Check CUDA
        if TORCH_AVAILABLE and torch.cuda.is_available():
            print("✅ PyTorch CUDA available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n   GPU {i}: {props.name}")
                print(f"   VRAM: {props.total_memory / 1024**3:.1f} GB")
                print(f"   Compute capability: {props.major}.{props.minor}")
        else:
            print("❌ PyTorch CUDA not available")

        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            print("\n✅ nvidia-smi available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\n❌ nvidia-smi not available")

        return

    # Run monitor
    monitor = GPUMonitor(log_file=args.log, interval=args.interval)
    monitor.run()

if __name__ == "__main__":
    main()
