#!/usr/bin/env python3
"""
Shaggoth AI Command Center - Terminal Dashboard
DeepSeek-R1 671B Training Interface for TTY Display
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, Button, Log, ProgressBar, Label
from textual.reactive import reactive
from textual import on
from datetime import datetime
import psutil
import os
import sys

class SystemPanel(Static):
    """System resource monitoring panel"""

    cpu_usage = reactive(0.0)
    ram_usage = reactive(0.0)
    gpu_status = reactive("N/A")

    def compose(self) -> ComposeResult:
        yield Static("", id="system-stats")

    def on_mount(self) -> None:
        self.set_interval(2.0, self.update_stats)
        self.update_display()

    def update_stats(self) -> None:
        self.cpu_usage = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        self.ram_usage = mem.percent

        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_status = f"{torch.cuda.get_device_name(0)} - {torch.cuda.memory_allocated(0) / 1e9:.1f}GB"
            else:
                self.gpu_status = "No CUDA GPU"
        except:
            self.gpu_status = "PyTorch not installed"

        self.update_display()

    def update_display(self) -> None:
        cpu_bar = "█" * int(self.cpu_usage / 5) + "░" * (20 - int(self.cpu_usage / 5))
        ram_bar = "█" * int(self.ram_usage / 5) + "░" * (20 - int(self.ram_usage / 5))

        content = []
        content.append("[bold cyan]SYSTEM RESOURCES[/bold cyan]\n")
        content.append(f"CPU:  [{cpu_bar}] {self.cpu_usage:5.1f}%\n")
        content.append(f"RAM:  [{ram_bar}] {self.ram_usage:5.1f}%\n")
        content.append(f"GPU:  {self.gpu_status}\n")

        self.query_one("#system-stats").update("\n".join(content))


class ModelPanel(Static):
    """Model information and status panel"""

    model_name = reactive("DeepSeek-R1-671B")
    model_status = reactive("Not Loaded")
    model_path = reactive("./DeepSeek-R1")

    def compose(self) -> ComposeResult:
        yield Static("", id="model-info")

    def on_mount(self) -> None:
        self.update_display()

    def update_display(self) -> None:
        content = []
        content.append("[bold magenta]MODEL CONFIGURATION[/bold magenta]\n")
        content.append(f"Name:   {self.model_name}\n")
        content.append(f"Status: {self.model_status}\n")
        content.append(f"Path:   {self.model_path}\n")

        # Check if model files exist
        if os.path.exists(self.model_path):
            size = sum(os.path.getsize(os.path.join(self.model_path, f))
                      for f in os.listdir(self.model_path)
                      if os.path.isfile(os.path.join(self.model_path, f)))
            content.append(f"Size:   {size / 1e9:.1f} GB\n")
        else:
            content.append(f"[yellow]⚠ Model path not found[/yellow]\n")

        self.query_one("#model-info").update("\n".join(content))


class TrainingPanel(Static):
    """Training status and controls"""

    training_status = reactive("Idle")
    current_epoch = reactive(0)
    total_epochs = reactive(0)
    progress = reactive(0.0)

    def compose(self) -> ComposeResult:
        yield Static("", id="training-info")
        yield ProgressBar(total=100, show_eta=False, id="training-progress")
        yield Horizontal(
            Button("Start Training", id="btn-start", variant="success"),
            Button("Stop Training", id="btn-stop", variant="error"),
            Button("Load Model", id="btn-load", variant="primary"),
            id="control-buttons"
        )

    def on_mount(self) -> None:
        self.update_display()

    def update_display(self) -> None:
        content = []
        content.append("[bold green]TRAINING STATUS[/bold green]\n")
        content.append(f"Status: {self.training_status}\n")

        if self.total_epochs > 0:
            content.append(f"Epoch:  {self.current_epoch}/{self.total_epochs}\n")
            content.append(f"Progress: {self.progress:.1f}%\n")
        else:
            content.append(f"Ready to start training\n")

        self.query_one("#training-info").update("\n".join(content))
        self.query_one("#training-progress", ProgressBar).update(progress=self.progress)

    @on(Button.Pressed, "#btn-start")
    def start_training(self) -> None:
        self.training_status = "Training..."
        self.current_epoch = 1
        self.total_epochs = 10
        self.progress = 0.0
        self.update_display()
        self.app.log_message("[green]✓[/green] Training started")

    @on(Button.Pressed, "#btn-stop")
    def stop_training(self) -> None:
        self.training_status = "Stopped"
        self.update_display()
        self.app.log_message("[yellow]⚠[/yellow] Training stopped by user")

    @on(Button.Pressed, "#btn-load")
    def load_model(self) -> None:
        self.app.log_message("[cyan]⟳[/cyan] Loading DeepSeek-R1 model...")
        self.app.log_message("[yellow]⚠[/yellow] Model loading requires PyTorch and Transformers")


class ShaggothDashboard(App):
    """Shaggoth AI Command Center - Terminal Dashboard"""

    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        background: $primary;
        color: $text;
        text-style: bold;
    }

    Footer {
        background: $primary-darken-1;
    }

    #main-container {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #top-panels {
        height: auto;
        margin-bottom: 1;
    }

    .panel {
        border: solid $primary;
        background: $surface-darken-1;
        padding: 1;
        margin: 0 1;
        height: auto;
    }

    SystemPanel {
        width: 1fr;
        min-height: 8;
    }

    ModelPanel {
        width: 1fr;
        min-height: 8;
    }

    TrainingPanel {
        width: 100%;
        height: auto;
        min-height: 12;
    }

    #training-progress {
        margin: 1 0;
    }

    #control-buttons {
        height: auto;
        align: center middle;
    }

    #control-buttons Button {
        margin: 0 1;
    }

    #log-container {
        border: solid $accent;
        background: $surface-darken-2;
        padding: 1;
        height: 1fr;
        margin-top: 1;
    }

    #log-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    Log {
        height: 100%;
        background: $surface-darken-2;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("c", "clear_log", "Clear Log"),
    ]

    TITLE = "Shaggoth AI Command Center"
    SUB_TITLE = "DeepSeek-R1 671B Parameter Reasoning Model Training"

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-container"):
            with Horizontal(id="top-panels"):
                yield SystemPanel(classes="panel")
                yield ModelPanel(classes="panel")

            yield TrainingPanel(classes="panel")

            with Container(id="log-container"):
                yield Static("[bold yellow]TRAINING LOGS[/bold yellow]", id="log-title")
                yield Log(id="training-log", auto_scroll=True)

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the dashboard"""
        self.log_message("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
        self.log_message("[bold cyan]  SHAGGOTH AI COMMAND CENTER - INITIALIZED[/bold cyan]")
        self.log_message("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
        self.log_message(f"[green]✓[/green] System started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"[green]✓[/green] Platform: {sys.platform}")
        self.log_message(f"[green]✓[/green] Python: {sys.version.split()[0]}")
        self.log_message("")

        # Check for PyTorch
        try:
            import torch
            self.log_message(f"[green]✓[/green] PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                self.log_message(f"[green]✓[/green] CUDA: {torch.version.cuda}")
                self.log_message(f"[green]✓[/green] GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.log_message(f"[yellow]⚠[/yellow] No CUDA GPU detected")
        except ImportError:
            self.log_message(f"[red]✗[/red] PyTorch not installed - install with:")
            self.log_message(f"[dim]    ./venv/bin/pip install torch transformers[/dim]")

        self.log_message("")
        self.log_message("[cyan]Ready to load model and start training[/cyan]")

    def log_message(self, message: str) -> None:
        """Add a message to the training log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log = self.query_one("#training-log", Log)
        log.write_line(f"[{timestamp}] {message}")

    def action_refresh(self) -> None:
        """Refresh all panels"""
        self.log_message("[cyan]⟳[/cyan] Refreshing dashboard...")
        self.query_one(SystemPanel).update_stats()
        self.query_one(ModelPanel).update_display()

    def action_clear_log(self) -> None:
        """Clear the training log"""
        log = self.query_one("#training-log", Log)
        log.clear()
        self.log_message("[cyan]⟳[/cyan] Log cleared")


if __name__ == "__main__":
    app = ShaggothDashboard()
    app.run()
