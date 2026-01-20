"""
Logging Utilities
Comprehensive logging for training and evaluation

Features:
- Console logging
- File logging
- TensorBoard logging
- Wandb logging
- JSON logging

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys


# ==================== Console Logger ====================

class ConsoleLogger:
    """
    Console logger with formatting
    """
    
    def __init__(
        self,
        name: str = 'AlphaNego',
        level: int = logging.INFO,
        use_color: bool = True,
    ):
        """
        Args:
            name: Logger name
            level: Logging level
            use_color: Use colored output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Formatter
        if use_color:
            formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        
        # Add handler
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)


# ==================== Colored Formatter ====================

class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',
    }
    
    def format(self, record):
        """Format with color"""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


# ==================== File Logger ====================

class FileLogger:
    """
    Log to file
    """
    
    def __init__(
        self,
        log_dir: str,
        name: str = 'training',
        level: int = logging.INFO,
    ):
        """
        Args:
            log_dir: Directory for log files
            name: Log file name
            level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'{name}_{timestamp}.log'
        
        self.logger = logging.getLogger(f'{name}_file')
        self.logger.setLevel(level)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
        
        print(f"[FileLogger] Logging to: {log_file}")
    
    def log(self, message: str, level: str = 'INFO'):
        """Log message"""
        level_fn = getattr(self.logger, level.lower(), self.logger.info)
        level_fn(message)


# ==================== JSON Logger ====================

class JSONLogger:
    """
    Log metrics to JSON file
    """
    
    def __init__(self, log_file: str):
        """
        Args:
            log_file: Path to JSON log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logs = []
        
        print(f"[JSONLogger] Logging to: {log_file}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics
        
        Args:
            metrics: Metrics dictionary
            step: Training step
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'metrics': metrics,
        }
        
        self.logs.append(entry)
    
    def save(self):
        """Save logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def load(self):
        """Load logs from file"""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)
        return self.logs


# ==================== TensorBoard Logger ====================

class TensorBoardLogger:
    """
    Log to TensorBoard
    """
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: TensorBoard log directory
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
            
            print(f"[TensorBoardLogger] Logging to: {log_dir}")
            print(f"  View with: tensorboard --logdir {log_dir}")
        except ImportError:
            print("[TensorBoardLogger] TensorBoard not available")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars"""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        if self.enabled:
            self.writer.add_text(tag, text, step)
    
    def close(self):
        """Close writer"""
        if self.enabled:
            self.writer.close()


# ==================== Wandb Logger ====================

class WandbLogger:
    """
    Log to Weights & Biases
    """
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            project: Wandb project name
            name: Run name
            config: Configuration dictionary
        """
        try:
            import wandb
            
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
            )
            self.enabled = True
            
            print(f"[WandbLogger] Logging to project: {project}")
            print(f"  Run: {self.run.name}")
            print(f"  URL: {self.run.url}")
        except ImportError:
            print("[WandbLogger] Wandb not available")
            self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics
        
        Args:
            metrics: Metrics dictionary
            step: Training step
        """
        if self.enabled:
            self.wandb.log(metrics, step=step)
    
    def finish(self):
        """Finish run"""
        if self.enabled:
            self.run.finish()


# ==================== Multi Logger ====================

class MultiLogger:
    """
    Log to multiple destinations
    """
    
    def __init__(
        self,
        log_dir: str,
        use_console: bool = True,
        use_file: bool = True,
        use_json: bool = True,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ):
        """
        Args:
            log_dir: Base logging directory
            use_console: Use console logging
            use_file: Use file logging
            use_json: Use JSON logging
            use_tensorboard: Use TensorBoard logging
            use_wandb: Use Wandb logging
            wandb_project: Wandb project name
            wandb_config: Wandb configuration
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.loggers = {}
        
        if use_console:
            self.loggers['console'] = ConsoleLogger()
        
        if use_file:
            self.loggers['file'] = FileLogger(log_dir)
        
        if use_json:
            json_file = self.log_dir / 'metrics.json'
            self.loggers['json'] = JSONLogger(json_file)
        
        if use_tensorboard:
            tb_dir = self.log_dir / 'tensorboard'
            self.loggers['tensorboard'] = TensorBoardLogger(str(tb_dir))
        
        if use_wandb and wandb_project:
            self.loggers['wandb'] = WandbLogger(
                project=wandb_project,
                config=wandb_config,
            )
        
        print(f"[MultiLogger] Initialized with {len(self.loggers)} loggers")
    
    def log_message(self, message: str, level: str = 'INFO'):
        """
        Log text message
        
        Args:
            message: Message to log
            level: Log level
        """
        if 'console' in self.loggers:
            level_fn = getattr(self.loggers['console'], level.lower(), self.loggers['console'].info)
            level_fn(message)
        
        if 'file' in self.loggers:
            self.loggers['file'].log(message, level)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics
        
        Args:
            metrics: Metrics dictionary
            step: Training step
        """
        # JSON logger
        if 'json' in self.loggers:
            self.loggers['json'].log(metrics, step)
        
        # TensorBoard logger
        if 'tensorboard' in self.loggers and step is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.loggers['tensorboard'].log_scalar(key, value, step)
        
        # Wandb logger
        if 'wandb' in self.loggers:
            self.loggers['wandb'].log(metrics, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram to TensorBoard"""
        if 'tensorboard' in self.loggers:
            self.loggers['tensorboard'].log_histogram(tag, values, step)
    
    def save(self):
        """Save all logs"""
        if 'json' in self.loggers:
            self.loggers['json'].save()
    
    def close(self):
        """Close all loggers"""
        if 'tensorboard' in self.loggers:
            self.loggers['tensorboard'].close()
        
        if 'wandb' in self.loggers:
            self.loggers['wandb'].finish()
        
        self.save()


# ==================== Training Logger ====================

class TrainingLogger:
    """
    Specialized logger for training
    """
    
    def __init__(self, log_dir: str, **kwargs):
        """
        Args:
            log_dir: Logging directory
            **kwargs: Arguments for MultiLogger
        """
        self.logger = MultiLogger(log_dir, **kwargs)
        self.epoch = 0
        self.step = 0
    
    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """
        Log epoch metrics
        
        Args:
            epoch: Epoch number
            metrics: Metrics dictionary
        """
        self.epoch = epoch
        
        # Log message
        message = f"Epoch {epoch}: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        self.logger.log_message(message, level='INFO')
        
        # Log metrics
        metrics['epoch'] = epoch
        self.logger.log_metrics(metrics, step=epoch)
    
    def log_step(self, step: int, metrics: Dict[str, Any]):
        """
        Log step metrics
        
        Args:
            step: Step number
            metrics: Metrics dictionary
        """
        self.step = step
        metrics['step'] = step
        self.logger.log_metrics(metrics, step=step)
    
    def log_evaluation(self, epoch: int, eval_metrics: Dict[str, Any]):
        """
        Log evaluation metrics
        
        Args:
            epoch: Epoch number
            eval_metrics: Evaluation metrics
        """
        message = f"Evaluation at epoch {epoch}: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in eval_metrics.items()
        )
        self.logger.log_message(message, level='INFO')
        
        # Add prefix to metrics
        prefixed_metrics = {f'eval/{k}': v for k, v in eval_metrics.items()}
        self.logger.log_metrics(prefixed_metrics, step=epoch)
    
    def close(self):
        """Close logger"""
        self.logger.close()

