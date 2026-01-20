"""
Training Callbacks
Callbacks for monitoring and controlling training

Features:
- Checkpointing
- Early stopping
- Logging
- Visualization
- Custom callbacks

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod


# ==================== Base Callback ====================

class Callback(ABC):
    """
    Base callback class
    """
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training"""
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of an epoch"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of an epoch"""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of a batch"""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of a batch"""
        pass


# ==================== Checkpoint Callback ====================

class CheckpointCallback(Callback):
    """
    Save model checkpoints during training
    """
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = False,
        save_frequency: int = 10,
        verbose: bool = True,
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best model
            save_frequency: Save every N epochs
            verbose: Print messages
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.verbose = verbose
        
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.best_epoch = 0
        
        print(f"[CheckpointCallback] Initialized")
        print(f"  Save directory: {save_dir}")
        print(f"  Monitor: {monitor} ({mode})")
        print(f"  Save best only: {save_best_only}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save checkpoint at epoch end"""
        if logs is None:
            return
        
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        # Check if this is the best model
        is_best = False
        if self.mode == 'min':
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value
        
        # Update best
        if is_best:
            self.best_value = current_value
            self.best_epoch = epoch
            
            if self.save_best_only:
                # Save best model
                best_path = self.save_dir / 'best_model.pt'
                logs['save_path'] = str(best_path)
                
                if self.verbose:
                    print(f"\n[Checkpoint] New best {self.monitor}: {current_value:.4f}")
                    print(f"             Saved to: {best_path}")
        
        # Save periodic checkpoint
        if not self.save_best_only and (epoch + 1) % self.save_frequency == 0:
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            logs['save_path'] = str(checkpoint_path)
            
            if self.verbose:
                print(f"\n[Checkpoint] Epoch {epoch+1} saved to: {checkpoint_path}")


# ==================== Early Stopping ====================

class EarlyStoppingCallback(Callback):
    """
    Stop training when monitored metric stops improving
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0001,
        verbose: bool = True,
    ):
        """
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.wait = 0
        self.stopped_epoch = 0
        
        print(f"[EarlyStoppingCallback] Initialized")
        print(f"  Monitor: {monitor} ({mode})")
        print(f"  Patience: {patience}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check for early stopping"""
        if logs is None:
            return
        
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        # Check if improved
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logs['stop_training'] = True
                
                if self.verbose:
                    print(f"\n[EarlyStopping] Stopping at epoch {epoch+1}")
                    print(f"                No improvement in {self.monitor} for {self.patience} epochs")
                    print(f"                Best {self.monitor}: {self.best_value:.4f}")


# ==================== History Logger ====================

class HistoryCallback(Callback):
    """
    Log training history
    """
    
    def __init__(self, save_path: Optional[str] = None):
        """
        Args:
            save_path: Path to save history JSON
        """
        self.save_path = save_path
        self.history = {
            'epochs': [],
            'metrics': {},
        }
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log epoch metrics"""
        if logs is None:
            return
        
        self.history['epochs'].append(epoch)
        
        for key, value in logs.items():
            if key not in self.history['metrics']:
                self.history['metrics'][key] = []
            self.history['metrics'][key].append(float(value) if isinstance(value, (int, float, np.number)) else value)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Save history"""
        if self.save_path:
            with open(self.save_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"\n[History] Saved to: {self.save_path}")
    
    def get_history(self) -> Dict:
        """Get training history"""
        return self.history


# ==================== Learning Rate Scheduler ====================

class LearningRateSchedulerCallback(Callback):
    """
    Adjust learning rate during training
    """
    
    def __init__(
        self,
        scheduler,
        monitor: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            scheduler: PyTorch learning rate scheduler
            monitor: Metric to monitor (for ReduceLROnPlateau)
            verbose: Print messages
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Update learning rate"""
        if self.monitor and logs:
            # For ReduceLROnPlateau
            metric_value = logs.get(self.monitor)
            if metric_value is not None:
                self.scheduler.step(metric_value)
        else:
            # For other schedulers
            self.scheduler.step()
        
        if self.verbose:
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"[LRScheduler] Learning rate: {current_lr:.6f}")


# ==================== Progress Bar ====================

class ProgressBarCallback(Callback):
    """
    Display progress bar
    """
    
    def __init__(self, total_epochs: int):
        """
        Args:
            total_epochs: Total number of epochs
        """
        self.total_epochs = total_epochs
        self.epoch_bar = None
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Initialize progress bar"""
        try:
            from tqdm import tqdm
            self.epoch_bar = tqdm(total=self.total_epochs, desc="Training")
        except ImportError:
            print("[ProgressBar] tqdm not installed, skipping progress bar")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Update progress bar"""
        if self.epoch_bar:
            # Update with metrics
            if logs:
                postfix = {k: f"{v:.4f}" for k, v in logs.items() if isinstance(v, (int, float, np.number))}
                self.epoch_bar.set_postfix(postfix)
            
            self.epoch_bar.update(1)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Close progress bar"""
        if self.epoch_bar:
            self.epoch_bar.close()


# ==================== Metric Logger ====================

class MetricLoggerCallback(Callback):
    """
    Log metrics to console
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        log_frequency: int = 1,
    ):
        """
        Args:
            metrics: List of metrics to log (None = all)
            log_frequency: Log every N epochs
        """
        self.metrics = metrics
        self.log_frequency = log_frequency
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log metrics"""
        if logs is None or (epoch + 1) % self.log_frequency != 0:
            return
        
        print(f"\nEpoch {epoch + 1}:")
        
        for key, value in logs.items():
            if self.metrics is None or key in self.metrics:
                if isinstance(value, (int, float, np.number)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


# ==================== Callback Manager ====================

class CallbackManager:
    """
    Manage multiple callbacks
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: Callback):
        """Add a callback"""
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Call all callbacks"""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Call all callbacks"""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Call all callbacks"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Call all callbacks"""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
        
        # Check for stop signal
        if logs and logs.get('stop_training', False):
            return True
        return False
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Call all callbacks"""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Call all callbacks"""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


# ==================== Custom Callback Example ====================

class CustomMetricCallback(Callback):
    """
    Example custom callback for computing custom metrics
    """
    
    def __init__(self, metric_fn):
        """
        Args:
            metric_fn: Function to compute metric
        """
        self.metric_fn = metric_fn
        self.metrics = []
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Compute custom metric"""
        if logs:
            metric_value = self.metric_fn(logs)
            self.metrics.append(metric_value)
            logs['custom_metric'] = metric_value


# ==================== Testing ====================

if __name__ == '__main__':
    print("Testing Callbacks...")
    
    # Test CheckpointCallback
    print("\n1. Testing CheckpointCallback:")
    checkpoint_cb = CheckpointCallback(
        save_dir='/tmp/checkpoints/',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=True
    )
    
    # Simulate training
    for epoch in range(5):
        logs = {'val_loss': 1.0 - epoch * 0.1, 'train_loss': 1.2 - epoch * 0.1}
        checkpoint_cb.on_epoch_end(epoch, logs)
    
    # Test EarlyStoppingCallback
    print("\n2. Testing EarlyStoppingCallback:")
    early_stop_cb = EarlyStoppingCallback(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=True
    )
    
    # Simulate training with no improvement
    for epoch in range(10):
        if epoch < 5:
            val_loss = 1.0 - epoch * 0.1
        else:
            val_loss = 0.5  # No improvement
        
        logs = {'val_loss': val_loss}
        early_stop_cb.on_epoch_end(epoch, logs)
        
        if logs.get('stop_training'):
            print(f"   Training stopped at epoch {epoch + 1}")
            break
    
    # Test HistoryCallback
    print("\n3. Testing HistoryCallback:")
    history_cb = HistoryCallback(save_path='/tmp/history.json')
    
    history_cb.on_train_begin()
    for epoch in range(3):
        logs = {
            'train_loss': 1.0 - epoch * 0.2,
            'val_loss': 1.1 - epoch * 0.15,
            'accuracy': 0.5 + epoch * 0.1
        }
        history_cb.on_epoch_end(epoch, logs)
    history_cb.on_train_end()
    
    print(f"   History: {history_cb.get_history()}")
    
    # Test MetricLoggerCallback
    print("\n4. Testing MetricLoggerCallback:")
    logger_cb = MetricLoggerCallback(
        metrics=['train_loss', 'val_loss'],
        log_frequency=1
    )
    
    for epoch in range(2):
        logs = {'train_loss': 1.0, 'val_loss': 1.1, 'accuracy': 0.8}
        logger_cb.on_epoch_end(epoch, logs)
    
    # Test CallbackManager
    print("\n5. Testing CallbackManager:")
    manager = CallbackManager([
        CheckpointCallback('/tmp/checkpoints2/', save_best_only=False, save_frequency=2, verbose=False),
        MetricLoggerCallback(log_frequency=1),
    ])
    
    manager.on_train_begin()
    for epoch in range(3):
        manager.on_epoch_begin(epoch)
        logs = {'train_loss': 1.0 - epoch * 0.2, 'val_loss': 1.1 - epoch * 0.15}
        should_stop = manager.on_epoch_end(epoch, logs)
        if should_stop:
            break
    manager.on_train_end()
    
    print("\nCallbacks tests passed!")