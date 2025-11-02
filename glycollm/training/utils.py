"""
Training utilities for GlycoLLM.

This module provides utility functions and classes for training
including learning rate scheduling, checkpoint management, and
training loop helpers.
"""

import os
import json
import logging
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import glob
import shutil

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import _LRScheduler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    _LRScheduler = object


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    # Loss metrics
    total_loss: float = 0.0
    structure_loss: float = 0.0
    spectra_loss: float = 0.0
    text_loss: float = 0.0
    contrastive_loss: float = 0.0
    
    # Accuracy metrics
    structure_accuracy: float = 0.0
    spectra_accuracy: float = 0.0
    text_accuracy: float = 0.0
    
    # Training metadata
    epoch: int = 0
    step: int = 0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    batch_size: int = 0
    samples_per_second: float = 0.0
    
    # Memory usage
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Features:
    - Automatic checkpoint saving
    - Best model tracking
    - Checkpoint rotation
    - Resume capability
    """
    
    def __init__(self, 
                 checkpoint_dir: str,
                 max_checkpoints: int = 5,
                 save_best_only: bool = False,
                 monitor_metric: str = "total_loss",
                 mode: str = "min"):
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_history = []
        
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[_LRScheduler],
                       epoch: int,
                       step: int,
                       metrics: TrainingMetrics,
                       additional_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            additional_state: Additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        
        if not HAS_TORCH:
            logger.warning("PyTorch not available - cannot save checkpoint")
            return ""
            
        # Check if we should save based on metric
        current_metric = getattr(metrics, self.monitor_metric, 0.0)
        is_best = self._is_better_metric(current_metric)
        
        if self.save_best_only and not is_best:
            return ""
            
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': asdict(metrics),
            'best_metric': self.best_metric,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        if additional_state:
            checkpoint['additional_state'] = additional_state
            
        # Generate checkpoint filename
        if is_best:
            filename = f"best_model_epoch_{epoch}_step_{step}.pt"
            self.best_metric = current_metric
        else:
            filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
            
        checkpoint_path = self.checkpoint_dir / filename
        
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Update checkpoint history
            self.checkpoint_history.append({
                'path': str(checkpoint_path),
                'epoch': epoch,
                'step': step,
                'metric': current_metric,
                'is_best': is_best,
                'timestamp': time.time()
            })
            
            # Rotate old checkpoints
            if not is_best:  # Never delete best checkpoints
                self._rotate_checkpoints()
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return ""
            
        return str(checkpoint_path)
        
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[_LRScheduler] = None,
                       load_optimizer: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Checkpoint metadata
        """
        
        if not HAS_TORCH:
            logger.warning("PyTorch not available - cannot load checkpoint")
            return {}
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model state from {checkpoint_path}")
            
            # Load optimizer state
            if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state")
                
            # Load scheduler state
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Loaded scheduler state")
                
            # Update best metric
            if 'best_metric' in checkpoint:
                self.best_metric = checkpoint['best_metric']
                
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
            
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        
        checkpoint_pattern = self.checkpoint_dir / "checkpoint_*.pt"
        checkpoints = glob.glob(str(checkpoint_pattern))
        
        if not checkpoints:
            return None
            
        # Sort by modification time
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]
        
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get path to the best checkpoint.
        
        Returns:
            Path to best checkpoint or None
        """
        
        best_pattern = self.checkpoint_dir / "best_model_*.pt"
        best_checkpoints = glob.glob(str(best_pattern))
        
        if not best_checkpoints:
            return None
            
        # Sort by modification time (latest best)
        best_checkpoints.sort(key=os.path.getmtime, reverse=True)
        return best_checkpoints[0]
        
    def _is_better_metric(self, metric_value: float) -> bool:
        """Check if metric value is better than current best."""
        
        if self.mode == 'min':
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric
            
    def _rotate_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        
        if self.max_checkpoints <= 0:
            return
            
        # Get all regular checkpoints (not best)
        checkpoint_pattern = self.checkpoint_dir / "checkpoint_*.pt"
        checkpoints = glob.glob(str(checkpoint_pattern))
        
        if len(checkpoints) <= self.max_checkpoints:
            return
            
        # Sort by modification time (oldest first)
        checkpoints.sort(key=os.path.getmtime)
        
        # Remove oldest checkpoints
        to_remove = len(checkpoints) - self.max_checkpoints
        for i in range(to_remove):
            try:
                os.remove(checkpoints[i])
                logger.info(f"Removed old checkpoint: {checkpoints[i]}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoints[i]}: {e}")


if HAS_TORCH:
    class WarmupCosineScheduler(_LRScheduler):
        """
        Learning rate scheduler with linear warmup and cosine annealing.
        
        Features:
        - Linear warmup phase
        - Cosine annealing decay
        - Minimum learning rate
        """
        
        def __init__(self, 
                     optimizer: torch.optim.Optimizer,
                     warmup_steps: int,
                     total_steps: int,
                     min_lr: float = 1e-7,
                     last_epoch: int = -1):
            
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps
            self.min_lr = min_lr
            
            super().__init__(optimizer, last_epoch)
            
        def get_lr(self):
            """Calculate learning rate for current step."""
            
            if self.last_epoch < self.warmup_steps:
                # Linear warmup
                lr_scale = self.last_epoch / self.warmup_steps
                return [base_lr * lr_scale for base_lr in self.base_lrs]
            else:
                # Cosine annealing
                progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                progress = min(progress, 1.0)
                
                cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                
                return [
                    self.min_lr + (base_lr - self.min_lr) * cosine_factor
                    for base_lr in self.base_lrs
                ]


    class PolynomialDecayScheduler(_LRScheduler):
        """
        Polynomial learning rate decay scheduler.
        
        Features:
        - Polynomial decay with configurable power
        - End learning rate specification
        """
        
        def __init__(self, 
                     optimizer: torch.optim.Optimizer,
                     total_steps: int,
                     end_lr: float = 1e-7,
                     power: float = 1.0,
                     last_epoch: int = -1):
            
            self.total_steps = total_steps
            self.end_lr = end_lr
            self.power = power
            
            super().__init__(optimizer, last_epoch)
            
        def get_lr(self):
            """Calculate learning rate for current step."""
            
            if self.last_epoch >= self.total_steps:
                return [self.end_lr for _ in self.base_lrs]
                
            decay_factor = (1 - self.last_epoch / self.total_steps) ** self.power
            
            return [
                self.end_lr + (base_lr - self.end_lr) * decay_factor
                for base_lr in self.base_lrs
            ]

else:
    # Mock classes when PyTorch is not available
    class WarmupCosineScheduler:
        def __init__(self, **kwargs):
            pass
            
    class PolynomialDecayScheduler:
        def __init__(self, **kwargs):
            pass


class MetricsLogger:
    """
    Logs training metrics to various outputs.
    
    Supports:
    - Console logging
    - JSON file logging
    - TensorBoard integration (optional)
    """
    
    def __init__(self, 
                 log_dir: str,
                 console_logging: bool = True,
                 file_logging: bool = True,
                 tensorboard_logging: bool = False):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.console_logging = console_logging
        self.file_logging = file_logging
        self.tensorboard_logging = tensorboard_logging
        
        # Initialize file logger
        if self.file_logging:
            self.metrics_file = self.log_dir / "training_metrics.jsonl"
            
        # Initialize TensorBoard logger
        if self.tensorboard_logging:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                logger.warning("TensorBoard not available")
                self.tensorboard_logging = False
                self.tb_writer = None
        else:
            self.tb_writer = None
            
    def log_metrics(self, 
                   metrics: TrainingMetrics,
                   step: Optional[int] = None,
                   prefix: str = "train"):
        """
        Log training metrics to all configured outputs.
        
        Args:
            metrics: Training metrics to log
            step: Global step number
            prefix: Metric name prefix (train/val/test)
        """
        
        step = step or metrics.step
        
        # Console logging
        if self.console_logging:
            self._log_to_console(metrics, step, prefix)
            
        # File logging
        if self.file_logging:
            self._log_to_file(metrics, step, prefix)
            
        # TensorBoard logging
        if self.tensorboard_logging and self.tb_writer:
            self._log_to_tensorboard(metrics, step, prefix)
            
    def _log_to_console(self, metrics: TrainingMetrics, step: int, prefix: str):
        """Log metrics to console."""
        
        logger.info(
            f"[{prefix.upper()}] Epoch {metrics.epoch}, Step {step}: "
            f"Loss={metrics.total_loss:.4f}, "
            f"Struct_Acc={metrics.structure_accuracy:.3f}, "
            f"Spectra_Acc={metrics.spectra_accuracy:.3f}, "
            f"Text_Acc={metrics.text_accuracy:.3f}, "
            f"LR={metrics.learning_rate:.2e}, "
            f"Samples/s={metrics.samples_per_second:.1f}"
        )
        
    def _log_to_file(self, metrics: TrainingMetrics, step: int, prefix: str):
        """Log metrics to JSON file."""
        
        try:
            log_entry = {
                'timestamp': time.time(),
                'prefix': prefix,
                'step': step,
                'metrics': asdict(metrics)
            }
            
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.warning(f"Failed to log metrics to file: {e}")
            
    def _log_to_tensorboard(self, metrics: TrainingMetrics, step: int, prefix: str):
        """Log metrics to TensorBoard."""
        
        try:
            # Loss metrics
            self.tb_writer.add_scalar(f'{prefix}/total_loss', metrics.total_loss, step)
            self.tb_writer.add_scalar(f'{prefix}/structure_loss', metrics.structure_loss, step)
            self.tb_writer.add_scalar(f'{prefix}/spectra_loss', metrics.spectra_loss, step)
            self.tb_writer.add_scalar(f'{prefix}/text_loss', metrics.text_loss, step)
            self.tb_writer.add_scalar(f'{prefix}/contrastive_loss', metrics.contrastive_loss, step)
            
            # Accuracy metrics
            self.tb_writer.add_scalar(f'{prefix}/structure_accuracy', metrics.structure_accuracy, step)
            self.tb_writer.add_scalar(f'{prefix}/spectra_accuracy', metrics.spectra_accuracy, step)
            self.tb_writer.add_scalar(f'{prefix}/text_accuracy', metrics.text_accuracy, step)
            
            # Training metadata
            self.tb_writer.add_scalar(f'{prefix}/learning_rate', metrics.learning_rate, step)
            self.tb_writer.add_scalar(f'{prefix}/gradient_norm', metrics.gradient_norm, step)
            self.tb_writer.add_scalar(f'{prefix}/samples_per_second', metrics.samples_per_second, step)
            
            # GPU memory
            if metrics.gpu_memory_used > 0:
                self.tb_writer.add_scalar(f'{prefix}/gpu_memory_used', metrics.gpu_memory_used, step)
                memory_util = metrics.gpu_memory_used / metrics.gpu_memory_total if metrics.gpu_memory_total > 0 else 0
                self.tb_writer.add_scalar(f'{prefix}/gpu_memory_utilization', memory_util, step)
                
            self.tb_writer.flush()
            
        except Exception as e:
            logger.warning(f"Failed to log metrics to TensorBoard: {e}")
            
    def close(self):
        """Close logger and cleanup resources."""
        
        if self.tb_writer:
            self.tb_writer.close()


class GradientClipping:
    """
    Gradient clipping utilities for stable training.
    
    Features:
    - Gradient norm clipping
    - Gradient value clipping
    - Adaptive clipping
    """
    
    @staticmethod
    def clip_grad_norm(model: nn.Module, max_norm: float) -> float:
        """
        Clip gradients by norm.
        
        Args:
            model: Model with parameters
            max_norm: Maximum gradient norm
            
        Returns:
            Total gradient norm before clipping
        """
        
        if not HAS_TORCH:
            return 0.0
            
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
    @staticmethod
    def clip_grad_value(model: nn.Module, clip_value: float):
        """
        Clip gradients by value.
        
        Args:
            model: Model with parameters
            clip_value: Maximum gradient value
        """
        
        if not HAS_TORCH:
            return
            
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        
    @staticmethod
    def get_grad_norm(model: nn.Module) -> float:
        """
        Get total gradient norm.
        
        Args:
            model: Model with parameters
            
        Returns:
            Total gradient norm
        """
        
        if not HAS_TORCH:
            return 0.0
            
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
        return total_norm ** (1. / 2)


def create_scheduler(optimizer: torch.optim.Optimizer, 
                    scheduler_config: Dict[str, Any]) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_config: Scheduler configuration
        
    Returns:
        Learning rate scheduler
    """
    
    if not HAS_TORCH:
        return None
        
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=scheduler_config.get('warmup_steps', 1000),
            total_steps=scheduler_config.get('total_steps', 10000),
            min_lr=scheduler_config.get('min_lr', 1e-7)
        )
    elif scheduler_type == 'polynomial':
        return PolynomialDecayScheduler(
            optimizer,
            total_steps=scheduler_config.get('total_steps', 10000),
            end_lr=scheduler_config.get('end_lr', 1e-7),
            power=scheduler_config.get('power', 1.0)
        )
    elif scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 1000),
            eta_min=scheduler_config.get('eta_min', 1e-7)
        )
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 1000),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}")
        return None


def get_gpu_memory_info() -> Tuple[float, float]:
    """
    Get GPU memory usage information.
    
    Returns:
        Tuple of (used_memory_gb, total_memory_gb)
    """
    
    if not HAS_TORCH or not torch.cuda.is_available():
        return 0.0, 0.0
        
    try:
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return used, total
    except:
        return 0.0, 0.0


def format_time(seconds: float) -> str:
    """
    Format time duration as human-readable string.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"