"""
Training pipeline for GlycoLLM multimodal transformer.

This module implements a comprehensive training system with support for
multimodal learning, contrastive learning, distributed training, and
advanced optimization strategies.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import random

# Optional imports - available when packages installed
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.cuda.amp import GradScaler, autocast
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Mock classes for development
    class DataLoader:
        pass
    class DistributedSampler:
        pass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for GlycoLLM training"""
    
    # Model settings
    model_name: str = "glycollm-base"
    model_config_path: Optional[str] = None
    
    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "linear", "cosine", "polynomial"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Loss configuration
    structure_weight: float = 1.0
    spectra_weight: float = 1.0
    text_weight: float = 1.0
    contrastive_weight: float = 0.5
    contrastive_temperature: float = 0.07
    label_smoothing: float = 0.1
    
    # Contrastive learning
    hard_negative_mining: bool = True
    negative_sample_ratio: float = 0.1
    temperature_scheduling: bool = True
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    modality_dropout: float = 0.1  # Randomly drop modalities during training
    
    # Mixed precision training
    use_amp: bool = True
    amp_opt_level: str = "O1"
    
    # Distributed training
    use_ddp: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 3
    output_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    # Evaluation
    eval_steps: int = 500
    eval_accumulation_steps: int = 1
    eval_batch_size: int = 16
    
    # Logging
    logging_steps: int = 50
    log_level: str = "INFO"
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Advanced training strategies
    curriculum_learning: bool = False
    gradient_checkpointing: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0001
    
    # Seed for reproducibility
    seed: int = 42


@dataclass 
class TrainingState:
    """Current state of training"""
    epoch: int = 0
    global_step: int = 0
    total_steps: int = 0
    
    # Loss tracking
    train_loss: float = 0.0
    eval_loss: float = float('inf')
    best_eval_loss: float = float('inf')
    
    # Component losses
    structure_loss: float = 0.0
    spectra_loss: float = 0.0
    text_loss: float = 0.0
    contrastive_loss: float = 0.0
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Timing
    train_time: float = 0.0
    eval_time: float = 0.0
    
    # Early stopping
    patience_counter: int = 0
    should_stop: bool = False


class GlycoLLMTrainer:
    """
    Trainer class for GlycoLLM with multimodal capabilities.
    
    Handles training loop, evaluation, checkpointing, and distributed training.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 eval_dataloader: Optional[DataLoader] = None,
                 config: Optional[TrainingConfig] = None):
        """
        Initialize the trainer.
        
        Args:
            model: GlycoLLM model to train
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            config: Training configuration
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available - trainer will have limited functionality")
            
        self.config = config or TrainingConfig()
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Training state
        self.state = TrainingState()
        
        # Set random seeds for reproducibility
        self._set_seed(self.config.seed)
        
        # Setup device and distributed training
        self._setup_devices()
        
        # Initialize model for training
        self._setup_model()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup loss function
        self._setup_loss_function()
        
        # Setup mixed precision training
        if self.config.use_amp and HAS_TORCH:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            
        # Setup logging
        self._setup_logging()
        
        # Calculate total training steps
        self.state.total_steps = len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
        
    def _setup_devices(self):
        """Setup device configuration and distributed training."""
        if not HAS_TORCH:
            self.device = "cpu"
            return
            
        # Determine device
        if torch.cuda.is_available():
            if self.config.local_rank != -1:
                self.device = torch.device(f"cuda:{self.config.local_rank}")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # Initialize distributed training
        if self.config.use_ddp and self.config.local_rank != -1:
            dist.init_process_group(backend="nccl")
            self.config.world_size = dist.get_world_size()
            
        logger.info(f"Training device: {self.device}")
        logger.info(f"World size: {self.config.world_size}")
        
    def _setup_model(self):
        """Setup model for training."""
        if not HAS_TORCH:
            return
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        # Wrap model for distributed training
        if self.config.use_ddp and self.config.local_rank != -1:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=True
            )
            
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        if not HAS_TORCH:
            return
            
        # Get model parameters
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize learning rate scheduler
        self._setup_scheduler()
        
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if not HAS_TORCH:
            return
            
        if self.config.lr_scheduler == "linear":
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step):
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                return max(
                    0.0, 
                    float(self.state.total_steps - current_step) / 
                    float(max(1, self.state.total_steps - self.config.warmup_steps))
                )
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            
        elif self.config.lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.state.total_steps - self.config.warmup_steps,
                eta_min=0.0
            )
            
        else:
            self.scheduler = None
            
    def _setup_loss_function(self):
        """Setup multimodal loss function."""
        # Import here to avoid circular imports
        try:
            from ..models.loss_functions import MultiModalLoss, LossConfig
            
            loss_config = LossConfig(
                structure_weight=self.config.structure_weight,
                spectra_weight=self.config.spectra_weight,
                text_weight=self.config.text_weight,
                contrastive_weight=self.config.contrastive_weight,
                temperature=self.config.contrastive_temperature,
                label_smoothing=self.config.label_smoothing
            )
            
            self.loss_function = MultiModalLoss(loss_config)
            
        except ImportError:
            logger.warning("Could not import loss functions - using mock")
            self.loss_function = None
            
    def _setup_logging(self):
        """Setup logging and experiment tracking."""
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup tensorboard logging
        if "tensorboard" in self.config.report_to:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(
                    log_dir=os.path.join(self.config.output_dir, "logs")
                )
            except ImportError:
                logger.warning("TensorBoard not available")
                self.tb_writer = None
        else:
            self.tb_writer = None
            
    def train(self) -> TrainingState:
        """
        Execute the complete training loop.
        
        Returns:
            Final training state
        """
        logger.info("Starting GlycoLLM training...")
        logger.info(f"Total epochs: {self.config.num_epochs}")
        logger.info(f"Total steps: {self.state.total_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)
            
        # Training loop
        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            
            # Train one epoch
            epoch_loss = self._train_epoch()
            
            # Evaluate if requested
            if self.eval_dataloader is not None:
                eval_loss = self._evaluate()
                
                # Check for improvement
                if eval_loss < self.state.best_eval_loss - self.config.early_stopping_threshold:
                    self.state.best_eval_loss = eval_loss
                    self.state.patience_counter = 0
                    # Save best model
                    self._save_checkpoint(is_best=True)
                else:
                    self.state.patience_counter += 1
                    
                # Early stopping check
                if self.state.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    self.state.should_stop = True
                    break
                    
            # Save checkpoint
            if (epoch + 1) % max(1, self.config.num_epochs // 10) == 0:
                self._save_checkpoint()
                
        logger.info("Training completed!")
        return self.state
        
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        if not HAS_TORCH:
            return 0.0
            
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        # Setup distributed sampler
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.state.epoch)
            
        for step, batch in enumerate(self.train_dataloader):
            # Forward pass
            batch_loss = self._training_step(batch)
            
            total_loss += batch_loss
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                if self.scheduler:
                    self.scheduler.step()
                    
                self.optimizer.zero_grad()
                self.state.global_step += 1
                
                # Update learning rate in state
                self.state.learning_rate = self.optimizer.param_groups[0]['lr']
                
                # Logging
                if self.state.global_step % self.config.logging_steps == 0:
                    self._log_training_step(batch_loss)
                    
                # Evaluation during training
                if (self.eval_dataloader is not None and 
                    self.state.global_step % self.config.eval_steps == 0):
                    eval_loss = self._evaluate()
                    
                # Save checkpoint
                if self.state.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                    
        epoch_time = time.time() - epoch_start_time
        self.state.train_time += epoch_time
        
        avg_loss = total_loss / max(1, num_batches)
        logger.info(f"Epoch {self.state.epoch + 1} completed in {epoch_time:.2f}s, avg loss: {avg_loss:.6f}")
        
        return avg_loss
        
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute one training step.
        
        Args:
            batch: Input batch
            
        Returns:
            Loss value
        """
        if not HAS_TORCH:
            return 0.0
            
        # Move batch to device
        batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                for k, v in batch.items()}
        
        # Apply modality dropout for regularization
        if self.config.modality_dropout > 0 and self.training:
            batch = self._apply_modality_dropout(batch)
            
        # Forward pass with mixed precision
        if self.scaler and self.config.use_amp:
            with autocast():
                outputs = self.model(**batch)
                loss_dict = self._compute_loss(outputs, batch)
                loss = loss_dict['total_loss']
                
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()
            
        else:
            outputs = self.model(**batch)
            loss_dict = self._compute_loss(outputs, batch)
            loss = loss_dict['total_loss']
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()
            
        # Update state with component losses
        self.state.train_loss = loss.item()
        self.state.structure_loss = loss_dict.get('structure_loss', torch.tensor(0.0)).item()
        self.state.spectra_loss = loss_dict.get('spectra_loss', torch.tensor(0.0)).item()
        self.state.text_loss = loss_dict.get('text_loss', torch.tensor(0.0)).item()
        self.state.contrastive_loss = loss_dict.get('contrastive_loss', torch.tensor(0.0)).item()
        
        return loss.item()
        
    def _apply_modality_dropout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply modality dropout for regularization.
        
        Args:
            batch: Input batch
            
        Returns:
            Modified batch with some modalities dropped
        """
        if random.random() < self.config.modality_dropout:
            # Randomly choose which modality to drop
            modalities = ['text_input_ids', 'structure_input_ids', 'spectra_input_ids']
            available_modalities = [m for m in modalities if m in batch and batch[m] is not None]
            
            if len(available_modalities) > 1:  # Keep at least one modality
                drop_modality = random.choice(available_modalities)
                batch[drop_modality] = None
                
        return batch
        
    def _compute_loss(self, 
                     outputs: Dict[str, torch.Tensor],
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multimodal loss.
        
        Args:
            outputs: Model outputs
            batch: Input batch with targets
            
        Returns:
            Dictionary of loss components
        """
        if self.loss_function is None:
            # Mock loss for development
            return {'total_loss': torch.tensor(0.0)}
            
        # Prepare targets
        targets = {}
        if 'structure_targets' in batch:
            targets['structure_targets'] = batch['structure_targets']
        if 'spectra_targets' in batch:
            targets['spectra_targets'] = batch['spectra_targets']
        if 'text_targets' in batch:
            targets['text_targets'] = batch['text_targets']
            
        # Extract embeddings for contrastive learning if available
        embeddings = None
        if 'retrieval_embeddings' in outputs:
            embeddings = {
                'text_embeddings': outputs['retrieval_embeddings'][:, :batch.get('text_length', 0)].mean(1),
                'structure_embeddings': outputs['retrieval_embeddings'][:, batch.get('text_length', 0):].mean(1)
            }
            
        return self.loss_function(outputs, targets, embeddings)
        
    def _evaluate(self) -> float:
        """
        Run evaluation on validation set.
        
        Returns:
            Average evaluation loss
        """
        if not HAS_TORCH or self.eval_dataloader is None:
            return 0.0
            
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        eval_start_time = time.time()
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss_dict = self._compute_loss(outputs, batch)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
                
        eval_time = time.time() - eval_start_time
        self.state.eval_time += eval_time
        
        avg_loss = total_loss / max(1, num_batches)
        self.state.eval_loss = avg_loss
        
        logger.info(f"Evaluation completed in {eval_time:.2f}s, avg loss: {avg_loss:.6f}")
        
        # Log to tensorboard
        if self.tb_writer:
            self.tb_writer.add_scalar("eval/loss", avg_loss, self.state.global_step)
            
        self.model.train()
        return avg_loss
        
    def _log_training_step(self, loss: float):
        """Log training step metrics."""
        logger.info(
            f"Step {self.state.global_step}, Loss: {loss:.6f}, "
            f"LR: {self.state.learning_rate:.2e}"
        )
        
        # Tensorboard logging
        if self.tb_writer:
            self.tb_writer.add_scalar("train/loss", loss, self.state.global_step)
            self.tb_writer.add_scalar("train/learning_rate", self.state.learning_rate, self.state.global_step)
            self.tb_writer.add_scalar("train/structure_loss", self.state.structure_loss, self.state.global_step)
            self.tb_writer.add_scalar("train/spectra_loss", self.state.spectra_loss, self.state.global_step)
            self.tb_writer.add_scalar("train/text_loss", self.state.text_loss, self.state.global_step)
            self.tb_writer.add_scalar("train/contrastive_loss", self.state.contrastive_loss, self.state.global_step)
            
    def _save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        if not HAS_TORCH:
            return
            
        checkpoint_dir = Path(self.config.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model without DDP wrapper
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': asdict(self.config),
            'state': asdict(self.state),
            'rng_state': torch.get_rng_state(),
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint-{self.state.global_step}.pt"
            
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if self.config.save_total_limit <= 0:
            return
            
        checkpoint_dir = Path(self.config.output_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint-*.pt"))
        
        if len(checkpoints) > self.config.save_total_limit:
            # Sort by step number and remove oldest
            checkpoints.sort(key=lambda x: int(x.stem.split('-')[1]))
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
                
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        if not HAS_TORCH:
            return
            
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load scaler state
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        # Load training state
        state_dict = checkpoint.get('state', {})
        for key, value in state_dict.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
                
        # Load random state
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
            
        logger.info(f"Checkpoint loaded from step {self.state.global_step}")


def create_trainer(model: nn.Module,
                  train_dataloader: DataLoader,
                  eval_dataloader: Optional[DataLoader] = None,
                  config: Optional[TrainingConfig] = None) -> GlycoLLMTrainer:
    """
    Create a GlycoLLM trainer with default configuration.
    
    Args:
        model: Model to train
        train_dataloader: Training data
        eval_dataloader: Evaluation data
        config: Training configuration
        
    Returns:
        Configured trainer
    """
    if config is None:
        config = TrainingConfig()
        
    return GlycoLLMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config
    )


def train_glycollm(model: nn.Module,
                   train_dataloader: DataLoader,
                   eval_dataloader: Optional[DataLoader] = None,
                   **training_kwargs) -> TrainingState:
    """
    Convenience function to train GlycoLLM model.
    
    Args:
        model: Model to train
        train_dataloader: Training data
        eval_dataloader: Evaluation data
        **training_kwargs: Training configuration parameters
        
    Returns:
        Final training state
    """
    # Create config from kwargs
    config_dict = {
        'num_epochs': training_kwargs.get('num_epochs', 10),
        'batch_size': training_kwargs.get('batch_size', 8),
        'learning_rate': training_kwargs.get('learning_rate', 5e-5),
        'output_dir': training_kwargs.get('output_dir', './checkpoints'),
        **training_kwargs
    }
    
    config = TrainingConfig(**config_dict)
    
    # Create and run trainer
    trainer = create_trainer(model, train_dataloader, eval_dataloader, config)
    return trainer.train()