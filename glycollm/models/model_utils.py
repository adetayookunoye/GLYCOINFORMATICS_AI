"""
Model utilities and helper functions for GlycoLLM.

This module provides utilities for model initialization, parameter counting,
gradient analysis, and other model management functions.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict

# Optional imports
try:
    import torch
    import torch.nn as nn
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """Statistics about a model"""
    total_params: int
    trainable_params: int
    non_trainable_params: int
    model_size_mb: float
    memory_usage_mb: float
    
    # Layer-wise statistics
    embedding_params: int = 0
    encoder_params: int = 0
    task_head_params: int = 0
    
    # Parameter distribution
    linear_params: int = 0
    embedding_params_count: int = 0
    layernorm_params: int = 0
    other_params: int = 0


class ModelAnalyzer:
    """
    Analyzer for GlycoLLM model architecture and parameters.
    """
    
    def __init__(self):
        self.model = None
        self.stats = None
        
    def analyze_model(self, model: nn.Module) -> ModelStats:
        """
        Analyze model architecture and compute statistics.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Model statistics
        """
        if not HAS_TORCH:
            return ModelStats(0, 0, 0, 0.0, 0.0)
            
        self.model = model
        
        # Count parameters
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        # Layer-wise counts
        embedding_params = 0
        encoder_params = 0
        task_head_params = 0
        
        # Parameter type counts
        linear_params = 0
        embedding_params_count = 0
        layernorm_params = 0
        other_params = 0
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            else:
                non_trainable_params += param_count
                
            # Categorize by layer type
            if 'embedding' in name.lower():
                embedding_params += param_count
                embedding_params_count += param_count
            elif 'encoder' in name.lower() or 'layer' in name.lower():
                encoder_params += param_count
            elif 'head' in name.lower() or 'projection' in name.lower():
                task_head_params += param_count
                
            # Categorize by parameter type
            if 'weight' in name and len(param.shape) == 2:
                linear_params += param_count
            elif 'embedding' in name.lower():
                pass  # Already counted above
            elif 'norm' in name.lower():
                layernorm_params += param_count
            else:
                other_params += param_count
        
        # Estimate memory usage (rough approximation)
        model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
        
        # Estimate runtime memory (activations, gradients, etc.)
        # This is a rough estimate - actual usage depends on batch size
        memory_usage_mb = model_size_mb * 3  # Model + gradients + activations
        
        self.stats = ModelStats(
            total_params=total_params,
            trainable_params=trainable_params,
            non_trainable_params=non_trainable_params,
            model_size_mb=model_size_mb,
            memory_usage_mb=memory_usage_mb,
            embedding_params=embedding_params,
            encoder_params=encoder_params,
            task_head_params=task_head_params,
            linear_params=linear_params,
            embedding_params_count=embedding_params_count,
            layernorm_params=layernorm_params,
            other_params=other_params
        )
        
        return self.stats
        
    def print_model_summary(self, model: Optional[nn.Module] = None):
        """
        Print detailed model summary.
        
        Args:
            model: Model to analyze (uses stored model if None)
        """
        if model is not None:
            self.analyze_model(model)
        elif self.stats is None:
            print("No model analyzed yet. Please provide a model.")
            return
            
        stats = self.stats
        
        print("\n" + "="*60)
        print("GlycoLLM Model Summary")
        print("="*60)
        
        print(f"\n📊 Parameter Statistics:")
        print(f"   Total Parameters:      {stats.total_params:,}")
        print(f"   Trainable Parameters:  {stats.trainable_params:,}")
        print(f"   Non-trainable:         {stats.non_trainable_params:,}")
        
        print(f"\n🏗️  Architecture Breakdown:")
        print(f"   Embedding Layer:       {stats.embedding_params:,} params")
        print(f"   Encoder Layers:        {stats.encoder_params:,} params")
        print(f"   Task Heads:           {stats.task_head_params:,} params")
        
        print(f"\n🔧 Parameter Types:")
        print(f"   Linear Weights:        {stats.linear_params:,}")
        print(f"   Embedding Tables:      {stats.embedding_params_count:,}")
        print(f"   Layer Normalization:   {stats.layernorm_params:,}")
        print(f"   Other Parameters:      {stats.other_params:,}")
        
        print(f"\n💾 Memory Requirements:")
        print(f"   Model Size:           {stats.model_size_mb:.1f} MB")
        print(f"   Estimated Runtime:    {stats.memory_usage_mb:.1f} MB")
        
        # Calculate percentages
        if stats.total_params > 0:
            emb_pct = (stats.embedding_params / stats.total_params) * 100
            enc_pct = (stats.encoder_params / stats.total_params) * 100  
            head_pct = (stats.task_head_params / stats.total_params) * 100
            
            print(f"\n📈 Distribution:")
            print(f"   Embeddings:           {emb_pct:.1f}%")
            print(f"   Encoder:              {enc_pct:.1f}%") 
            print(f"   Task Heads:           {head_pct:.1f}%")
        
        print("\n" + "="*60)
        
    def get_layer_wise_params(self, model: nn.Module) -> Dict[str, int]:
        """
        Get parameter count for each layer.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary mapping layer names to parameter counts
        """
        if not HAS_TORCH:
            return {}
            
        layer_params = {}
        
        for name, param in model.named_parameters():
            layer_name = name.split('.')[0]  # Top-level module name
            
            if layer_name not in layer_params:
                layer_params[layer_name] = 0
                
            layer_params[layer_name] += param.numel()
            
        return layer_params


class GradientAnalyzer:
    """
    Analyzer for gradient flow and optimization diagnostics.
    """
    
    def __init__(self):
        self.gradient_history = []
        
    def analyze_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze gradient statistics for model parameters.
        
        Args:
            model: Model with computed gradients
            
        Returns:
            Gradient analysis results
        """
        if not HAS_TORCH:
            return {}
            
        gradient_stats = {
            'layer_stats': {},
            'global_stats': {},
            'problematic_layers': []
        }
        
        all_gradients = []
        zero_gradient_layers = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_max = param.grad.max().item()
                grad_min = param.grad.min().item()
                
                gradient_stats['layer_stats'][name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'max': grad_max,
                    'min': grad_min,
                    'param_count': param.numel()
                }
                
                all_gradients.append(grad_norm)
                
                # Check for problematic gradients
                if grad_norm < 1e-8:
                    zero_gradient_layers.append(name)
                elif grad_norm > 10.0:
                    gradient_stats['problematic_layers'].append({
                        'layer': name,
                        'issue': 'large_gradient',
                        'norm': grad_norm
                    })
                    
            else:
                gradient_stats['layer_stats'][name] = {
                    'norm': 0.0,
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0.0,
                    'min': 0.0,
                    'param_count': param.numel()
                }
                zero_gradient_layers.append(name)
        
        # Global gradient statistics
        if all_gradients:
            gradient_stats['global_stats'] = {
                'total_norm': sum(all_gradients),
                'mean_norm': np.mean(all_gradients),
                'std_norm': np.std(all_gradients),
                'max_norm': max(all_gradients),
                'min_norm': min(all_gradients),
                'zero_gradient_layers': zero_gradient_layers
            }
        
        self.gradient_history.append(gradient_stats)
        return gradient_stats
        
    def print_gradient_summary(self, gradient_stats: Dict[str, Any]):
        """Print summary of gradient analysis."""
        
        print("\n" + "="*50)
        print("Gradient Analysis Summary")
        print("="*50)
        
        global_stats = gradient_stats.get('global_stats', {})
        
        if global_stats:
            print(f"\n🌐 Global Statistics:")
            print(f"   Total Gradient Norm:   {global_stats.get('total_norm', 0):.6f}")
            print(f"   Mean Layer Norm:       {global_stats.get('mean_norm', 0):.6f}")
            print(f"   Max Layer Norm:        {global_stats.get('max_norm', 0):.6f}")
            print(f"   Min Layer Norm:        {global_stats.get('min_norm', 0):.6f}")
            
        # Problematic layers
        problematic = gradient_stats.get('problematic_layers', [])
        if problematic:
            print(f"\n⚠️  Problematic Layers ({len(problematic)}):")
            for issue in problematic[:5]:  # Show first 5
                print(f"   {issue['layer']}: {issue['issue']} (norm: {issue['norm']:.6f})")
                
        # Zero gradient layers
        zero_grad = global_stats.get('zero_gradient_layers', [])
        if zero_grad:
            print(f"\n❄️  Layers with Zero Gradients ({len(zero_grad)}):")
            for layer in zero_grad[:5]:  # Show first 5
                print(f"   {layer}")
                
        print("\n" + "="*50)


def save_model_checkpoint(model: nn.Module,
                         optimizer,
                         epoch: int,
                         loss: float,
                         save_path: str,
                         config: Optional[Dict] = None):
    """
    Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        save_path: Path to save checkpoint
        config: Model configuration
    """
    if not HAS_TORCH:
        logger.warning("PyTorch not available - cannot save checkpoint")
        return
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_model_checkpoint(model: nn.Module,
                         optimizer,
                         checkpoint_path: str) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint metadata
    """
    if not HAS_TORCH:
        logger.warning("PyTorch not available - cannot load checkpoint")
        return {}
        
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
        'config': checkpoint.get('config', {})
    }


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    if not HAS_TORCH:
        return 0, 0
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Freeze specified layers in the model.
    
    Args:
        model: Model to modify
        layer_names: Names of layers to freeze
    """
    if not HAS_TORCH:
        return
        
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")
                break


def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Unfreeze specified layers in the model.
    
    Args:
        model: Model to modify
        layer_names: Names of layers to unfreeze
    """
    if not HAS_TORCH:
        return
        
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")
                break


def get_model_device(model: nn.Module) -> str:
    """
    Get the device where model parameters are located.
    
    Args:
        model: PyTorch model
        
    Returns:
        Device string ('cpu', 'cuda:0', etc.)
    """
    if not HAS_TORCH:
        return "cpu"
        
    return next(model.parameters()).device.type


def move_model_to_device(model: nn.Module, device: str) -> nn.Module:
    """
    Move model to specified device.
    
    Args:
        model: Model to move
        device: Target device
        
    Returns:
        Model on target device
    """
    if not HAS_TORCH:
        return model
        
    return model.to(device)


def initialize_weights(model: nn.Module, 
                      init_type: str = "normal",
                      init_std: float = 0.02):
    """
    Initialize model weights with specified strategy.
    
    Args:
        model: Model to initialize
        init_type: Initialization type ("normal", "xavier", "kaiming")
        init_std: Standard deviation for normal initialization
    """
    if not HAS_TORCH:
        return
        
    def init_func(m):
        if isinstance(m, nn.Linear):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_std)
            elif init_type == "xavier":
                nn.init.xavier_uniform_(m.weight.data)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(m.weight.data)
                
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight.data, 0.0, init_std)
            
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias.data, 0.0)
            nn.init.constant_(m.weight.data, 1.0)
    
    model.apply(init_func)
    logger.info(f"Model weights initialized with {init_type} strategy")


class ModelProfiler:
    """
    Profiler for model performance analysis.
    """
    
    def __init__(self):
        self.timing_results = {}
        self.memory_results = {}
        
    def profile_forward_pass(self, 
                           model: nn.Module,
                           sample_input: Dict[str, torch.Tensor],
                           num_runs: int = 10) -> Dict[str, float]:
        """
        Profile model forward pass timing.
        
        Args:
            model: Model to profile
            sample_input: Sample input batch
            num_runs: Number of runs for averaging
            
        Returns:
            Timing statistics
        """
        if not HAS_TORCH:
            return {}
            
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(3):
                _ = model(**sample_input)
        
        # Timing runs
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(**sample_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                times.append(end_time - start_time)
        
        timing_stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': min(times),
            'max_time': max(times),
            'num_runs': num_runs
        }
        
        self.timing_results = timing_stats
        return timing_stats
        
    def print_profiling_results(self):
        """Print profiling results summary."""
        
        print("\n" + "="*40)
        print("Model Profiling Results")
        print("="*40)
        
        if self.timing_results:
            print(f"\n⏱️  Timing Statistics:")
            print(f"   Mean Forward Time:  {self.timing_results['mean_time']*1000:.2f} ms")
            print(f"   Std Deviation:      {self.timing_results['std_time']*1000:.2f} ms")
            print(f"   Min Time:           {self.timing_results['min_time']*1000:.2f} ms")
            print(f"   Max Time:           {self.timing_results['max_time']*1000:.2f} ms")
            print(f"   Number of Runs:     {self.timing_results['num_runs']}")
            
        print("\n" + "="*40)