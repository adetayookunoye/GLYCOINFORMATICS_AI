"""
Training initialization module for GlycoLLM.

This module provides __init__.py for the training package and
convenience imports for training components.
"""

from typing import Optional, Dict, Any

# Import availability flags
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Core trainer (always available)
from .trainer import GlycoLLMTrainer, TrainingConfig, TrainingState

# Evaluation components
from .evaluation import (
    GlycoLLMEvaluator, 
    EvaluationResults,
    StructureEvaluator,
    SpectraEvaluator, 
    TextEvaluator,
    CrossModalEvaluator,
    create_evaluator
)

# Curriculum learning
from .curriculum import (
    CurriculumManager,
    CurriculumStage,
    DifficultyLevel,
    StructureComplexityAnalyzer,
    SpectraComplexityAnalyzer,
    create_curriculum_manager
)

# Contrastive learning
from .contrastive import (
    ContrastiveLearningManager,
    NegativeSampler,
    RandomNegativeSampler,
    HardNegativeSampler,
    StructuralNegativeSampler,
    create_contrastive_manager,
    create_contrastive_loss
)

# Training utilities
from .utils import (
    TrainingMetrics,
    CheckpointManager,
    MetricsLogger,
    GradientClipping,
    create_scheduler,
    get_gpu_memory_info,
    format_time
)

# Optional PyTorch-dependent imports
if HAS_TORCH:
    from .contrastive import MultiModalContrastiveLoss, AdaptiveContrastiveLoss
    from .utils import WarmupCosineScheduler, PolynomialDecayScheduler


__all__ = [
    # Main trainer
    'GlycoLLMTrainer',
    'TrainingConfig', 
    'TrainingState',
    
    # Evaluation
    'GlycoLLMEvaluator',
    'EvaluationResults',
    'StructureEvaluator',
    'SpectraEvaluator',
    'TextEvaluator', 
    'CrossModalEvaluator',
    'create_evaluator',
    
    # Curriculum learning
    'CurriculumManager',
    'CurriculumStage',
    'DifficultyLevel',
    'StructureComplexityAnalyzer',
    'SpectraComplexityAnalyzer',
    'create_curriculum_manager',
    
    # Contrastive learning
    'ContrastiveLearningManager',
    'NegativeSampler',
    'RandomNegativeSampler', 
    'HardNegativeSampler',
    'StructuralNegativeSampler',
    'create_contrastive_manager',
    'create_contrastive_loss',
    
    # Utilities
    'TrainingMetrics',
    'CheckpointManager',
    'MetricsLogger',
    'GradientClipping',
    'create_scheduler',
    'get_gpu_memory_info',
    'format_time',
]

# Add PyTorch-dependent exports
if HAS_TORCH:
    __all__.extend([
        'MultiModalContrastiveLoss',
        'AdaptiveContrastiveLoss',
        'WarmupCosineScheduler',
        'PolynomialDecayScheduler'
    ])


def create_training_pipeline(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a complete training pipeline with all components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing all training components
    """
    
    if config is None:
        config = {}
        
    # Create trainer configuration
    trainer_config = TrainingConfig(**config.get('trainer', {}))
    
    # Create evaluator
    evaluator = create_evaluator()
    
    # Create curriculum manager
    curriculum_config = config.get('curriculum', {})
    curriculum_manager = create_curriculum_manager(curriculum_config)
    
    # Create contrastive learning manager
    contrastive_config = config.get('contrastive', {})
    contrastive_manager = create_contrastive_manager(contrastive_config)
    contrastive_loss = create_contrastive_loss(contrastive_config)
    
    # Create checkpoint manager
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_config.get('checkpoint_dir', './checkpoints'),
        max_checkpoints=checkpoint_config.get('max_checkpoints', 5),
        save_best_only=checkpoint_config.get('save_best_only', False),
        monitor_metric=checkpoint_config.get('monitor_metric', 'total_loss'),
        mode=checkpoint_config.get('mode', 'min')
    )
    
    # Create metrics logger
    logging_config = config.get('logging', {})
    metrics_logger = MetricsLogger(
        log_dir=logging_config.get('log_dir', './logs'),
        console_logging=logging_config.get('console_logging', True),
        file_logging=logging_config.get('file_logging', True),
        tensorboard_logging=logging_config.get('tensorboard_logging', False)
    )
    
    return {
        'trainer_config': trainer_config,
        'evaluator': evaluator,
        'curriculum_manager': curriculum_manager,
        'contrastive_manager': contrastive_manager,
        'contrastive_loss': contrastive_loss,
        'checkpoint_manager': checkpoint_manager,
        'metrics_logger': metrics_logger,
        'has_torch': HAS_TORCH
    }


def get_training_info() -> Dict[str, Any]:
    """
    Get information about training capabilities and dependencies.
    
    Returns:
        Dictionary with training system information
    """
    
    info = {
        'torch_available': HAS_TORCH,
        'training_components': {
            'trainer': True,
            'evaluator': True,
            'curriculum': True,
            'contrastive': True,
            'checkpoints': True,
            'metrics_logging': True
        },
        'supported_features': [
            'Multi-modal training',
            'Curriculum learning', 
            'Contrastive learning',
            'Evaluation metrics',
            'Checkpoint management',
            'Metrics logging'
        ]
    }
    
    if HAS_TORCH:
        info['pytorch_features'] = [
            'GPU training',
            'Distributed training',
            'Mixed precision',
            'Gradient accumulation',
            'Learning rate scheduling',
            'TensorBoard logging'
        ]
    else:
        info['limitations'] = [
            'PyTorch not available - some features disabled',
            'No GPU training support',
            'No actual model training (development mode)'
        ]
        
    return info