"""
Models module for GlycoLLM.

This module provides the complete implementation of GlycoLLM,
a multimodal large language model for glycoinformatics.
"""

from .glycollm import (
    GlycoLLM,
    GlycoLLMConfig,
    GlycoLLMEncoder,
    GlycoLLMTaskHeads,
    MultiModalEmbedding,
    PositionalEncoding,
    CrossModalAttention,
    GlycoTransformerLayer,
    create_glycollm_model
)

from .loss_functions import (
    MultiModalLoss,
    ContrastiveLoss,
    StructurePredictionLoss,
    SpectraPredictionLoss,
    TripletLoss,
    CosineSimilarityLoss,
    LossConfig,
    LossType,
    create_loss_function,
    compute_accuracy,
    compute_perplexity
)

from .model_utils import (
    ModelAnalyzer,
    GradientAnalyzer,
    ModelProfiler,
    ModelStats,
    save_model_checkpoint,
    load_model_checkpoint,
    count_parameters,
    freeze_layers,
    unfreeze_layers,
    get_model_device,
    move_model_to_device,
    initialize_weights
)

from .llm_finetuning import (
    GlycoLLMWithFineTuning, 
    GlycoLLMFineTuner,
    LLMFineTuningConfig,
    LLMType,
    FineTuningMethod,
    GlycanDatasetForLLM,
    create_glycomics_training_data,
    load_llm_fine_tuning_config
)

__all__ = [
    # Main model classes
    'GlycoLLM',
    'GlycoLLMConfig',
    'GlycoLLMEncoder',
    'GlycoLLMTaskHeads',
    
    # Architecture components
    'MultiModalEmbedding',
    'PositionalEncoding',
    'CrossModalAttention',
    'GlycoTransformerLayer',
    
    # Loss functions
    'MultiModalLoss',
    'ContrastiveLoss',
    'StructurePredictionLoss',
    'SpectraPredictionLoss',
    'TripletLoss',
    'CosineSimilarityLoss',
    'LossConfig',
    'LossType',
    
    # Utilities and analysis
    'ModelAnalyzer',
    'GradientAnalyzer', 
    'ModelProfiler',
    'ModelStats',
    
    # LLM Fine-tuning
    'GlycoLLMWithFineTuning',
    'GlycoLLMFineTuner',
    'LLMFineTuningConfig',
    'LLMType',
    'FineTuningMethod',
    'GlycanDatasetForLLM',
    
    # Factory functions
    'create_glycollm_model',
    'create_loss_function',
    'create_glycomics_training_data',
    'load_llm_fine_tuning_config',
    
    # Utility functions
    'save_model_checkpoint',
    'load_model_checkpoint',
    'count_parameters',
    'freeze_layers',
    'unfreeze_layers',
    'get_model_device',
    'move_model_to_device',
    'initialize_weights',
    'compute_accuracy',
    'compute_perplexity'
]

# Version information
__version__ = "0.1.0"
__author__ = "GlycoInformatics AI Team"
__description__ = "Multimodal transformer architecture for glycoinformatics"