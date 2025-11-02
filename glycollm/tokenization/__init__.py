"""
Tokenization module for glycoinformatics data.

This module provides specialized tokenizers for WURCS notation, mass spectra,
and glycan-related text for multimodal AI model training.
"""

from .glyco_tokenizer import (
    GlycoTokenizer,
    TokenizationConfig,
    WURCSTokenizer, 
    SpectraTokenizer,
    GlycanTextTokenizer
)

from .tokenizer_utils import (
    WURCSValidator,
    SpectrumAnalyzer,
    GlycanTextProcessor,
    TokenizerTester,
    WURCSParseResult,
    SpectrumAnalysis,
    load_sample_data,
    create_sample_data_files
)

from .tokenizer_training import (
    TokenizerTrainer,
    TrainingConfig,
    TrainingStats,
    train_glyco_tokenizer,
    create_demo_training_data
)

__all__ = [
    # Main tokenizer classes
    'GlycoTokenizer',
    'TokenizationConfig',
    'WURCSTokenizer',
    'SpectraTokenizer', 
    'GlycanTextTokenizer',
    
    # Utility classes
    'WURCSValidator',
    'SpectrumAnalyzer',
    'GlycanTextProcessor',
    'TokenizerTester',
    
    # Data classes
    'WURCSParseResult',
    'SpectrumAnalysis',
    'TrainingConfig',
    'TrainingStats',
    
    # Training functions
    'TokenizerTrainer',
    'train_glyco_tokenizer',
    
    # Utility functions
    'load_sample_data',
    'create_sample_data_files',
    'create_demo_training_data'
]

# Version information
__version__ = "0.1.0"
__author__ = "GlycoInformatics AI Team"
__description__ = "Specialized tokenization for glycoinformatics multimodal data"