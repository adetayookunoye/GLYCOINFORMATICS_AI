"""
Training utilities for the glycoinformatics tokenizer.

This module provides tools for training and evaluating the specialized
tokenizer on glycan-related data.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import random

# Optional imports that will be available when packages are installed
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    from tokenizers.implementations import ByteLevelBPETokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

from .glyco_tokenizer import TokenizationConfig
from .tokenizer_utils import (
    WURCSValidator, SpectrumAnalyzer, GlycanTextProcessor,
    load_sample_data, create_sample_data_files
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for tokenizer training"""
    
    # Data settings
    min_frequency: int = 2
    vocab_size: int = 50000
    
    # Training settings
    show_progress: bool = True
    
    # BPE settings
    dropout: Optional[float] = None
    continuing_subword_prefix: str = "##"
    end_of_word_suffix: str = ""
    
    # Special tokens training
    special_tokens_ratio: float = 0.1  # Ratio of vocab reserved for special tokens
    
    # Validation settings
    validation_split: float = 0.1
    seed: int = 42


@dataclass
class TrainingStats:
    """Statistics from tokenizer training"""
    total_texts: int
    total_tokens_before: int
    total_tokens_after: int
    vocab_coverage: float
    oov_rate: float
    compression_ratio: float
    
    # Modality-specific stats
    wurcs_stats: Dict[str, any]
    spectra_stats: Dict[str, any]
    text_stats: Dict[str, any]


class TokenizerTrainer:
    """
    Trainer for the glycoinformatics tokenizer.
    
    Handles corpus preparation, vocabulary building, and tokenizer training
    for multimodal glycan data.
    """
    
    def __init__(self, 
                 tokenization_config: Optional[TokenizationConfig] = None,
                 training_config: Optional[TrainingConfig] = None):
        """
        Initialize the tokenizer trainer.
        
        Args:
            tokenization_config: Tokenizer configuration
            training_config: Training configuration
        """
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Initialize utility classes
        self.wurcs_validator = WURCSValidator()
        self.spectrum_analyzer = SpectrumAnalyzer()
        self.text_processor = GlycanTextProcessor()
        
        # Training data storage
        self.training_corpus = []
        self.validation_corpus = []
        
        # Statistics
        self.training_stats = None
        
        # Set random seed
        random.seed(self.training_config.seed)
        
    def prepare_training_data(self, 
                            text_samples: List[str],
                            wurcs_samples: Optional[List[str]] = None,
                            spectra_samples: Optional[List[List[Tuple[float, float]]]] = None) -> Tuple[List[str], List[str]]:
        """
        Prepare training corpus from multimodal data.
        
        Args:
            text_samples: List of scientific text samples
            wurcs_samples: List of WURCS notation strings
            spectra_samples: List of mass spectra (peak lists)
            
        Returns:
            Tuple of (training_corpus, validation_corpus)
        """
        logger.info("Preparing training corpus from multimodal data...")
        
        corpus = []
        
        # Process text samples
        if text_samples:
            logger.info(f"Processing {len(text_samples)} text samples...")
            for text in text_samples:
                if text and text.strip():
                    processed_text = self.text_processor.preprocess_text(text)
                    if processed_text:
                        corpus.append(processed_text)
                        
        # Process WURCS samples
        if wurcs_samples:
            logger.info(f"Processing {len(wurcs_samples)} WURCS sequences...")
            for wurcs in wurcs_samples:
                if wurcs and wurcs.strip():
                    # Validate and extract components
                    parse_result = self.wurcs_validator.validate_wurcs(wurcs)
                    if parse_result.is_valid:
                        # Create text representation of WURCS components
                        wurcs_text = self._wurcs_to_text(parse_result)
                        if wurcs_text:
                            corpus.append(wurcs_text)
                    else:
                        logger.warning(f"Invalid WURCS skipped: {wurcs}")
                        
        # Process spectra samples
        if spectra_samples:
            logger.info(f"Processing {len(spectra_samples)} mass spectra...")
            for peaks in spectra_samples:
                if peaks:
                    # Convert spectrum to text representation
                    spectrum_text = self._spectrum_to_text(peaks)
                    if spectrum_text:
                        corpus.append(spectrum_text)
                        
        # Shuffle corpus
        random.shuffle(corpus)
        
        # Split into training and validation
        split_idx = int(len(corpus) * (1 - self.training_config.validation_split))
        training_corpus = corpus[:split_idx]
        validation_corpus = corpus[split_idx:]
        
        self.training_corpus = training_corpus
        self.validation_corpus = validation_corpus
        
        logger.info(f"Prepared corpus: {len(training_corpus)} training, {len(validation_corpus)} validation samples")
        
        return training_corpus, validation_corpus
        
    def _wurcs_to_text(self, parse_result) -> str:
        """Convert WURCS parse result to text representation"""
        components = []
        
        # Add version info
        components.append(f"wurcs_version_{parse_result.version.replace('.', '_')}")
        
        # Add counts
        components.append(f"residue_count_{parse_result.residue_count}")
        components.append(f"linkage_count_{parse_result.linkage_count}")
        
        # Add monosaccharide composition
        mono_counts = self.wurcs_validator.identify_monosaccharides(parse_result.residues)
        for mono_type, count in mono_counts.items():
            components.append(f"mono_{mono_type.lower()}_{count}")
            
        # Add connection patterns
        for connection in parse_result.connections:
            if connection:
                # Simplify connection notation
                simplified = connection.replace('-', '_').replace('/', '_')
                components.append(f"connection_{simplified}")
                
        return " ".join(components)
        
    def _spectrum_to_text(self, peaks: List[Tuple[float, float]]) -> str:
        """Convert spectrum to text representation"""
        if not peaks:
            return ""
            
        # Analyze spectrum
        analysis = self.spectrum_analyzer.analyze_spectrum(peaks)
        
        components = []
        
        # Add basic spectrum info
        components.append(f"spectrum_peaks_{analysis.total_peaks}")
        components.append(f"base_peak_mz_{int(analysis.base_peak_mz)}")
        
        # Add mass range info
        mass_range = analysis.mass_range
        components.append(f"mass_range_{int(mass_range[0])}_{int(mass_range[1])}")
        
        # Add identified fragments
        for fragment in analysis.identified_fragments:
            fragment_type = fragment.split('@')[0]
            components.append(f"fragment_{fragment_type.lower()}")
            
        # Add neutral losses
        for loss in analysis.neutral_losses:
            loss_type = loss.split('@')[0].replace('-', '')
            components.append(f"loss_{loss_type.lower()}")
            
        # Add signal quality indicator
        if analysis.signal_to_noise > 10:
            components.append("high_quality_spectrum")
        elif analysis.signal_to_noise > 3:
            components.append("medium_quality_spectrum")
        else:
            components.append("low_quality_spectrum")
            
        return " ".join(components)
        
    def build_vocabulary(self, corpus: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Build vocabulary from corpus with frequency counting.
        
        Args:
            corpus: Text corpus (uses training corpus if None)
            
        Returns:
            Vocabulary dictionary (token -> frequency)
        """
        if corpus is None:
            corpus = self.training_corpus
            
        if not corpus:
            raise ValueError("No corpus available for vocabulary building")
            
        logger.info("Building vocabulary from corpus...")
        
        # Count token frequencies
        token_counts = Counter()
        
        for text in corpus:
            tokens = text.split()
            token_counts.update(tokens)
            
        # Filter by minimum frequency
        filtered_vocab = {
            token: count for token, count in token_counts.items()
            if count >= self.training_config.min_frequency
        }
        
        logger.info(f"Built vocabulary: {len(filtered_vocab)} tokens (min_freq={self.training_config.min_frequency})")
        
        return filtered_vocab
        
    def train_tokenizer(self, 
                       corpus: Optional[List[str]] = None,
                       vocab_dict: Optional[Dict[str, int]] = None) -> 'Tokenizer':
        """
        Train the base tokenizer using BPE.
        
        Args:
            corpus: Training corpus
            vocab_dict: Pre-built vocabulary
            
        Returns:
            Trained tokenizer
        """
        if not HAS_TOKENIZERS:
            raise ImportError("tokenizers library not installed. Run: pip install tokenizers")
            
        if corpus is None:
            corpus = self.training_corpus
            
        if not corpus:
            raise ValueError("No corpus available for training")
            
        logger.info("Training BPE tokenizer...")
        
        # Create BPE tokenizer
        tokenizer = ByteLevelBPETokenizer()
        
        # Prepare special tokens
        special_tokens = self.tokenization_config.get_all_special_tokens()
        
        # Create temporary corpus file for training
        corpus_file = Path("temp_training_corpus.txt")
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for text in corpus:
                f.write(text + '\n')
                
        try:
            # Train the tokenizer
            tokenizer.train(
                files=[str(corpus_file)],
                vocab_size=self.training_config.vocab_size,
                min_frequency=self.training_config.min_frequency,
                special_tokens=special_tokens,
                show_progress=self.training_config.show_progress
            )
            
            # Set special tokens
            tokenizer.add_special_tokens(special_tokens)
            
            logger.info(f"Tokenizer trained with vocabulary size: {tokenizer.get_vocab_size()}")
            
        finally:
            # Clean up temporary file
            if corpus_file.exists():
                corpus_file.unlink()
                
        return tokenizer
        
    def evaluate_tokenizer(self, tokenizer: 'Tokenizer', 
                         test_corpus: Optional[List[str]] = None) -> TrainingStats:
        """
        Evaluate trained tokenizer performance.
        
        Args:
            tokenizer: Trained tokenizer to evaluate
            test_corpus: Test corpus (uses validation corpus if None)
            
        Returns:
            Evaluation statistics
        """
        if not HAS_TOKENIZERS:
            raise ImportError("tokenizers library not installed")
            
        if test_corpus is None:
            test_corpus = self.validation_corpus
            
        if not test_corpus:
            raise ValueError("No test corpus available for evaluation")
            
        logger.info("Evaluating tokenizer performance...")
        
        total_texts = len(test_corpus)
        total_chars_before = sum(len(text) for text in test_corpus)
        total_tokens_after = 0
        oov_count = 0
        total_tokens_before = sum(len(text.split()) for text in test_corpus)
        
        # Get vocabulary
        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)
        
        # Process each text
        for text in test_corpus:
            encoded = tokenizer.encode(text)
            tokens = encoded.tokens
            
            total_tokens_after += len(tokens)
            
            # Count OOV tokens (assuming [UNK] token exists)
            unk_token = self.tokenization_config.unk_token
            if unk_token in vocab:
                unk_id = vocab[unk_token]
                oov_count += sum(1 for token_id in encoded.ids if token_id == unk_id)
                
        # Calculate statistics
        compression_ratio = total_chars_before / total_tokens_after if total_tokens_after > 0 else 0
        oov_rate = oov_count / total_tokens_after if total_tokens_after > 0 else 0
        vocab_coverage = (total_tokens_after - oov_count) / total_tokens_after if total_tokens_after > 0 else 0
        
        # Create statistics object
        stats = TrainingStats(
            total_texts=total_texts,
            total_tokens_before=total_tokens_before,
            total_tokens_after=total_tokens_after,
            vocab_coverage=vocab_coverage,
            oov_rate=oov_rate,
            compression_ratio=compression_ratio,
            wurcs_stats={},  # Placeholder for detailed stats
            spectra_stats={},
            text_stats={}
        )
        
        self.training_stats = stats
        
        logger.info(f"Tokenizer evaluation complete:")
        logger.info(f"  - Vocabulary coverage: {vocab_coverage:.3f}")
        logger.info(f"  - OOV rate: {oov_rate:.3f}")
        logger.info(f"  - Compression ratio: {compression_ratio:.2f}")
        
        return stats
        
    def save_training_artifacts(self, 
                              tokenizer: 'Tokenizer',
                              output_dir: str,
                              save_corpus: bool = True):
        """
        Save training artifacts for future use.
        
        Args:
            tokenizer: Trained tokenizer
            output_dir: Directory to save artifacts
            save_corpus: Whether to save training corpus
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        tokenizer_path = output_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        
        # Save configurations
        tokenization_config_path = output_path / "tokenization_config.json"
        with open(tokenization_config_path, 'w') as f:
            json.dump(asdict(self.tokenization_config), f, indent=2)
            
        training_config_path = output_path / "training_config.json"
        with open(training_config_path, 'w') as f:
            json.dump(asdict(self.training_config), f, indent=2)
            
        # Save statistics if available
        if self.training_stats:
            stats_path = output_path / "training_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(asdict(self.training_stats), f, indent=2)
                
        # Save corpus if requested
        if save_corpus:
            if self.training_corpus:
                corpus_path = output_path / "training_corpus.json"
                with open(corpus_path, 'w') as f:
                    json.dump(self.training_corpus, f, indent=2)
                    
            if self.validation_corpus:
                val_corpus_path = output_path / "validation_corpus.json"
                with open(val_corpus_path, 'w') as f:
                    json.dump(self.validation_corpus, f, indent=2)
                    
        logger.info(f"Training artifacts saved to {output_dir}")
        
    @classmethod
    def from_saved_artifacts(cls, model_dir: str) -> 'TokenizerTrainer':
        """
        Load trainer from saved artifacts.
        
        Args:
            model_dir: Directory containing saved artifacts
            
        Returns:
            Configured trainer instance
        """
        model_path = Path(model_dir)
        
        # Load configurations
        tokenization_config_path = model_path / "tokenization_config.json"
        training_config_path = model_path / "training_config.json"
        
        tokenization_config = None
        training_config = None
        
        if tokenization_config_path.exists():
            with open(tokenization_config_path, 'r') as f:
                config_dict = json.load(f)
                tokenization_config = TokenizationConfig(**config_dict)
                
        if training_config_path.exists():
            with open(training_config_path, 'r') as f:
                config_dict = json.load(f)
                training_config = TrainingConfig(**config_dict)
                
        # Create trainer
        trainer = cls(tokenization_config, training_config)
        
        # Load corpus if available
        corpus_path = model_path / "training_corpus.json"
        val_corpus_path = model_path / "validation_corpus.json"
        
        if corpus_path.exists():
            with open(corpus_path, 'r') as f:
                trainer.training_corpus = json.load(f)
                
        if val_corpus_path.exists():
            with open(val_corpus_path, 'r') as f:
                trainer.validation_corpus = json.load(f)
                
        # Load statistics if available
        stats_path = model_path / "training_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats_dict = json.load(f)
                trainer.training_stats = TrainingStats(**stats_dict)
                
        logger.info(f"Trainer loaded from {model_dir}")
        return trainer


def train_glyco_tokenizer(
    text_data: List[str],
    wurcs_data: Optional[List[str]] = None,
    spectra_data: Optional[List[List[Tuple[float, float]]]] = None,
    output_dir: str = "trained_tokenizer",
    vocab_size: int = 50000,
    min_frequency: int = 2
) -> 'TokenizerTrainer':
    """
    Convenience function to train a glycoinformatics tokenizer.
    
    Args:
        text_data: List of text samples
        wurcs_data: List of WURCS sequences
        spectra_data: List of mass spectra
        output_dir: Directory to save trained tokenizer
        vocab_size: Target vocabulary size
        min_frequency: Minimum token frequency
        
    Returns:
        Trained tokenizer trainer
    """
    # Configure tokenizer
    tokenization_config = TokenizationConfig()
    
    training_config = TrainingConfig(
        vocab_size=vocab_size,
        min_frequency=min_frequency
    )
    
    # Create trainer
    trainer = TokenizerTrainer(tokenization_config, training_config)
    
    # Prepare data
    training_corpus, validation_corpus = trainer.prepare_training_data(
        text_samples=text_data,
        wurcs_samples=wurcs_data,
        spectra_samples=spectra_data
    )
    
    # Train tokenizer
    if HAS_TOKENIZERS:
        tokenizer = trainer.train_tokenizer(training_corpus)
        
        # Evaluate
        stats = trainer.evaluate_tokenizer(tokenizer, validation_corpus)
        
        # Save artifacts
        trainer.save_training_artifacts(tokenizer, output_dir)
        
        logger.info("Tokenizer training completed successfully!")
    else:
        logger.warning("Tokenizers library not available - saving corpus only")
        trainer.save_training_artifacts(None, output_dir)
        
    return trainer


def create_demo_training_data() -> Dict[str, List]:
    """
    Create demonstration training data for tokenizer development.
    
    Returns:
        Dictionary with demo data for each modality
    """
    demo_data = {
        'text_samples': [
            "N-linked glycans contain GlcNAc residues attached to asparagine via β1-4 linkages to galactose.",
            "Mass spectrometry analysis revealed fucosylated structures with characteristic fragmentation patterns.",
            "The core pentasaccharide consists of two GlcNAc and three mannose residues in α1-6 and α1-3 configurations.",
            "MALDI-TOF MS showed peaks at m/z 1663.6 corresponding to biantennary complex N-glycans.",
            "Sialic acid modifications were identified through neutral loss of 291 Da from precursor ions.",
            "LC-MS/MS fragmentation yielded diagnostic ions for terminal fucose and core fucosylation.",
            "The glycoprotein exhibited high-mannose type structures with 5-9 mannose residues.",
            "Collision-induced dissociation revealed Y-type ions characteristic of glycosidic bond cleavage.",
            "Neuraminidase treatment confirmed the presence of α2-3 and α2-6 linked sialic acids.",
            "Cross-ring cleavage fragments provided linkage-specific information for structural elucidation."
        ],
        'wurcs_sequences': [
            "WURCS=2.0/3,3,2/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1221m-1a_1-5]/1-2-3/a4-b1_b4-c1",
            "WURCS=2.0/2,2,1/[a1122h-1b_1-5][a1221m-1a_1-5]/1-2/a4-b1",
            "WURCS=2.0/1,1,0/[a112h-1b_1-5]/1/",
            "WURCS=2.0/4,4,3/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1221m-1a_1-5][a112h-1b_1-5]/1-2-3-4/a4-b1_b3-c1_c6-d1",
            "WURCS=2.0/5,5,4/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1221m-1a_1-5][a1221m-1a_1-5][a1122A-2a_2-6]/1-2-3-4-5/a4-b1_b4-c1_c3-d1_d6-e2"
        ],
        'spectra_data': [
            # High-mannose N-glycan spectrum
            [(163.060, 100.0), (325.113, 85.2), (487.166, 67.3), (649.219, 45.8), (811.272, 23.4)],
            # Complex N-glycan with fucose
            [(146.058, 100.0), (204.087, 92.5), (366.139, 78.1), (512.197, 56.7), (674.250, 34.2), (836.303, 18.9)],
            # Sialylated N-glycan 
            [(274.093, 100.0), (366.139, 68.4), (657.235, 45.7), (819.288, 32.1), (981.341, 19.8), (1272.427, 12.3)],
            # Core fucosylated biantennary
            [(204.087, 100.0), (350.145, 83.6), (512.197, 71.2), (658.255, 54.9), (820.308, 38.7), (966.366, 25.1)]
        ]
    }
    
    return demo_data