"""
Specialized Tokenizer for Glycoinformatics Data

This module implements a custom tokenizer designed specifically for processing
WURCS notation, mass spectra, and glycan-related scientific text.
"""

import logging
import re
import json
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
import torch

logger = logging.getLogger(__name__)


@dataclass
class TokenizationConfig:
    """Configuration for glyco tokenizer"""
    # Vocabulary settings
    vocab_size: int = 50000
    min_frequency: int = 2
    
    # Special tokens
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    
    # Modality-specific tokens
    glycan_start: str = "[GLYCAN_START]"
    glycan_end: str = "[GLYCAN_END]"
    spectra_start: str = "[SPECTRA_START]"
    spectra_end: str = "[SPECTRA_END]"
    text_start: str = "[TEXT_START]"
    text_end: str = "[TEXT_END]"
    
    # WURCS-specific tokens
    wurcs_version: str = "[WURCS_VER]"
    wurcs_counts: str = "[WURCS_COUNTS]"
    wurcs_residues: str = "[WURCS_RES]"
    wurcs_connections: str = "[WURCS_CONN]"
    
    # Linkage tokens (glycosidic bonds)
    linkage_alpha_1_2: str = "[LINK_α1-2]"
    linkage_alpha_1_3: str = "[LINK_α1-3]"
    linkage_alpha_1_4: str = "[LINK_α1-4]"
    linkage_alpha_1_6: str = "[LINK_α1-6]"
    linkage_beta_1_2: str = "[LINK_β1-2]"
    linkage_beta_1_3: str = "[LINK_β1-3]"
    linkage_beta_1_4: str = "[LINK_β1-4]"
    linkage_beta_1_6: str = "[LINK_β1-6]"
    
    # Monosaccharide tokens
    mono_glc: str = "[MONO_Glc]"
    mono_gal: str = "[MONO_Gal]"
    mono_man: str = "[MONO_Man]"
    mono_fuc: str = "[MONO_Fuc]"
    mono_xyl: str = "[MONO_Xyl]"
    mono_glcnac: str = "[MONO_GlcNAc]"
    mono_galnac: str = "[MONO_GalNAc]"
    mono_mannac: str = "[MONO_ManNAc]"
    mono_neu5ac: str = "[MONO_Neu5Ac]"
    mono_neu5gc: str = "[MONO_Neu5Gc]"
    
    # Structural motifs
    motif_core: str = "[MOTIF_CORE]"
    motif_antenna: str = "[MOTIF_ANTENNA]"
    motif_branch: str = "[MOTIF_BRANCH]"
    motif_terminal: str = "[MOTIF_TERMINAL]"
    
    # Mass spectra tokens
    spectra_peak: str = "[PEAK]"
    spectra_precursor: str = "[PRECURSOR]"
    spectra_fragment: str = "[FRAGMENT]"
    spectra_neutral_loss: str = "[LOSS]"
    
    # Quantitative bins for m/z and intensity
    mz_bins: int = 1000  # Number of m/z bins for discretization
    intensity_bins: int = 100  # Number of intensity bins
    
    # Text processing
    max_sequence_length: int = 512
    lowercase: bool = True
    
    def get_all_special_tokens(self) -> List[str]:
        """Get all special tokens as a list"""
        tokens = []
        for field_name, field_value in asdict(self).items():
            if isinstance(field_value, str) and field_value.startswith("[") and field_value.endswith("]"):
                tokens.append(field_value)
        return tokens


class WURCSTokenizer:
    """
    Specialized tokenizer for WURCS (Web3 Unique Representation of Carbohydrate Structures) notation.
    
    WURCS format: WURCS=version/unique_count,residue_count,linkage_count/[residue_definitions]/connections
    Example: WURCS=2.0/3,3,2/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5]/1-2-3/a4-b1_b4-c1
    """
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        
        # WURCS parsing patterns
        self.wurcs_pattern = re.compile(r'WURCS=([^/]+)/([^/]+)/([^/]+)/([^/]*)')
        self.residue_pattern = re.compile(r'\[([^\]]+)\]')
        self.connection_pattern = re.compile(r'([a-z]\d+-[a-z]\d+(?:_[a-z]\d+-[a-z]\d+)*)')
        
        # Monosaccharide mappings
        self.mono_mappings = {
            'a2122h': config.mono_glcnac,  # GlcNAc
            'a1122h': config.mono_gal,     # Gal
            'a1221m': config.mono_man,     # Man
            'a1221h': config.mono_glc,     # Glc
            'a112h': config.mono_fuc,      # Fuc
            'a1122A': config.mono_neu5ac,  # Neu5Ac
            'a1122G': config.mono_neu5gc,  # Neu5Gc
        }
        
        # Linkage mappings
        self.linkage_mappings = {
            'a1-b2': config.linkage_alpha_1_2,
            'a1-b3': config.linkage_alpha_1_3, 
            'a1-b4': config.linkage_alpha_1_4,
            'a1-b6': config.linkage_alpha_1_6,
            'b1-a2': config.linkage_beta_1_2,
            'b1-a3': config.linkage_beta_1_3,
            'b1-a4': config.linkage_beta_1_4,
            'b1-a6': config.linkage_beta_1_6,
        }
        
    def tokenize_wurcs(self, wurcs_sequence: str) -> List[str]:
        """
        Tokenize a WURCS sequence into specialized tokens.
        
        Args:
            wurcs_sequence: WURCS notation string
            
        Returns:
            List of specialized tokens
        """
        tokens = [self.config.glycan_start]
        
        try:
            # Parse WURCS components
            match = self.wurcs_pattern.match(wurcs_sequence)
            if not match:
                logger.warning(f"Invalid WURCS format: {wurcs_sequence}")
                tokens.extend([self.config.unk_token, self.config.glycan_end])
                return tokens
                
            version, counts, residues, connections = match.groups()
            
            # Add version token
            tokens.append(self.config.wurcs_version)
            tokens.append(f"v{version}")
            
            # Add counts token and parse counts
            tokens.append(self.config.wurcs_counts)
            count_parts = counts.split(',')
            for count in count_parts:
                tokens.append(f"c{count}")
                
            # Add residues section
            tokens.append(self.config.wurcs_residues)
            residue_matches = self.residue_pattern.findall(residues)
            
            for i, residue in enumerate(residue_matches):
                # Map to monosaccharide token if recognized
                mono_token = self._identify_monosaccharide(residue)
                tokens.append(mono_token)
                
                # Add position indicator
                tokens.append(f"pos{i+1}")
                
                # Parse modifications if present
                modifications = self._parse_modifications(residue)
                tokens.extend(modifications)
                
            # Add connections section
            if connections:
                tokens.append(self.config.wurcs_connections)
                connection_parts = connections.split('/')
                
                for conn_part in connection_parts:
                    if conn_part:
                        linkage_tokens = self._parse_connections(conn_part)
                        tokens.extend(linkage_tokens)
                        
        except Exception as e:
            logger.error(f"Error parsing WURCS {wurcs_sequence}: {e}")
            tokens = [self.config.glycan_start, self.config.unk_token]
            
        tokens.append(self.config.glycan_end)
        return tokens
        
    def _identify_monosaccharide(self, residue: str) -> str:
        """Identify monosaccharide from residue definition"""
        # Look for known patterns in residue definition
        for pattern, token in self.mono_mappings.items():
            if pattern in residue:
                return token
                
        # Check for specific modifications that indicate monosaccharide type
        if 'NCC' in residue and '3=O' in residue:
            return self.config.mono_glcnac  # N-acetyl modification
        elif 'h-1b_1-5' in residue:
            return self.config.mono_gal     # Galactose pattern
        elif 'h-1a_1-5' in residue:
            return self.config.mono_man     # Mannose pattern
            
        return self.config.unk_token
        
    def _parse_modifications(self, residue: str) -> List[str]:
        """Parse chemical modifications from residue definition"""
        modifications = []
        
        # N-acetyl modification
        if 'NCC' in residue and '3=O' in residue:
            modifications.append('[MOD_NAc]')
            
        # Sulfation
        if 'S' in residue:
            modifications.append('[MOD_Sulf]')
            
        # Phosphorylation
        if 'P' in residue:
            modifications.append('[MOD_Phos]')
            
        # Methylation
        if 'Me' in residue:
            modifications.append('[MOD_Me]')
            
        return modifications
        
    def _parse_connections(self, connections: str) -> List[str]:
        """Parse linkage connections"""
        tokens = []
        
        # Parse connection patterns like "a4-b1_b4-c1"
        connection_matches = self.connection_pattern.findall(connections)
        
        for connection in connection_matches:
            # Split multiple connections
            links = connection.split('_')
            
            for link in links:
                # Map to linkage token
                linkage_token = self.linkage_mappings.get(link, f'[LINK_{link}]')
                tokens.append(linkage_token)
                
        return tokens


class SpectraTokenizer:
    """
    Specialized tokenizer for mass spectrometry data.
    
    Converts peak lists into discrete tokens representing m/z values,
    intensities, and spectral patterns.
    """
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        
        # m/z binning (50-2000 Da typical range)
        self.mz_min = 50.0
        self.mz_max = 2000.0
        self.mz_bin_width = (self.mz_max - self.mz_min) / config.mz_bins
        
        # Intensity binning (relative intensities 0-100%)
        self.intensity_bins = config.intensity_bins
        
        # Common fragment patterns for glycans
        self.common_fragments = {
            163.060: '[FRAG_Hex]',          # Hexose fragment
            204.087: '[FRAG_HexNAc]',       # HexNAc fragment  
            274.093: '[FRAG_NeuAc]',        # Sialic acid fragment
            146.058: '[FRAG_Fuc]',          # Fucose fragment
            126.055: '[FRAG_C2H6O4]',       # Cross-ring fragment
            366.139: '[FRAG_HexHexNAc]',    # Hex-HexNAc disaccharide
        }
        
        # Neutral loss patterns
        self.neutral_losses = {
            18.010: '[LOSS_H2O]',    # Water loss
            32.026: '[LOSS_CH2O2]',  # Formaldehyde loss
            60.021: '[LOSS_C2H4O2]', # Acetic acid loss
            162.053: '[LOSS_Hex]',   # Hexose loss
            203.079: '[LOSS_HexNAc]' # HexNAc loss
        }
        
    def tokenize_spectrum(self, 
                        peaks: List[Tuple[float, float]],
                        precursor_mz: Optional[float] = None,
                        normalize: bool = True) -> List[str]:
        """
        Tokenize mass spectrum into discrete tokens.
        
        Args:
            peaks: List of (m/z, intensity) tuples
            precursor_mz: Precursor m/z value
            normalize: Whether to normalize intensities
            
        Returns:
            List of spectrum tokens
        """
        tokens = [self.config.spectra_start]
        
        if not peaks:
            tokens.extend([self.config.unk_token, self.config.spectra_end])
            return tokens
            
        # Normalize intensities if requested
        if normalize:
            max_intensity = max(peak[1] for peak in peaks)
            if max_intensity > 0:
                peaks = [(mz, intensity / max_intensity * 100) 
                        for mz, intensity in peaks]
                
        # Add precursor information
        if precursor_mz:
            tokens.append(self.config.spectra_precursor)
            tokens.append(self._mz_to_token(precursor_mz))
            
        # Sort peaks by m/z
        sorted_peaks = sorted(peaks, key=lambda x: x[0])
        
        # Tokenize each peak
        for mz, intensity in sorted_peaks:
            # Skip very low intensity peaks
            if intensity < 1.0:
                continue
                
            # Check for known fragments
            fragment_token = self._identify_fragment(mz)
            if fragment_token:
                tokens.append(fragment_token)
            else:
                tokens.append(self.config.spectra_peak)
                tokens.append(self._mz_to_token(mz))
                
            # Add intensity token
            intensity_token = self._intensity_to_token(intensity)
            tokens.append(intensity_token)
            
        # Identify neutral losses if precursor is known
        if precursor_mz:
            loss_tokens = self._identify_neutral_losses(sorted_peaks, precursor_mz)
            tokens.extend(loss_tokens)
            
        tokens.append(self.config.spectra_end)
        return tokens
        
    def _mz_to_token(self, mz: float) -> str:
        """Convert m/z value to discrete token"""
        if mz < self.mz_min or mz > self.mz_max:
            return '[MZ_OOR]'  # Out of range
            
        bin_idx = int((mz - self.mz_min) / self.mz_bin_width)
        bin_idx = min(bin_idx, self.config.mz_bins - 1)
        return f'[MZ_{bin_idx:04d}]'
        
    def _intensity_to_token(self, intensity: float) -> str:
        """Convert intensity to discrete token"""
        bin_idx = int(intensity / 100.0 * self.intensity_bins)
        bin_idx = min(bin_idx, self.intensity_bins - 1)
        return f'[INT_{bin_idx:03d}]'
        
    def _identify_fragment(self, mz: float, tolerance: float = 0.05) -> Optional[str]:
        """Identify known fragment patterns"""
        for ref_mz, token in self.common_fragments.items():
            if abs(mz - ref_mz) <= tolerance:
                return token
        return None
        
    def _identify_neutral_losses(self, 
                               peaks: List[Tuple[float, float]],
                               precursor_mz: float,
                               tolerance: float = 0.05) -> List[str]:
        """Identify neutral loss patterns"""
        loss_tokens = []
        
        for mz, intensity in peaks:
            mass_diff = precursor_mz - mz
            
            for loss_mass, token in self.neutral_losses.items():
                if abs(mass_diff - loss_mass) <= tolerance:
                    loss_tokens.append(token)
                    break
                    
        return loss_tokens


class GlycanTextTokenizer:
    """
    Specialized tokenizer for glycan-related scientific text.
    
    Handles glycomics terminology, chemical names, and domain-specific language.
    """
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        
        # Glycan terminology dictionary
        self.glycan_terms = {
            # Monosaccharides (full names)
            'glucose': config.mono_glc,
            'galactose': config.mono_gal,
            'mannose': config.mono_man,
            'fucose': config.mono_fuc,
            'xylose': config.mono_xyl,
            'n-acetylglucosamine': config.mono_glcnac,
            'n-acetylgalactosamine': config.mono_galnac,
            'n-acetylmannosamine': config.mono_mannac,
            'n-acetylneuraminic acid': config.mono_neu5ac,
            'sialic acid': config.mono_neu5ac,
            
            # Abbreviations
            'glc': config.mono_glc,
            'gal': config.mono_gal,
            'man': config.mono_man,
            'fuc': config.mono_fuc,
            'xyl': config.mono_xyl,
            'glcnac': config.mono_glcnac,
            'galnac': config.mono_galnac,
            'mannac': config.mono_mannac,
            'neu5ac': config.mono_neu5ac,
            'neu5gc': config.mono_neu5gc,
            
            # Structural terms
            'core': config.motif_core,
            'antenna': config.motif_antenna,
            'branch': config.motif_branch,
            'terminal': config.motif_terminal,
        }
        
        # Linkage patterns
        self.linkage_patterns = {
            r'α1?-?2': config.linkage_alpha_1_2,
            r'α1?-?3': config.linkage_alpha_1_3,
            r'α1?-?4': config.linkage_alpha_1_4,
            r'α1?-?6': config.linkage_alpha_1_6,
            r'β1?-?2': config.linkage_beta_1_2,
            r'β1?-?3': config.linkage_beta_1_3,
            r'β1?-?4': config.linkage_beta_1_4,
            r'β1?-?6': config.linkage_beta_1_6,
        }
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess glycomics text with domain-specific normalization.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text with normalized terminology
        """
        if not text:
            return ""
            
        # Convert to lowercase if configured
        if self.config.lowercase:
            text = text.lower()
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Replace glycan terms with special tokens
        for term, token in self.glycan_terms.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, f' {token} ', text, flags=re.IGNORECASE)
            
        # Replace linkage patterns
        for pattern, token in self.linkage_patterns.items():
            text = re.sub(pattern, f' {token} ', text, flags=re.IGNORECASE)
            
        # Normalize multiple spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract glycan-related entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted entities by category
        """
        entities = {
            'monosaccharides': [],
            'linkages': [],
            'modifications': [],
            'structures': []
        }
        
        # Find monosaccharides
        mono_pattern = r'\b(glc|gal|man|fuc|xyl|glcnac|galnac|neu5ac|glucose|galactose|mannose)\b'
        entities['monosaccharides'] = re.findall(mono_pattern, text, re.IGNORECASE)
        
        # Find linkages
        link_pattern = r'[αβ]1?-?[1-6]'
        entities['linkages'] = re.findall(link_pattern, text, re.IGNORECASE)
        
        # Find modifications
        mod_pattern = r'\b(acetyl|sulfat|phosphat|methylat)\w*\b'
        entities['modifications'] = re.findall(mod_pattern, text, re.IGNORECASE)
        
        # Find structure types
        struct_pattern = r'\b(n-glycan|o-glycan|glycosaminoglycan|proteoglycan|glycolipid)\b'
        entities['structures'] = re.findall(struct_pattern, text, re.IGNORECASE)
        
        # Remove duplicates
        for category in entities:
            entities[category] = list(set(entities[category]))
            
        return entities


class GlycoTokenizer:
    """
    Unified tokenizer for multimodal glycoinformatics data.
    
    Combines WURCS, spectra, and text tokenization into a single interface
    with cross-modal alignment capabilities.
    """
    
    def __init__(self, config: Optional[TokenizationConfig] = None):
        """
        Initialize the unified glyco tokenizer.
        
        Args:
            config: Tokenization configuration
        """
        self.config = config or TokenizationConfig()
        
        # Initialize specialized tokenizers
        self.wurcs_tokenizer = WURCSTokenizer(self.config)
        self.spectra_tokenizer = SpectraTokenizer(self.config)
        self.text_tokenizer = GlycanTextTokenizer(self.config)
        
        # Build vocabulary
        self.vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)
        
        # Create token-to-ID and ID-to-token mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Base tokenizer for regular text
        self.base_tokenizer = self._create_base_tokenizer()
        
    def _build_vocabulary(self) -> List[str]:
        """Build complete vocabulary including all special tokens"""
        vocab = []
        
        # Add all special tokens from config
        vocab.extend(self.config.get_all_special_tokens())
        
        # Add common text tokens (this would be expanded with actual corpus)
        common_tokens = [
            'glycan', 'protein', 'mass', 'spectrum', 'structure', 'linkage',
            'monosaccharide', 'disaccharide', 'oligosaccharide', 'polysaccharide',
            'n-linked', 'o-linked', 'glycosylation', 'glycoprotein', 'glycolipid',
            'lectin', 'carbohydrate', 'sugar', 'residue', 'bond', 'configuration',
            'ms', 'lc', 'maldi', 'esi', 'tof', 'fragmentation', 'cid', 'hcd',
            'precursor', 'fragment', 'peak', 'intensity', 'mz', 'charge',
            'human', 'mouse', 'tissue', 'serum', 'plasma', 'cell', 'membrane',
            'cancer', 'disease', 'biomarker', 'diagnosis', 'therapeutic'
        ]
        
        vocab.extend(common_tokens)
        
        # Add numeric tokens for m/z bins
        for i in range(self.config.mz_bins):
            vocab.append(f'[MZ_{i:04d}]')
            
        # Add numeric tokens for intensity bins  
        for i in range(self.config.intensity_bins):
            vocab.append(f'[INT_{i:03d}]')
            
        # Add positional tokens
        for i in range(20):  # Up to 20 residue positions
            vocab.append(f'pos{i+1}')
            
        # Add count tokens
        for i in range(50):  # Up to 50 residues/linkages
            vocab.append(f'c{i}')
            
        # Add version tokens
        vocab.extend(['v1.0', 'v2.0', 'v2.1'])
        
        # Add common subwords (this would be learned from corpus)
        subwords = ['yl', 'ose', 'ac', 'nac', 'an', 'ine', 'ate', 'tion', 'ing', 'ed']
        vocab.extend(subwords)
        
        return vocab
        
    def _create_base_tokenizer(self) -> Tokenizer:
        """Create base tokenizer for regular text"""
        # Use BPE for subword tokenization
        tokenizer = Tokenizer(models.BPE())
        
        # Pre-tokenization 
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Post-processing
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.config.cls_token} $A {self.config.sep_token}",
            pair=f"{self.config.cls_token} $A {self.config.sep_token} $B:1 {self.config.sep_token}:1",
            special_tokens=[
                (self.config.cls_token, self.token_to_id[self.config.cls_token]),
                (self.config.sep_token, self.token_to_id[self.config.sep_token]),
            ],
        )
        
        # Decoder
        tokenizer.decoder = decoders.BPE()
        
        return tokenizer
        
    def tokenize_multimodal(self, 
                          text: Optional[str] = None,
                          wurcs_sequence: Optional[str] = None,
                          spectra_peaks: Optional[List[Tuple[float, float]]] = None,
                          precursor_mz: Optional[float] = None) -> Dict[str, List[int]]:
        """
        Tokenize multimodal input and return token IDs.
        
        Args:
            text: Scientific text
            wurcs_sequence: WURCS glycan notation
            spectra_peaks: Mass spectrum peaks
            precursor_mz: Precursor m/z for spectrum
            
        Returns:
            Dictionary with token IDs for each modality
        """
        result = {}
        
        # Tokenize text
        if text:
            text_tokens = [self.config.text_start]
            
            # Preprocess and tokenize
            preprocessed_text = self.text_tokenizer.preprocess_text(text)
            
            # Use base tokenizer for regular words
            encoded = self.base_tokenizer.encode(preprocessed_text)
            text_tokens.extend(encoded.tokens)
            
            text_tokens.append(self.config.text_end)
            
            # Convert to IDs
            result['text'] = self._tokens_to_ids(text_tokens)
            
        # Tokenize WURCS sequence
        if wurcs_sequence:
            wurcs_tokens = self.wurcs_tokenizer.tokenize_wurcs(wurcs_sequence)
            result['structure'] = self._tokens_to_ids(wurcs_tokens)
            
        # Tokenize spectrum
        if spectra_peaks:
            spectra_tokens = self.spectra_tokenizer.tokenize_spectrum(
                spectra_peaks, precursor_mz
            )
            result['spectra'] = self._tokens_to_ids(spectra_tokens)
            
        return result
        
    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs, handling unknown tokens"""
        ids = []
        unk_id = self.token_to_id.get(self.config.unk_token, 0)
        
        for token in tokens:
            token_id = self.token_to_id.get(token, unk_id)
            ids.append(token_id)
            
        return ids
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        tokens = []
        special_tokens = set(self.config.get_all_special_tokens())
        
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.config.unk_token)
            
            if skip_special_tokens and token in special_tokens:
                continue
                
            tokens.append(token)
            
        return ' '.join(tokens)
        
    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration and vocabulary"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = save_path / "tokenizer_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
            
        # Save vocabulary
        vocab_path = save_path / "vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump(self.token_to_id, f, indent=2)
            
        # Save base tokenizer
        tokenizer_path = save_path / "tokenizer.json"
        self.base_tokenizer.save(str(tokenizer_path))
        
        logger.info(f"Tokenizer saved to {save_directory}")
        
    @classmethod
    def from_pretrained(cls, model_directory: str):
        """Load tokenizer from saved files"""
        model_path = Path(model_directory)
        
        # Load configuration
        config_path = model_path / "tokenizer_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        config = TokenizationConfig(**config_dict)
        
        # Create tokenizer instance
        tokenizer = cls(config)
        
        # Load base tokenizer
        tokenizer_path = model_path / "tokenizer.json"
        tokenizer.base_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        logger.info(f"Tokenizer loaded from {model_directory}")
        return tokenizer
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.vocab_size
        
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get mapping of special token names to IDs"""
        special_tokens = {
            'pad_token_id': self.token_to_id.get(self.config.pad_token),
            'unk_token_id': self.token_to_id.get(self.config.unk_token),
            'cls_token_id': self.token_to_id.get(self.config.cls_token),
            'sep_token_id': self.token_to_id.get(self.config.sep_token),
            'mask_token_id': self.token_to_id.get(self.config.mask_token),
        }
        return {k: v for k, v in special_tokens.items() if v is not None}