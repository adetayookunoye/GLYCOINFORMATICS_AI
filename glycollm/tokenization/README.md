# GlycoLLM Tokenization

This directory contains the specialized tokenization system for the GlycoLLM multimodal language model. The tokenizer is designed to handle three types of glycoinformatics data:

1. **WURCS notation** - Structured glycan sequences
2. **Mass spectra** - Experimental glycan fragmentation data  
3. **Scientific text** - Glycomics literature and annotations

## Features

### Specialized Tokenization
- **WURCS Tokenizer**: Parses Web3 Unique Representation of Carbohydrate Structures
- **Spectra Tokenizer**: Converts mass spectra peaks to discrete tokens
- **Text Tokenizer**: Processes glycomics terminology and scientific text
- **Multimodal Integration**: Unified interface for all data types

### Domain Knowledge Integration
- Monosaccharide recognition (GlcNAc, Gal, Man, Fuc, Neu5Ac, etc.)
- Linkage pattern identification (α1-2, β1-4, etc.)
- Fragment pattern recognition (Hex, HexNAc, cross-ring cleavages)
- Glycomics terminology normalization

### Quality Assurance
- WURCS format validation
- Spectrum quality analysis
- Entity extraction from text
- Comprehensive testing framework

## Quick Start

```python
from glycollm.tokenization import GlycoTokenizer, TokenizationConfig

# Initialize tokenizer
config = TokenizationConfig()
tokenizer = GlycoTokenizer(config)

# Tokenize multimodal data
result = tokenizer.tokenize_multimodal(
    text="N-linked glycan with GlcNAc residues",
    wurcs_sequence="WURCS=2.0/3,3,2/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1221m-1a_1-5]/1-2-3/a4-b1_b4-c1",
    spectra_peaks=[(163.060, 100.0), (204.087, 85.5), (366.139, 45.2)]
)

print(f"Text tokens: {len(result['text'])}")
print(f"Structure tokens: {len(result['structure'])}")  
print(f"Spectra tokens: {len(result['spectra'])}")
```

## Components

### Core Classes

- **`GlycoTokenizer`**: Main unified tokenizer interface
- **`WURCSTokenizer`**: Specialized WURCS sequence processor
- **`SpectraTokenizer`**: Mass spectrum peak encoder
- **`GlycanTextTokenizer`**: Scientific text processor

### Utility Classes

- **`WURCSValidator`**: Validates and parses WURCS notation
- **`SpectrumAnalyzer`**: Analyzes mass spectra for quality and fragments
- **`GlycanTextProcessor`**: Preprocesses glycomics text
- **`TokenizerTester`**: Comprehensive testing utilities

### Training Support

- **`TokenizerTrainer`**: Trains custom vocabulary from corpus
- **`TrainingConfig`**: Configuration for tokenizer training
- **`TrainingStats`**: Training evaluation metrics

## Configuration

The `TokenizationConfig` class controls tokenizer behavior:

```python
config = TokenizationConfig(
    vocab_size=50000,           # Target vocabulary size
    min_frequency=2,            # Minimum token frequency
    max_sequence_length=512,    # Maximum sequence length
    
    # Special tokens
    pad_token="[PAD]",
    unk_token="[UNK]", 
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    
    # Modality markers
    glycan_start="[GLYCAN_START]",
    spectra_start="[SPECTRA_START]",
    text_start="[TEXT_START]",
    
    # Domain-specific tokens
    mono_glcnac="[MONO_GlcNAc]",
    linkage_beta_1_4="[LINK_β1-4]",
    spectra_peak="[PEAK]",
    
    # Quantization
    mz_bins=1000,              # m/z discretization bins
    intensity_bins=100         # Intensity discretization bins
)
```

## WURCS Tokenization

The WURCS tokenizer handles Web3 Unique Representation of Carbohydrate Structures:

```python
wurcs = "WURCS=2.0/3,3,2/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1221m-1a_1-5]/1-2-3/a4-b1_b4-c1"

tokens = tokenizer.wurcs_tokenizer.tokenize_wurcs(wurcs)
# Returns: ['[GLYCAN_START]', '[WURCS_VER]', 'v2.0', '[WURCS_COUNTS]', 'c3', 'c3', 'c2', 
#          '[WURCS_RES]', '[MONO_GlcNAc]', 'pos1', '[MOD_NAc]', '[MONO_Gal]', 'pos2', ...]
```

Features:
- Version and count parsing
- Monosaccharide identification
- Chemical modification recognition  
- Linkage pattern extraction
- Position tracking

## Mass Spectrum Tokenization

The spectrum tokenizer converts peak lists to discrete representations:

```python
peaks = [(163.060, 100.0), (204.087, 85.5), (366.139, 45.2), (512.197, 22.1)]

tokens = tokenizer.spectra_tokenizer.tokenize_spectrum(peaks, precursor_mz=674.250)
# Returns: ['[SPECTRA_START]', '[PRECURSOR]', '[MZ_0624]', '[FRAG_Hex]', '[INT_100]', 
#          '[FRAG_HexNAc]', '[INT_085]', '[MZ_0366]', '[INT_045]', ...]
```

Features:
- m/z and intensity discretization
- Fragment pattern recognition
- Neutral loss identification
- Precursor ion tracking
- Quality assessment

## Scientific Text Processing

The text tokenizer handles glycomics terminology:

```python
text = "N-linked glycans contain GlcNAc residues with α1-6 linkages analyzed by MALDI-TOF MS."

processed = tokenizer.text_tokenizer.preprocess_text(text)
# Returns: "n_glycan contain [MONO_GlcNAc] residues with [LINK_α1-6] linkages analyzed by ms"

entities = tokenizer.text_tokenizer.extract_entities(text)
# Returns: {'monosaccharides': ['glcnac'], 'linkages': ['α1-6'], 'methods': ['maldi-tof', 'ms'], ...}
```

Features:
- Terminology normalization
- Entity extraction
- Linkage pattern recognition
- Method identification
- Chemical formula handling

## Training Custom Tokenizers

Train domain-specific vocabularies from your data:

```python
from glycollm.tokenization import train_glyco_tokenizer

# Prepare training data
text_data = ["List of scientific texts..."]
wurcs_data = ["List of WURCS sequences..."]  
spectra_data = [[(mz, intensity), ...], ...]

# Train tokenizer
trainer = train_glyco_tokenizer(
    text_data=text_data,
    wurcs_data=wurcs_data,
    spectra_data=spectra_data,
    output_dir="custom_tokenizer",
    vocab_size=30000,
    min_frequency=3
)

# Load trained tokenizer
tokenizer = GlycoTokenizer.from_pretrained("custom_tokenizer")
```

## Testing and Validation

Comprehensive testing utilities:

```python
from glycollm.tokenization import TokenizerTester

tester = TokenizerTester()

# Test WURCS samples
wurcs_results = tester.test_wurcs_samples(wurcs_list)
print(f"Valid: {wurcs_results['valid_samples']}/{wurcs_results['total_samples']}")

# Test spectra processing  
spectrum_results = tester.test_spectrum_processing(spectra_list)
print(f"Fragments identified: {spectrum_results['fragment_identification']}")

# Generate report
report = tester.generate_test_report(wurcs_results, spectrum_results, text_samples)
```

## Demo Script

Run the comprehensive demo:

```bash
cd glycollm/tokenization
python demo_tokenizer.py
```

The demo showcases:
- WURCS sequence tokenization
- Mass spectrum processing
- Scientific text handling
- Multimodal integration
- Validation and analysis
- Testing framework
- Save/load functionality

## Dependencies

Core dependencies:
- `json`, `re`, `pathlib` (built-in)
- `dataclasses`, `typing` (built-in)
- `collections`, `logging` (built-in)

Optional dependencies (for training):
- `tokenizers` - Hugging Face tokenizers library
- `numpy` - Numerical processing
- `torch` - PyTorch tensors

Install optional dependencies:
```bash
pip install tokenizers numpy torch
```

## Architecture

```
glycollm/tokenization/
├── __init__.py              # Package interface
├── glyco_tokenizer.py       # Core tokenizer classes
├── tokenizer_utils.py       # Validation and analysis utilities
├── tokenizer_training.py    # Training framework
├── demo_tokenizer.py        # Demonstration script
└── README.md               # This documentation
```

## Integration with GlycoLLM

The tokenizer integrates seamlessly with the GlycoLLM model:

1. **Preprocessing**: Raw data → tokenized sequences
2. **Encoding**: Tokens → input embeddings
3. **Model**: Multimodal transformer processing  
4. **Decoding**: Output tokens → structured predictions

The specialized vocabulary ensures the model understands:
- Glycan structural motifs
- Mass spectrometry patterns
- Scientific terminology
- Cross-modal relationships

## Future Enhancements

Planned improvements:
- Extended monosaccharide library
- Advanced fragment prediction
- Cross-modal alignment tokens
- Adaptive vocabulary expansion
- Real-time quality assessment
- Integration with glycan databases

---

*This tokenizer is part of the GlycoInformatics AI platform for multimodal glycan analysis and prediction.*