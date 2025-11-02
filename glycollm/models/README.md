# GlycoLLM Models

This directory contains the complete implementation of **GlycoLLM**, a multimodal large language model specifically designed for glycoinformatics. The architecture combines transformer technology with domain-specific innovations for processing glycan structures, mass spectra, and scientific text.

## Architecture Overview

### 🧬 Multimodal Design

GlycoLLM processes three types of data simultaneously:

1. **Text**: Scientific literature, annotations, and descriptions
2. **Structure**: WURCS notation for glycan sequences  
3. **Spectra**: Mass spectrometry peak data

### 🏗️ Core Components

- **`GlycoLLM`**: Main model class integrating all components
- **`MultiModalEmbedding`**: Handles different input modalities
- **`CrossModalAttention`**: Enables information fusion across modalities
- **`GlycoTransformerLayer`**: Custom transformer layer with glycan-specific features
- **`GlycoLLMTaskHeads`**: Specialized prediction heads for different tasks

## Quick Start

```python
from glycollm.models import create_glycollm_model, create_loss_function

# Create model
model = create_glycollm_model(
    vocab_size=50000,
    d_model=768,
    n_layers=12,
    n_heads=12
)

# Create loss function
loss_fn = create_loss_function()

# Process multimodal input
outputs = model(
    text_input_ids=text_tokens,
    structure_input_ids=wurcs_tokens,
    spectra_input_ids=spectra_tokens,
    task="all"
)
```

## Model Architecture

### Embedding Layer (`MultiModalEmbedding`)

```
Input Modalities → Separate Embeddings → Projection → Common Dimension
     ↓                    ↓                 ↓              ↓
   Text IDs          Text Embeddings   768-dim       Combined
   WURCS IDs    →    Structure Emb  →   512-dim  →    Sequence
   Spectra IDs       Spectra Emb       512-dim       
```

Features:
- Modality-specific embedding dimensions
- Positional encoding for each modality
- Modality type embeddings for differentiation
- Unified projection to common dimension

### Transformer Layers (`GlycoTransformerLayer`)

Each layer includes:
1. **Self-attention** within modality
2. **Cross-modal attention** between modalities  
3. **Feed-forward network** with residual connections
4. **Layer normalization** at each step

### Task-Specific Heads

1. **Structure Prediction**: Generates WURCS sequences
2. **Spectra Prediction**: Predicts mass spectrum peaks
3. **Text Generation**: Produces scientific descriptions
4. **Cross-Modal Retrieval**: Creates aligned embeddings

## Configuration

### Model Configuration (`GlycoLLMConfig`)

```python
config = GlycoLLMConfig(
    # Core architecture
    d_model=768,              # Model dimension
    n_layers=12,              # Number of transformer layers
    n_heads=12,               # Attention heads
    d_ff=3072,                # Feed-forward dimension
    
    # Sequence lengths
    max_seq_length=2048,      # Total sequence length
    text_max_length=512,      # Text component length
    structure_max_length=256, # WURCS component length
    spectra_max_length=1024,  # Spectra component length
    
    # Modality dimensions
    text_d_model=768,         # Text embedding dimension
    structure_d_model=512,    # Structure embedding dimension  
    spectra_d_model=512,      # Spectra embedding dimension
    
    # Cross-modal fusion
    fusion_layers=4,          # Cross-modal attention layers
    fusion_heads=8,           # Cross-modal attention heads
    
    # Regularization
    dropout=0.1,              # Dropout rate
    attention_dropout=0.1,    # Attention dropout
    
    # Task heads
    enable_structure_prediction=True,
    enable_spectra_prediction=True,
    enable_text_generation=True,
    enable_cross_modal_retrieval=True
)
```

## Loss Functions

### Multimodal Loss (`MultiModalLoss`)

Combines multiple loss components:

1. **Structure Prediction Loss**: Cross-entropy with label smoothing
2. **Spectra Prediction Loss**: Combined token + intensity prediction
3. **Text Generation Loss**: Standard language modeling loss
4. **Contrastive Loss**: Cross-modal alignment

### Loss Configuration (`LossConfig`)

```python
loss_config = LossConfig(
    # Task weights
    structure_weight=1.0,      # Structure prediction importance
    spectra_weight=1.0,        # Spectra prediction importance
    text_weight=1.0,           # Text generation importance
    contrastive_weight=0.5,    # Cross-modal alignment importance
    
    # Contrastive learning
    temperature=0.07,          # Contrastive temperature
    margin=0.2,               # Triplet loss margin
    
    # Regularization
    label_smoothing=0.1,       # Label smoothing factor
    ignore_index=-100         # Padding token index
)
```

## Specialized Features

### 1. Cross-Modal Attention

```python
# Text attending to structure
structure_context = cross_modal_attention(
    query=text_embeddings,
    key=structure_embeddings,
    value=structure_embeddings
)

# Bidirectional information flow
text_context = cross_modal_attention(
    query=structure_embeddings, 
    key=text_embeddings,
    value=text_embeddings
)
```

### 2. Contrastive Learning

Aligns representations across modalities:
- Text ↔ Structure alignment
- Text ↔ Spectra alignment  
- Structure ↔ Spectra alignment

### 3. Structure-Aware Loss

Weights important structural tokens more heavily:
- Monosaccharide tokens
- Linkage patterns
- Modification markers

### 4. Spectra-Specific Processing

- m/z value discretization
- Intensity quantization
- Fragment pattern recognition
- Neutral loss identification

## Model Analysis Tools

### Parameter Analysis (`ModelAnalyzer`)

```python
from glycollm.models import ModelAnalyzer

analyzer = ModelAnalyzer()
stats = analyzer.analyze_model(model)

# Detailed statistics
print(f"Total parameters: {stats.total_params:,}")
print(f"Model size: {stats.model_size_mb:.1f} MB")

# Layer-wise breakdown
analyzer.print_model_summary()
```

### Gradient Analysis (`GradientAnalyzer`)

```python
from glycollm.models import GradientAnalyzer

grad_analyzer = GradientAnalyzer()
gradient_stats = grad_analyzer.analyze_gradients(model)

# Check for gradient problems
grad_analyzer.print_gradient_summary(gradient_stats)
```

### Performance Profiling (`ModelProfiler`)

```python
from glycollm.models import ModelProfiler

profiler = ModelProfiler()
timing_stats = profiler.profile_forward_pass(model, sample_input)

# Timing analysis
profiler.print_profiling_results()
```

## Training Considerations

### 1. Multi-Task Learning
- Balance between different task losses
- Curriculum learning for complex tasks
- Task-specific learning rates

### 2. Memory Management
- Gradient checkpointing for large models
- Mixed precision training
- Batch size optimization

### 3. Cross-Modal Alignment
- Contrastive learning temperature tuning
- Hard negative mining
- Modality dropout for robustness

### 4. Domain Adaptation
- Pre-training on large text corpora
- Fine-tuning on glycomics data
- Transfer learning strategies

## Model Variants

### Base Model (12 layers, 768 hidden)
```python
model = create_glycollm_model(
    vocab_size=50000,
    d_model=768,
    n_layers=12,
    n_heads=12
)
# ~110M parameters
```

### Large Model (24 layers, 1024 hidden)
```python
model = create_glycollm_model(
    vocab_size=50000,
    d_model=1024,
    n_layers=24,
    n_heads=16
)
# ~340M parameters
```

### Efficient Model (6 layers, 512 hidden)
```python
model = create_glycollm_model(
    vocab_size=50000,
    d_model=512,
    n_layers=6,
    n_heads=8
)
# ~25M parameters
```

## Integration with Pipeline

### 1. Data Pipeline
```
Raw Data → Tokenizer → Model → Predictions
    ↓          ↓         ↓         ↓
  Text      Token IDs  Logits   Sequences
  WURCS     Structure  Attention Text
  Spectra   Embeddings Weights   Structures
```

### 2. Training Pipeline
```
Multimodal → Forward → Loss → Backward → Update
Batch        Pass     Computation Pass    Parameters
```

### 3. Inference Pipeline
```
Input → Encode → Generate → Decode → Output
Data    Tokens   Sequence  Text     Results
```

## Demo and Testing

Run the comprehensive model demo:

```bash
cd glycollm/models
python demo_model.py
```

The demo showcases:
- Model configuration and creation
- Architecture analysis
- Loss function setup
- Sample data processing
- Forward pass simulation
- Performance profiling
- Use case examples

## Dependencies

Core dependencies:
- `torch` - PyTorch deep learning framework
- `numpy` - Numerical computing
- `dataclasses` - Configuration management
- `logging` - Logging utilities

Optional dependencies:
- `transformers` - Hugging Face transformers (for comparison)
- `wandb` - Experiment tracking
- `tensorboard` - Visualization

Install dependencies:
```bash
pip install torch numpy
pip install transformers wandb tensorboard  # Optional
```

## File Structure

```
glycollm/models/
├── __init__.py              # Package interface
├── glycollm.py             # Main model architecture
├── loss_functions.py       # Specialized loss functions
├── model_utils.py          # Analysis and utility functions
├── demo_model.py           # Demonstration script
└── README.md              # This documentation
```

## Research Applications

### 1. Structure Elucidation
- Predict glycan structures from MS/MS spectra
- Validate proposed structures against experimental data
- Generate alternative structure hypotheses

### 2. Fragmentation Prediction
- Predict fragmentation patterns for glycan structures
- Simulate MS/MS spectra for database matching
- Understand fragmentation mechanisms

### 3. Knowledge Discovery
- Mine literature for glycan-related information
- Discover structure-function relationships
- Identify novel glycan biomarkers

### 4. Database Curation
- Automated annotation of glycan databases
- Quality assessment of structural assignments
- Cross-reference validation between databases

## Future Enhancements

Planned improvements:
- Larger pre-trained models
- Additional modalities (NMR, etc.)
- Improved cross-modal attention
- Reinforcement learning integration
- Real-time inference optimization
- Mobile deployment support

---

*This architecture represents a novel approach to multimodal learning in glycoinformatics, combining the power of transformer models with domain-specific knowledge and specialized loss functions.*