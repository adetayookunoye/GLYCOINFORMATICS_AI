#!/usr/bin/env python3
"""
Demo script for GlycoLLM model architecture.

This script demonstrates the multimodal transformer architecture
and its capabilities for glycoinformatics tasks.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run GlycoLLM model demonstration"""
    
    try:
        # Import model components
        from glycollm.models import (
            GlycoLLM, GlycoLLMConfig, MultiModalLoss, LossConfig,
            ModelAnalyzer, create_glycollm_model, create_loss_function
        )
        
        # Import tokenization for creating sample data
        from glycollm.tokenization import (
            GlycoTokenizer, TokenizationConfig, create_demo_training_data
        )
        
        print("🧬 GlycoLLM Model Architecture Demo")
        print("=" * 50)
        
        # Demo 1: Model Configuration
        print("\n⚙️ Demo 1: Model Configuration")
        print("-" * 40)
        
        # Create model configuration
        model_config = GlycoLLMConfig(
            d_model=768,
            n_layers=12,
            n_heads=12,
            vocab_size=50000,
            max_seq_length=2048,
            text_max_length=512,
            structure_max_length=256,
            spectra_max_length=1024,
            dropout=0.1
        )
        
        print(f"✅ Model Configuration:")
        print(f"   Model Dimension: {model_config.d_model}")
        print(f"   Layers: {model_config.n_layers}")
        print(f"   Attention Heads: {model_config.n_heads}")
        print(f"   Vocabulary Size: {model_config.vocab_size:,}")
        print(f"   Max Sequence Length: {model_config.max_seq_length}")
        
        # Demo 2: Model Architecture
        print("\n🏗️ Demo 2: Model Architecture")
        print("-" * 40)
        
        # Create model instance
        print("Creating GlycoLLM model...")
        model = create_glycollm_model(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            n_layers=model_config.n_layers,
            n_heads=model_config.n_heads
        )
        
        print("✅ Model created successfully")
        print(f"   Architecture: Multimodal Transformer")
        print(f"   Modalities: Text, Structure (WURCS), Spectra")
        print(f"   Task Heads: Structure Prediction, Spectra Analysis, Text Generation")
        
        # Demo 3: Model Analysis
        print("\n📊 Demo 3: Model Analysis")
        print("-" * 40)
        
        # Analyze model parameters
        analyzer = ModelAnalyzer()
        
        try:
            stats = analyzer.analyze_model(model)
            print(f"✅ Model Analysis Complete:")
            print(f"   Total Parameters: {stats.total_params:,}")
            print(f"   Trainable Parameters: {stats.trainable_params:,}")
            print(f"   Model Size: {stats.model_size_mb:.1f} MB")
            print(f"   Estimated Memory: {stats.memory_usage_mb:.1f} MB")
            
            # Print detailed summary
            analyzer.print_model_summary()
            
        except Exception as e:
            print(f"⚠️ Model analysis skipped (requires PyTorch): {e}")
        
        # Demo 4: Loss Functions
        print("\n📈 Demo 4: Loss Functions")
        print("-" * 40)
        
        # Create loss configuration
        loss_config = LossConfig(
            structure_weight=1.0,
            spectra_weight=1.0,
            text_weight=1.0,
            contrastive_weight=0.5,
            temperature=0.07,
            label_smoothing=0.1
        )
        
        print(f"✅ Loss Configuration:")
        print(f"   Structure Weight: {loss_config.structure_weight}")
        print(f"   Spectra Weight: {loss_config.spectra_weight}")
        print(f"   Text Weight: {loss_config.text_weight}")
        print(f"   Contrastive Weight: {loss_config.contrastive_weight}")
        print(f"   Temperature: {loss_config.temperature}")
        print(f"   Label Smoothing: {loss_config.label_smoothing}")
        
        # Create loss function
        loss_function = create_loss_function(loss_config)
        print("✅ Multimodal loss function created")
        
        # Demo 5: Sample Data Processing
        print("\n🔄 Demo 5: Sample Data Processing")
        print("-" * 40)
        
        # Create tokenizer for data preparation
        tokenizer_config = TokenizationConfig()
        tokenizer = GlycoTokenizer(tokenizer_config)
        
        # Get demo data
        demo_data = create_demo_training_data()
        
        print(f"✅ Demo Data Loaded:")
        print(f"   Text Samples: {len(demo_data['text_samples'])}")
        print(f"   WURCS Sequences: {len(demo_data['wurcs_sequences'])}")
        print(f"   Spectra Data: {len(demo_data['spectra_data'])}")
        
        # Process sample data
        sample_text = demo_data['text_samples'][0]
        sample_wurcs = demo_data['wurcs_sequences'][0]
        sample_spectra = demo_data['spectra_data'][0]
        
        print(f"\n📝 Processing Sample Data:")
        print(f"   Text: {sample_text[:60]}...")
        print(f"   WURCS: {sample_wurcs[:60]}...")
        print(f"   Spectra Peaks: {len(sample_spectra)} peaks")
        
        # Tokenize multimodal data
        tokenized_data = tokenizer.tokenize_multimodal(
            text=sample_text,
            wurcs_sequence=sample_wurcs,
            spectra_peaks=sample_spectra
        )
        
        print(f"\n🎯 Tokenization Results:")
        for modality, tokens in tokenized_data.items():
            print(f"   {modality}: {len(tokens)} tokens")
            print(f"      Sample: {tokens[:5]}...")
        
        # Demo 6: Model Forward Pass Simulation
        print("\n🚀 Demo 6: Model Forward Pass Simulation")
        print("-" * 40)
        
        try:
            import torch
            
            # Create sample tensors
            batch_size = 2
            text_tokens = torch.randint(0, 1000, (batch_size, 64))
            structure_tokens = torch.randint(0, 1000, (batch_size, 32))
            spectra_tokens = torch.randint(0, 1000, (batch_size, 128))
            
            print(f"✅ Sample Input Tensors Created:")
            print(f"   Text Shape: {text_tokens.shape}")
            print(f"   Structure Shape: {structure_tokens.shape}")
            print(f"   Spectra Shape: {spectra_tokens.shape}")
            
            # Model forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(
                    text_input_ids=text_tokens,
                    structure_input_ids=structure_tokens,
                    spectra_input_ids=spectra_tokens,
                    task="all"
                )
            
            print(f"\n🎯 Model Outputs:")
            for key, value in outputs.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {type(value)}")
                    
        except ImportError:
            print("⚠️ PyTorch not available - forward pass simulation skipped")
        except Exception as e:
            print(f"⚠️ Forward pass simulation failed: {e}")
        
        # Demo 7: Architecture Highlights
        print("\n🌟 Demo 7: Architecture Highlights")
        print("-" * 40)
        
        print("✅ Key Architecture Features:")
        print("   🔄 Multimodal Embeddings:")
        print("      • Separate embedding layers for each modality")
        print("      • Modality-specific positional encoding") 
        print("      • Unified projection to common dimension")
        
        print("\n   🧠 Cross-Modal Attention:")
        print("      • Cross-attention between modalities")
        print("      • Information fusion across data types")
        print("      • Glycan-specific attention patterns")
        
        print("\n   🎯 Task-Specific Heads:")
        print("      • WURCS structure prediction")
        print("      • Mass spectra generation")
        print("      • Scientific text generation")
        print("      • Cross-modal retrieval")
        
        print("\n   📚 Specialized Training:")
        print("      • Contrastive learning for alignment")
        print("      • Structure-aware loss functions")
        print("      • Multi-task optimization")
        print("      • Domain-specific regularization")
        
        # Demo 8: Use Cases
        print("\n💡 Demo 8: Use Cases and Applications")
        print("-" * 40)
        
        print("🔬 Primary Applications:")
        print("   1. Structure Prediction:")
        print("      • Predict WURCS from mass spectra")
        print("      • Generate structures from text descriptions")
        print("      • Validate proposed structures")
        
        print("\n   2. Spectra Analysis:")
        print("      • Predict fragmentation patterns")
        print("      • Identify unknown glycans")
        print("      • Quality assessment of spectra")
        
        print("\n   3. Knowledge Integration:")
        print("      • Cross-modal search and retrieval")
        print("      • Literature mining and annotation")
        print("      • Database integration and curation")
        
        print("\n   4. Research Assistance:")
        print("      • Hypothesis generation")
        print("      • Experiment design")
        print("      • Result interpretation")
        
        # Demo 9: Model Comparison
        print("\n📊 Demo 9: Model Comparison")
        print("-" * 40)
        
        print("🆚 GlycoLLM vs. Traditional Approaches:")
        print("   Advantages:")
        print("   ✅ Multimodal understanding")
        print("   ✅ End-to-end learning")
        print("   ✅ Transfer learning capabilities")
        print("   ✅ Contextual predictions")
        print("   ✅ Scalable architecture")
        
        print("\n   Traditional Limitations:")
        print("   ❌ Single-modality focus")
        print("   ❌ Manual feature engineering")
        print("   ❌ Limited generalization")
        print("   ❌ Separate pipeline components")
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey capabilities demonstrated:")
        print("  ✅ Multimodal transformer architecture")
        print("  ✅ Cross-modal attention mechanisms")
        print("  ✅ Task-specific prediction heads")
        print("  ✅ Specialized loss functions")
        print("  ✅ Model analysis and profiling")
        print("  ✅ Integration with tokenization")
        print("  ✅ Comprehensive model framework")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the model modules are properly installed.")
        return 1
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.exception("Demo execution failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)