#!/usr/bin/env python3
"""
Demo script for glycoinformatics tokenizer.

This script demonstrates the capabilities of the specialized tokenizer
for WURCS notation, mass spectra, and glycan-related text.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run tokenizer demonstration"""
    
    try:
        # Import tokenization components
        from glycollm.tokenization import (
            GlycoTokenizer, TokenizationConfig, WURCSValidator,
            SpectrumAnalyzer, GlycanTextProcessor, TokenizerTester,
            create_demo_training_data
        )
        
        print("🧬 Glycoinformatics Tokenizer Demo")
        print("=" * 50)
        
        # Create tokenizer with default configuration
        print("\n📋 Initializing tokenizer...")
        config = TokenizationConfig()
        tokenizer = GlycoTokenizer(config)
        
        print(f"✅ Tokenizer initialized with vocabulary size: {tokenizer.get_vocab_size()}")
        print(f"   Special tokens: {len(config.get_all_special_tokens())}")
        
        # Demo 1: WURCS tokenization
        print("\n🔗 Demo 1: WURCS Sequence Tokenization")
        print("-" * 40)
        
        sample_wurcs = "WURCS=2.0/3,3,2/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1221m-1a_1-5]/1-2-3/a4-b1_b4-c1"
        print(f"Input WURCS: {sample_wurcs}")
        
        wurcs_tokens = tokenizer.wurcs_tokenizer.tokenize_wurcs(sample_wurcs)
        print(f"Tokenized: {wurcs_tokens[:10]}...")  # Show first 10 tokens
        print(f"Token count: {len(wurcs_tokens)}")
        
        # Demo 2: Mass spectrum tokenization
        print("\n📊 Demo 2: Mass Spectrum Tokenization")
        print("-" * 40)
        
        sample_peaks = [(163.060, 100.0), (204.087, 85.5), (366.139, 45.2), (512.197, 22.1)]
        print(f"Input peaks: {sample_peaks}")
        
        spectrum_tokens = tokenizer.spectra_tokenizer.tokenize_spectrum(sample_peaks, precursor_mz=674.250)
        print(f"Tokenized: {spectrum_tokens}")
        print(f"Token count: {len(spectrum_tokens)}")
        
        # Demo 3: Scientific text tokenization
        print("\n📝 Demo 3: Scientific Text Tokenization")
        print("-" * 40)
        
        sample_text = "N-linked glycans contain GlcNAc residues with α1-6 linkages analyzed by MALDI-TOF MS."
        print(f"Input text: {sample_text}")
        
        processed_text = tokenizer.text_tokenizer.preprocess_text(sample_text)
        print(f"Preprocessed: {processed_text}")
        
        # Extract entities
        entities = tokenizer.text_tokenizer.extract_entities(sample_text)
        print(f"Extracted entities: {entities}")
        
        # Demo 4: Multimodal tokenization
        print("\n🔄 Demo 4: Multimodal Tokenization")
        print("-" * 40)
        
        multimodal_result = tokenizer.tokenize_multimodal(
            text=sample_text,
            wurcs_sequence=sample_wurcs,
            spectra_peaks=sample_peaks,
            precursor_mz=674.250
        )
        
        print("Multimodal token IDs:")
        for modality, token_ids in multimodal_result.items():
            print(f"  {modality}: {len(token_ids)} tokens")
            print(f"    Sample IDs: {token_ids[:5]}...")
            
        # Demo 5: Validation and analysis
        print("\n🔍 Demo 5: Validation and Analysis")
        print("-" * 40)
        
        # WURCS validation
        validator = WURCSValidator()
        parse_result = validator.validate_wurcs(sample_wurcs)
        print(f"WURCS validation: {'✅ Valid' if parse_result.is_valid else '❌ Invalid'}")
        if parse_result.is_valid:
            mono_counts = validator.identify_monosaccharides(parse_result.residues)
            print(f"Monosaccharides: {mono_counts}")
            
        # Spectrum analysis
        analyzer = SpectrumAnalyzer()
        spectrum_analysis = analyzer.analyze_spectrum(sample_peaks)
        print(f"Spectrum analysis:")
        print(f"  Total peaks: {spectrum_analysis.total_peaks}")
        print(f"  Base peak: {spectrum_analysis.base_peak_mz:.3f} m/z")
        print(f"  Mass range: {spectrum_analysis.mass_range[0]:.1f}-{spectrum_analysis.mass_range[1]:.1f}")
        print(f"  Identified fragments: {spectrum_analysis.identified_fragments}")
        
        # Demo 6: Comprehensive testing
        print("\n🧪 Demo 6: Comprehensive Testing")
        print("-" * 40)
        
        # Create demo data
        demo_data = create_demo_training_data()
        
        # Test with multiple samples
        tester = TokenizerTester()
        
        # Test WURCS samples
        wurcs_results = tester.test_wurcs_samples(demo_data['wurcs_sequences'])
        print(f"WURCS testing: {wurcs_results['valid_samples']}/{wurcs_results['total_samples']} valid")
        print(f"Monosaccharide distribution: {wurcs_results['monosaccharide_distribution']}")
        
        # Test spectra samples
        spectrum_results = tester.test_spectrum_processing(demo_data['spectra_data'])
        print(f"Spectrum testing: {spectrum_results['total_spectra']} spectra processed")
        print(f"Average peaks: {spectrum_results['peak_statistics']['avg_peaks']:.1f}")
        print(f"Fragment identification: {spectrum_results['fragment_identification']}")
        
        # Generate test report
        report = tester.generate_test_report(
            wurcs_results=wurcs_results,
            spectrum_results=spectrum_results,
            text_samples=demo_data['text_samples']
        )
        
        print("\n📈 Test Report")
        print("-" * 40)
        print(report[:500] + "..." if len(report) > 500 else report)
        
        # Demo 7: Save and load tokenizer
        print("\n💾 Demo 7: Save and Load Tokenizer")
        print("-" * 40)
        
        output_dir = "demo_tokenizer"
        print(f"Saving tokenizer to {output_dir}...")
        
        try:
            tokenizer.save_pretrained(output_dir)
            print("✅ Tokenizer saved successfully")
            
            # Load tokenizer
            loaded_tokenizer = GlycoTokenizer.from_pretrained(output_dir)
            print("✅ Tokenizer loaded successfully")
            
            # Verify functionality
            test_result = loaded_tokenizer.tokenize_multimodal(text="Test glycan structure")
            print(f"✅ Loaded tokenizer working: {len(test_result['text'])} tokens")
            
        except Exception as e:
            print(f"⚠️  Save/load demo skipped: {e}")
            
        print("\n🎉 Demo completed successfully!")
        print("\nKey capabilities demonstrated:")
        print("  ✅ WURCS sequence tokenization")
        print("  ✅ Mass spectrum tokenization") 
        print("  ✅ Scientific text processing")
        print("  ✅ Multimodal data integration")
        print("  ✅ Validation and quality analysis")
        print("  ✅ Comprehensive testing framework")
        print("  ✅ Serialization and persistence")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the tokenization module is properly installed.")
        return 1
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.exception("Demo execution failed")
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)