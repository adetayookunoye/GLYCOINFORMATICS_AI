#!/usr/bin/env python3
"""
Test GlycoWorks + Candycrunch Integration
=========================================

Quick validation test to ensure the complete GlycoLLM experimental training
pipeline works with both GlycoWorks training data and candycrunch validation.

Author: Glycoinformatics AI Team
Date: November 5, 2025
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from glycollm.data.glycoworks_processor import GlycoWorksProcessor
from glycollm.data.candycrunch_validator import CandycrunchValidator


def test_glycoworks_processor():
    """Test GlycoWorks processor functionality."""
    print("🧪 Testing GlycoWorks Processor...")

    try:
        # Initialize processor
        processor = GlycoWorksProcessor(
            data_dir="data/raw/dataset/glycoworks_glycan_data",
            output_dir="data/processed/test_glycoworks"
        )

        # Test processing (limited to avoid long runtime)
        print("   📊 Processing GlycoWorks data...")
        dataset = processor.process_all_csv_files()

        print(f"   ✅ Loaded {len(dataset.samples)} experimental samples")
        print(f"   ✅ Found {len(dataset.glycans)} unique glycans")

        # Test multimodal sample creation
        print("   🎯 Creating multimodal samples...")
        samples = processor.create_multimodal_samples(['abundance_prediction'])

        print(f"   ✅ Created {len(samples)} multimodal samples")

        return True

    except Exception as e:
        print(f"   ❌ GlycoWorks processor test failed: {e}")
        return False


def test_candycrunch_validator():
    """Test candycrunch validator functionality."""
    print("🧪 Testing Candycrunch Validator...")

    try:
        # Initialize validator
        validator = CandycrunchValidator(
            data_dir="data/raw/dataset/candycrunch_glycan_data"
        )

        # Test knowledge base loading
        print("   📚 Loading validation knowledge base...")
        kb = validator.load_knowledge_base()

        print(f"   ✅ Loaded {len(kb.entries)} validation entries")
        print(f"   ✅ GlyTouCan coverage: {kb.glytoucan_coverage:.1%}")

        # Test validation
        print("   🔍 Testing structure validation...")
        test_structures = [
            kb.entries[0].glycan_sequence if kb.entries else "test_glycan",
            "unknown_structure"
        ]

        results = validator.validate_predictions(test_structures)
        print(f"   ✅ Validation results: {results['summary_statistics']['exact_match_rate']:.1%} exact matches")

        return True

    except Exception as e:
        print(f"   ❌ Candycrunch validator test failed: {e}")
        return False


def test_integration():
    """Test the integration between processor and validator."""
    print("🧪 Testing GlycoWorks + Candycrunch Integration...")

    try:
        # Initialize both components
        processor = GlycoWorksProcessor(
            data_dir="data/raw/dataset/glycoworks_glycan_data",
            output_dir="data/processed/test_integration"
        )

        validator = CandycrunchValidator(
            data_dir="data/raw/dataset/candycrunch_glycan_data"
        )

        # Load both knowledge bases
        print("   🔄 Loading integrated knowledge bases...")
        glycoworks_data = processor.process_all_csv_files()
        validator_kb = validator.load_knowledge_base()

        print(f"   ✅ GlycoWorks: {len(glycoworks_data.samples)} experimental samples")
        print(f"   ✅ Candycrunch: {len(validator_kb.entries)} validation structures")

        # Test cross-validation
        print("   🔗 Testing cross-validation...")

        # Sample some GlycoWorks glycans for validation
        sample_glycans = list(glycoworks_data.glycans)[:10]  # First 10 glycans

        validation_results = validator.validate_predictions(
            sample_glycans,
            task_type='structure_prediction'
        )

        print(f"   ✅ Cross-validation: {validation_results['summary_statistics']['known_structure_rate']:.1%} known structures")

        # Test statistics
        glycoworks_stats = glycoworks_data.statistics
        validator_stats = validator_kb.get_validation_statistics()

        print("   📊 Integration Statistics:")
        print(f"      GlycoWorks measurements: {glycoworks_stats['total_measurements']:,}")
        print(f"      Candycrunch structures: {validator_stats['total_entries']:,}")
        print(f"      Combined coverage: {(len(glycoworks_data.glycans) + len(validator_kb.entries)):,} total structures")

        return True

    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🚀 GlycoLLM Experimental Training Integration Test")
    print("=" * 60)

    tests = [
        ("GlycoWorks Processor", test_glycoworks_processor),
        ("Candycrunch Validator", test_candycrunch_validator),
        ("Full Integration", test_integration)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")

    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not success:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All integration tests PASSED!")
        print("   📈 Your GlycoLLM experimental training pipeline is ready!")
        print("   🚀 You can now run: python scripts/integrate_glycoworks_training.py --all")
    else:
        print("❌ Some tests FAILED!")
        print("   🔧 Please check the error messages above and fix any issues.")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)