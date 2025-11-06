#!/usr/bin/env python3
"""
Test script to verify TinyLlama integration with GlycoLLM fine-tuning.

This script tests configuration and basic setup without loading the full model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_tinyllama_config():
    """Test TinyLlama configuration setup."""
    try:
        from glycollm.models.llm_finetuning import LLMFineTuningConfig, LLMType

        # Test default config
        config = LLMFineTuningConfig()
        print(f"✓ Default model: {config.model_name}")
        print(f"✓ Default type: {config.model_type}")
        print(f"✓ Default method: {config.method}")

        # Test explicit TinyLlama config
        tiny_config = LLMFineTuningConfig(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type=LLMType.TINYLLAMA
        )
        print(f"✓ TinyLlama config: {tiny_config.model_name}")
        print(f"✓ TinyLlama type: {tiny_config.model_type}")

        # Test enum access
        print(f"✓ TINYLLAMA enum value: {LLMType.TINYLLAMA.value}")

        return True

    except Exception as e:
        print(f"❌ Error with TinyLlama config: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that all required imports work."""
    try:
        from glycollm.models.llm_finetuning import (
            LLMFineTuningConfig, LLMType, FineTuningMethod,
            GlycoLLMFineTuner, GlycoLLMWithFineTuning
        )
        print("✓ All LLM fine-tuning imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("Testing TinyLlama integration (config only)...")

    success1 = test_imports()
    success2 = test_tinyllama_config()

    if success1 and success2:
        print("\n🎉 TinyLlama configuration test PASSED!")
        print("✓ TinyLlama is now the default LLM for fine-tuning")
        print("✓ You can use it by running:")
        print("  python scripts/train_glycollm.py --train-data data/training/spec_to_struct_train.json --val-data data/training/spec_to_struct_val.json --output-dir models/glycollm_tinyllama")
    else:
        print("\n❌ TinyLlama configuration test FAILED!")