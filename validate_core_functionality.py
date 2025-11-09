#!/usr/bin/env python3
"""
Quick validation that our core GlyGen integration still works after protein enhancements
"""

import asyncio
import json
from enhance_real_dataset import EnhancedGlycanDataEnricher

async def validate_core_functionality():
    """Test that core GlyGen integration still works perfectly"""
    
    # Test with IDs we know work from our earlier successful test
    test_samples = [
        {"sample_id": "test_1", "glytoucan_id": "G00002CF", "wurcs_sequence": None},
        {"sample_id": "test_2", "glytoucan_id": "G00012MO", "wurcs_sequence": None},
        {"sample_id": "test_3", "glytoucan_id": "G00016MO", "wurcs_sequence": None},
    ]
    
    print("🔧 Validating Core GlyGen Integration")
    print("=" * 40)
    
    async with EnhancedGlycanDataEnricher(rate_limit_delay=1.0) as enricher:
        
        # Temporarily disable external APIs by monkey-patching
        async def mock_uniprot(*args, **kwargs):
            await asyncio.sleep(0.1)  # Small delay to simulate
            return None
            
        async def mock_pdb(*args, **kwargs):
            await asyncio.sleep(0.1)  # Small delay to simulate
            return None
            
        # Replace methods with faster mocks
        enricher.fetch_uniprot_glycosylation = mock_uniprot
        enricher.fetch_pdb_glycan_structures = mock_pdb
        
        success_count = 0
        
        for i, sample in enumerate(test_samples, 1):
            glytoucan_id = sample['glytoucan_id']
            print(f"\n🔬 Test {i}/3: {glytoucan_id}")
            
            try:
                enhanced = await enricher.enhance_sample(sample)
                
                # Check structural data
                wurcs = enhanced.get('wurcs_sequence')
                iupac = enhanced.get('iupac_name')
                data_sources = enhanced.get('data_sources', {})
                
                if wurcs:
                    print(f"   ✅ WURCS: {wurcs[:30]}...")
                    print(f"   ✅ IUPAC: {iupac[:30] if iupac else 'None'}")
                    print(f"   ✅ Source: {data_sources.get('structure', 'Unknown')}")
                    success_count += 1
                else:
                    print(f"   ❌ No structural data retrieved")
                    
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
        
        success_rate = (success_count / len(test_samples)) * 100
        print(f"\n📊 CORE VALIDATION RESULTS:")
        print(f"✅ Success Rate: {success_count}/{len(test_samples)} ({success_rate:.1f}%)")
        
        return success_rate >= 100  # Expect 100% success for known working IDs

if __name__ == "__main__":
    success = asyncio.run(validate_core_functionality())
    if success:
        print("🎉 CORE FUNCTIONALITY VALIDATED!")
    else:
        print("⚠️ CORE FUNCTIONALITY ISSUES DETECTED")