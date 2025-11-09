#!/usr/bin/env python3
"""
Test enhance pipeline with known working glycan IDs
"""

import asyncio
import json
from enhance_real_dataset import EnhancedGlycanDataEnricher

async def test_known_working():
    """Test with glycan IDs we know work"""
    
    # Create test samples with known working IDs from our coverage test
    test_samples = [
        {
            "sample_id": "test_1",
            "glytoucan_id": "G00002CF",  # This worked in our coverage test
            "wurcs_sequence": None,
            "molecular_mass": None,
            "iupac_name": None
        },
        {
            "sample_id": "test_2", 
            "glytoucan_id": "G00047MO",  # This is our known working one
            "wurcs_sequence": None,
            "molecular_mass": None,
            "iupac_name": None
        }
    ]
    
    async with EnhancedGlycanDataEnricher() as enricher:
        for sample in test_samples:
            print(f"\n🔄 Testing {sample['glytoucan_id']}")
            
            try:
                enhanced = await enricher.enhance_sample(sample)
                
                print(f"✅ Enhanced sample:")
                print(f"   WURCS: {enhanced.get('wurcs_sequence', 'None')[:50] if enhanced.get('wurcs_sequence') else 'None'}")
                print(f"   IUPAC: {enhanced.get('iupac_name', 'None')[:50] if enhanced.get('iupac_name') else 'None'}")
                print(f"   Mass: {enhanced.get('molecular_mass', 'None')}")
                
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_known_working())