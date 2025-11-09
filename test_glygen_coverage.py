#!/usr/bin/env python3
"""
Sample GlyTouCan IDs from our dataset and test GlyGen coverage
"""

import json
import asyncio
import aiohttp
from pathlib import Path

async def test_glygen_coverage():
    """Test what percentage of our GlyTouCan IDs exist in GlyGen"""
    
    dataset_path = Path("data/processed/ultimate_real_training/train_dataset.json")
    
    # Load dataset and get first 10 samples
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    sample_ids = []
    for sample in dataset[:50]:  # First 50 samples
        glytoucan_id = sample.get('glytoucan_id')
        if glytoucan_id:
            sample_ids.append(glytoucan_id)
    
    print(f"Testing {len(sample_ids)} GlyTouCan IDs:")
    for id in sample_ids:
        print(f"  {id}")
    print()
    
    # Test each ID
    success_count = 0
    async with aiohttp.ClientSession() as session:
        for glytoucan_id in sample_ids:
            glygen_url = f"https://api.glygen.org/glycan/detail/{glytoucan_id}"
            
            try:
                async with session.get(glygen_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'wurcs' in data and data['wurcs']:
                            print(f"✅ {glytoucan_id}: Found with WURCS")
                            success_count += 1
                        else:
                            print(f"⚠️ {glytoucan_id}: Found but no WURCS")
                    else:
                        print(f"❌ {glytoucan_id}: Not found (status {response.status})")
                        
                # Rate limit
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ {glytoucan_id}: Error - {e}")
    
    coverage = (success_count / len(sample_ids)) * 100 if sample_ids else 0
    print(f"\n📊 GlyGen Coverage: {success_count}/{len(sample_ids)} = {coverage:.1f}%")
    
    return coverage

if __name__ == "__main__":
    asyncio.run(test_glygen_coverage())