#!/usr/bin/env python3
"""
Test with a known working GlyTouCan ID
"""

import asyncio
import aiohttp
import json

async def test_working_glycan():
    """Test with G00047MO which we know works"""
    
    test_id = "G00047MO"
    glygen_url = f"https://api.glygen.org/glycan/detail/{test_id}"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(glygen_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    print(f"✅ GlyGen Success for {test_id}")
                    print(f"WURCS: {data.get('wurcs', 'None')}")
                    print(f"IUPAC: {data.get('iupac', 'None')}")
                    print(f"Mass: {data.get('mass', 'None')}")
                    print(f"GlycoCT: {data.get('glycoct', 'None')}")
                    
                    species = data.get('species', [])
                    if species:
                        print(f"Species: {[s.get('name') for s in species[:3]]}")
                    
                    return True
                else:
                    print(f"❌ Status: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

async def main():
    await test_working_glycan()

if __name__ == "__main__":
    asyncio.run(main())