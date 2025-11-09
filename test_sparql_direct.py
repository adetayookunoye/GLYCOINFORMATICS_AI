#!/usr/bin/env python3
"""
Test enhanced SPARQL query directly
"""

import asyncio
import aiohttp
import json

async def test_enhanced_sparql():
    """Test our enhanced SPARQL query directly"""
    
    test_ids = ["G00047MO", "G01415VY", "G00000CV"]  # Mix of working and problematic IDs
    
    for glytoucan_id in test_ids:
        print(f"\n🔬 Testing SPARQL for {glytoucan_id}")
        
        # Our enhanced SPARQL query
        sparql_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
        PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>
        PREFIX wurcs: <http://www.glycoinfo.org/glyco/owl/relation#>
        
        SELECT ?wurcs ?mass ?composition ?iupac WHERE {{
            ?saccharide rdf:type glycan:saccharide .
            ?saccharide glytoucan:has_primary_id "{glytoucan_id}" .
            OPTIONAL {{ ?saccharide glycan:has_wurcs ?wurcs }}
            OPTIONAL {{ ?saccharide glycan:has_monoisotopic_mass ?mass }}
            OPTIONAL {{ ?saccharide glycan:has_composition ?composition }}
            OPTIONAL {{ ?saccharide glycan:has_iupac ?iupac }}
        }}
        """
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    "https://ts.glytoucan.org/sparql",
                    params={'query': sparql_query, 'format': 'json'},
                    timeout=15
                ) as response:
                    print(f"   Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', {}).get('bindings', [])
                        
                        if results:
                            result = results[0]
                            wurcs = result.get('wurcs', {}).get('value')
                            mass = result.get('mass', {}).get('value')
                            composition = result.get('composition', {}).get('value')
                            iupac = result.get('iupac', {}).get('value')
                            
                            print(f"   ✅ SPARQL Results:")
                            print(f"      WURCS: {bool(wurcs)} - {wurcs[:50] if wurcs else 'None'}")
                            print(f"      Mass: {mass}")
                            print(f"      IUPAC: {bool(iupac)} - {iupac[:50] if iupac else 'None'}")
                            print(f"      Composition: {composition}")
                        else:
                            print(f"   ❌ No SPARQL results")
                            print(f"   Response keys: {list(data.keys())}")
                    else:
                        text = await response.text()
                        print(f"   ❌ Error: {text[:200]}")
                        
            except Exception as e:
                print(f"   ❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_sparql())