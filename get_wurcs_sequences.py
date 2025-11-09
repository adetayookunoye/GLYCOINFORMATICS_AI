#!/usr/bin/env python3
"""
Build working SPARQL query that retrieves actual WURCS sequences
"""

import asyncio
import aiohttp
import json

async def get_actual_wurcs_sequences():
    """Get actual WURCS sequences from the URIs we discovered"""
    
    print("🎯 RETRIEVING ACTUAL WURCS SEQUENCES")
    print("=" * 50)
    
    test_ids = ["G00047MO", "G00002CF"]
    
    for glytoucan_id in test_ids:
        print(f"\n🔬 Testing {glytoucan_id}")
        print("-" * 30)
        
        # First, get the WURCS URI
        uri_query = f"""
        SELECT ?wurcs_uri WHERE {{
            <http://rdf.glycoinfo.org/glycan/{glytoucan_id}> ?pred ?wurcs_uri .
            FILTER(CONTAINS(STR(?wurcs_uri), "wurcs") && CONTAINS(STR(?wurcs_uri), "2.0"))
        }}
        """
        
        async with aiohttp.ClientSession() as session:
            try:
                # Get WURCS URI
                async with session.get(
                    "https://ts.glytoucan.org/sparql",
                    params={'query': uri_query, 'format': 'json'},
                    timeout=10
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', {}).get('bindings', [])
                        
                        if results:
                            wurcs_uri = results[0].get('wurcs_uri', {}).get('value')
                            print(f"Found WURCS URI: {wurcs_uri}")
                            
                            # Now get the actual WURCS sequence from that URI
                            sequence_query = f"""
                            PREFIX wurcs: <http://www.glycoinfo.org/glyco/owl/wurcs#>
                            
                            SELECT ?sequence WHERE {{
                                <{wurcs_uri}> wurcs:has_sequence ?sequence .
                            }}
                            """
                            
                            async with session.get(
                                "https://ts.glytoucan.org/sparql",
                                params={'query': sequence_query, 'format': 'json'},
                                timeout=10
                            ) as seq_response:
                                
                                if seq_response.status == 200:
                                    seq_data = await seq_response.json()
                                    seq_results = seq_data.get('results', {}).get('bindings', [])
                                    
                                    if seq_results:
                                        wurcs_seq = seq_results[0].get('sequence', {}).get('value')
                                        print(f"✅ WURCS SEQUENCE FOUND: {wurcs_seq}")
                                    else:
                                        print("❌ No WURCS sequence found")
                                        
                                        # Try alternative property names
                                        alt_query = f"""
                                        SELECT ?prop ?val WHERE {{
                                            <{wurcs_uri}> ?prop ?val .
                                        }}
                                        """
                                        
                                        async with session.get(
                                            "https://ts.glytoucan.org/sparql",
                                            params={'query': alt_query, 'format': 'json'},
                                            timeout=10
                                        ) as alt_response:
                                            if alt_response.status == 200:
                                                alt_data = await alt_response.json()
                                                alt_results = alt_data.get('results', {}).get('bindings', [])
                                                
                                                print("Available properties on WURCS URI:")
                                                for prop in alt_results[:5]:
                                                    p = prop.get('prop', {}).get('value', '')
                                                    v = prop.get('val', {}).get('value', '')
                                                    print(f"  {p.split('#')[-1]}: {v[:50]}")
                                else:
                                    print(f"❌ Sequence query failed: {seq_response.status}")
                        else:
                            print(f"❌ No WURCS URI found for {glytoucan_id}")
                    else:
                        print(f"❌ URI query failed: {response.status}")
                        
            except Exception as e:
                print(f"❌ Exception: {e}")
            
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(get_actual_wurcs_sequences())