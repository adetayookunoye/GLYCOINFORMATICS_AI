#!/usr/bin/env python3
"""
Test working SPARQL queries based on namespace debugging results
"""

import asyncio
import aiohttp
import json

async def test_working_sparql_queries():
    """Test SPARQL queries with discovered working namespaces"""
    
    print("🧪 TESTING WORKING SPARQL QUERIES")
    print("=" * 50)
    
    # Working namespace: http://www.glytoucan.org/glyco/owl/glytoucan#
    # Subject pattern: http://rdf.glycoinfo.org/glycan/{ID}
    
    test_ids = ["G00047MO", "G00002CF", "G00012MO"]
    
    for glytoucan_id in test_ids:
        print(f"\n🔬 Testing {glytoucan_id}")
        print("-" * 30)
        
        # Working SPARQL query based on discovered patterns
        sparql_query = f"""
        PREFIX gtc: <http://www.glytoucan.org/glyco/owl/glytoucan#>
        PREFIX wurcs: <http://www.glycoinfo.org/glyco/owl/wurcs#>
        PREFIX glyco: <http://purl.jp/bio/12/glyco/glycan#>
        PREFIX mass: <http://www.glycoinfo.org/glyco/owl/relation#>
        PREFIX glycoinfo: <http://www.glycoinfo.org/glyco/owl/relation#>
        
        SELECT ?wurcs_seq ?mass ?iupac ?composition WHERE {{
            <http://rdf.glycoinfo.org/glycan/{glytoucan_id}> ?wurcs_pred ?wurcs_seq .
            OPTIONAL {{ <http://rdf.glycoinfo.org/glycan/{glytoucan_id}> glycoinfo:has_monoisotopic_mass ?mass }}
            OPTIONAL {{ <http://rdf.glycoinfo.org/glycan/{glytoucan_id}> glycoinfo:has_iupac ?iupac }}
            OPTIONAL {{ <http://rdf.glycoinfo.org/glycan/{glytoucan_id}> glycoinfo:has_composition ?composition }}
            FILTER(CONTAINS(STR(?wurcs_pred), "wurcs") || CONTAINS(STR(?wurcs_pred), "sequence"))
        }}
        """
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    "https://ts.glytoucan.org/sparql",
                    params={'query': sparql_query, 'format': 'json'},
                    timeout=15
                ) as response:
                    
                    print(f"Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', {}).get('bindings', [])
                        
                        print(f"Results found: {len(results)}")
                        
                        if results:
                            result = results[0]
                            wurcs = result.get('wurcs_seq', {}).get('value')
                            mass = result.get('mass', {}).get('value')
                            iupac = result.get('iupac', {}).get('value')
                            composition = result.get('composition', {}).get('value')
                            
                            print(f"✅ SPARQL SUCCESS for {glytoucan_id}:")
                            print(f"   WURCS: {wurcs[:50] if wurcs else 'None'}...")
                            print(f"   Mass: {mass}")
                            print(f"   IUPAC: {iupac[:50] if iupac else 'None'}")
                            print(f"   Composition: {composition}")
                        else:
                            print(f"❌ No results for {glytoucan_id}")
                            
                            # Try a simpler query to see what properties exist
                            simple_query = f"""
                            SELECT ?predicate ?object WHERE {{
                                <http://rdf.glycoinfo.org/glycan/{glytoucan_id}> ?predicate ?object .
                            }}
                            LIMIT 10
                            """
                            
                            async with session.get(
                                "https://ts.glytoucan.org/sparql",
                                params={'query': simple_query, 'format': 'json'},
                                timeout=10
                            ) as simple_response:
                                if simple_response.status == 200:
                                    simple_data = await simple_response.json()
                                    simple_results = simple_data.get('results', {}).get('bindings', [])
                                    
                                    print(f"   Available properties for {glytoucan_id}:")
                                    for prop in simple_results[:5]:
                                        pred = prop.get('predicate', {}).get('value', '')
                                        obj = prop.get('object', {}).get('value', '')[:50]
                                        print(f"     {pred.split('#')[-1] if '#' in pred else pred.split('/')[-1]}: {obj}...")
                    else:
                        error_text = await response.text()
                        print(f"❌ Error: {error_text[:100]}...")
                        
            except Exception as e:
                print(f"❌ Exception: {e}")
            
            await asyncio.sleep(2)  # Rate limiting

if __name__ == "__main__":
    asyncio.run(test_working_sparql_queries())