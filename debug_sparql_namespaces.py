#!/usr/bin/env python3
"""
Debug GlyTouCan SPARQL namespaces and find working query structure
"""

import asyncio
import aiohttp
import json

async def debug_glytoucan_sparql():
    """Debug what namespaces and predicates actually work in GlyTouCan"""
    
    print("🔍 SPARQL NAMESPACE DEBUGGING")
    print("=" * 50)
    
    # Test queries with different namespace combinations
    test_queries = [
        {
            "name": "Basic namespace exploration",
            "query": """
            SELECT DISTINCT ?predicate (COUNT(*) as ?count) WHERE {
                ?subject ?predicate ?object .
            }
            GROUP BY ?predicate
            ORDER BY DESC(?count)
            LIMIT 10
            """
        },
        {
            "name": "Find any G00047MO references",
            "query": """
            SELECT ?subject ?predicate ?object WHERE {
                {?subject ?predicate "G00047MO"} UNION
                {?subject ?predicate ?object . FILTER(CONTAINS(STR(?object), "G00047MO"))} UNION
                {"G00047MO" ?predicate ?object} UNION
                {?subject ?predicate ?object . FILTER(CONTAINS(STR(?subject), "G00047MO"))}
            }
            LIMIT 20
            """
        },
        {
            "name": "Alternative namespace test 1",
            "query": """
            PREFIX glycan: <https://glytoucan.org/glyco/owl/glytoucan#>
            SELECT ?subject ?predicate ?object WHERE {
                ?subject glycan:has_primary_id "G00047MO" .
                ?subject ?predicate ?object .
            }
            LIMIT 10
            """
        },
        {
            "name": "Alternative namespace test 2", 
            "query": """
            PREFIX gtc: <http://www.glytoucan.org/glyco/owl/glytoucan#>
            SELECT ?subject ?predicate ?object WHERE {
                ?subject gtc:has_primary_id "G00047MO" .
                ?subject ?predicate ?object .
            }
            LIMIT 10
            """
        },
        {
            "name": "Simple URI-based search",
            "query": """
            SELECT ?subject ?predicate ?object WHERE {
                ?subject ?predicate ?object .
                FILTER(CONTAINS(STR(?subject), "G00047MO") || CONTAINS(STR(?object), "G00047MO"))
            }
            LIMIT 10
            """
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, test in enumerate(test_queries, 1):
            print(f"\n🔬 Test {i}: {test['name']}")
            print("-" * 30)
            
            try:
                async with session.get(
                    "https://ts.glytoucan.org/sparql",
                    params={'query': test['query'], 'format': 'json'},
                    timeout=20
                ) as response:
                    
                    print(f"Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', {}).get('bindings', [])
                        
                        print(f"Results found: {len(results)}")
                        
                        if results:
                            print("Sample results:")
                            for j, result in enumerate(results[:3]):
                                print(f"  {j+1}. Subject: {result.get('subject', {}).get('value', 'N/A')[:60]}...")
                                print(f"     Predicate: {result.get('predicate', {}).get('value', 'N/A')}")
                                print(f"     Object: {result.get('object', {}).get('value', 'N/A')[:60]}...")
                                print()
                        else:
                            print("  No results returned")
                            print(f"  Response keys: {list(data.keys())}")
                    else:
                        error_text = await response.text()
                        print(f"Error: {error_text[:200]}...")
                        
            except asyncio.TimeoutError:
                print("❌ Request timed out")
            except Exception as e:
                print(f"❌ Exception: {e}")
            
            await asyncio.sleep(2)  # Rate limiting
    
    print(f"\n🎯 DEBUGGING COMPLETE!")
    print("Look for working patterns to fix our SPARQL queries")

if __name__ == "__main__":
    asyncio.run(debug_glytoucan_sparql())