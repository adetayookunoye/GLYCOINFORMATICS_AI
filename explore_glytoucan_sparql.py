#!/usr/bin/env python3
"""
Explore GlyTouCan SPARQL namespaces and predicates
"""

import asyncio
import aiohttp
import json

async def explore_glytoucan_rdf():
    """Explore what namespaces and predicates are actually available"""
    
    # Basic exploration query
    exploration_query = """
    SELECT DISTINCT ?predicate (COUNT(*) as ?count) WHERE {
        ?subject ?predicate ?object .
        FILTER(CONTAINS(STR(?subject), "G00047MO") || CONTAINS(STR(?object), "G00047MO"))
    }
    GROUP BY ?predicate
    ORDER BY DESC(?count)
    LIMIT 20
    """
    
    print("🔍 Exploring GlyTouCan RDF structure...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                "https://ts.glytoucan.org/sparql",
                params={'query': exploration_query, 'format': 'json'},
                timeout=15
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', {}).get('bindings', [])
                    
                    print(f"\n📊 Found {len(results)} predicates used with G00047MO:")
                    for result in results:
                        predicate = result.get('predicate', {}).get('value', 'Unknown')
                        count = result.get('count', {}).get('value', '0')
                        print(f"   {predicate} (used {count} times)")
                        
                else:
                    text = await response.text()
                    print(f"❌ Error: {text[:200]}")
                    
        except Exception as e:
            print(f"❌ Exception: {e}")

    # Try to find any glycan with G00047MO
    simple_query = """
    SELECT ?subject ?predicate ?object WHERE {
        ?subject ?predicate ?object .
        FILTER(CONTAINS(STR(?subject), "G00047MO") || CONTAINS(STR(?object), "G00047MO"))
    }
    LIMIT 10
    """
    
    print(f"\n🔍 Looking for any triples containing G00047MO...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                "https://ts.glytoucan.org/sparql",
                params={'query': simple_query, 'format': 'json'},
                timeout=15
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', {}).get('bindings', [])
                    
                    print(f"\n📋 Found {len(results)} triples:")
                    for i, result in enumerate(results):
                        subject = result.get('subject', {}).get('value', 'Unknown')
                        predicate = result.get('predicate', {}).get('value', 'Unknown')
                        obj = result.get('object', {}).get('value', 'Unknown')
                        print(f"   {i+1}. S: {subject}")
                        print(f"      P: {predicate}")
                        print(f"      O: {obj[:100]}..." if len(obj) > 100 else f"      O: {obj}")
                        print()
                        
                else:
                    text = await response.text()
                    print(f"❌ Error: {text[:200]}")
                    
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(explore_glytoucan_rdf())