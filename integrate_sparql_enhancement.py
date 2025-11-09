#!/usr/bin/env python3
"""
Integrate working SPARQL patterns into enhanced real dataset pipeline
"""

import asyncio
import aiohttp
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

async def enhanced_sparql_query(session, glytoucan_id):
    """Get comprehensive glycan data from GlyTouCan SPARQL endpoint"""
    
    # Working query that gets WURCS sequences and related data
    sparql_query = f"""
    PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
    PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT DISTINCT ?wurcs_uri ?wurcs_sequence ?label ?mass ?formula WHERE {{
        # Get the main glycan
        ?glycan_uri rdf:type glytoucan:glycan .
        ?glycan_uri glytoucan:has_id "{glytoucan_id}" .
        
        # Get WURCS sequence
        OPTIONAL {{
            ?glycan_uri ?pred ?wurcs_uri .
            FILTER(CONTAINS(STR(?wurcs_uri), "wurcs") && CONTAINS(STR(?wurcs_uri), "2.0"))
            ?wurcs_uri glycan:has_sequence ?wurcs_sequence .
        }}
        
        # Get additional properties
        OPTIONAL {{ ?glycan_uri rdfs:label ?label . }}
        OPTIONAL {{ ?glycan_uri glycan:has_glycan_mass ?mass . }}
        OPTIONAL {{ ?glycan_uri glycan:has_molecular_formula ?formula . }}
    }}
    """
    
    try:
        async with session.get(
            "https://ts.glytoucan.org/sparql",
            params={'query': sparql_query, 'format': 'json'},
            timeout=15
        ) as response:
            
            if response.status == 200:
                data = await response.json()
                results = data.get('results', {}).get('bindings', [])
                
                if results:
                    result = results[0]
                    return {
                        'glytoucan_id': glytoucan_id,
                        'wurcs_sequence': result.get('wurcs_sequence', {}).get('value'),
                        'label': result.get('label', {}).get('value'),
                        'mass': result.get('mass', {}).get('value'),
                        'formula': result.get('formula', {}).get('value'),
                        'sparql_success': True
                    }
                else:
                    # Fallback: direct property query
                    fallback_query = f"""
                    SELECT ?prop ?val WHERE {{
                        <http://rdf.glycoinfo.org/glycan/{glytoucan_id}/wurcs/2.0> ?prop ?val .
                        FILTER(?prop = <http://purl.jp/bio/12/glyco/glycan#has_sequence>)
                    }}
                    """
                    
                    async with session.get(
                        "https://ts.glytoucan.org/sparql",
                        params={'query': fallback_query, 'format': 'json'},
                        timeout=10
                    ) as fallback_response:
                        
                        if fallback_response.status == 200:
                            fallback_data = await fallback_response.json()
                            fallback_results = fallback_data.get('results', {}).get('bindings', [])
                            
                            if fallback_results:
                                wurcs_seq = fallback_results[0].get('val', {}).get('value')
                                return {
                                    'glytoucan_id': glytoucan_id,
                                    'wurcs_sequence': wurcs_seq,
                                    'sparql_success': True
                                }
                    
                    return {'glytoucan_id': glytoucan_id, 'sparql_success': False}
            else:
                return {'glytoucan_id': glytoucan_id, 'sparql_success': False, 'error': f"HTTP {response.status}"}
                
    except Exception as e:
        return {'glytoucan_id': glytoucan_id, 'sparql_success': False, 'error': str(e)}

async def test_enhanced_sparql():
    """Test the enhanced SPARQL integration"""
    
    print("🚀 TESTING ENHANCED SPARQL INTEGRATION")
    print("=" * 50)
    
    test_ids = ["G00047MO", "G00002CF", "G00012MO", "G00055FR", "G00000CV"]
    
    async with aiohttp.ClientSession() as session:
        tasks = [enhanced_sparql_query(session, gid) for gid in test_ids]
        results = await asyncio.gather(*tasks)
        
        successful = 0
        for result in results:
            gid = result.get('glytoucan_id')
            success = result.get('sparql_success', False)
            wurcs = result.get('wurcs_sequence', 'N/A')
            
            if success and wurcs and wurcs != 'N/A':
                print(f"✅ {gid}: {wurcs[:50]}...")
                successful += 1
            else:
                print(f"❌ {gid}: Failed")
        
        print(f"\n📊 SUCCESS RATE: {successful}/{len(test_ids)} ({successful/len(test_ids)*100:.1f}%)")
        
        return results

async def integrate_sparql_into_enhancement():
    """Integrate SPARQL into the main enhancement pipeline"""
    
    print("\n🔧 INTEGRATING SPARQL INTO MAIN PIPELINE")
    print("=" * 50)
    
    # Read current dataset
    dataset_path = Path("data/interim/ultimate_real_glycoinformatics_dataset.json")
    
    if not dataset_path.exists():
        print("❌ Dataset not found. Using sample data.")
        return
    
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    print(f"📊 Found {len(dataset)} samples in dataset")
    
    # Sample some IDs to test enhancement
    sample_size = min(10, len(dataset))
    sample_data = dataset[:sample_size]
    
    enhanced_count = 0
    async with aiohttp.ClientSession() as session:
        for sample in sample_data:
            gid = sample.get('glytoucan_id')
            if gid and not sample.get('sparql_enhanced'):
                
                sparql_result = await enhanced_sparql_query(session, gid)
                
                if sparql_result.get('sparql_success') and sparql_result.get('wurcs_sequence'):
                    # Enhance the sample
                    sample['sparql_enhanced'] = True
                    sample['wurcs_sequence'] = sparql_result.get('wurcs_sequence')
                    sample['sparql_label'] = sparql_result.get('label')
                    sample['sparql_mass'] = sparql_result.get('mass')
                    sample['sparql_formula'] = sparql_result.get('formula')
                    
                    enhanced_count += 1
                    print(f"✅ Enhanced {gid} with SPARQL data")
                
                await asyncio.sleep(0.5)  # Rate limiting
    
    print(f"\n📊 SPARQL ENHANCEMENT: {enhanced_count}/{sample_size} samples enhanced")
    
    # Save enhanced sample
    output_path = Path("data/interim/sparql_enhanced_sample.json")
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"💾 Enhanced sample saved to: {output_path}")

if __name__ == "__main__":
    async def main():
        # Test SPARQL queries
        await test_enhanced_sparql()
        
        # Integrate into pipeline
        await integrate_sparql_into_enhancement()
    
    asyncio.run(main())