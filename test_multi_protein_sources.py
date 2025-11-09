#!/usr/bin/env python3
"""
Test enhanced multi-source protein integration
"""

import asyncio
import json
from enhance_real_dataset import EnhancedGlycanDataEnricher

async def test_multi_source_protein_integration():
    """Test protein integration with UniProt + PDB + GlyGen"""
    
    # Test with samples that should have good protein coverage
    test_samples = [
        # Known GlyGen working IDs that should also have UniProt/PDB data
        {"sample_id": "test_multi_1", "glytoucan_id": "G00047MO", "wurcs_sequence": None},
        {"sample_id": "test_multi_2", "glytoucan_id": "G00002CF", "wurcs_sequence": None},
        {"sample_id": "test_multi_3", "glytoucan_id": "G00012MO", "wurcs_sequence": None},
        {"sample_id": "test_multi_4", "glytoucan_id": "G00016MO", "wurcs_sequence": None},
        {"sample_id": "test_multi_5", "glytoucan_id": "G00017MO", "wurcs_sequence": None},
    ]
    
    print("🧪 Testing Multi-Source Protein Integration")
    print("=" * 50)
    
    results = {
        "total": len(test_samples),
        "glygen_hits": 0,
        "uniprot_hits": 0, 
        "pdb_hits": 0,
        "multi_source_hits": 0,
        "protein_enhanced": 0,
        "details": []
    }
    
    async with EnhancedGlycanDataEnricher(rate_limit_delay=2.0) as enricher:
        for i, sample in enumerate(test_samples, 1):
            glytoucan_id = sample['glytoucan_id']
            print(f"\n🔬 Test {i}/{len(test_samples)}: {glytoucan_id}")
            
            try:
                enhanced = await enricher.enhance_sample(sample)
                
                # Check protein data sources
                data_sources = enhanced.get('data_sources', {})
                protein_sources = data_sources.get('proteins', 'None')
                protein_enhanced = data_sources.get('real_components', {}).get('proteins', False)
                
                # Count different data sources
                sources_list = protein_sources.split('+') if protein_sources != 'None' else []
                has_glygen = 'GlyGen' in sources_list
                has_uniprot = 'UniProt' in sources_list
                has_pdb = 'PDB' in sources_list
                
                # Check specific protein data
                protein_associations = enhanced.get('protein_associations', [])
                structural_context = enhanced.get('structural_context', [])
                
                print(f"   📊 Protein Sources: {protein_sources}")
                print(f"   🔗 Protein Associations: {len(protein_associations)}")
                print(f"   🏗️ Structural Context: {len(structural_context)}")
                
                if has_glygen:
                    results["glygen_hits"] += 1
                    print(f"   ✅ GlyGen: Species data available")
                
                if has_uniprot:
                    results["uniprot_hits"] += 1
                    print(f"   ✅ UniProt: {len(protein_associations)} protein associations")
                    for assoc in protein_associations[:2]:  # Show first 2
                        print(f"      - {assoc.get('protein_name', 'Unknown')[:40]}... ({assoc.get('uniprot_id')})")
                
                if has_pdb:
                    results["pdb_hits"] += 1
                    print(f"   ✅ PDB: {len(structural_context)} 3D structures")
                    for struct in structural_context[:2]:  # Show first 2
                        print(f"      - {struct.get('pdb_id')}: {struct.get('structure_title', 'Unknown')[:40]}...")
                
                if len(sources_list) > 1:
                    results["multi_source_hits"] += 1
                    print(f"   🎯 MULTI-SOURCE SUCCESS!")
                
                if protein_enhanced:
                    results["protein_enhanced"] += 1
                
                results["details"].append({
                    "glytoucan_id": glytoucan_id,
                    "sources": sources_list,
                    "protein_enhanced": protein_enhanced,
                    "protein_count": len(protein_associations),
                    "structure_count": len(structural_context)
                })
                
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                results["details"].append({
                    "glytoucan_id": glytoucan_id,
                    "error": str(e)
                })
    
    # Summary
    print(f"\n📊 MULTI-SOURCE PROTEIN INTEGRATION RESULTS:")
    print(f"=" * 50)
    print(f"✅ Total protein enhanced: {results['protein_enhanced']}/{results['total']} ({(results['protein_enhanced']/results['total']*100):.1f}%)")
    print(f"🧬 GlyGen hits: {results['glygen_hits']}/{results['total']} ({(results['glygen_hits']/results['total']*100):.1f}%)")
    print(f"🔗 UniProt hits: {results['uniprot_hits']}/{results['total']} ({(results['uniprot_hits']/results['total']*100):.1f}%)")
    print(f"🏗️ PDB hits: {results['pdb_hits']}/{results['total']} ({(results['pdb_hits']/results['total']*100):.1f}%)")
    print(f"🎯 Multi-source success: {results['multi_source_hits']}/{results['total']} ({(results['multi_source_hits']/results['total']*100):.1f}%)")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_multi_source_protein_integration())