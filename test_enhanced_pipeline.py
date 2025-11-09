#!/usr/bin/env python3
"""
Test enhanced pipeline with known working glycan IDs to validate improvements
"""

import asyncio
import json
from enhance_real_dataset import EnhancedGlycanDataEnricher

async def test_enhanced_improvements():
    """Test enhanced 4-level fallback strategy"""
    
    # Test with known working IDs and some that should test fallbacks
    test_samples = [
        # Known GlyGen working IDs
        {"sample_id": "test_glygen_1", "glytoucan_id": "G00047MO", "wurcs_sequence": None, "molecular_mass": None},
        {"sample_id": "test_glygen_2", "glytoucan_id": "G00002CF", "wurcs_sequence": None, "molecular_mass": None},
        
        # Complex structures for SPARQL testing
        {"sample_id": "test_sparql_1", "glytoucan_id": "G01415VY", "wurcs_sequence": None, "molecular_mass": None},
        {"sample_id": "test_sparql_2", "glytoucan_id": "G00065MO", "wurcs_sequence": None, "molecular_mass": None},
        
        # IDs that may test REST API fallback
        {"sample_id": "test_rest_1", "glytoucan_id": "G00000CV", "wurcs_sequence": None, "molecular_mass": None},
        {"sample_id": "test_rest_2", "glytoucan_id": "G00001LE", "wurcs_sequence": None, "molecular_mass": None},
    ]
    
    print("🧪 Testing Enhanced 4-Level Fallback Pipeline")
    print("=" * 50)
    
    results = {"successes": 0, "total": len(test_samples), "details": []}
    
    async with EnhancedGlycanDataEnricher(rate_limit_delay=1.5) as enricher:
        for i, sample in enumerate(test_samples, 1):
            glytoucan_id = sample['glytoucan_id']
            print(f"\n🔬 Test {i}/{len(test_samples)}: {glytoucan_id}")
            
            try:
                enhanced = await enricher.enhance_sample(sample)
                
                # Check what data we retrieved
                wurcs = enhanced.get('wurcs_sequence')
                mass = enhanced.get('molecular_mass')
                iupac = enhanced.get('iupac_name')
                source = enhanced.get('data_sources', {}).get('structure', 'Unknown')
                
                has_structure = bool(wurcs or mass or iupac)
                
                if has_structure:
                    results["successes"] += 1
                    print(f"✅ SUCCESS - Source: {source}")
                    if wurcs:
                        print(f"   WURCS: {wurcs[:50]}..." if len(wurcs) > 50 else f"   WURCS: {wurcs}")
                    if iupac:
                        print(f"   IUPAC: {iupac[:50]}..." if len(iupac) > 50 else f"   IUPAC: {iupac}")
                    if mass:
                        print(f"   Mass: {mass}")
                else:
                    print(f"❌ NO STRUCTURE DATA - Source: {source}")
                
                results["details"].append({
                    "glytoucan_id": glytoucan_id,
                    "success": has_structure,
                    "source": source,
                    "has_wurcs": bool(wurcs),
                    "has_mass": bool(mass),
                    "has_iupac": bool(iupac)
                })
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                results["details"].append({
                    "glytoucan_id": glytoucan_id,
                    "success": False,
                    "error": str(e)
                })
    
    # Summary
    success_rate = (results["successes"] / results["total"]) * 100
    print(f"\n📊 ENHANCED PIPELINE RESULTS:")
    print(f"=" * 40)
    print(f"✅ Successful: {results['successes']}/{results['total']} ({success_rate:.1f}%)")
    
    # Breakdown by source
    source_counts = {}
    for detail in results["details"]:
        if detail.get("success"):
            source = detail.get("source", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n📈 Success breakdown by source:")
    for source, count in source_counts.items():
        print(f"   {source}: {count} samples")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_enhanced_improvements())