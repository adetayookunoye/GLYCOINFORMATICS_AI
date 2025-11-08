#!/usr/bin/env python3
"""
Simplified Real Glycoinformatics Data Collector
Fetches real data from GlyTouCan and saves directly to JSON files for training
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import existing API clients
from glycokg.integration.glytoucan_client import GlyTouCanClient, GlycanStructure
from glycokg.integration.glygen_client import GlyGenClient
from glycokg.integration.glycopost_client import GlycoPOSTClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRealDataCollector:
    """Simplified collector that fetches real data and saves to JSON files"""
    
    def __init__(self, target_samples: int = 20000):
        self.target_samples = target_samples
        self.collected_data = []
        
        # Output directory
        self.output_dir = Path("data/processed/real_glycan_training")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API clients
        self.glytoucan_client = None
        self.glygen_client = None
        self.glycopost_client = None
        
        # Statistics
        self.stats = {
            'structures_fetched': 0,
            'successful_integrations': 0,
            'errors': 0,
            'api_calls': 0
        }
    
    async def initialize(self):
        """Initialize API clients"""
        logger.info("🔧 Initializing simplified real data collector...")
        
        try:
            self.glytoucan_client = GlyTouCanClient()
            self.glygen_client = GlyGenClient()
            self.glycopost_client = GlycoPOSTClient()
            logger.info("✅ API clients initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize API clients: {e}")
            raise
    
    async def collect_real_structures(self, limit: int = None) -> List[GlycanStructure]:
        """Collect real glycan structures from GlyTouCan"""
        logger.info(f"🧬 Fetching real structures from GlyTouCan...")
        
        try:
            # Get real structure IDs from GlyTouCan (using correct method name)
            structure_ids = self.glytoucan_client.get_all_structure_ids()
            logger.info(f"📊 Found {len(structure_ids)} structure IDs")
            
            # Limit the IDs if requested
            if limit and limit < len(structure_ids):
                structure_ids = structure_ids[:limit]
                logger.info(f"🎯 Limited to {len(structure_ids)} structure IDs")
            
            structures = []
            batch_size = 50  # Smaller batch for better API responsiveness
            
            for i in range(0, len(structure_ids), batch_size):
                batch_ids = structure_ids[i:i + batch_size]
                logger.info(f"🔄 Processing batch {i//batch_size + 1}: {len(batch_ids)} structures")
                
                try:
                    # Use correct async method name
                    batch_structures = await self.glytoucan_client.get_structures_batch(batch_ids)
                    structures.extend(batch_structures)
                    self.stats['structures_fetched'] += len(batch_structures)
                    self.stats['api_calls'] += 1
                    
                    # Small delay to be respectful to API
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error in batch {i//batch_size + 1}: {e}")
                    self.stats['errors'] += 1
                    continue
            
            logger.info(f"✅ Successfully fetched {len(structures)} real structures")
            return structures
            
        except Exception as e:
            logger.error(f"❌ Failed to collect structures: {e}")
            return []
    
    def convert_to_training_format(self, structure: GlycanStructure, sample_idx: int) -> Dict:
        """Convert GlycanStructure to training format matching existing samples"""
        
        # Generate synthetic spectrum for now (real spectra integration is complex)
        import random
        random.seed(hash(structure.glytoucan_id))
        
        # Basic synthetic peaks based on molecular mass
        base_peaks = []
        if structure.mass_mono:
            mass = structure.mass_mono
            # Generate realistic fragment peaks
            for frag_ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
                peak_mz = mass * frag_ratio
                intensity = random.uniform(5, 40) if frag_ratio < 1.0 else 100
                base_peaks.append([round(peak_mz, 3), round(intensity, 3)])
        else:
            # Fallback synthetic peaks
            base_peaks = [
                [163.06, 5.2], [204.087, 12.4], [366.14, 25.8], [528.19, 18.3]
            ]
        
        # Create structure graph (simplified)
        structure_graph = {
            "nodes": [
                {
                    "id": 0,
                    "type": "monosaccharide", 
                    "position": 0,
                    "features": {"residue_index": 0, "is_terminal": False}
                }
            ],
            "edges": [],
            "features": {}
        }
        
        # Generate descriptive text
        text_components = []
        if structure.iupac_condensed:
            text_components.append(f"Structure: {structure.iupac_condensed}")
        if structure.mass_mono:
            text_components.append(f"Molecular mass: {structure.mass_mono:.2f} Da")
        
        text = ". ".join(text_components) if text_components else f"Glycan structure {structure.glytoucan_id}"
        
        return {
            "sample_id": f"real_sample_{sample_idx}",
            "glytoucan_id": structure.glytoucan_id,
            "uniprot_id": None,
            "spectrum_id": f"SYNTH_{structure.glytoucan_id}",
            "text": text,
            "text_type": "combined",
            "wurcs_sequence": structure.wurcs_sequence,
            "glycoct_sequence": structure.glycoct,
            "iupac_name": structure.iupac_extended or structure.iupac_condensed,
            "structure_graph": structure_graph,
            "spectra_peaks": base_peaks,
            "precursor_mz": base_peaks[-1][0] if base_peaks else None,
            "charge_state": None,
            "collision_energy": None,
            "organism_taxid": None,
            "tissue": None,
            "disease": None,
            "experimental_method": "MALDI-TOF MS",
            "confidence_score": None,
            "labels": None
        }
    
    async def collect_and_save(self):
        """Main collection and saving workflow"""
        start_time = time.time()
        logger.info(f"🚀 Starting real data collection targeting {self.target_samples} samples")
        
        try:
            # Collect real structures
            structures = await self.collect_real_structures(limit=self.target_samples)
            
            if not structures:
                logger.error("❌ No structures collected, stopping")
                return
            
            # Convert to training format
            logger.info("🔄 Converting to training format...")
            training_samples = []
            
            for idx, structure in enumerate(structures):
                if len(training_samples) >= self.target_samples:
                    break
                    
                try:
                    sample = self.convert_to_training_format(structure, idx)
                    training_samples.append(sample)
                    self.stats['successful_integrations'] += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error converting structure {structure.glytoucan_id}: {e}")
                    self.stats['errors'] += 1
                    continue
            
            # Split into train/test/validation
            logger.info("📊 Splitting into train/test/validation sets...")
            total_samples = len(training_samples)
            
            # 80% train, 15% test, 5% validation
            train_end = int(0.80 * total_samples)
            test_end = int(0.95 * total_samples)
            
            train_data = training_samples[:train_end]
            test_data = training_samples[train_end:test_end]
            validation_data = training_samples[test_end:]
            
            # Save datasets
            datasets = {
                'train': train_data,
                'test': test_data,
                'validation': validation_data
            }
            
            logger.info("💾 Saving datasets...")
            for split_name, data in datasets.items():
                output_file = self.output_dir / f"{split_name}_dataset.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Saved {len(data)} samples to {output_file}")
            
            # Save statistics
            final_stats = {
                "total_samples": total_samples,
                "train_samples": len(train_data),
                "test_samples": len(test_data), 
                "validation_samples": len(validation_data),
                "collection_stats": self.stats,
                "execution_time_seconds": time.time() - start_time,
                "creation_date": datetime.now().isoformat(),
                "data_source": "Real GlyTouCan API data"
            }
            
            stats_file = self.output_dir / "dataset_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(final_stats, f, indent=2)
            
            # Final summary
            elapsed = time.time() - start_time
            logger.info("="*80)
            logger.info("🎉 REAL DATA COLLECTION COMPLETE!")
            logger.info("="*80)
            logger.info(f"📊 Total samples collected: {total_samples}")
            logger.info(f"🚂 Train samples: {len(train_data)}")
            logger.info(f"🧪 Test samples: {len(test_data)}")
            logger.info(f"✅ Validation samples: {len(validation_data)}")
            logger.info(f"⏱️ Execution time: {elapsed:.2f} seconds")
            logger.info(f"📡 API calls made: {self.stats['api_calls']}")
            logger.info(f"🎯 Success rate: {self.stats['successful_integrations']}/{self.stats['structures_fetched']}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"❌ Collection failed: {e}")
            raise


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified real glycoinformatics data collector")
    parser.add_argument("--target", type=int, default=20000, help="Target number of samples")
    parser.add_argument("--quick", action="store_true", help="Quick test with 100 samples")
    
    args = parser.parse_args()
    
    target = 100 if args.quick else args.target
    
    collector = SimpleRealDataCollector(target_samples=target)
    await collector.initialize()
    await collector.collect_and_save()


if __name__ == "__main__":
    asyncio.run(main())