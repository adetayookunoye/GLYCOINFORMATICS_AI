"""
Training Data Formatter for GlycoLLM

Converts database records into multimodal training format combining:
- Spectra data (MS/MS peaks)
- Structure data (WURCS, GlycoCT, IUPAC)
- Text data (descriptions, annotations)

Author: Adetayo Research Team
Date: November 2025
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import asyncio

import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient
import redis

from glycollm.data.spectra_parser import MSSpectrum, SpectraParser

logger = logging.getLogger(__name__)


@dataclass
class MultimodalTrainingSample:
    """Single training sample with all modalities"""
    sample_id: str
    
    # Structure modality
    structure_wurcs: Optional[str] = None
    structure_glycoct: Optional[str] = None
    structure_iupac: Optional[str] = None
    
    # Spectra modality
    spectrum_peaks: Optional[np.ndarray] = None
    precursor_mz: Optional[float] = None
    precursor_charge: Optional[int] = None
    
    # Text modality
    description: Optional[str] = None
    biological_source: Optional[str] = None
    disease_associations: Optional[List[str]] = None
    protein_interactions: Optional[List[str]] = None
    
    # Metadata
    source: Optional[str] = None  # glytoucan, glygen, glycopost
    mass: Optional[float] = None
    composition: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert numpy arrays to lists
        if self.spectrum_peaks is not None:
            data['spectrum_peaks'] = self.spectrum_peaks.tolist()
        return data
    
    def has_structure(self) -> bool:
        """Check if sample has any structure representation"""
        return any([self.structure_wurcs, self.structure_glycoct, self.structure_iupac])
    
    def has_spectra(self) -> bool:
        """Check if sample has spectra data"""
        return self.spectrum_peaks is not None and len(self.spectrum_peaks) > 0
    
    def has_text(self) -> bool:
        """Check if sample has text data"""
        return any([self.description, self.biological_source, 
                   self.disease_associations, self.protein_interactions])


class TrainingDataFormatter:
    """Format database records into training samples"""
    
    def __init__(self,
                 postgres_config: Dict,
                 mongodb_config: Dict,
                 redis_config: Dict,
                 spectra_parser: Optional[SpectraParser] = None):
        """
        Initialize formatter with database connections
        
        Args:
            postgres_config: PostgreSQL connection config
            mongodb_config: MongoDB connection config
            redis_config: Redis connection config
            spectra_parser: Optional spectra parser for processing raw spectra
        """
        self.postgres_config = postgres_config
        self.mongodb_config = mongodb_config
        self.redis_config = redis_config
        self.spectra_parser = spectra_parser or SpectraParser()
        
        # Database connections
        self.pg_conn = None
        self.mongo_client = None
        self.redis_client = None
        
        logger.info("TrainingDataFormatter initialized")
    
    def connect(self):
        """Establish database connections"""
        logger.info("Connecting to databases...")
        
        # PostgreSQL
        self.pg_conn = psycopg2.connect(**self.postgres_config)
        
        # MongoDB
        mongo_uri = f"mongodb://{self.mongodb_config['user']}:{self.mongodb_config['password']}@{self.mongodb_config['host']}:{self.mongodb_config['port']}"
        self.mongo_client = MongoClient(mongo_uri)
        self.mongo_db = self.mongo_client[self.mongodb_config['database']]
        
        # Redis
        self.redis_client = redis.Redis(**self.redis_config, decode_responses=True)
        
        logger.info("Database connections established")
    
    def disconnect(self):
        """Close database connections"""
        if self.pg_conn:
            self.pg_conn.close()
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            self.redis_client.close()
        logger.info("Database connections closed")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def fetch_glycan_from_postgres(self, glytoucan_id: str) -> Optional[Dict]:
        """Fetch glycan data from PostgreSQL"""
        query = """
        SELECT 
            glytoucan_id, mass, wurcs, glycoct, iupac_extended,
            composition, num_monosaccharides, num_linkages,
            motif_id, biological_source, tissue_location
        FROM glycans
        WHERE glytoucan_id = %s
        """
        
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (glytoucan_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def fetch_annotations_from_mongodb(self, glytoucan_id: str) -> Dict:
        """Fetch annotations from MongoDB"""
        collection = self.mongo_db['glycan_annotations']
        doc = collection.find_one({'glytoucan_id': glytoucan_id})
        
        if not doc:
            return {}
        
        return {
            'description': doc.get('description'),
            'biological_source': doc.get('biological_source'),
            'disease_associations': doc.get('disease_associations', []),
            'protein_interactions': doc.get('protein_interactions', []),
            'functions': doc.get('functions', []),
            'pathways': doc.get('pathways', [])
        }
    
    def fetch_spectra_from_mongodb(self, glytoucan_id: str) -> Optional[Dict]:
        """Fetch MS/MS spectra from MongoDB"""
        collection = self.mongo_db['glycan_spectra']
        doc = collection.find_one({'glytoucan_id': glytoucan_id})
        
        if not doc or 'peaks' not in doc:
            return None
        
        return {
            'peaks': np.array(doc['peaks']),
            'precursor_mz': doc.get('precursor_mz'),
            'precursor_charge': doc.get('precursor_charge'),
            'collision_energy': doc.get('collision_energy'),
            'instrument': doc.get('instrument')
        }
    
    def fetch_metadata_from_redis(self, glytoucan_id: str) -> Dict:
        """Fetch cached metadata from Redis"""
        key = f"glycan:{glytoucan_id}"
        data = self.redis_client.hgetall(key)
        
        if not data:
            return {}
        
        return {
            'mass': float(data.get('mass', 0)) if data.get('mass') else None,
            'composition': data.get('composition'),
            'source': data.get('source'),
            'last_updated': data.get('last_updated')
        }
    
    def create_training_sample(self, glytoucan_id: str) -> Optional[MultimodalTrainingSample]:
        """
        Create a complete training sample from all data sources
        
        Args:
            glytoucan_id: GlyTouCan accession ID
            
        Returns:
            MultimodalTrainingSample or None if insufficient data
        """
        try:
            # Fetch from all sources
            pg_data = self.fetch_glycan_from_postgres(glytoucan_id)
            if not pg_data:
                return None
            
            annotations = self.fetch_annotations_from_mongodb(glytoucan_id)
            spectra = self.fetch_spectra_from_mongodb(glytoucan_id)
            metadata = self.fetch_metadata_from_redis(glytoucan_id)
            
            # Build sample
            sample = MultimodalTrainingSample(
                sample_id=glytoucan_id,
                
                # Structure
                structure_wurcs=pg_data.get('wurcs'),
                structure_glycoct=pg_data.get('glycoct'),
                structure_iupac=pg_data.get('iupac_extended'),
                
                # Spectra
                spectrum_peaks=spectra['peaks'] if spectra else None,
                precursor_mz=spectra.get('precursor_mz') if spectra else None,
                precursor_charge=spectra.get('precursor_charge') if spectra else None,
                
                # Text
                description=annotations.get('description'),
                biological_source=annotations.get('biological_source') or pg_data.get('biological_source'),
                disease_associations=annotations.get('disease_associations'),
                protein_interactions=annotations.get('protein_interactions'),
                
                # Metadata
                source=metadata.get('source', 'glytoucan'),
                mass=pg_data.get('mass') or metadata.get('mass'),
                composition=pg_data.get('composition') or metadata.get('composition')
            )
            
            return sample
            
        except Exception as e:
            logger.error(f"Error creating sample for {glytoucan_id}: {e}")
            return None
    
    def generate_training_samples(self, 
                                  limit: Optional[int] = None,
                                  require_spectra: bool = False,
                                  require_text: bool = False) -> Iterator[MultimodalTrainingSample]:
        """
        Generate training samples from database
        
        Args:
            limit: Maximum number of samples to generate
            require_spectra: Only include samples with spectra
            require_text: Only include samples with text annotations
            
        Yields:
            MultimodalTrainingSample objects
        """
        logger.info(f"Generating training samples (limit={limit}, require_spectra={require_spectra}, require_text={require_text})")
        
        # Get all glycan IDs from PostgreSQL
        query = "SELECT glytoucan_id FROM glycans ORDER BY glytoucan_id"
        if limit:
            query += f" LIMIT {limit}"
        
        with self.pg_conn.cursor() as cursor:
            cursor.execute(query)
            
            count = 0
            for row in cursor:
                glytoucan_id = row[0]
                sample = self.create_training_sample(glytoucan_id)
                
                if not sample:
                    continue
                
                # Apply filters
                if require_spectra and not sample.has_spectra():
                    continue
                if require_text and not sample.has_text():
                    continue
                
                yield sample
                count += 1
                
                if count % 1000 == 0:
                    logger.info(f"Generated {count} samples...")
    
    def build_task_specific_datasets(self, 
                                    output_dir: Path,
                                    limit: Optional[int] = None) -> Dict[str, int]:
        """
        Build task-specific datasets for different training objectives
        
        Tasks:
        1. spec_to_struct: Spectra → Structure prediction
        2. struct_to_text: Structure → Description generation
        3. text_to_struct: Description → Structure prediction
        4. multimodal_retrieval: Cross-modal matching
        
        Args:
            output_dir: Directory to save datasets
            limit: Maximum samples per task
            
        Returns:
            Dictionary with counts per task
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Task-specific sample lists
        spec_to_struct = []
        struct_to_text = []
        text_to_struct = []
        multimodal_retrieval = []
        
        logger.info("Building task-specific datasets...")
        
        for sample in self.generate_training_samples(limit=limit):
            sample_dict = sample.to_dict()
            
            # Task 1: Spectra → Structure (requires both spectra and structure)
            if sample.has_spectra() and sample.has_structure():
                spec_to_struct.append({
                    'sample_id': sample.sample_id,
                    'input_spectra': sample_dict['spectrum_peaks'],
                    'input_precursor_mz': sample.precursor_mz,
                    'input_precursor_charge': sample.precursor_charge,
                    'target_structure_wurcs': sample.structure_wurcs,
                    'target_structure_glycoct': sample.structure_glycoct,
                    'target_structure_iupac': sample.structure_iupac
                })
            
            # Task 2: Structure → Text (requires both structure and text)
            if sample.has_structure() and sample.has_text():
                struct_to_text.append({
                    'sample_id': sample.sample_id,
                    'input_structure_wurcs': sample.structure_wurcs,
                    'input_structure_glycoct': sample.structure_glycoct,
                    'input_structure_iupac': sample.structure_iupac,
                    'target_description': sample.description,
                    'target_biological_source': sample.biological_source,
                    'target_disease_associations': sample.disease_associations,
                    'target_protein_interactions': sample.protein_interactions
                })
            
            # Task 3: Text → Structure (requires both text and structure)
            if sample.has_text() and sample.has_structure():
                text_to_struct.append({
                    'sample_id': sample.sample_id,
                    'input_description': sample.description,
                    'input_biological_source': sample.biological_source,
                    'input_disease_associations': sample.disease_associations,
                    'target_structure_wurcs': sample.structure_wurcs,
                    'target_structure_glycoct': sample.structure_glycoct,
                    'target_structure_iupac': sample.structure_iupac
                })
            
            # Task 4: Multimodal retrieval (requires all modalities)
            if sample.has_spectra() and sample.has_structure() and sample.has_text():
                multimodal_retrieval.append(sample_dict)
        
        # Save datasets
        datasets = {
            'spec_to_struct': spec_to_struct,
            'struct_to_text': struct_to_text,
            'text_to_struct': text_to_struct,
            'multimodal_retrieval': multimodal_retrieval
        }
        
        counts = {}
        for task_name, dataset in datasets.items():
            output_path = output_dir / f"{task_name}.json"
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            counts[task_name] = len(dataset)
            logger.info(f"Saved {task_name}: {len(dataset)} samples to {output_path}")
        
        # Save summary
        summary = {
            'total_samples_processed': sum(counts.values()),
            'task_counts': counts,
            'output_directory': str(output_dir)
        }
        
        summary_path = output_dir / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dataset building complete. Summary saved to {summary_path}")
        return counts
    
    def create_train_val_test_split(self,
                                    dataset_path: Path,
                                    output_dir: Path,
                                    train_ratio: float = 0.8,
                                    val_ratio: float = 0.1,
                                    test_ratio: float = 0.1,
                                    seed: int = 42) -> Dict[str, int]:
        """
        Split dataset into train/val/test sets
        
        Args:
            dataset_path: Path to full dataset JSON
            output_dir: Directory to save splits
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed
            
        Returns:
            Dictionary with split counts
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        # Shuffle
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))
        
        # Split
        n_train = int(len(dataset) * train_ratio)
        n_val = int(len(dataset) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        splits = {
            'train': [dataset[i] for i in train_indices],
            'val': [dataset[i] for i in val_indices],
            'test': [dataset[i] for i in test_indices]
        }
        
        # Save splits
        counts = {}
        for split_name, split_data in splits.items():
            output_path = output_dir / f"{dataset_path.stem}_{split_name}.json"
            with open(output_path, 'w') as f:
                json.dump(split_data, f, indent=2)
            counts[split_name] = len(split_data)
            logger.info(f"Saved {split_name}: {len(split_data)} samples to {output_path}")
        
        return counts
    
    def get_dataset_statistics(self) -> Dict:
        """Compute comprehensive dataset statistics"""
        stats = {
            'total_glycans': 0,
            'with_wurcs': 0,
            'with_glycoct': 0,
            'with_iupac': 0,
            'with_spectra': 0,
            'with_annotations': 0,
            'with_disease_associations': 0,
            'with_protein_interactions': 0,
            'mass_distribution': [],
            'num_monosaccharides': []
        }
        
        # PostgreSQL stats
        with self.pg_conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM glycans")
            stats['total_glycans'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM glycans WHERE wurcs IS NOT NULL")
            stats['with_wurcs'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM glycans WHERE glycoct IS NOT NULL")
            stats['with_glycoct'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM glycans WHERE iupac_extended IS NOT NULL")
            stats['with_iupac'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT mass, num_monosaccharides FROM glycans WHERE mass IS NOT NULL")
            for row in cursor.fetchall():
                if row[0]:
                    stats['mass_distribution'].append(float(row[0]))
                if row[1]:
                    stats['num_monosaccharides'].append(int(row[1]))
        
        # MongoDB stats
        stats['with_spectra'] = self.mongo_db['glycan_spectra'].count_documents({})
        stats['with_annotations'] = self.mongo_db['glycan_annotations'].count_documents({})
        stats['with_disease_associations'] = self.mongo_db['glycan_annotations'].count_documents(
            {'disease_associations': {'$exists': True, '$ne': []}}
        )
        stats['with_protein_interactions'] = self.mongo_db['glycan_annotations'].count_documents(
            {'protein_interactions': {'$exists': True, '$ne': []}}
        )
        
        # Compute summary statistics
        if stats['mass_distribution']:
            stats['mass_stats'] = {
                'mean': float(np.mean(stats['mass_distribution'])),
                'std': float(np.std(stats['mass_distribution'])),
                'min': float(np.min(stats['mass_distribution'])),
                'max': float(np.max(stats['mass_distribution']))
            }
        
        if stats['num_monosaccharides']:
            stats['monosaccharide_stats'] = {
                'mean': float(np.mean(stats['num_monosaccharides'])),
                'std': float(np.std(stats['num_monosaccharides'])),
                'min': int(np.min(stats['num_monosaccharides'])),
                'max': int(np.max(stats['num_monosaccharides']))
            }
        
        # Compute coverage rates
        stats['coverage'] = {
            'structure_any': (stats['with_wurcs'] or stats['with_glycoct'] or stats['with_iupac']) / max(stats['total_glycans'], 1),
            'spectra': stats['with_spectra'] / max(stats['total_glycans'], 1),
            'annotations': stats['with_annotations'] / max(stats['total_glycans'], 1),
            'multimodal': min(stats['with_spectra'], stats['with_annotations'], stats['with_wurcs']) / max(stats['total_glycans'], 1)
        }
        
        return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Format training data from databases")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--limit", type=int, help="Maximum samples")
    parser.add_argument("--postgres-host", type=str, default="localhost")
    parser.add_argument("--postgres-port", type=int, default=5432)
    parser.add_argument("--postgres-db", type=str, default="glyco_db")
    parser.add_argument("--postgres-user", type=str, default="glyco_user")
    parser.add_argument("--postgres-password", type=str, default="glyco_pass")
    parser.add_argument("--mongodb-host", type=str, default="localhost")
    parser.add_argument("--mongodb-port", type=int, default=27017)
    parser.add_argument("--mongodb-db", type=str, default="glyco_annotations")
    parser.add_argument("--mongodb-user", type=str, default="glyco_user")
    parser.add_argument("--mongodb-password", type=str, default="glyco_pass")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--create-splits", action="store_true", help="Create train/val/test splits")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Database configs
    postgres_config = {
        'host': args.postgres_host,
        'port': args.postgres_port,
        'database': args.postgres_db,
        'user': args.postgres_user,
        'password': args.postgres_password
    }
    
    mongodb_config = {
        'host': args.mongodb_host,
        'port': args.mongodb_port,
        'database': args.mongodb_db,
        'user': args.mongodb_user,
        'password': args.mongodb_password
    }
    
    redis_config = {
        'host': args.redis_host,
        'port': args.redis_port,
        'db': args.redis_db
    }
    
    # Build datasets
    with TrainingDataFormatter(postgres_config, mongodb_config, redis_config) as formatter:
        # Get statistics
        stats = formatter.get_dataset_statistics()
        print("\n=== Dataset Statistics ===")
        print(json.dumps(stats, indent=2))
        
        # Build task-specific datasets
        counts = formatter.build_task_specific_datasets(
            output_dir=Path(args.output_dir),
            limit=args.limit
        )
        
        print("\n=== Dataset Counts ===")
        print(json.dumps(counts, indent=2))
        
        # Create splits if requested
        if args.create_splits:
            for task_name in counts.keys():
                dataset_path = Path(args.output_dir) / f"{task_name}.json"
                split_dir = Path(args.output_dir) / "splits" / task_name
                split_counts = formatter.create_train_val_test_split(
                    dataset_path=dataset_path,
                    output_dir=split_dir
                )
                print(f"\n=== {task_name} Splits ===")
                print(json.dumps(split_counts, indent=2))
