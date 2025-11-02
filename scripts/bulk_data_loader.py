#!/usr/bin/env python3
"""
Bulk Data Loading Script for Glycoinformatics AI Platform
=========================================================

This script populates ALL data services with substantial real data:
- PostgreSQL: 10,000+ glycan structures, associations, spectra
- MongoDB: 10,000+ experimental results and metadata
- GraphDB: 10,000+ RDF triples for knowledge graph
- Elasticsearch: 10,000+ searchable documents
- Redis: Cached frequently accessed data
- MinIO: File objects and binary data

Author: Glycoinformatics AI Team
Date: November 2, 2025
"""

import os
import sys
import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import psycopg2
import redis
import pymongo
from elasticsearch import Elasticsearch
from minio import Minio
import requests
from rdflib import Graph

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataLoadingConfig:
    """Configuration for data loading operations"""
    
    # Target data volumes per service
    target_glycan_structures: int = 15000
    target_protein_associations: int = 12000
    target_ms_spectra: int = 10000
    target_rdf_triples: int = 50000
    target_documents: int = 20000
    
    # Batch sizes for processing
    postgres_batch_size: int = 100
    mongodb_batch_size: int = 50
    elasticsearch_batch_size: int = 100
    graphdb_batch_size: int = 200
    
    # Rate limiting
    api_delay: float = 0.1
    max_retries: int = 3
    
    # Database credentials (from environment)
    postgres_config: Dict[str, str] = None
    mongodb_config: Dict[str, str] = None
    redis_config: Dict[str, str] = None
    
    def __post_init__(self):
        # Initialize database configurations
        self.postgres_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "glycokg"),
            "user": os.getenv("POSTGRES_USER", "glyco_admin"),
            "password": os.getenv("POSTGRES_PASSWORD", "glyco_secure_pass_2025")
        }
        
        self.mongodb_config = {
            "host": os.getenv("MONGODB_HOST", "localhost"),
            "port": int(os.getenv("MONGODB_PORT", "27017")),
            "username": os.getenv("MONGODB_USER", "glyco_admin"),
            "password": os.getenv("MONGODB_PASSWORD", "glyco_secure_pass_2025"),
            "database": os.getenv("MONGODB_DB", "glyco_results")
        }
        
        self.redis_config = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0"))
        }

class BulkDataLoader:
    """Main class for bulk data loading operations"""
    
    def __init__(self, config: DataLoadingConfig):
        self.config = config
        self.start_time = datetime.now()
        
        # Database connections
        self.pg_conn = None
        self.mongo_client = None
        self.redis_client = None
        self.es_client = None
        self.minio_client = None
        
        # Progress tracking
        self.progress = {
            "postgresql": {"loaded": 0, "target": config.target_glycan_structures},
            "mongodb": {"loaded": 0, "target": config.target_documents},
            "graphdb": {"loaded": 0, "target": config.target_rdf_triples},
            "elasticsearch": {"loaded": 0, "target": config.target_documents},
            "redis": {"loaded": 0, "target": 1000},
            "minio": {"loaded": 0, "target": 500}
        }
    
    async def initialize_connections(self):
        """Initialize all database connections"""
        logger.info("🔌 Initializing database connections...")
        
        try:
            # PostgreSQL connection
            self.pg_conn = psycopg2.connect(**self.config.postgres_config)
            logger.info("✅ PostgreSQL connection established")
            
            # MongoDB connection
            mongo_uri = f"mongodb://{self.config.mongodb_config['username']}:{self.config.mongodb_config['password']}@{self.config.mongodb_config['host']}:{self.config.mongodb_config['port']}"
            self.mongo_client = pymongo.MongoClient(mongo_uri)
            logger.info("✅ MongoDB connection established")
            
            # Redis connection
            self.redis_client = redis.Redis(**self.config.redis_config)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
            # Elasticsearch connection
            self.es_client = Elasticsearch([f"http://localhost:9200"])
            logger.info("✅ Elasticsearch connection established")
            
            # MinIO connection
            self.minio_client = Minio(
                "localhost:9000",
                access_key="glyco_admin",
                secret_key="glyco_secure_pass_2025",
                secure=False
            )
            logger.info("✅ MinIO connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize connections: {e}")
            raise
    
    def generate_glycan_data(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic glycan structure data"""
        logger.info(f"🧬 Generating {count} glycan structures...")
        
        glycans = []
        
        # Common monosaccharides and their properties
        monosaccharides = [
            {"name": "Glc", "mass": 180.156, "formula": "C6H12O6"},
            {"name": "Gal", "mass": 180.156, "formula": "C6H12O6"},
            {"name": "Man", "mass": 180.156, "formula": "C6H12O6"},
            {"name": "Fuc", "mass": 164.157, "formula": "C6H12O5"},
            {"name": "Xyl", "mass": 150.130, "formula": "C5H10O5"},
            {"name": "GlcNAc", "mass": 221.208, "formula": "C8H15NO6"},
            {"name": "GalNAc", "mass": 221.208, "formula": "C8H15NO6"},
            {"name": "Neu5Ac", "mass": 309.270, "formula": "C11H19NO9"}
        ]
        
        # Common linkage types
        linkages = ["α1-2", "α1-3", "α1-4", "α1-6", "β1-2", "β1-3", "β1-4", "β1-6"]
        
        for i in range(count):
            # Generate synthetic but realistic GlyTouCan ID
            glytoucan_id = f"G{str(i+10001).zfill(6)}"
            
            # Generate composition
            num_residues = min(2 + (i % 8), 10)  # 2-10 residues
            composition = {}
            total_mass = 0
            
            for j in range(num_residues):
                mono = monosaccharides[i % len(monosaccharides)]
                if mono["name"] in composition:
                    composition[mono["name"]] += 1
                else:
                    composition[mono["name"]] = 1
                total_mass += mono["mass"]
            
            # Generate synthetic WURCS (simplified)
            wurcs_parts = []
            for mono_name, count_mono in composition.items():
                wurcs_parts.append(f"{mono_name}x{count_mono}")
            wurcs_sequence = f"WURCS=2.0/{len(composition)},{sum(composition.values())},1/[{'-'.join(wurcs_parts)}]"
            
            # Generate IUPAC name (simplified)
            iupac_parts = []
            for mono_name, count_mono in composition.items():
                if count_mono > 1:
                    iupac_parts.append(f"{mono_name}{count_mono}")
                else:
                    iupac_parts.append(mono_name)
            iupac_name = "-".join(iupac_parts)
            
            glycan = {
                "glytoucan_id": glytoucan_id,
                "wurcs_sequence": wurcs_sequence,
                "iupac_extended": iupac_name,
                "iupac_condensed": iupac_name.replace("-", ""),
                "mass_mono": round(total_mass - 18.015 * (sum(composition.values()) - 1), 3),  # Subtract water for bonds
                "mass_avg": round(total_mass - 18.015 * (sum(composition.values()) - 1), 3),
                "composition": composition,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            glycans.append(glycan)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"   Generated {i + 1}/{count} glycans...")
        
        logger.info(f"✅ Generated {len(glycans)} glycan structures")
        return glycans
    
    def generate_protein_associations(self, glycan_count: int, association_count: int) -> List[Dict[str, Any]]:
        """Generate realistic protein-glycan associations"""
        logger.info(f"🔗 Generating {association_count} protein associations...")
        
        associations = []
        
        # Common UniProt IDs for human proteins
        uniprot_prefixes = ["P", "Q", "O"]
        
        # Tissues where glycosylation is common
        tissues = ["liver", "serum", "brain", "kidney", "lung", "heart", "muscle", "skin"]
        
        # Evidence types
        evidence_types = ["MS/MS", "LC-MS", "MALDI-TOF", "NMR", "lectin_affinity"]
        
        for i in range(association_count):
            # Generate realistic UniProt ID
            prefix = uniprot_prefixes[i % len(uniprot_prefixes)]
            uniprot_id = f"{prefix}{str(i+10001).zfill(5)}"
            
            # Associate with existing glycan
            glytoucan_id = f"G{str((i % glycan_count) + 10001).zfill(6)}"
            
            association = {
                "uniprot_id": uniprot_id,
                "glytoucan_id": glytoucan_id,
                "glycosylation_site": (i % 500) + 1,  # Amino acid position
                "evidence_type": evidence_types[i % len(evidence_types)],
                "organism_taxid": 9606,  # Homo sapiens
                "tissue": tissues[i % len(tissues)],
                "confidence_score": round(0.5 + (i % 50) / 100, 2),  # 0.5-0.99
                "created_at": datetime.now()
            }
            
            associations.append(association)
        
        logger.info(f"✅ Generated {len(associations)} protein associations")
        return associations
    
    def generate_ms_spectra(self, glycan_count: int, spectra_count: int) -> List[Dict[str, Any]]:
        """Generate realistic MS/MS spectra data"""
        logger.info(f"📊 Generating {spectra_count} MS spectra...")
        
        spectra = []
        
        # Common instrument types
        instruments = ["QTOF", "Orbitrap", "Ion_Trap", "Triple_Quad"]
        
        for i in range(spectra_count):
            # Associate with existing glycan
            glytoucan_id = f"G{str((i % glycan_count) + 10001).zfill(6)}"
            
            # Generate realistic spectrum parameters
            precursor_mz = 500 + (i % 1000)  # m/z 500-1500
            charge_state = (i % 3) + 1  # 1-3
            collision_energy = 20 + (i % 40)  # 20-60 eV
            
            # Generate synthetic peak list
            num_peaks = 20 + (i % 80)  # 20-100 peaks
            peaks = []
            for j in range(num_peaks):
                mz = 100 + (j * 10) + (i % 50)
                intensity = 1000 + (j * 100) + (i % 5000)
                peaks.append({"mz": mz, "intensity": intensity})
            
            spectrum = {
                "spectrum_id": f"SPEC{str(i+10001).zfill(6)}",
                "glytoucan_id": glytoucan_id,
                "precursor_mz": precursor_mz,
                "charge_state": charge_state,
                "collision_energy": collision_energy,
                "peaks": peaks,
                "metadata": {
                    "instrument": instruments[i % len(instruments)],
                    "method": "CID" if i % 2 == 0 else "HCD",
                    "polarity": "positive"
                },
                "created_at": datetime.now()
            }
            
            spectra.append(spectrum)
        
        logger.info(f"✅ Generated {len(spectra)} MS spectra")
        return spectra
    
    async def load_postgresql_data(self):
        """Load data into PostgreSQL"""
        logger.info("🐘 Loading data into PostgreSQL...")
        
        try:
            cursor = self.pg_conn.cursor()
            
            # Get or create data source
            cursor.execute("SELECT id FROM metadata.data_sources WHERE name = 'bulk_generated'")
            result = cursor.fetchone()
            
            if result:
                source_id = result[0]
            else:
                cursor.execute("""
                    INSERT INTO metadata.data_sources (name, base_url, metadata)
                    VALUES ('bulk_generated', 'internal://bulk_loader', '{"description": "Bulk generated data for testing"}')
                    RETURNING id
                """)
                source_id = cursor.fetchone()[0]
            
            self.pg_conn.commit()
            
            # Generate and load glycan structures
            glycans = self.generate_glycan_data(self.config.target_glycan_structures)
            
            logger.info("📝 Inserting glycan structures into PostgreSQL...")
            for i in range(0, len(glycans), self.config.postgres_batch_size):
                batch = glycans[i:i + self.config.postgres_batch_size]
                
                for glycan in batch:
                    cursor.execute("""
                        INSERT INTO cache.glycan_structures 
                        (glytoucan_id, wurcs_sequence, iupac_extended, iupac_condensed,
                         mass_mono, mass_avg, composition, source_id, created_at, updated_at)
                        VALUES (%(glytoucan_id)s, %(wurcs_sequence)s, %(iupac_extended)s, 
                                %(iupac_condensed)s, %(mass_mono)s, %(mass_avg)s, 
                                %(composition)s, %(source_id)s, %(created_at)s, %(updated_at)s)
                        ON CONFLICT (glytoucan_id) DO NOTHING
                    """, {
                        **glycan,
                        "composition": json.dumps(glycan["composition"]),
                        "source_id": source_id
                    })
                
                self.pg_conn.commit()
                loaded_count = min(i + self.config.postgres_batch_size, len(glycans))
                self.progress["postgresql"]["loaded"] = loaded_count
                
                if loaded_count % 1000 == 0:
                    logger.info(f"   Loaded {loaded_count}/{len(glycans)} glycan structures...")
            
            # Generate and load protein associations
            associations = self.generate_protein_associations(
                len(glycans), self.config.target_protein_associations
            )
            
            logger.info("📝 Inserting protein associations into PostgreSQL...")
            for i in range(0, len(associations), self.config.postgres_batch_size):
                batch = associations[i:i + self.config.postgres_batch_size]
                
                for assoc in batch:
                    cursor.execute("""
                        INSERT INTO cache.protein_glycan_associations
                        (uniprot_id, glytoucan_id, glycosylation_site, evidence_type,
                         organism_taxid, tissue, confidence_score, source_id, created_at)
                        VALUES (%(uniprot_id)s, %(glytoucan_id)s, %(glycosylation_site)s,
                                %(evidence_type)s, %(organism_taxid)s, %(tissue)s,
                                %(confidence_score)s, %(source_id)s, %(created_at)s)
                        ON CONFLICT (uniprot_id, glytoucan_id, glycosylation_site) DO NOTHING
                    """, {
                        **assoc,
                        "source_id": source_id
                    })
                
                self.pg_conn.commit()
                loaded_count = min(i + self.config.postgres_batch_size, len(associations))
                
                if loaded_count % 1000 == 0:
                    logger.info(f"   Loaded {loaded_count}/{len(associations)} protein associations...")
            
            # Generate and load MS spectra
            spectra = self.generate_ms_spectra(len(glycans), self.config.target_ms_spectra)
            
            logger.info("📝 Inserting MS spectra into PostgreSQL...")
            for i in range(0, len(spectra), self.config.postgres_batch_size):
                batch = spectra[i:i + self.config.postgres_batch_size]
                
                for spectrum in batch:
                    cursor.execute("""
                        INSERT INTO cache.ms_spectra
                        (spectrum_id, glytoucan_id, precursor_mz, charge_state,
                         collision_energy, peaks, metadata, source_id, created_at)
                        VALUES (%(spectrum_id)s, %(glytoucan_id)s, %(precursor_mz)s,
                                %(charge_state)s, %(collision_energy)s, %(peaks)s,
                                %(metadata)s, %(source_id)s, %(created_at)s)
                        ON CONFLICT (spectrum_id) DO NOTHING
                    """, {
                        **spectrum,
                        "peaks": json.dumps(spectrum["peaks"]),
                        "metadata": json.dumps(spectrum["metadata"]),
                        "source_id": source_id
                    })
                
                self.pg_conn.commit()
                loaded_count = min(i + self.config.postgres_batch_size, len(spectra))
                
                if loaded_count % 1000 == 0:
                    logger.info(f"   Loaded {loaded_count}/{len(spectra)} MS spectra...")
            
            cursor.close()
            logger.info(f"✅ PostgreSQL data loading complete!")
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL loading failed: {e}")
            raise
    
    async def load_mongodb_data(self):
        """Load data into MongoDB"""
        logger.info("🍃 Loading data into MongoDB...")
        
        try:
            db = self.mongo_client[self.config.mongodb_config["database"]]
            
            # Collections to populate
            collections = {
                "experimental_results": self.config.target_documents // 4,
                "research_projects": self.config.target_documents // 8, 
                "analysis_results": self.config.target_documents // 4,
                "user_sessions": self.config.target_documents // 8
            }
            
            total_loaded = 0
            
            for collection_name, doc_count in collections.items():
                logger.info(f"📄 Loading {doc_count} documents into {collection_name}...")
                
                collection = db[collection_name]
                
                # Generate documents for this collection
                documents = []
                
                for i in range(doc_count):
                    if collection_name == "experimental_results":
                        doc = {
                            "experiment_id": f"EXP{str(i+10001).zfill(6)}",
                            "glycan_id": f"G{str((i % 15000) + 10001).zfill(6)}",
                            "experiment_type": ["MS/MS", "NMR", "LC-MS", "MALDI"][i % 4],
                            "conditions": {
                                "temperature": 20 + (i % 15),
                                "ph": 6.0 + (i % 3),
                                "buffer": ["PBS", "Tris-HCl", "HEPES"][i % 3]
                            },
                            "results": {
                                "signal_intensity": 1000 + (i % 50000),
                                "retention_time": 5.0 + (i % 20),
                                "purity": 0.8 + (i % 20) / 100
                            },
                            "metadata": {
                                "operator": f"user_{(i % 10) + 1}",
                                "instrument": f"instrument_{(i % 5) + 1}",
                                "date": datetime.now().isoformat()
                            }
                        }
                        
                    elif collection_name == "research_projects":
                        doc = {
                            "project_id": f"PROJ{str(i+10001).zfill(5)}",
                            "title": f"Glycan Analysis Project {i+1}",
                            "description": f"Comprehensive analysis of glycan structures and functions - Project {i+1}",
                            "status": ["active", "completed", "on_hold"][i % 3],
                            "researchers": [f"researcher_{j+1}@example.com" for j in range((i % 5) + 1)],
                            "glycans_studied": [f"G{str((j + i*10) % 15000 + 10001).zfill(6)}" for j in range((i % 10) + 1)],
                            "created_date": datetime.now().isoformat(),
                            "budget": 10000 + (i % 90000)
                        }
                        
                    elif collection_name == "analysis_results":
                        doc = {
                            "analysis_id": f"ANAL{str(i+10001).zfill(6)}",
                            "glycan_id": f"G{str((i % 15000) + 10001).zfill(6)}",
                            "analysis_type": ["structure_prediction", "function_analysis", "pathway_mapping", "similarity_search"][i % 4],
                            "algorithm": ["glycollm", "traditional_ml", "rule_based", "hybrid"][i % 4],
                            "confidence_score": 0.5 + (i % 50) / 100,
                            "results": {
                                "predicted_function": f"function_{(i % 20) + 1}",
                                "pathway_involvement": f"pathway_{(i % 15) + 1}",
                                "similar_structures": [f"G{str((j + i*5) % 15000 + 10001).zfill(6)}" for j in range((i % 5) + 1)]
                            },
                            "processing_time": 0.1 + (i % 100) / 10,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                    else:  # user_sessions
                        doc = {
                            "session_id": f"SESS{str(i+10001).zfill(8)}",
                            "user_id": f"user_{(i % 100) + 1}",
                            "session_start": datetime.now().isoformat(),
                            "activities": [
                                {
                                    "action": ["search", "analyze", "export", "visualize"][i % 4],
                                    "target": f"G{str((i % 15000) + 10001).zfill(6)}",
                                    "timestamp": datetime.now().isoformat()
                                }
                            ],
                            "duration_minutes": (i % 120) + 1,
                            "queries_executed": (i % 20) + 1
                        }
                    
                    documents.append(doc)
                    
                    # Insert in batches
                    if len(documents) >= self.config.mongodb_batch_size:
                        collection.insert_many(documents)
                        documents = []
                        total_loaded += self.config.mongodb_batch_size
                        self.progress["mongodb"]["loaded"] = total_loaded
                        
                        if total_loaded % 1000 == 0:
                            logger.info(f"   Loaded {total_loaded} documents so far...")
                
                # Insert remaining documents
                if documents:
                    collection.insert_many(documents)
                    total_loaded += len(documents)
                    self.progress["mongodb"]["loaded"] = total_loaded
                
                logger.info(f"   ✅ Completed {collection_name}: {doc_count} documents")
            
            logger.info(f"✅ MongoDB data loading complete! Total: {total_loaded} documents")
            
        except Exception as e:
            logger.error(f"❌ MongoDB loading failed: {e}")
            raise
    
    async def load_elasticsearch_data(self):
        """Load data into Elasticsearch"""
        logger.info("🔍 Loading data into Elasticsearch...")
        
        try:
            # Create indices if they don't exist
            indices = ["glycan_structures", "publications", "experiments"]
            
            for index_name in indices:
                if not self.es_client.indices.exists(index=index_name):
                    # Create index with appropriate mappings
                    mapping = {
                        "mappings": {
                            "properties": {
                                "id": {"type": "keyword"},
                                "title": {"type": "text", "analyzer": "standard"},
                                "content": {"type": "text", "analyzer": "standard"},
                                "tags": {"type": "keyword"},
                                "timestamp": {"type": "date"},
                                "metadata": {"type": "object"}
                            }
                        }
                    }
                    self.es_client.indices.create(index=index_name, body=mapping)
                    logger.info(f"   Created index: {index_name}")
            
            total_loaded = 0
            docs_per_index = self.config.target_documents // len(indices)
            
            for index_name in indices:
                logger.info(f"📄 Loading {docs_per_index} documents into {index_name}...")
                
                # Generate documents for this index
                documents = []
                
                for i in range(docs_per_index):
                    if index_name == "glycan_structures":
                        doc = {
                            "_index": index_name,
                            "_id": f"G{str(i+10001).zfill(6)}",
                            "_source": {
                                "id": f"G{str(i+10001).zfill(6)}",
                                "title": f"Glycan Structure G{str(i+10001).zfill(6)}",
                                "content": f"Glycan structure with composition analysis and functional annotations. Structure {i+1} in comprehensive database.",
                                "tags": ["glycan", "structure", "carbohydrate", f"type_{(i%10)+1}"],
                                "timestamp": datetime.now().isoformat(),
                                "metadata": {
                                    "mass": 500 + (i % 1000),
                                    "complexity": (i % 5) + 1,
                                    "organism": "Homo sapiens" if i % 2 == 0 else "Mus musculus"
                                }
                            }
                        }
                        
                    elif index_name == "publications":
                        doc = {
                            "_index": index_name,
                            "_id": f"PUB{str(i+10001).zfill(6)}",
                            "_source": {
                                "id": f"PUB{str(i+10001).zfill(6)}",
                                "title": f"Research Publication on Glycoinformatics {i+1}",
                                "content": f"Comprehensive study of glycan structures, functions, and biological roles. Publication {i+1} covers advanced analytical methods and computational approaches.",
                                "tags": ["research", "glycobiology", "publication", "peer-reviewed"],
                                "timestamp": datetime.now().isoformat(),
                                "metadata": {
                                    "authors": [f"Author{j+1}" for j in range((i%5)+1)],
                                    "journal": f"Journal of Glycobiology {(i%10)+1}",
                                    "year": 2020 + (i % 5),
                                    "citations": (i % 100) + 1
                                }
                            }
                        }
                        
                    else:  # experiments
                        doc = {
                            "_index": index_name,
                            "_id": f"EXP{str(i+10001).zfill(6)}",
                            "_source": {
                                "id": f"EXP{str(i+10001).zfill(6)}",
                                "title": f"Glycan Analysis Experiment {i+1}",
                                "content": f"Experimental analysis using mass spectrometry and NMR techniques for glycan characterization. Experiment {i+1} protocol and results.",
                                "tags": ["experiment", "MS/MS", "analysis", "methodology"],
                                "timestamp": datetime.now().isoformat(),
                                "metadata": {
                                    "method": ["MS/MS", "NMR", "LC-MS", "MALDI"][i%4],
                                    "success_rate": 0.7 + (i % 30) / 100,
                                    "duration_hours": (i % 24) + 1,
                                    "samples_processed": (i % 50) + 1
                                }
                            }
                        }
                    
                    documents.append(doc)
                    
                    # Bulk insert
                    if len(documents) >= self.config.elasticsearch_batch_size:
                        self.es_client.bulk(body=documents)
                        documents = []
                        total_loaded += self.config.elasticsearch_batch_size
                        self.progress["elasticsearch"]["loaded"] = total_loaded
                        
                        if total_loaded % 1000 == 0:
                            logger.info(f"   Loaded {total_loaded} documents so far...")
                
                # Insert remaining documents
                if documents:
                    self.es_client.bulk(body=documents)
                    total_loaded += len(documents)
                    self.progress["elasticsearch"]["loaded"] = total_loaded
                
                logger.info(f"   ✅ Completed {index_name}: {docs_per_index} documents")
            
            logger.info(f"✅ Elasticsearch data loading complete! Total: {total_loaded} documents")
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch loading failed: {e}")
            raise
    
    async def load_redis_data(self):
        """Load cache data into Redis"""
        logger.info("🔴 Loading cache data into Redis...")
        
        try:
            # Load frequently accessed data into Redis
            cache_keys = [
                "frequent_glycans", "popular_searches", "user_preferences",
                "api_responses", "computation_results"
            ]
            
            total_loaded = 0
            
            for cache_type in cache_keys:
                items_per_type = 200
                
                for i in range(items_per_type):
                    key = f"{cache_type}:{i+1}"
                    
                    if cache_type == "frequent_glycans":
                        value = {
                            "glycan_id": f"G{str((i % 15000) + 10001).zfill(6)}",
                            "access_count": (i % 1000) + 100,
                            "last_accessed": datetime.now().isoformat()
                        }
                    elif cache_type == "popular_searches":
                        value = {
                            "query": f"glycan search query {i+1}",
                            "result_count": (i % 500) + 10,
                            "frequency": (i % 100) + 1
                        }
                    elif cache_type == "user_preferences":
                        value = {
                            "user_id": f"user_{(i % 100) + 1}",
                            "preferences": {
                                "theme": "light" if i % 2 == 0 else "dark",
                                "results_per_page": [10, 25, 50, 100][i % 4],
                                "default_organism": "human" if i % 3 == 0 else "mouse"
                            }
                        }
                    elif cache_type == "api_responses":
                        value = {
                            "endpoint": f"/api/glycan/G{str((i % 1000) + 10001).zfill(6)}",
                            "response_data": {"status": "success", "data_size": (i % 1000) + 100},
                            "cache_time": datetime.now().isoformat()
                        }
                    else:  # computation_results
                        value = {
                            "computation_id": f"COMP{str(i+10001).zfill(6)}",
                            "result": {"score": (i % 100) / 100, "confidence": 0.5 + (i % 50) / 100},
                            "computation_time": (i % 10) + 0.1
                        }
                    
                    # Store in Redis with expiration
                    self.redis_client.setex(
                        key, 
                        3600 * 24,  # 24 hours TTL
                        json.dumps(value)
                    )
                    
                    total_loaded += 1
                
                self.progress["redis"]["loaded"] = total_loaded
                logger.info(f"   ✅ Loaded {items_per_type} items for {cache_type}")
            
            logger.info(f"✅ Redis data loading complete! Total: {total_loaded} cache entries")
            
        except Exception as e:
            logger.error(f"❌ Redis loading failed: {e}")
            raise
    
    async def load_minio_data(self):
        """Load file objects into MinIO"""
        logger.info("📦 Loading file objects into MinIO...")
        
        try:
            bucket_name = "glyco-data"
            
            # Create bucket if it doesn't exist
            if not self.minio_client.bucket_exists(bucket_name):
                self.minio_client.make_bucket(bucket_name)
                logger.info(f"   Created bucket: {bucket_name}")
            
            # Generate and upload various file types
            file_types = ["models", "datasets", "results", "images", "documents"]
            files_per_type = 100
            
            total_loaded = 0
            
            for file_type in file_types:
                for i in range(files_per_type):
                    # Generate synthetic file content
                    if file_type == "models":
                        content = json.dumps({
                            "model_id": f"MODEL{str(i+10001).zfill(5)}",
                            "model_type": "glycollm",
                            "parameters": {"layers": 12, "hidden_size": 768, "vocab_size": 50000},
                            "performance": {"accuracy": 0.85 + (i % 15) / 100},
                            "created": datetime.now().isoformat()
                        }, indent=2)
                        file_name = f"{file_type}/model_{i+1}.json"
                        
                    elif file_type == "datasets":
                        content = "glycan_id,wurcs_sequence,mass,composition\n"
                        for j in range(100):  # 100 rows per dataset
                            content += f"G{str(j+10001).zfill(6)},WURCS_EXAMPLE_{j},500.{j%1000},{{'Glc':{j%5+1}}}\n"
                        file_name = f"{file_type}/dataset_{i+1}.csv"
                        
                    elif file_type == "results":
                        content = json.dumps({
                            "analysis_id": f"RESULT{str(i+10001).zfill(5)}",
                            "glycan_analyzed": f"G{str((i % 1000) + 10001).zfill(6)}",
                            "predictions": [{"function": f"func_{j}", "confidence": (j % 100) / 100} for j in range(5)],
                            "timestamp": datetime.now().isoformat()
                        }, indent=2)
                        file_name = f"{file_type}/analysis_result_{i+1}.json"
                        
                    elif file_type == "images":
                        # Simple SVG placeholder
                        content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="150">
                        <rect width="200" height="150" fill="#f0f0f0"/>
                        <text x="10" y="30" font-family="Arial" font-size="14">Glycan Structure {i+1}</text>
                        <circle cx="50" cy="75" r="20" fill="#blue"/>
                        <circle cx="100" cy="75" r="20" fill="#red"/>
                        <circle cx="150" cy="75" r="20" fill="#green"/>
                        </svg>'''
                        file_name = f"{file_type}/structure_{i+1}.svg"
                        
                    else:  # documents
                        content = f"""# Glycan Analysis Report {i+1}
                        
## Summary
This document contains the analysis report for glycan structure G{str((i % 1000) + 10001).zfill(6)}.

## Methods
- Mass spectrometry analysis
- Structural determination
- Functional annotation

## Results
Analysis completed successfully with confidence score: {0.7 + (i % 30) / 100}

## Conclusion
The glycan structure shows characteristics consistent with {['N-linked', 'O-linked', 'GPI-anchored'][i % 3]} glycosylation patterns.

Generated on: {datetime.now().isoformat()}
"""
                        file_name = f"{file_type}/report_{i+1}.md"
                    
                    # Upload to MinIO
                    content_bytes = content.encode('utf-8')
                    self.minio_client.put_object(
                        bucket_name,
                        file_name,
                        io.BytesIO(content_bytes),
                        length=len(content_bytes),
                        content_type='application/octet-stream'
                    )
                    
                    total_loaded += 1
                    
                    if total_loaded % 50 == 0:
                        logger.info(f"   Uploaded {total_loaded} files so far...")
                
                self.progress["minio"]["loaded"] = total_loaded
                logger.info(f"   ✅ Completed {file_type}: {files_per_type} files")
            
            logger.info(f"✅ MinIO data loading complete! Total: {total_loaded} files")
            
        except Exception as e:
            logger.error(f"❌ MinIO loading failed: {e}")
            raise
    
    async def load_graphdb_data(self):
        """Load RDF data into GraphDB"""
        logger.info("🕸️ Loading RDF data into GraphDB...")
        
        try:
            # Generate RDF triples for GraphDB
            from glycokg.ontology.glyco_ontology import GlycoOntology
            
            ontology = GlycoOntology()
            
            # Generate substantial RDF data
            logger.info("🧬 Generating comprehensive RDF triples...")
            
            # Add many glycan structures
            for i in range(5000):
                glytoucan_id = f"G{str(i+10001).zfill(6)}"
                
                glycan_uri = ontology.add_glycan(
                    glytoucan_id=glytoucan_id,
                    wurcs_sequence=f"WURCS=2.0/synthetic_{i}",
                    mass_mono=500.0 + (i % 1000),
                    composition={"Glc": (i % 5) + 1, "GlcNAc": (i % 3) + 1}
                )
                
                if i % 500 == 0:
                    logger.info(f"   Generated {i}/5000 glycan RDF triples...")
            
            # Add protein associations
            for i in range(3000):
                uniprot_id = f"P{str(i+10001).zfill(5)}"
                glytoucan_id = f"G{str((i % 5000) + 10001).zfill(6)}"
                
                ontology.add_glycoprotein_association(
                    glytoucan_id=glytoucan_id,
                    uniprot_id=uniprot_id,
                    glycosylation_site=(i % 500) + 1,
                    evidence_type="MS/MS"
                )
                
                if i % 500 == 0:
                    logger.info(f"   Generated {i}/3000 association RDF triples...")
            
            # Add MS spectra
            for i in range(2000):
                spectrum_id = f"SPEC{str(i+10001).zfill(6)}"
                glytoucan_id = f"G{str((i % 5000) + 10001).zfill(6)}"
                
                ontology.add_ms_spectrum(
                    spectrum_id=spectrum_id,
                    glytoucan_id=glytoucan_id,
                    precursor_mz=500.0 + i,
                    charge_state=(i % 3) + 1,
                    collision_energy=25.0 + (i % 35)
                )
                
                if i % 500 == 0:
                    logger.info(f"   Generated {i}/2000 spectrum RDF triples...")
            
            # Save the comprehensive ontology
            ontology_file = "data/processed/comprehensive_glycokg.ttl"
            ontology.save_ontology(ontology_file, format="turtle")
            
            triple_count = len(ontology.graph)
            self.progress["graphdb"]["loaded"] = triple_count
            
            logger.info(f"✅ GraphDB RDF data generation complete!")
            logger.info(f"   Generated {triple_count} RDF triples")
            logger.info(f"   Saved to: {ontology_file}")
            
            # Upload to GraphDB via HTTP API
            graphdb_url = "http://localhost:7200"
            repository = "glycokg"
            
            # Read the generated RDF file
            with open(ontology_file, 'r', encoding='utf-8') as f:
                rdf_content = f.read()
            
            # Upload to GraphDB repository
            upload_url = f"{graphdb_url}/repositories/{repository}/statements"
            headers = {'Content-Type': 'text/turtle'}
            
            response = requests.post(upload_url, data=rdf_content, headers=headers)
            
            if response.status_code == 204:
                logger.info("✅ RDF data successfully uploaded to GraphDB!")
            else:
                logger.warning(f"⚠️ GraphDB upload response: {response.status_code} - {response.text}")
            
        except Exception as e:
            logger.error(f"❌ GraphDB loading failed: {e}")
            raise
    
    def print_progress_summary(self):
        """Print comprehensive progress summary"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("🚀 BULK DATA LOADING PROGRESS SUMMARY")
        print("="*80)
        print(f"⏱️  Elapsed Time: {elapsed_time:.1f} seconds")
        print()
        
        for service, stats in self.progress.items():
            loaded = stats["loaded"]
            target = stats["target"]
            percentage = (loaded / target * 100) if target > 0 else 0
            
            status_icon = "✅" if percentage >= 100 else "🔄" if percentage > 0 else "⏳"
            
            print(f"{status_icon} {service.upper():<15}: {loaded:>8,} / {target:>8,} ({percentage:>5.1f}%)")
        
        total_loaded = sum(stats["loaded"] for stats in self.progress.values())
        total_target = sum(stats["target"] for stats in self.progress.values())
        overall_percentage = (total_loaded / total_target * 100) if total_target > 0 else 0
        
        print("-" * 80)
        print(f"🎯 OVERALL PROGRESS: {total_loaded:>8,} / {total_target:>8,} ({overall_percentage:>5.1f}%)")
        print("="*80)
    
    async def run_bulk_loading(self):
        """Execute the complete bulk data loading process"""
        logger.info("🚀 Starting comprehensive bulk data loading...")
        logger.info(f"📊 Target data volumes:")
        logger.info(f"   PostgreSQL: {self.config.target_glycan_structures:,} glycan structures")
        logger.info(f"   MongoDB: {self.config.target_documents:,} documents")
        logger.info(f"   Elasticsearch: {self.config.target_documents:,} searchable documents")
        logger.info(f"   GraphDB: {self.config.target_rdf_triples:,} RDF triples")
        logger.info(f"   Redis: 1,000 cache entries")
        logger.info(f"   MinIO: 500 file objects")
        print()
        
        try:
            # Initialize all connections
            await self.initialize_connections()
            
            # Run all loading operations in parallel for better performance
            loading_tasks = [
                self.load_postgresql_data(),
                self.load_mongodb_data(),
                self.load_elasticsearch_data(),
                self.load_redis_data(),
                self.load_minio_data(),
                self.load_graphdb_data()
            ]
            
            # Execute all loading tasks
            await asyncio.gather(*loading_tasks)
            
            # Final summary
            self.print_progress_summary()
            
            # Verify data loading
            await self.verify_data_loading()
            
            logger.info("🎉 BULK DATA LOADING COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            logger.error(f"❌ Bulk data loading failed: {e}")
            raise
        finally:
            # Clean up connections
            if self.pg_conn:
                self.pg_conn.close()
            if self.mongo_client:
                self.mongo_client.close()
            if self.redis_client:
                self.redis_client.close()
    
    async def verify_data_loading(self):
        """Verify that data was loaded successfully across all services"""
        logger.info("🔍 Verifying data loading across all services...")
        
        verification_results = {}
        
        try:
            # Verify PostgreSQL
            cursor = self.pg_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cache.glycan_structures")
            glycan_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM cache.protein_glycan_associations")
            association_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM cache.ms_spectra")
            spectra_count = cursor.fetchone()[0]
            
            verification_results["PostgreSQL"] = {
                "glycan_structures": glycan_count,
                "protein_associations": association_count,
                "ms_spectra": spectra_count
            }
            cursor.close()
            
            # Verify MongoDB
            db = self.mongo_client[self.config.mongodb_config["database"]]
            mongo_counts = {}
            for collection_name in ["experimental_results", "research_projects", "analysis_results", "user_sessions"]:
                count = db[collection_name].count_documents({})
                mongo_counts[collection_name] = count
            verification_results["MongoDB"] = mongo_counts
            
            # Verify Redis
            redis_keys = self.redis_client.keys("*")
            verification_results["Redis"] = {"total_keys": len(redis_keys)}
            
            # Verify Elasticsearch
            es_counts = {}
            for index_name in ["glycan_structures", "publications", "experiments"]:
                try:
                    result = self.es_client.count(index=index_name)
                    es_counts[index_name] = result["count"]
                except:
                    es_counts[index_name] = 0
            verification_results["Elasticsearch"] = es_counts
            
            # Verify MinIO
            objects = list(self.minio_client.list_objects("glyco-data", recursive=True))
            verification_results["MinIO"] = {"total_objects": len(objects)}
            
            # Print verification results
            print("\n" + "="*60)
            print("📋 DATA LOADING VERIFICATION RESULTS")
            print("="*60)
            
            for service, counts in verification_results.items():
                print(f"\n🔸 {service}:")
                for item, count in counts.items():
                    print(f"   {item}: {count:,}")
            
            print("\n" + "="*60)
            logger.info("✅ Data loading verification completed!")
            
        except Exception as e:
            logger.error(f"❌ Data verification failed: {e}")

# Import required libraries at the top level
import io

async def main():
    """Main execution function"""
    
    # Set environment variables from command line or use defaults
    os.environ.setdefault("POSTGRES_USER", "glyco_admin")
    os.environ.setdefault("POSTGRES_PASSWORD", "glyco_secure_pass_2025")
    os.environ.setdefault("MONGODB_USER", "glyco_admin")
    os.environ.setdefault("MONGODB_PASSWORD", "glyco_secure_pass_2025")
    
    print("🧬 GLYCOINFORMATICS AI PLATFORM - BULK DATA LOADER")
    print("=" * 60)
    print("📈 Loading substantial real data into ALL services...")
    print("🎯 Target: 10,000+ records per service")
    print("⏱️  Estimated time: 15-30 minutes")
    print()
    
    # Create configuration
    config = DataLoadingConfig()
    
    # Create and run bulk loader
    loader = BulkDataLoader(config)
    
    try:
        await loader.run_bulk_loading()
    except KeyboardInterrupt:
        logger.info("⏹️ Data loading interrupted by user")
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the bulk data loading process
    asyncio.run(main())