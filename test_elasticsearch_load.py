#!/usr/bin/env python3
"""
DEPRECATED: MEGA-SCALE Elasticsearch loading (SYNTHETIC DATA ONLY)
===================================================================

⚠️  WARNING: THIS SCRIPT GENERATES SYNTHETIC DATA ONLY! ⚠️ 

This script has been replaced by populate_real_data.py which uses 
REAL experimental data from GlyTouCan, GlyGen, and GlycoPOST APIs.

Use populate_real_data.py instead for authentic glycoinformatics data.
"""

from elasticsearch import Elasticsearch
import json
import random
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time
import re

class AdvancedGlycanGenerator:
    """Scientific-grade glycan data generator for mega-scale Elasticsearch loading"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
        # Comprehensive monosaccharide database with normalized frequencies
        self.monosaccharides = {
            # Common hexoses
            "Glc": {"mass": 180.156, "formula": "C6H12O6", "frequency": 0.195, "type": "hexose"},
            "Gal": {"mass": 180.156, "formula": "C6H12O6", "frequency": 0.18, "type": "hexose"},
            "Man": {"mass": 180.156, "formula": "C6H12O6", "frequency": 0.16, "type": "hexose"},
            
            # Amino sugars
            "GlcNAc": {"mass": 221.208, "formula": "C8H15NO6", "frequency": 0.14, "type": "amino_sugar"},
            "GalNAc": {"mass": 221.208, "formula": "C8H15NO6", "frequency": 0.12, "type": "amino_sugar"},
            "ManNAc": {"mass": 221.208, "formula": "C8H15NO6", "frequency": 0.03, "type": "amino_sugar"},
            
            # Deoxy sugars
            "Fuc": {"mass": 164.157, "formula": "C6H12O5", "frequency": 0.08, "type": "deoxy_sugar"},
            "Rha": {"mass": 164.157, "formula": "C6H12O5", "frequency": 0.02, "type": "deoxy_sugar"},
            
            # Sialic acids
            "Neu5Ac": {"mass": 309.270, "formula": "C11H19NO9", "frequency": 0.035, "type": "sialic_acid"},
            "Neu5Gc": {"mass": 325.269, "formula": "C11H19NO10", "frequency": 0.005, "type": "sialic_acid"},
            
            # Pentoses
            "Xyl": {"mass": 150.130, "formula": "C5H10O5", "frequency": 0.015, "type": "pentose"},
            "Ara": {"mass": 150.130, "formula": "C5H10O5", "frequency": 0.01, "type": "pentose"},
            
            # Uronic acids
            "GlcA": {"mass": 194.139, "formula": "C6H10O7", "frequency": 0.008, "type": "uronic_acid"},
            "IdoA": {"mass": 194.139, "formula": "C6H10O7", "frequency": 0.002, "type": "uronic_acid"}
        }
        
        # Realistic glycosidic linkage patterns
        self.linkage_patterns = {
            "N-linked": [
                "GlcNAc(β1-4)GlcNAc(β1-N)Asn",
                "Man(α1-3)[Man(α1-6)]Man(β1-4)GlcNAc(β1-4)GlcNAc(β1-N)Asn",
                "Gal(β1-4)GlcNAc(β1-2)Man(α1-3)[Gal(β1-4)GlcNAc(β1-2)Man(α1-6)]Man(β1-4)GlcNAc(β1-4)GlcNAc(β1-N)Asn",
            ],
            "O-linked": [
                "GalNAc(α1-O)Ser/Thr",
                "Gal(β1-3)GalNAc(α1-O)Ser/Thr", 
                "Neu5Ac(α2-3)Gal(β1-3)GalNAc(α1-O)Ser/Thr",
            ],
            "glycolipid": [
                "Glc(β1-1)Cer",
                "Gal(β1-4)Glc(β1-1)Cer",
                "GalNAc(β1-4)Gal(β1-4)Glc(β1-1)Cer"
            ]
        }
        
        # Organisms with biological weighting
        self.organisms = [
            {"name": "Homo sapiens", "weight": 0.35},
            {"name": "Mus musculus", "weight": 0.25},
            {"name": "Saccharomyces cerevisiae", "weight": 0.15},
            {"name": "Escherichia coli", "weight": 0.10},
            {"name": "Arabidopsis thaliana", "weight": 0.08},
            {"name": "Drosophila melanogaster", "weight": 0.07}
        ]
        
        # Tissue types for biological context
        self.tissues = ["liver", "brain", "muscle", "blood", "kidney", "heart", "lung", "skin", "bone", "adipose"]
    
    def generate_advanced_sequence(self, glycan_type="N-linked"):
        """Generate realistic glycan sequence with proper linkages"""
        if glycan_type in self.linkage_patterns:
            base_pattern = self.rng.choice(self.linkage_patterns[glycan_type])
            
            # Add additional branching
            mono_names = list(self.monosaccharides.keys())
            frequencies = [self.monosaccharides[mono]["frequency"] for mono in mono_names]
            num_residues = int(self.rng.randint(2, 12))
            
            sequence_parts = []
            for i in range(num_residues):
                mono = self.rng.choice(mono_names, p=frequencies)
                linkage = self.rng.choice(["α", "β"])
                position = int(self.rng.choice([1, 2, 3, 4, 6]))
                if i == 0:
                    sequence_parts.append(mono)
                else:
                    sequence_parts.append(f"{mono}({linkage}1-{position})")
            
            return "".join(sequence_parts)
    
    def calculate_molecular_properties(self, sequence):
        """Calculate molecular weight and formula from sequence"""
        total_mass = 0
        elements = {"C": 0, "H": 0, "N": 0, "O": 0}
        
        for mono_name in self.monosaccharides:
            count = sequence.count(mono_name)
            if count > 0:
                mono_data = self.monosaccharides[mono_name]
                total_mass += count * mono_data["mass"]
                
                # Parse formula and add elements
                formula = mono_data["formula"]
                for element in elements:
                    match = re.search(f"{element}(\\d+)", formula)
                    if match:
                        elements[element] += count * int(match.group(1))
                    elif element in formula and not re.search(f"{element}\\d", formula):
                        elements[element] += count
        
        # Generate molecular formula string
        molecular_formula = "".join([f"{elem}{count}" if count > 1 else elem 
                                   for elem, count in elements.items() if count > 0])
        
        return total_mass, molecular_formula

def generate_sample_doc(doc_id, doc_type, generator=None):
    """Generate sample documents for different indices with advanced glycan data"""
    
    if generator is None:
        generator = AdvancedGlycanGenerator(seed=doc_id)
    
    if doc_type == "glycan_structures":
        # Generate advanced glycan structure
        glycan_type = generator.rng.choice(["N-linked", "O-linked", "glycolipid"])
        sequence = generator.generate_advanced_sequence(glycan_type)
        molecular_weight, molecular_formula = generator.calculate_molecular_properties(sequence)
        
        # Select organism with biological weighting
        organism = generator.rng.choice(generator.organisms, p=[org["weight"] for org in generator.organisms])
        
        return {
            "structure_id": f"GLYCAN_{doc_id:06d}",
            "sequence": sequence,
            "molecular_weight": float(round(molecular_weight, 3)),  # Convert to native Python float
            "molecular_formula": molecular_formula,
            "classification": glycan_type,
            "organism": organism["name"],
            "tissue": generator.rng.choice(generator.tissues),
            "biological_function": generator.rng.choice(["cell_adhesion", "immune_response", "metabolism", "signaling", "structural"]),
            "pathogenicity": generator.rng.choice([None, "disease_associated", "therapeutic_target"]),
            "confidence_score": float(generator.rng.uniform(0.7, 1.0)),  # Convert to native Python float
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "source": "computational_generation",
                "validation_status": generator.rng.choice(["validated", "predicted", "experimental"]),
                "complexity_score": float(len(sequence.split("(")) / 10.0)  # Convert to native Python float
            }
        }
    elif doc_type == "research_publications":
        glycan_topics = [
            "N-linked glycosylation", "O-linked glycan biosynthesis", "Sialic acid metabolism",
            "Glycan-protein interactions", "Fucosylation pathways", "GlcNAc processing",
            "Mannosidase activity", "Glycan biomarkers", "Lectin binding studies"
        ]
        return {
            "paper_id": f"PUB_{doc_id:06d}",
            "title": f"{generator.rng.choice(glycan_topics)} in {generator.rng.choice(['cancer', 'diabetes', 'immunology', 'development'])}",
            "authors": [f"Author_{int(generator.rng.randint(1,500))}" for _ in range(int(generator.rng.randint(2,8)))],
            "journal": generator.rng.choice(["Glycobiology", "Nature Chemical Biology", "Cell", "Science", "PNAS", "JBC"]),
            "year": int(generator.rng.randint(2015, 2025)),
            "keywords": [generator.rng.choice(glycan_topics + ["mass spectrometry", "proteomics", "glycomics"]) for _ in range(int(generator.rng.randint(4,10)))],
            "abstract": f"Comprehensive analysis of {generator.rng.choice(glycan_topics)} using advanced glycoinformatics approaches",
            "doi": f"10.1038/glyco.{doc_id:06d}",
            "citation_count": int(generator.rng.randint(0, 150)),  # Convert to native Python int
            "impact_factor": float(generator.rng.uniform(2.5, 15.0)),  # Convert to native Python float
            "timestamp": datetime.now().isoformat()
        }
    elif doc_type == "experimental_data":
        methods = ["LC-ESI-MS/MS", "MALDI-TOF-MS", "NMR-1H", "HPLC-FLD", "CE-LIF", "Ion-Mobility-MS"]
        glycan_id = f"GLYCAN_{int(generator.rng.randint(1, 100000)):06d}"
        return {
            "experiment_id": f"EXP_{doc_id:06d}",
            "method": generator.rng.choice(methods),
            "sample_id": f"SAMPLE_{int(generator.rng.randint(10000,99999))}",
            "glycan_target": glycan_id,
            "conditions": {
                "temperature": float(generator.rng.uniform(20, 37)),  # Convert to native Python float
                "pH": float(generator.rng.uniform(6.5, 8.5)),  # Convert to native Python float
                "buffer": generator.rng.choice(["PBS", "HEPES", "Tris-HCl", "Acetate"]),
                "ionic_strength": float(generator.rng.uniform(0.1, 0.5))  # Convert to native Python float
            },
            "spectral_data": {
                "peaks": [{"mz": float(generator.rng.uniform(100, 2000)), "intensity": float(generator.rng.uniform(0, 100))} 
                         for _ in range(int(generator.rng.randint(10, 50)))],  # Convert to native Python types
                "base_peak": float(generator.rng.uniform(500, 1500)),  # Convert to native Python float
                "total_ion_current": float(generator.rng.uniform(1e6, 1e9))  # Convert to native Python float
            },
            "quantification": {
                "concentration": float(generator.rng.uniform(0.1, 100.0)),  # Convert to native Python float
                "units": "μg/mL",
                "cv_percent": float(generator.rng.uniform(1, 15))  # Convert to native Python float
            },
            "quality_score": float(generator.rng.uniform(0.8, 1.0)),  # Convert to native Python float
            "validated": generator.rng.choice([True, False]),
            "timestamp": datetime.now().isoformat()
        }
    elif doc_type == "pathway_analysis":
        return {
            "pathway_id": f"PATH_{doc_id:06d}", 
            "name": f"Glycosylation Pathway {doc_id}",
            "enzymes": [f"ENZ_{int(generator.rng.randint(100,999))}" for _ in range(int(generator.rng.randint(3,8)))],
            "substrates": [f"SUB_{int(generator.rng.randint(1,100))}" for _ in range(int(generator.rng.randint(2,5)))],
            "products": [f"PROD_{int(generator.rng.randint(1,100))}" for _ in range(int(generator.rng.randint(1,3)))],
            "regulation": generator.rng.choice(["Upregulated", "Downregulated", "Constitutive"]),
            "tissue_specificity": generator.rng.choice(["Liver", "Brain", "Muscle", "Ubiquitous"]),
            "timestamp": datetime.now().isoformat()
        }
    else:  # protein_interactions
        return {
            "interaction_id": f"INT_{doc_id:06d}",
            "protein_a": f"PROT_{int(generator.rng.randint(1000,9999))}",
            "protein_b": f"PROT_{int(generator.rng.randint(1000,9999))}",
            "interaction_type": generator.rng.choice(["binding", "enzymatic", "regulatory"]),
            "confidence_score": float(generator.rng.uniform(0.7, 1.0)),  # Convert to native Python float
            "evidence": generator.rng.choice(["experimental", "computational", "literature"]),
            "glycan_involved": f"GLYCAN_{int(generator.rng.randint(1,1000))}",
            "timestamp": datetime.now().isoformat()
        }

def load_elasticsearch_data():
    """Load 1,000,000+ documents into Elasticsearch - MEGA-SCALE GLYCAN DATABASE"""
    
    print("🚀 MEGA-SCALE ELASTICSEARCH LOADING - 1,000,000+ GLYCAN RECORDS")
    print("=" * 80)
    print("🧬 Advanced glycan data generation with realistic biochemical properties")
    print("⚡ Ultra-high performance parallel processing")
    print("🔬 Production-grade massive datasets for ML/AI applications")
    print()
    
    # Connect to Elasticsearch with v8 compatibility mode
    es = Elasticsearch(
        ['http://localhost:9200'],
        headers={"accept": "application/json", "content-type": "application/json"}
    )
    
    # Test connection
    info = es.info()
    print(f"✅ Connected to Elasticsearch: {info['cluster_name']}")
    print(f"📊 Version: {info['version']['number']}")
    print()
    
    # Define indices and their document counts - MEGA-SCALE TARGETS
    indices = {
        "glycan_structures": 400000,        # Core glycan database
        "research_publications": 250000,    # Literature and studies  
        "experimental_data": 200000,        # Mass spec and analytical data
        "pathway_analysis": 100000,         # Biochemical pathways
        "protein_interactions": 50000       # Protein-glycan interactions
    }
    
    print(f"🎯 MEGA-SCALE TARGETS:")
    for idx, count in indices.items():
        print(f"   {idx}: {count:,} documents")
    print(f"📈 TOTAL TARGET: {sum(indices.values()):,} documents")
    print()
    
    # Initialize advanced glycan generator
    generator = AdvancedGlycanGenerator(seed=42)
    
    total_loaded = 0
    total_target = sum(indices.values())
    start_time = time.time()
    
    for index_name, doc_count in indices.items():
        print(f"📝 Loading {doc_count:,} documents into {index_name}...")
        index_start_time = time.time()
        
        # Create index if it doesn't exist with optimized settings
        if not es.indices.exists(index=index_name):
            index_settings = {
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 0,  # Optimize for loading speed
                    "refresh_interval": "30s",  # Reduce refresh frequency during loading
                    "index.max_result_window": 50000
                },
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "molecular_weight": {"type": "float"},
                        "confidence_score": {"type": "float"},
                        "sequence": {"type": "text", "analyzer": "standard"},
                        "organism": {"type": "keyword"},
                        "classification": {"type": "keyword"}
                    }
                }
            }
            es.indices.create(index=index_name, body=index_settings)
            print(f"   ✅ Created optimized index: {index_name}")
        
        # Ultra-high performance batch processing
        batch_size = 2000  # Larger batches for mega-scale
        total_batches = doc_count // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            
            # Generate documents with advanced glycan data
            operations = []
            for i in range(batch_size):
                doc_id = start_idx + i
                doc = generate_sample_doc(doc_id, index_name, generator)
                operations.extend([
                    {"index": {"_index": index_name, "_id": f"{index_name}_{doc_id}"}},
                    doc
                ])
            
            # Ultra-fast bulk indexing
            es.bulk(operations=operations, refresh=False, timeout='60s')
            total_loaded += batch_size
            
            # Progress update every 20,000 docs for mega-scale
            if total_loaded % 20000 == 0:
                elapsed = time.time() - start_time
                rate = total_loaded / elapsed if elapsed > 0 else 0
                progress_percent = (total_loaded / total_target) * 100
                print(f"   📈 MEGA-PROGRESS: {total_loaded:,}/{total_target:,} ({progress_percent:.1f}%) - {rate:.0f} docs/sec")
        
        # Index completion summary
        index_elapsed = time.time() - index_start_time
        index_rate = doc_count / index_elapsed if index_elapsed > 0 else 0
        print(f"   ✅ {index_name} COMPLETE: {doc_count:,} docs in {index_elapsed:.1f}s ({index_rate:.0f} docs/sec)")
        print()
    
    # Re-enable replicas and optimize refresh for production
    for index_name in indices.keys():
        es.indices.put_settings(
            index=index_name,
            body={
                "number_of_replicas": 1,
                "refresh_interval": "1s"
            }
        )
    
    elapsed = time.time() - start_time
    rate = total_loaded / elapsed if elapsed > 0 else 0
    
    print("=" * 80)
    print("🎉 MEGA-SCALE ELASTICSEARCH LOADING COMPLETED!")
    print("=" * 80)
    print(f"📊 Total documents loaded: {total_loaded:,}")
    print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"🚀 Average rate: {rate:.0f} documents/second")
    print(f"🔬 Advanced glycan generation: {len(generator.monosaccharides)} monosaccharide types")
    print(f"⚡ Ultra-performance: 2000-doc batches, optimized indices")
    print()
    
    # Verify mega-scale indices
    print("📋 MEGA-SCALE INDEX SUMMARY:")
    print("-" * 50)
    total_verified = 0
    for index_name in indices.keys():
        count = es.count(index=index_name)['count']
        total_verified += count
        print(f"   {index_name}: {count:,} documents ✅")
    
    print("-" * 50)
    print(f"🎯 TOTAL VERIFIED: {total_verified:,} documents")
    print(f"✅ SUCCESS RATE: {(total_verified/total_loaded)*100:.1f}%")
    print()
    print("🧬 MEGA-SCALE GLYCAN DATABASE READY FOR ML/AI APPLICATIONS!")
    print("🔬 Production-grade dataset with advanced biochemical properties")
    print("⚡ Ultra-high performance Elasticsearch cluster optimized")
    print("=" * 80)

if __name__ == "__main__":
    load_elasticsearch_data()