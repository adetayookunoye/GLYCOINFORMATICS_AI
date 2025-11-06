"""
RDF Graph Population Pipeline for GlycoKG

Converts database records (PostgreSQL, MongoDB) into RDF triples
and populates the knowledge graph using the GlycoKG ontology.

Handles:
- Glycan structures and properties
- Protein-glycan interactions
- Disease associations
- MS/MS experimental data
- Pathways and functions

Author: Adetayo Research Team
Date: November 2025
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Set
from dataclasses import dataclass
import json
from datetime import datetime
import asyncio

import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient
import redis

from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD
from rdflib.namespace import DC, DCTERMS, FOAF

from glycokg.ontology.glyco_ontology import GlycoOntology

logger = logging.getLogger(__name__)


@dataclass
class GraphStatistics:
    """Statistics for populated graph"""
    total_triples: int = 0
    glycan_count: int = 0
    protein_count: int = 0
    disease_count: int = 0
    experiment_count: int = 0
    linkage_count: int = 0
    interaction_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'total_triples': self.total_triples,
            'glycan_count': self.glycan_count,
            'protein_count': self.protein_count,
            'disease_count': self.disease_count,
            'experiment_count': self.experiment_count,
            'linkage_count': self.linkage_count,
            'interaction_count': self.interaction_count
        }


class GraphPopulator:
    """Populate RDF knowledge graph from database records"""
    
    def __init__(self,
                 postgres_config: Dict,
                 mongodb_config: Dict,
                 redis_config: Dict,
                 ontology: Optional[GlycoOntology] = None):
        """
        Initialize graph populator
        
        Args:
            postgres_config: PostgreSQL connection config
            mongodb_config: MongoDB connection config
            redis_config: Redis connection config
            ontology: GlycoKG ontology instance
        """
        self.postgres_config = postgres_config
        self.mongodb_config = mongodb_config
        self.redis_config = redis_config
        
        # Initialize ontology and graph
        self.ontology = ontology or GlycoOntology()
        self.graph = self.ontology.graph
        
        # Database connections
        self.pg_conn = None
        self.mongo_client = None
        self.redis_client = None
        
        # Statistics
        self.stats = GraphStatistics()
        
        # Cache for URIs to avoid duplicates
        self.glycan_uris: Dict[str, URIRef] = {}
        self.protein_uris: Dict[str, URIRef] = {}
        self.disease_uris: Dict[str, URIRef] = {}
        
        logger.info("GraphPopulator initialized")
    
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
    
    def get_glycan_uri(self, glytoucan_id: str) -> URIRef:
        """Get or create URI for glycan"""
        if glytoucan_id not in self.glycan_uris:
            self.glycan_uris[glytoucan_id] = self.ontology.GLYCO[f"glycan_{glytoucan_id}"]
        return self.glycan_uris[glytoucan_id]
    
    def get_protein_uri(self, protein_id: str) -> URIRef:
        """Get or create URI for protein"""
        if protein_id not in self.protein_uris:
            self.protein_uris[protein_id] = self.ontology.GLYCO[f"protein_{protein_id}"]
        return self.protein_uris[protein_id]
    
    def get_disease_uri(self, disease_name: str) -> URIRef:
        """Get or create URI for disease"""
        # Clean disease name for URI
        clean_name = disease_name.replace(' ', '_').replace('/', '_')
        if disease_name not in self.disease_uris:
            self.disease_uris[disease_name] = self.ontology.GLYCO[f"disease_{clean_name}"]
        return self.disease_uris[disease_name]
    
    def populate_glycan_from_postgres(self, glytoucan_id: str) -> bool:
        """
        Populate glycan data from PostgreSQL
        
        Args:
            glytoucan_id: GlyTouCan accession ID
            
        Returns:
            True if successful
        """
        try:
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
                record = cursor.fetchone()
                
                if not record:
                    return False
                
                # Create glycan URI
                glycan_uri = self.get_glycan_uri(glytoucan_id)
                
                # Add type
                self.graph.add((glycan_uri, RDF.type, self.ontology.Glycan))
                
                # Add properties
                self.graph.add((glycan_uri, self.ontology.glytoucan_id, Literal(glytoucan_id)))
                
                if record['mass']:
                    self.graph.add((glycan_uri, self.ontology.has_mass, Literal(float(record['mass']), datatype=XSD.float)))
                
                if record['wurcs']:
                    self.graph.add((glycan_uri, self.ontology.wurcs, Literal(record['wurcs'])))
                
                if record['glycoct']:
                    self.graph.add((glycan_uri, self.ontology.glycoct, Literal(record['glycoct'])))
                
                if record['iupac_extended']:
                    self.graph.add((glycan_uri, self.ontology.iupac, Literal(record['iupac_extended'])))
                
                if record['composition']:
                    self.graph.add((glycan_uri, self.ontology.composition, Literal(record['composition'])))
                
                if record['num_monosaccharides']:
                    self.graph.add((glycan_uri, self.ontology.num_monosaccharides, Literal(int(record['num_monosaccharides']), datatype=XSD.integer)))
                
                if record['num_linkages']:
                    self.graph.add((glycan_uri, self.ontology.num_linkages, Literal(int(record['num_linkages']), datatype=XSD.integer)))
                
                if record['biological_source']:
                    self.graph.add((glycan_uri, self.ontology.biological_source, Literal(record['biological_source'])))
                
                if record['tissue_location']:
                    self.graph.add((glycan_uri, self.ontology.tissue_location, Literal(record['tissue_location'])))
                
                self.stats.glycan_count += 1
                return True
                
        except Exception as e:
            logger.error(f"Error populating glycan {glytoucan_id}: {e}")
            return False
    
    def populate_annotations_from_mongodb(self, glytoucan_id: str) -> bool:
        """
        Populate glycan annotations from MongoDB
        
        Args:
            glytoucan_id: GlyTouCan accession ID
            
        Returns:
            True if successful
        """
        try:
            collection = self.mongo_db['glycan_annotations']
            doc = collection.find_one({'glytoucan_id': glytoucan_id})
            
            if not doc:
                return False
            
            glycan_uri = self.get_glycan_uri(glytoucan_id)
            
            # Add description
            if doc.get('description'):
                self.graph.add((glycan_uri, DCTERMS.description, Literal(doc['description'])))
            
            # Add disease associations
            if doc.get('disease_associations'):
                for disease_name in doc['disease_associations']:
                    disease_uri = self.get_disease_uri(disease_name)
                    
                    # Create disease if not exists
                    if (disease_uri, RDF.type, self.ontology.Disease) not in self.graph:
                        self.graph.add((disease_uri, RDF.type, self.ontology.Disease))
                        self.graph.add((disease_uri, RDFS.label, Literal(disease_name)))
                        self.stats.disease_count += 1
                    
                    # Link glycan to disease
                    self.graph.add((glycan_uri, self.ontology.associated_with_disease, disease_uri))
                    self.stats.interaction_count += 1
            
            # Add protein interactions
            if doc.get('protein_interactions'):
                for protein_id in doc['protein_interactions']:
                    protein_uri = self.get_protein_uri(protein_id)
                    
                    # Create protein if not exists
                    if (protein_uri, RDF.type, self.ontology.Protein) not in self.graph:
                        self.graph.add((protein_uri, RDF.type, self.ontology.Protein))
                        self.graph.add((protein_uri, self.ontology.protein_id, Literal(protein_id)))
                        self.stats.protein_count += 1
                    
                    # Link glycan to protein
                    self.graph.add((glycan_uri, self.ontology.binds_to_protein, protein_uri))
                    self.stats.interaction_count += 1
            
            # Add functions
            if doc.get('functions'):
                for function in doc['functions']:
                    self.graph.add((glycan_uri, self.ontology.has_function, Literal(function)))
            
            # Add pathways
            if doc.get('pathways'):
                for pathway in doc['pathways']:
                    pathway_uri = self.ontology.GLYCO[f"pathway_{pathway.replace(' ', '_')}"]
                    if (pathway_uri, RDF.type, self.ontology.BiologicalPathway) not in self.graph:
                        self.graph.add((pathway_uri, RDF.type, self.ontology.BiologicalPathway))
                        self.graph.add((pathway_uri, RDFS.label, Literal(pathway)))
                    self.graph.add((glycan_uri, self.ontology.participates_in_pathway, pathway_uri))
            
            return True
            
        except Exception as e:
            logger.error(f"Error populating annotations for {glytoucan_id}: {e}")
            return False
    
    def populate_spectra_from_mongodb(self, glytoucan_id: str) -> bool:
        """
        Populate MS/MS spectra data from MongoDB
        
        Args:
            glytoucan_id: GlyTouCan accession ID
            
        Returns:
            True if successful
        """
        try:
            collection = self.mongo_db['glycan_spectra']
            doc = collection.find_one({'glytoucan_id': glytoucan_id})
            
            if not doc:
                return False
            
            glycan_uri = self.get_glycan_uri(glytoucan_id)
            
            # Create MS experiment
            experiment_id = f"exp_{glytoucan_id}_{doc.get('spectrum_id', 'unknown')}"
            experiment_uri = self.ontology.GLYCO[experiment_id]
            
            self.graph.add((experiment_uri, RDF.type, self.ontology.MSExperiment))
            self.graph.add((glycan_uri, self.ontology.has_experimental_data, experiment_uri))
            
            # Add experiment properties
            if doc.get('precursor_mz'):
                self.graph.add((experiment_uri, self.ontology.precursor_mz, Literal(float(doc['precursor_mz']), datatype=XSD.float)))
            
            if doc.get('precursor_charge'):
                self.graph.add((experiment_uri, self.ontology.precursor_charge, Literal(int(doc['precursor_charge']), datatype=XSD.integer)))
            
            if doc.get('collision_energy'):
                self.graph.add((experiment_uri, self.ontology.collision_energy, Literal(float(doc['collision_energy']), datatype=XSD.float)))
            
            if doc.get('instrument'):
                self.graph.add((experiment_uri, self.ontology.instrument, Literal(doc['instrument'])))
            
            # Note: We don't store full peak lists in RDF due to size
            # They remain in MongoDB for data access
            if doc.get('peaks'):
                peak_count = len(doc['peaks'])
                self.graph.add((experiment_uri, self.ontology.num_peaks, Literal(peak_count, datatype=XSD.integer)))
            
            self.stats.experiment_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error populating spectra for {glytoucan_id}: {e}")
            return False
    
    def populate_all_glycans(self, limit: Optional[int] = None, batch_size: int = 1000):
        """
        Populate all glycans from databases
        
        Args:
            limit: Maximum number of glycans to populate
            batch_size: Commit graph every N glycans
        """
        logger.info(f"Populating glycans (limit={limit}, batch_size={batch_size})...")
        
        query = "SELECT glytoucan_id FROM glycans ORDER BY glytoucan_id"
        if limit:
            query += f" LIMIT {limit}"
        
        with self.pg_conn.cursor() as cursor:
            cursor.execute(query)
            
            count = 0
            for row in cursor:
                glytoucan_id = row[0]
                
                # Populate from all sources
                self.populate_glycan_from_postgres(glytoucan_id)
                self.populate_annotations_from_mongodb(glytoucan_id)
                self.populate_spectra_from_mongodb(glytoucan_id)
                
                count += 1
                
                if count % batch_size == 0:
                    self.stats.total_triples = len(self.graph)
                    logger.info(f"Populated {count} glycans, {self.stats.total_triples} triples")
                
                if limit and count >= limit:
                    break
        
        self.stats.total_triples = len(self.graph)
        logger.info(f"Completed: {count} glycans, {self.stats.total_triples} total triples")
    
    def populate_from_batch(self, glytoucan_ids: List[str]):
        """
        Populate specific glycans by batch
        
        Args:
            glytoucan_ids: List of GlyTouCan IDs
        """
        logger.info(f"Populating batch of {len(glytoucan_ids)} glycans...")
        
        for glytoucan_id in glytoucan_ids:
            self.populate_glycan_from_postgres(glytoucan_id)
            self.populate_annotations_from_mongodb(glytoucan_id)
            self.populate_spectra_from_mongodb(glytoucan_id)
        
        self.stats.total_triples = len(self.graph)
        logger.info(f"Batch complete: {self.stats.total_triples} triples")
    
    def save_graph(self, output_path: Path, format: str = "turtle"):
        """
        Save RDF graph to file
        
        Args:
            output_path: Path to save graph
            format: RDF serialization format (turtle, xml, n3, nt)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving graph to {output_path} (format={format})...")
        
        self.graph.serialize(destination=str(output_path), format=format)
        
        logger.info(f"Graph saved: {self.stats.total_triples} triples")
    
    def load_graph(self, input_path: Path, format: str = "turtle"):
        """
        Load RDF graph from file
        
        Args:
            input_path: Path to graph file
            format: RDF serialization format
        """
        logger.info(f"Loading graph from {input_path}...")
        
        self.graph.parse(str(input_path), format=format)
        self.stats.total_triples = len(self.graph)
        
        logger.info(f"Graph loaded: {self.stats.total_triples} triples")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        stats_dict = self.stats.to_dict()
        
        # Add query-based stats
        try:
            # Count distinct types
            stats_dict['class_instances'] = {}
            for class_uri in [self.ontology.Glycan, self.ontology.Protein, 
                             self.ontology.Disease, self.ontology.MSExperiment]:
                class_name = str(class_uri).split('/')[-1]
                count = len(list(self.graph.subjects(RDF.type, class_uri)))
                stats_dict['class_instances'][class_name] = count
            
            # Count distinct properties used
            properties = set(self.graph.predicates())
            stats_dict['distinct_properties'] = len(properties)
            
            # Count distinct subjects and objects
            stats_dict['distinct_subjects'] = len(set(self.graph.subjects()))
            stats_dict['distinct_objects'] = len(set(self.graph.objects()))
            
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
        
        return stats_dict
    
    def export_summary(self, output_path: Path):
        """
        Export population summary
        
        Args:
            output_path: Path to save summary JSON
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'graph_info': {
                'total_triples': len(self.graph),
                'namespaces': {prefix: str(ns) for prefix, ns in self.graph.namespaces()}
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary exported to {output_path}")


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate RDF knowledge graph")
    parser.add_argument("--output", type=str, required=True, help="Output graph file")
    parser.add_argument("--format", type=str, default="turtle", choices=['turtle', 'xml', 'n3', 'nt'], help="RDF format")
    parser.add_argument("--limit", type=int, help="Maximum glycans to populate")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
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
    parser.add_argument("--summary", type=str, help="Export summary to JSON file")
    
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
    
    # Populate graph
    with GraphPopulator(postgres_config, mongodb_config, redis_config) as populator:
        populator.populate_all_glycans(limit=args.limit, batch_size=args.batch_size)
        
        # Save graph
        populator.save_graph(Path(args.output), format=args.format)
        
        # Export summary if requested
        if args.summary:
            populator.export_summary(Path(args.summary))
        
        # Print statistics
        stats = populator.get_statistics()
        print("\n=== Graph Population Statistics ===")
        print(json.dumps(stats, indent=2))
