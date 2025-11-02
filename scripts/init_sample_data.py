#!/usr/bin/env python3
"""
Sample Data Initialization Script for GlycoKG

This script demonstrates how to initialize the GlycoKG with sample data
from various sources and create a working knowledge graph.
"""

import asyncio
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_sample_data():
    """Initialize GlycoKG with sample data"""
    
    logger.info("Starting GlycoKG sample data initialization")
    
    # Configuration
    postgres_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DB", "glycokg"),
        "user": os.getenv("POSTGRES_USER", "glycoinfo"),
        "password": os.getenv("POSTGRES_PASSWORD", "glycoinfo_pass")
    }
    
    redis_config = {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "db": int(os.getenv("REDIS_DB_CACHE", "1"))
    }
    
    try:
        # Import after configuration
        from glycokg.integration import DataIntegrationCoordinator
        from glycokg.ontology.glyco_ontology import GlycoOntology
        
        # Initialize ontology
        logger.info("Initializing glyco ontology...")
        ontology = GlycoOntology()
        
        # Add sample glycan structures
        sample_glycans = [
            {
                "glytoucan_id": "G00001MO",
                "wurcs_sequence": "WURCS=2.0/3,3,2/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5]/1-2-3/a4-b1_b4-c1",
                "mass_mono": 910.327,
                "composition": {"Hex": 3, "HexNAc": 2}
            },
            {
                "glytoucan_id": "G00002MO", 
                "wurcs_sequence": "WURCS=2.0/2,2,1/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1a_1-5]/1-2/a4-b1",
                "mass_mono": 365.132,
                "composition": {"Hex": 1, "HexNAc": 1}
            }
        ]
        
        for glycan_data in sample_glycans:
            ontology.add_glycan(**glycan_data)
            logger.info(f"Added glycan {glycan_data['glytoucan_id']} to ontology")
        
        # Add sample protein-glycan associations
        sample_associations = [
            {
                "uniprot_id": "P01834",
                "glytoucan_id": "G00001MO", 
                "glycosylation_site": 52,
                "site_type": "N-linked",
                "organism_taxid": 9606,  # Homo sapiens
                "tissue": "serum",
                "confidence_score": 0.95
            },
            {
                "uniprot_id": "P02768",
                "glytoucan_id": "G00002MO",
                "glycosylation_site": 81,
                "site_type": "N-linked", 
                "organism_taxid": 9606,
                "tissue": "liver",
                "confidence_score": 0.87
            }
        ]
        
        for assoc_data in sample_associations:
            ontology.add_protein_glycan_association(**assoc_data)
            logger.info(f"Added association {assoc_data['uniprot_id']}-{assoc_data['glytoucan_id']}")
        
        # Add sample MS spectra
        sample_spectra = [
            {
                "spectrum_id": "SPEC001",
                "glytoucan_id": "G00001MO",
                "precursor_mz": 911.334,
                "charge_state": 1,
                "collision_energy": 20.0,
                "peaks": [[163.06, 1000], [204.09, 2500], [366.14, 5000], [528.19, 3000]]
            },
            {
                "spectrum_id": "SPEC002", 
                "glytoucan_id": "G00002MO",
                "precursor_mz": 366.139,
                "charge_state": 1,
                "collision_energy": 15.0,
                "peaks": [[126.05, 800], [163.06, 1200], [204.09, 1500]]
            }
        ]
        
        for spectrum_data in sample_spectra:
            ontology.add_ms_spectrum(**spectrum_data)
            logger.info(f"Added spectrum {spectrum_data['spectrum_id']}")
        
        # Save ontology to file
        ontology_file = "data/processed/sample_glycokg.ttl"
        os.makedirs("data/processed", exist_ok=True)
        ontology.save_to_file(ontology_file, format="turtle")
        
        # Get and display statistics
        stats = ontology.get_statistics()
        logger.info(f"Ontology statistics: {json.dumps(stats, indent=2)}")
        
        # Initialize data integration coordinator (if services are running)
        try:
            async with DataIntegrationCoordinator(postgres_config, redis_config) as coordinator:
                
                # Test small sync from GlyTouCan (limit to 10 structures)
                logger.info("Testing GlyTouCan synchronization...")
                glytoucan_stats = await coordinator.sync_glytoucan_structures(limit=10)
                logger.info(f"GlyTouCan sync results: {glytoucan_stats}")
                
                # Test GlyGen sync for human (limit to 5 associations)
                logger.info("Testing GlyGen synchronization...")
                glygen_stats = await coordinator.sync_glygen_associations(
                    organism_taxid=9606, limit=5
                )
                logger.info(f"GlyGen sync results: {glygen_stats}")
                
                # Test GlycoPOST sync (limit to 3 spectra)
                logger.info("Testing GlycoPOST synchronization...")
                glycopost_stats = await coordinator.sync_glycopost_spectra(limit=3)
                logger.info(f"GlycoPOST sync results: {glycopost_stats}")
                
        except Exception as e:
            logger.warning(f"Could not test data integration (services may not be running): {e}")
            logger.info("Run 'make start' to start services for full integration testing")
        
        # Create sample query examples
        sample_queries = create_sample_queries()
        
        # Save sample queries
        queries_file = "data/processed/sample_queries.sparql"
        with open(queries_file, 'w') as f:
            f.write(sample_queries)
        
        logger.info("Sample data initialization completed successfully!")
        
        # Display summary
        print("\n" + "="*60)
        print("GLYCOKG SAMPLE DATA INITIALIZATION COMPLETE")
        print("="*60)
        print(f"✅ Ontology saved to: {ontology_file}")
        print(f"✅ Sample queries saved to: {queries_file}")
        print(f"✅ Ontology contains:")
        for key, value in stats.items():
            if key != "last_updated":
                print(f"   - {value} {key}")
        print("\nNext steps:")
        print("1. Run 'make start' to start all services")
        print("2. Visit http://localhost:7200 to access GraphDB")
        print("3. Upload the ontology file to create a repository")
        print("4. Run sample SPARQL queries to explore the data")
        print("="*60)
        
    except ImportError as e:
        logger.error(f"Import error (install dependencies first): {e}")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise


def create_sample_queries() -> str:
    """Create sample SPARQL queries for testing"""
    
    queries = """
# Sample SPARQL Queries for GlycoKG
# ====================================

# Query 1: Find all glycans with their masses
PREFIX glyco: <http://purl.glycoinfo.org/ontology/>
PREFIX glycan: <http://purl.glycoinfo.org/glycan/>

SELECT ?glycan ?glytoucan_id ?wurcs ?mass WHERE {
    ?glycan a glyco:Glycan ;
           glyco:hasGlyTouCanID ?glytoucan_id ;
           glyco:hasWURCSSequence ?wurcs ;
           glyco:hasMonoisotopicMass ?mass .
}

# ====================================

# Query 2: Find protein-glycan associations with sites
PREFIX glyco: <http://purl.glycoinfo.org/ontology/>
PREFIX uniprot: <http://purl.uniprot.org/uniprot/>

SELECT ?protein ?glycan ?site_position ?confidence WHERE {
    ?assoc a glyco:ProteinGlycanAssociation ;
           glyco:hasProtein ?protein ;
           glyco:hasGlycan ?glycan ;
           glyco:hasConfidenceScore ?confidence .
    
    ?assoc glyco:atSite ?site .
    ?site glyco:hasGlycosylationPosition ?site_position .
}

# ====================================

# Query 3: Find MS spectra for specific glycans
PREFIX glyco: <http://purl.glycoinfo.org/ontology/>
PREFIX glycan: <http://purl.glycoinfo.org/glycan/>

SELECT ?spectrum ?glycan ?precursor_mz ?charge WHERE {
    ?spectrum a glyco:MSSpectrum ;
             glyco:identifiesGlycan ?glycan ;
             glyco:hasPrecursorMZ ?precursor_mz ;
             glyco:hasChargeState ?charge .
}

# ====================================

# Query 4: Find glycans by composition
PREFIX glyco: <http://purl.glycoinfo.org/ontology/>

SELECT ?glycan ?glytoucan_id ?hex_count ?hexnac_count WHERE {
    ?glycan a glyco:Glycan ;
           glyco:hasGlyTouCanID ?glytoucan_id ;
           glyco:hasComposition ?comp .
    
    ?comp glyco:hasHexCount ?hex_count ;
          glyco:hasHexNAcCount ?hexnac_count .
}

# ====================================

# Query 5: Find tissue-specific glycan expression
PREFIX glyco: <http://purl.glycoinfo.org/ontology/>
PREFIX tissue: <http://purl.obolibrary.org/obo/UBERON_>

SELECT ?glycan ?tissue ?protein WHERE {
    ?assoc a glyco:ProteinGlycanAssociation ;
           glyco:hasGlycan ?glycan ;
           glyco:hasProtein ?protein ;
           glyco:expressedIn ?tissue .
}

# ====================================

# Query 6: Count statistics by entity type
PREFIX glyco: <http://purl.glycoinfo.org/ontology/>

SELECT ?type (COUNT(?entity) as ?count) WHERE {
    ?entity a ?type .
    FILTER(?type IN (glyco:Glycan, glyco:Protein, glyco:MSSpectrum, 
                     glyco:ProteinGlycanAssociation, glyco:GlycosylationSite))
}
GROUP BY ?type
ORDER BY DESC(?count)
"""
    
    return queries


if __name__ == "__main__":
    # Run the initialization
    asyncio.run(initialize_sample_data())