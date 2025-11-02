"""
Data Integration Coordinator for GlycoKG

This module coordinates data integration from multiple sources
(GlyTouCan, GlyGen, GlycoPOST) into the unified GlycoKG.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
import json
from dataclasses import asdict
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .glytoucan_client import GlyTouCanClient, GlycanStructure
from .glygen_client import GlyGenClient, ProteinGlycanAssociation, ProteinInfo
from .glycopost_client import GlycoPOSTClient, MSSpectrum, ExperimentalEvidence

logger = logging.getLogger(__name__)


class DataIntegrationCoordinator:
    """
    Coordinates data integration from multiple glycoinformatics sources.
    
    Manages batch processing, deduplication, and synchronized updates
    to the GlycoKG database.
    """
    
    def __init__(self,
                 postgres_config: Dict[str, str],
                 redis_config: Dict[str, Any],
                 batch_size: int = 1000):
        """
        Initialize the data integration coordinator.
        
        Args:
            postgres_config: PostgreSQL connection configuration
            redis_config: Redis connection configuration  
            batch_size: Number of records to process per batch
        """
        self.postgres_config = postgres_config
        self.redis_config = redis_config
        self.batch_size = batch_size
        
        # Database connections
        self.pg_engine = None
        self.redis_client = None
        
        # API clients
        self.glytoucan_client = None
        self.glygen_client = None
        self.glycopost_client = None
        
        # Statistics tracking
        self.sync_stats = {
            "glytoucan": {"processed": 0, "added": 0, "updated": 0, "errors": 0},
            "glygen": {"processed": 0, "added": 0, "updated": 0, "errors": 0},
            "glycopost": {"processed": 0, "added": 0, "updated": 0, "errors": 0}
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_connections()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_connections()
        
    async def initialize_connections(self):
        """Initialize database connections and API clients"""
        # PostgreSQL connection
        pg_url = (f"postgresql://{self.postgres_config['user']}:"
                 f"{self.postgres_config['password']}@"
                 f"{self.postgres_config['host']}:"
                 f"{self.postgres_config['port']}/"
                 f"{self.postgres_config['database']}")
        
        self.pg_engine = create_engine(pg_url)
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=self.redis_config['host'],
            port=self.redis_config['port'],
            db=self.redis_config['db'],
            decode_responses=True
        )
        
        # API clients
        self.glytoucan_client = GlyTouCanClient(batch_size=self.batch_size)
        self.glygen_client = GlyGenClient(batch_size=self.batch_size)
        self.glycopost_client = GlycoPOSTClient(batch_size=self.batch_size)
        
        await self.glytoucan_client.__aenter__()
        await self.glygen_client.__aenter__()
        await self.glycopost_client.__aenter__()
        
        logger.info("Data integration coordinator initialized")
        
    async def close_connections(self):
        """Close all connections"""
        if self.glytoucan_client:
            await self.glytoucan_client.__aexit__(None, None, None)
        if self.glygen_client:
            await self.glygen_client.__aexit__(None, None, None)
        if self.glycopost_client:
            await self.glycopost_client.__aexit__(None, None, None)
            
        if self.pg_engine:
            self.pg_engine.dispose()
        if self.redis_client:
            self.redis_client.close()
            
        logger.info("Data integration coordinator closed")
        
    def _get_cache_key(self, source: str, entity_id: str) -> str:
        """Generate cache key for entity"""
        return f"glycokg:cache:{source}:{entity_id}"
        
    def _is_cached(self, source: str, entity_id: str) -> bool:
        """Check if entity is cached in Redis"""
        cache_key = self._get_cache_key(source, entity_id)
        return self.redis_client.exists(cache_key)
        
    def _cache_entity(self, source: str, entity_id: str, data: Dict[str, Any], ttl: int = 3600):
        """Cache entity data in Redis"""
        cache_key = self._get_cache_key(source, entity_id)
        self.redis_client.setex(cache_key, ttl, json.dumps(data))
        
    async def sync_glytoucan_structures(self, 
                                      limit: Optional[int] = None,
                                      force_update: bool = False) -> Dict[str, int]:
        """
        Synchronize glycan structures from GlyTouCan.
        
        Args:
            limit: Maximum number of structures to sync
            force_update: Force update even if cached
            
        Returns:
            Sync statistics
        """
        logger.info("Starting GlyTouCan structure synchronization")
        
        stats = {"processed": 0, "added": 0, "updated": 0, "errors": 0}
        
        try:
            async for structure_batch in self.glytoucan_client.get_all_structures(limit):
                batch_data = []
                
                for structure in structure_batch:
                    try:
                        # Check if already cached and skip if not force updating
                        if not force_update and self._is_cached("glytoucan", structure.glytoucan_id):
                            stats["processed"] += 1
                            continue
                            
                        # Prepare data for database insertion
                        structure_data = {
                            'glytoucan_id': structure.glytoucan_id,
                            'wurcs_sequence': structure.wurcs_sequence,
                            'glycoct': structure.glycoct,
                            'iupac_extended': structure.iupac_extended,
                            'iupac_condensed': structure.iupac_condensed,
                            'mass_mono': structure.mass_mono,
                            'mass_avg': structure.mass_avg,
                            'composition': json.dumps(structure.composition) if structure.composition else None,
                            'source_id': await self._get_source_id("GlyTouCan")
                        }
                        
                        batch_data.append(structure_data)
                        
                        # Cache the structure
                        self._cache_entity("glytoucan", structure.glytoucan_id, asdict(structure))
                        
                        stats["processed"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing structure {structure.glytoucan_id}: {e}")
                        stats["errors"] += 1
                        
                # Batch insert to database
                if batch_data:
                    inserted, updated = await self._batch_upsert_structures(batch_data)
                    stats["added"] += inserted
                    stats["updated"] += updated
                    
                logger.info(f"Processed batch: {len(structure_batch)} structures, "
                           f"Total: {stats['processed']}")
                           
        except Exception as e:
            logger.error(f"Error in GlyTouCan sync: {e}")
            stats["errors"] += 1
            
        self.sync_stats["glytoucan"] = stats
        logger.info(f"GlyTouCan sync completed: {stats}")
        
        return stats
        
    async def sync_glygen_associations(self,
                                     organism_taxid: Optional[int] = None,
                                     limit: Optional[int] = None,
                                     force_update: bool = False) -> Dict[str, int]:
        """
        Synchronize protein-glycan associations from GlyGen.
        
        Args:
            organism_taxid: Optional organism filter
            limit: Maximum number of associations to sync
            force_update: Force update even if cached
            
        Returns:
            Sync statistics
        """
        logger.info(f"Starting GlyGen association synchronization for taxid: {organism_taxid}")
        
        stats = {"processed": 0, "added": 0, "updated": 0, "errors": 0}
        
        try:
            async for association_batch in self.glygen_client.get_all_protein_glycan_associations(
                organism_taxid, limit):
                
                batch_data = []
                
                for association in association_batch:
                    try:
                        # Check cache
                        cache_key = f"{association.uniprot_id}:{association.glytoucan_id}:{association.glycosylation_site}"
                        if not force_update and self._is_cached("glygen", cache_key):
                            stats["processed"] += 1
                            continue
                            
                        # Prepare data for database insertion
                        association_data = {
                            'uniprot_id': association.uniprot_id,
                            'glytoucan_id': association.glytoucan_id,
                            'glycosylation_site': association.glycosylation_site,
                            'evidence_type': association.evidence_type,
                            'organism_taxid': association.organism_taxid,
                            'tissue': association.tissue,
                            'disease': association.disease,
                            'confidence_score': association.confidence_score,
                            'source_id': await self._get_source_id("GlyGen")
                        }
                        
                        batch_data.append(association_data)
                        
                        # Cache the association
                        self._cache_entity("glygen", cache_key, asdict(association))
                        
                        stats["processed"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing association: {e}")
                        stats["errors"] += 1
                        
                # Batch insert to database
                if batch_data:
                    inserted, updated = await self._batch_upsert_associations(batch_data)
                    stats["added"] += inserted
                    stats["updated"] += updated
                    
                logger.info(f"Processed batch: {len(association_batch)} associations, "
                           f"Total: {stats['processed']}")
                           
        except Exception as e:
            logger.error(f"Error in GlyGen sync: {e}")
            stats["errors"] += 1
            
        self.sync_stats["glygen"] = stats
        logger.info(f"GlyGen sync completed: {stats}")
        
        return stats
        
    async def sync_glycopost_spectra(self,
                                   organism_taxid: Optional[int] = None,
                                   limit: Optional[int] = None,
                                   force_update: bool = False) -> Dict[str, int]:
        """
        Synchronize MS/MS spectra from GlycoPOST.
        
        Args:
            organism_taxid: Optional organism filter
            limit: Maximum number of spectra to sync
            force_update: Force update even if cached
            
        Returns:
            Sync statistics
        """
        logger.info(f"Starting GlycoPOST spectra synchronization for taxid: {organism_taxid}")
        
        stats = {"processed": 0, "added": 0, "updated": 0, "errors": 0}
        
        try:
            async for spectra_batch in self.glycopost_client.get_all_spectra(organism_taxid, limit):
                batch_data = []
                
                for spectrum in spectra_batch:
                    try:
                        # Check cache
                        if not force_update and self._is_cached("glycopost", spectrum.spectrum_id):
                            stats["processed"] += 1
                            continue
                            
                        # Prepare data for database insertion
                        spectrum_data = {
                            'spectrum_id': spectrum.spectrum_id,
                            'glytoucan_id': spectrum.glytoucan_id,
                            'precursor_mz': spectrum.precursor_mz,
                            'charge_state': spectrum.charge_state,
                            'collision_energy': spectrum.collision_energy,
                            'peaks': json.dumps(spectrum.peaks) if spectrum.peaks else None,
                            'metadata': json.dumps(spectrum.metadata) if spectrum.metadata else None,
                            'source_id': await self._get_source_id("GlycoPOST")
                        }
                        
                        batch_data.append(spectrum_data)
                        
                        # Cache the spectrum
                        self._cache_entity("glycopost", spectrum.spectrum_id, asdict(spectrum))
                        
                        stats["processed"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing spectrum {spectrum.spectrum_id}: {e}")
                        stats["errors"] += 1
                        
                # Batch insert to database
                if batch_data:
                    inserted, updated = await self._batch_upsert_spectra(batch_data)
                    stats["added"] += inserted
                    stats["updated"] += updated
                    
                logger.info(f"Processed batch: {len(spectra_batch)} spectra, "
                           f"Total: {stats['processed']}")
                           
        except Exception as e:
            logger.error(f"Error in GlycoPOST sync: {e}")
            stats["errors"] += 1
            
        self.sync_stats["glycopost"] = stats
        logger.info(f"GlycoPOST sync completed: {stats}")
        
        return stats
        
    async def _get_source_id(self, source_name: str) -> str:
        """Get source ID from metadata.data_sources table"""
        query = "SELECT id FROM metadata.data_sources WHERE name = :source_name"
        
        with self.pg_engine.connect() as conn:
            result = conn.execute(text(query), {"source_name": source_name})
            row = result.fetchone()
            return str(row[0]) if row else None
            
    async def _batch_upsert_structures(self, batch_data: List[Dict[str, Any]]) -> tuple[int, int]:
        """Batch upsert glycan structures to database"""
        if not batch_data:
            return 0, 0
            
        upsert_query = """
        INSERT INTO cache.glycan_structures 
        (glytoucan_id, wurcs_sequence, glycoct, iupac_extended, iupac_condensed,
         mass_mono, mass_avg, composition, source_id)
        VALUES (%(glytoucan_id)s, %(wurcs_sequence)s, %(glycoct)s, 
                %(iupac_extended)s, %(iupac_condensed)s, %(mass_mono)s, 
                %(mass_avg)s, %(composition)s, %(source_id)s)
        ON CONFLICT (glytoucan_id) DO UPDATE SET
        wurcs_sequence = EXCLUDED.wurcs_sequence,
        glycoct = EXCLUDED.glycoct,
        iupac_extended = EXCLUDED.iupac_extended,
        iupac_condensed = EXCLUDED.iupac_condensed,
        mass_mono = EXCLUDED.mass_mono,
        mass_avg = EXCLUDED.mass_avg,
        composition = EXCLUDED.composition,
        updated_at = NOW()
        """
        
        inserted = 0
        updated = 0
        
        with self.pg_engine.connect() as conn:
            for data in batch_data:
                try:
                    result = conn.execute(text(upsert_query), data)
                    if result.rowcount == 1:
                        inserted += 1
                    else:
                        updated += 1
                except Exception as e:
                    logger.error(f"Error upserting structure: {e}")
                    
        return inserted, updated
        
    async def _batch_upsert_associations(self, batch_data: List[Dict[str, Any]]) -> tuple[int, int]:
        """Batch upsert protein-glycan associations to database"""
        if not batch_data:
            return 0, 0
            
        upsert_query = """
        INSERT INTO cache.protein_glycan_associations
        (uniprot_id, glytoucan_id, glycosylation_site, evidence_type,
         organism_taxid, tissue, disease, confidence_score, source_id)
        VALUES (%(uniprot_id)s, %(glytoucan_id)s, %(glycosylation_site)s,
                %(evidence_type)s, %(organism_taxid)s, %(tissue)s,
                %(disease)s, %(confidence_score)s, %(source_id)s)
        ON CONFLICT (uniprot_id, glytoucan_id, glycosylation_site) DO UPDATE SET
        evidence_type = EXCLUDED.evidence_type,
        organism_taxid = EXCLUDED.organism_taxid,
        tissue = EXCLUDED.tissue,
        disease = EXCLUDED.disease,
        confidence_score = EXCLUDED.confidence_score,
        created_at = NOW()
        """
        
        inserted = 0
        updated = 0
        
        with self.pg_engine.connect() as conn:
            for data in batch_data:
                try:
                    result = conn.execute(text(upsert_query), data)
                    if result.rowcount == 1:
                        inserted += 1
                    else:
                        updated += 1
                except Exception as e:
                    logger.error(f"Error upserting association: {e}")
                    
        return inserted, updated
        
    async def _batch_upsert_spectra(self, batch_data: List[Dict[str, Any]]) -> tuple[int, int]:
        """Batch upsert MS spectra to database"""
        if not batch_data:
            return 0, 0
            
        upsert_query = """
        INSERT INTO cache.ms_spectra
        (spectrum_id, glytoucan_id, precursor_mz, charge_state,
         collision_energy, peaks, metadata, source_id)
        VALUES (%(spectrum_id)s, %(glytoucan_id)s, %(precursor_mz)s,
                %(charge_state)s, %(collision_energy)s, %(peaks)s,
                %(metadata)s, %(source_id)s)
        ON CONFLICT (spectrum_id) DO UPDATE SET
        glytoucan_id = EXCLUDED.glytoucan_id,
        precursor_mz = EXCLUDED.precursor_mz,
        charge_state = EXCLUDED.charge_state,
        collision_energy = EXCLUDED.collision_energy,
        peaks = EXCLUDED.peaks,
        metadata = EXCLUDED.metadata,
        created_at = NOW()
        """
        
        inserted = 0
        updated = 0
        
        with self.pg_engine.connect() as conn:
            for data in batch_data:
                try:
                    result = conn.execute(text(upsert_query), data)
                    if result.rowcount == 1:
                        inserted += 1
                    else:
                        updated += 1
                except Exception as e:
                    logger.error(f"Error upserting spectrum: {e}")
                    
        return inserted, updated
        
    async def full_synchronization(self,
                                 organism_taxids: Optional[List[int]] = None,
                                 limit_per_source: Optional[int] = None,
                                 force_update: bool = False) -> Dict[str, Any]:
        """
        Perform full data synchronization from all sources.
        
        Args:
            organism_taxids: Optional list of organism filters
            limit_per_source: Maximum records per source
            force_update: Force update even if cached
            
        Returns:
            Combined synchronization statistics
        """
        logger.info("Starting full data synchronization")
        
        start_time = datetime.now()
        
        # Synchronize structures from GlyTouCan
        glytoucan_stats = await self.sync_glytoucan_structures(
            limit=limit_per_source,
            force_update=force_update
        )
        
        # Synchronize associations from GlyGen for each organism
        glygen_stats = {"processed": 0, "added": 0, "updated": 0, "errors": 0}
        
        if organism_taxids:
            for taxid in organism_taxids:
                taxid_stats = await self.sync_glygen_associations(
                    organism_taxid=taxid,
                    limit=limit_per_source,
                    force_update=force_update
                )
                for key in glygen_stats:
                    glygen_stats[key] += taxid_stats[key]
        else:
            glygen_stats = await self.sync_glygen_associations(
                limit=limit_per_source,
                force_update=force_update
            )
            
        # Synchronize spectra from GlycoPOST for each organism  
        glycopost_stats = {"processed": 0, "added": 0, "updated": 0, "errors": 0}
        
        if organism_taxids:
            for taxid in organism_taxids:
                taxid_stats = await self.sync_glycopost_spectra(
                    organism_taxid=taxid,
                    limit=limit_per_source,
                    force_update=force_update
                )
                for key in glycopost_stats:
                    glycopost_stats[key] += taxid_stats[key]
        else:
            glycopost_stats = await self.sync_glycopost_spectra(
                limit=limit_per_source,
                force_update=force_update
            )
            
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "glytoucan": glytoucan_stats,
            "glygen": glygen_stats,
            "glycopost": glycopost_stats,
            "total_processed": (glytoucan_stats["processed"] + 
                              glygen_stats["processed"] + 
                              glycopost_stats["processed"]),
            "total_added": (glytoucan_stats["added"] + 
                          glygen_stats["added"] + 
                          glycopost_stats["added"]),
            "total_updated": (glytoucan_stats["updated"] + 
                            glygen_stats["updated"] + 
                            glycopost_stats["updated"]),
            "total_errors": (glytoucan_stats["errors"] + 
                           glygen_stats["errors"] + 
                           glycopost_stats["errors"])
        }
        
        logger.info(f"Full synchronization completed: {summary}")
        
        return summary


# Alias for API compatibility
DataCoordinator = DataIntegrationCoordinator