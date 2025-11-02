"""
GlyGen API Client for Protein-Glycan Association Data

This module provides a client interface to the GlyGen database,
focusing on protein glycosylation data and associations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncIterator, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
import pandas as pd
from urllib.parse import urlencode
import json

logger = logging.getLogger(__name__)


@dataclass 
class ProteinGlycanAssociation:
    """Data class for protein-glycan association information"""
    uniprot_id: str
    glytoucan_id: str
    glycosylation_site: Optional[int] = None
    site_type: Optional[str] = None  # N-linked, O-linked, etc.
    evidence_type: Optional[str] = None
    organism_taxid: Optional[int] = None
    organism_name: Optional[str] = None
    tissue: Optional[str] = None
    disease: Optional[str] = None
    confidence_score: Optional[float] = None
    publication_pmid: Optional[str] = None
    experimental_method: Optional[str] = None
    glycan_type: Optional[str] = None
    created_date: Optional[datetime] = None


@dataclass
class ProteinInfo:
    """Data class for protein information from GlyGen"""
    uniprot_id: str
    gene_name: Optional[str] = None
    protein_name: Optional[str] = None
    organism_taxid: Optional[int] = None
    organism_name: Optional[str] = None
    sequence: Optional[str] = None
    glycosylation_sites: Optional[List[Dict[str, Any]]] = None
    pathways: Optional[List[str]] = None
    functions: Optional[List[str]] = None
    diseases: Optional[List[str]] = None


class GlyGenClient:
    """
    Client for interacting with the GlyGen database API.
    
    Provides access to protein glycosylation data, protein-glycan associations,
    and related biological annotations.
    """
    
    def __init__(self, 
                 base_url: str = "https://api.glygen.org",
                 api_version: str = "v2",
                 batch_size: int = 100,
                 rate_limit_delay: float = 0.2):
        """
        Initialize GlyGen client.
        
        Args:
            base_url: Base URL for GlyGen API
            api_version: API version to use
            batch_size: Number of records to fetch per batch
            rate_limit_delay: Delay between requests (seconds)
        """
        self.base_url = base_url.rstrip('/')
        self.api_version = api_version
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    def _build_url(self, endpoint: str, **params) -> str:
        """Build complete API URL with parameters"""
        url = f"{self.base_url}/{self.api_version}/{endpoint.lstrip('/')}"
        if params:
            url += f"?{urlencode(params)}"
        return url
        
    async def _make_request(self, endpoint: str, **params) -> Optional[Dict[str, Any]]:
        """Make HTTP request to GlyGen API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
            
        url = self._build_url(endpoint, **params)
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                else:
                    logger.error(f"API request failed: {response.status} - {url}")
                    return None
                    
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
            return None
            
    async def get_protein_info(self, uniprot_id: str) -> Optional[ProteinInfo]:
        """
        Get comprehensive protein information from GlyGen.
        
        Args:
            uniprot_id: UniProt accession ID
            
        Returns:
            ProteinInfo object or None if not found
        """
        data = await self._make_request(f"protein/{uniprot_id}")
        
        if not data:
            return None
            
        try:
            # Extract glycosylation sites
            glycosylation_sites = []
            if "glycosylation" in data:
                for site_data in data["glycosylation"]:
                    glycosylation_sites.append({
                        "position": site_data.get("start_pos"),
                        "site_type": site_data.get("type"),
                        "residue": site_data.get("residue"),
                        "evidence": site_data.get("evidence", [])
                    })
                    
            return ProteinInfo(
                uniprot_id=uniprot_id,
                gene_name=data.get("gene_name"),
                protein_name=data.get("protein_name"),
                organism_taxid=data.get("organism", {}).get("taxid"),
                organism_name=data.get("organism", {}).get("name"),
                sequence=data.get("sequence"),
                glycosylation_sites=glycosylation_sites,
                pathways=data.get("pathways", []),
                functions=data.get("function", []),
                diseases=data.get("disease", [])
            )
            
        except Exception as e:
            logger.error(f"Error parsing protein data for {uniprot_id}: {e}")
            return None
            
    async def get_protein_glycan_associations(self, 
                                           uniprot_id: str) -> List[ProteinGlycanAssociation]:
        """
        Get glycan associations for a specific protein.
        
        Args:
            uniprot_id: UniProt accession ID
            
        Returns:
            List of ProteinGlycanAssociation objects
        """
        data = await self._make_request(f"protein/{uniprot_id}/glycans")
        
        if not data or "glycans" not in data:
            return []
            
        associations = []
        
        for glycan_data in data["glycans"]:
            try:
                association = ProteinGlycanAssociation(
                    uniprot_id=uniprot_id,
                    glytoucan_id=glycan_data.get("glytoucan_id"),
                    glycosylation_site=glycan_data.get("site_position"),
                    site_type=glycan_data.get("site_type"),
                    evidence_type=glycan_data.get("evidence_type"),
                    organism_taxid=glycan_data.get("taxid"),
                    organism_name=glycan_data.get("organism"),
                    tissue=glycan_data.get("tissue"),
                    disease=glycan_data.get("disease"),
                    confidence_score=glycan_data.get("confidence"),
                    publication_pmid=glycan_data.get("pmid"),
                    experimental_method=glycan_data.get("method"),
                    glycan_type=glycan_data.get("glycan_type")
                )
                associations.append(association)
                
            except Exception as e:
                logger.warning(f"Error parsing glycan association: {e}")
                continue
                
        return associations
        
    async def search_proteins_by_organism(self, 
                                        taxid: int,
                                        limit: Optional[int] = None) -> AsyncIterator[List[str]]:
        """
        Search for proteins by organism taxonomy ID.
        
        Args:
            taxid: NCBI Taxonomy ID
            limit: Maximum number of proteins to return
            
        Yields:
            Batches of UniProt IDs
        """
        offset = 0
        total_fetched = 0
        
        while True:
            params = {
                "taxid": taxid,
                "offset": offset,
                "limit": self.batch_size
            }
            
            data = await self._make_request("search/protein", **params)
            
            if not data or "results" not in data or not data["results"]:
                break
                
            uniprot_ids = [result.get("uniprot_canonical_ac") 
                          for result in data["results"] 
                          if result.get("uniprot_canonical_ac")]
                          
            if not uniprot_ids:
                break
                
            yield uniprot_ids
            
            total_fetched += len(uniprot_ids)
            offset += self.batch_size
            
            # Check limit
            if limit and total_fetched >= limit:
                break
                
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
    async def get_glycan_proteins(self, 
                                glytoucan_id: str) -> List[ProteinGlycanAssociation]:
        """
        Get all proteins associated with a specific glycan.
        
        Args:
            glytoucan_id: GlyTouCan accession ID
            
        Returns:
            List of ProteinGlycanAssociation objects
        """
        data = await self._make_request(f"glycan/{glytoucan_id}/proteins")
        
        if not data or "proteins" not in data:
            return []
            
        associations = []
        
        for protein_data in data["proteins"]:
            try:
                association = ProteinGlycanAssociation(
                    uniprot_id=protein_data.get("uniprot_id"),
                    glytoucan_id=glytoucan_id,
                    glycosylation_site=protein_data.get("site_position"),
                    site_type=protein_data.get("site_type"),
                    evidence_type=protein_data.get("evidence_type"),
                    organism_taxid=protein_data.get("taxid"),
                    organism_name=protein_data.get("organism"),
                    tissue=protein_data.get("tissue"),
                    disease=protein_data.get("disease"),
                    confidence_score=protein_data.get("confidence"),
                    publication_pmid=protein_data.get("pmid"),
                    experimental_method=protein_data.get("method"),
                    glycan_type=protein_data.get("glycan_type")
                )
                associations.append(association)
                
            except Exception as e:
                logger.warning(f"Error parsing protein association: {e}")
                continue
                
        return associations
        
    async def get_tissue_specific_glycans(self, 
                                        tissue: str,
                                        organism_taxid: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get glycans specific to a tissue type.
        
        Args:
            tissue: Tissue name or type
            organism_taxid: Optional organism filter
            
        Returns:
            List of tissue-specific glycan data
        """
        params = {"tissue": tissue}
        if organism_taxid:
            params["taxid"] = organism_taxid
            
        data = await self._make_request("search/tissue_glycans", **params)
        
        if not data or "results" not in data:
            return []
            
        return data["results"]
        
    async def get_disease_associated_glycans(self, 
                                          disease: str,
                                          organism_taxid: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get glycans associated with a specific disease.
        
        Args:
            disease: Disease name or identifier
            organism_taxid: Optional organism filter
            
        Returns:
            List of disease-associated glycan data
        """
        params = {"disease": disease}
        if organism_taxid:
            params["taxid"] = organism_taxid
            
        data = await self._make_request("search/disease_glycans", **params)
        
        if not data or "results" not in data:
            return []
            
        return data["results"]
        
    async def get_all_protein_glycan_associations(self, 
                                                organism_taxid: Optional[int] = None,
                                                limit: Optional[int] = None) -> AsyncIterator[List[ProteinGlycanAssociation]]:
        """
        Get all protein-glycan associations, optionally filtered by organism.
        
        Args:
            organism_taxid: Optional organism filter
            limit: Maximum number of associations to return
            
        Yields:
            Batches of ProteinGlycanAssociation objects
        """
        # First get all proteins for the organism
        if organism_taxid:
            async for protein_batch in self.search_proteins_by_organism(organism_taxid, limit):
                associations_batch = []
                
                for uniprot_id in protein_batch:
                    associations = await self.get_protein_glycan_associations(uniprot_id)
                    associations_batch.extend(associations)
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                if associations_batch:
                    yield associations_batch
        else:
            # Get associations without organism filter - would need pagination endpoint
            logger.warning("Getting all associations without organism filter not efficiently supported")
            yield []
            
    async def get_statistics(self, organism_taxid: Optional[int] = None) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Args:
            organism_taxid: Optional organism filter
            
        Returns:
            Dictionary with database statistics
        """
        params = {}
        if organism_taxid:
            params["taxid"] = organism_taxid
            
        try:
            data = await self._make_request("statistics", **params)
            
            if data:
                return {
                    "total_proteins": data.get("total_proteins", 0),
                    "total_glycans": data.get("total_glycans", 0),
                    "total_associations": data.get("total_associations", 0),
                    "organisms_covered": data.get("organisms", []),
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            
        return {
            "error": "Failed to retrieve statistics",
            "last_updated": datetime.now().isoformat()
        }