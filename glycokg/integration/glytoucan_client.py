"""
GlyTouCan API Client for Glycan Structure Integration

This module provides a client interface to the GlyTouCan glycan repository,
supporting batch downloads and structure queries.
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncIterator, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import time
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class GlycanStructure:
    """Data class for glycan structure information"""
    glytoucan_id: str
    wurcs_sequence: Optional[str] = None
    glycoct: Optional[str] = None
    iupac_extended: Optional[str] = None
    iupac_condensed: Optional[str] = None
    mass_mono: Optional[float] = None
    mass_avg: Optional[float] = None
    composition: Optional[Dict[str, int]] = None
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    synonyms: Optional[List[str]] = None
    cross_references: Optional[Dict[str, str]] = None


class GlyTouCanClient:
    """
    Client for interacting with the GlyTouCan glycan repository.
    
    Supports both REST API and SPARQL endpoint access for comprehensive
    glycan structure data retrieval.
    """
    
    def __init__(self, 
                 sparql_endpoint: str = "https://ts.glytoucan.org/sparql",
                 rest_endpoint: str = "https://glytoucan.org/api",
                 batch_size: int = 1000,
                 rate_limit_delay: float = 0.1):
        """
        Initialize GlyTouCan client.
        
        Args:
            sparql_endpoint: SPARQL endpoint URL
            rest_endpoint: REST API endpoint URL  
            batch_size: Number of structures to fetch per batch
            rate_limit_delay: Delay between requests (seconds)
        """
        self.sparql_endpoint = sparql_endpoint
        self.rest_endpoint = rest_endpoint
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize SPARQL wrapper
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    def get_all_structure_ids(self) -> List[str]:
        """
        Get all GlyTouCan IDs from the repository.
        
        Returns:
            List of GlyTouCan accession IDs
        """
        query = """
        PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
        PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>
        
        SELECT DISTINCT ?id
        WHERE {
            ?uri a glycan:Saccharide ;
                 glytoucan:has_primary_id ?id .
        }
        ORDER BY ?id
        """
        
        self.sparql.setQuery(query)
        results = self.sparql.query().convert()
        
        return [result["id"]["value"] for result in results["results"]["bindings"]]
        
    def get_structure_details(self, glytoucan_id: str) -> Optional[GlycanStructure]:
        """
        Get detailed structure information for a specific GlyTouCan ID.
        
        Args:
            glytoucan_id: GlyTouCan accession ID
            
        Returns:
            GlycanStructure object or None if not found
        """
        query = f"""
        PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
        PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>
        PREFIX wurcs: <http://www.glycoinfo.org/glyco/owl/wurcs#>
        PREFIX glycoct: <http://www.glycoinfo.org/glyco/owl/glycoct#>
        PREFIX iupac: <http://www.glycoinfo.org/glyco/owl/iupac#>
        PREFIX mass: <http://www.glycoinfo.org/glyco/owl/mass#>
        PREFIX comp: <http://www.glycoinfo.org/glyco/owl/composition#>
        
        SELECT ?wurcs ?glycoct ?iupac_ext ?iupac_cond ?mass_mono ?mass_avg ?composition
        WHERE {{
            ?uri a glycan:Saccharide ;
                 glytoucan:has_primary_id "{glytoucan_id}" .
                 
            OPTIONAL {{ ?uri wurcs:has_wurcs ?wurcs }}
            OPTIONAL {{ ?uri glycoct:has_sequence ?glycoct }}
            OPTIONAL {{ ?uri iupac:has_extended_iupac ?iupac_ext }}
            OPTIONAL {{ ?uri iupac:has_condensed_iupac ?iupac_cond }}
            OPTIONAL {{ ?uri mass:has_monoisotopic_mass ?mass_mono }}
            OPTIONAL {{ ?uri mass:has_average_mass ?mass_avg }}
            OPTIONAL {{ ?uri comp:has_composition ?composition }}
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            if not results["results"]["bindings"]:
                return None
                
            result = results["results"]["bindings"][0]
            
            return GlycanStructure(
                glytoucan_id=glytoucan_id,
                wurcs_sequence=result.get("wurcs", {}).get("value"),
                glycoct=result.get("glycoct", {}).get("value"),
                iupac_extended=result.get("iupac_ext", {}).get("value"),
                iupac_condensed=result.get("iupac_cond", {}).get("value"),
                mass_mono=float(result["mass_mono"]["value"]) if "mass_mono" in result else None,
                mass_avg=float(result["mass_avg"]["value"]) if "mass_avg" in result else None,
                composition=self._parse_composition(result.get("composition", {}).get("value"))
            )
            
        except Exception as e:
            logger.error(f"Error fetching structure {glytoucan_id}: {e}")
            return None
            
    def _parse_composition(self, composition_str: Optional[str]) -> Optional[Dict[str, int]]:
        """Parse monosaccharide composition string"""
        if not composition_str:
            return None
            
        try:
            # Parse format like "Hex:3,HexNAc:2,Fuc:1"
            composition = {}
            for pair in composition_str.split(","):
                mono, count = pair.split(":")
                composition[mono.strip()] = int(count.strip())
            return composition
        except Exception as e:
            logger.warning(f"Failed to parse composition '{composition_str}': {e}")
            return None
            
    async def get_structures_batch(self, 
                                 glytoucan_ids: List[str]) -> List[GlycanStructure]:
        """
        Get structure details for a batch of GlyTouCan IDs.
        
        Args:
            glytoucan_ids: List of GlyTouCan IDs to fetch
            
        Returns:
            List of GlycanStructure objects
        """
        structures = []
        
        for glytoucan_id in glytoucan_ids:
            structure = self.get_structure_details(glytoucan_id)
            if structure:
                structures.append(structure)
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
        return structures
        
    async def get_all_structures(self, 
                               limit: Optional[int] = None) -> AsyncIterator[List[GlycanStructure]]:
        """
        Generator that yields batches of glycan structures.
        
        Args:
            limit: Maximum number of structures to fetch (None for all)
            
        Yields:
            Batches of GlycanStructure objects
        """
        logger.info("Fetching all GlyTouCan structure IDs...")
        all_ids = self.get_all_structure_ids()
        
        if limit:
            all_ids = all_ids[:limit]
            
        logger.info(f"Found {len(all_ids)} structures to process")
        
        # Process in batches
        for i in range(0, len(all_ids), self.batch_size):
            batch_ids = all_ids[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}: IDs {i+1}-{min(i+self.batch_size, len(all_ids))}")
            
            batch_structures = await self.get_structures_batch(batch_ids)
            yield batch_structures
            
    def search_by_mass(self, 
                      mass: float, 
                      tolerance_ppm: float = 10.0) -> List[GlycanStructure]:
        """
        Search for glycan structures by monoisotopic mass.
        
        Args:
            mass: Target monoisotopic mass
            tolerance_ppm: Mass tolerance in ppm
            
        Returns:
            List of matching GlycanStructure objects
        """
        tolerance_da = mass * tolerance_ppm / 1e6
        min_mass = mass - tolerance_da
        max_mass = mass + tolerance_da
        
        query = f"""
        PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
        PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>
        PREFIX mass: <http://www.glycoinfo.org/glyco/owl/mass#>
        
        SELECT ?id ?mass_mono
        WHERE {{
            ?uri a glycan:Saccharide ;
                 glytoucan:has_primary_id ?id ;
                 mass:has_monoisotopic_mass ?mass_mono .
                 
            FILTER(?mass_mono >= {min_mass} && ?mass_mono <= {max_mass})
        }}
        ORDER BY ABS(?mass_mono - {mass})
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            structures = []
            for result in results["results"]["bindings"]:
                glytoucan_id = result["id"]["value"]
                structure = self.get_structure_details(glytoucan_id)
                if structure:
                    structures.append(structure)
                    
            return structures
            
        except Exception as e:
            logger.error(f"Error in mass search: {e}")
            return []
            
    def search_by_composition(self, 
                            composition: Dict[str, int]) -> List[GlycanStructure]:
        """
        Search for glycan structures by monosaccharide composition.
        
        Args:
            composition: Dictionary mapping monosaccharide names to counts
            
        Returns:
            List of matching GlycanStructure objects
        """
        # Build composition filter string
        comp_filters = []
        for mono, count in composition.items():
            comp_filters.append(f'CONTAINS(?composition, "{mono}:{count}")')
            
        comp_filter = " && ".join(comp_filters)
        
        query = f"""
        PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
        PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>
        PREFIX comp: <http://www.glycoinfo.org/glyco/owl/composition#>
        
        SELECT ?id
        WHERE {{
            ?uri a glycan:Saccharide ;
                 glytoucan:has_primary_id ?id ;
                 comp:has_composition ?composition .
                 
            FILTER({comp_filter})
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            structures = []
            for result in results["results"]["bindings"]:
                glytoucan_id = result["id"]["value"]
                structure = self.get_structure_details(glytoucan_id)
                if structure:
                    structures.append(structure)
                    
            return structures
            
        except Exception as e:
            logger.error(f"Error in composition search: {e}")
            return []
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with statistics about the repository
        """
        query = """
        PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
        PREFIX wurcs: <http://www.glycoinfo.org/glyco/owl/wurcs#>
        PREFIX mass: <http://www.glycoinfo.org/glyco/owl/mass#>
        
        SELECT 
            (COUNT(DISTINCT ?uri) as ?total_structures)
            (COUNT(DISTINCT ?wurcs) as ?structures_with_wurcs)
            (AVG(?mass_mono) as ?avg_mass)
            (MIN(?mass_mono) as ?min_mass)
            (MAX(?mass_mono) as ?max_mass)
        WHERE {
            ?uri a glycan:Saccharide .
            OPTIONAL { ?uri wurcs:has_wurcs ?wurcs }
            OPTIONAL { ?uri mass:has_monoisotopic_mass ?mass_mono }
        }
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            if results["results"]["bindings"]:
                result = results["results"]["bindings"][0]
                return {
                    "total_structures": int(result["total_structures"]["value"]),
                    "structures_with_wurcs": int(result["structures_with_wurcs"]["value"]),
                    "average_mass": float(result["avg_mass"]["value"]) if "avg_mass" in result else None,
                    "min_mass": float(result["min_mass"]["value"]) if "min_mass" in result else None,
                    "max_mass": float(result["max_mass"]["value"]) if "max_mass" in result else None,
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            
        return {"error": "Failed to retrieve statistics"}