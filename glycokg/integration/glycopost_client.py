"""
GlycoPOST API Client for Mass Spectrometry Data

This module provides a client interface to the GlycoPOST database,
focusing on MS/MS glycomics spectra and experimental data.
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncIterator, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
import json
import numpy as np
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


@dataclass
class MSSpectrum:
    """Data class for mass spectrum information"""
    spectrum_id: str
    glytoucan_id: Optional[str] = None
    precursor_mz: Optional[float] = None
    charge_state: Optional[int] = None
    collision_energy: Optional[float] = None
    ionization_mode: Optional[str] = None  # ESI+, ESI-, etc.
    instrument: Optional[str] = None
    peaks: Optional[List[Tuple[float, float]]] = None  # [(mz, intensity), ...]
    metadata: Optional[Dict[str, Any]] = None
    experimental_conditions: Optional[Dict[str, Any]] = None
    publication_pmid: Optional[str] = None
    sample_type: Optional[str] = None
    organism_taxid: Optional[int] = None
    created_date: Optional[datetime] = None


@dataclass  
class ExperimentalEvidence:
    """Data class for experimental evidence linking spectra to structures"""
    spectrum_id: str
    glytoucan_id: str
    evidence_type: str  # "MS/MS", "CID", "HCD", etc.
    confidence_score: Optional[float] = None
    annotation_method: Optional[str] = None
    validated: Optional[bool] = None
    curator: Optional[str] = None
    publication_pmid: Optional[str] = None
    notes: Optional[str] = None


class GlycoPOSTClient:
    """
    Client for interacting with the GlycoPOST MS/MS glycomics database.
    
    Provides access to mass spectra, structure annotations, and 
    experimental evidence for glycan identification.
    """
    
    def __init__(self,
                 base_url: str = "https://glycopost.glycosmos.org/api", 
                 api_version: str = "v1",
                 batch_size: int = 50,
                 rate_limit_delay: float = 0.3):
        """
        Initialize GlycoPOST client.
        
        Args:
            base_url: Base URL for GlycoPOST API
            api_version: API version to use
            batch_size: Number of spectra to fetch per batch
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
        """Make HTTP request to GlycoPOST API"""
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
            
    def _parse_peaks(self, peaks_data: Any) -> List[Tuple[float, float]]:
        """Parse peak data from various formats"""
        if not peaks_data:
            return []
            
        peaks = []
        
        if isinstance(peaks_data, str):
            # Parse string format like "100.0:1000,200.0:2000"
            for peak_str in peaks_data.split(','):
                if ':' in peak_str:
                    mz_str, intensity_str = peak_str.split(':')
                    try:
                        mz = float(mz_str.strip())
                        intensity = float(intensity_str.strip())
                        peaks.append((mz, intensity))
                    except ValueError:
                        continue
                        
        elif isinstance(peaks_data, list):
            # Parse list format [[mz, intensity], ...]
            for peak in peaks_data:
                if len(peak) >= 2:
                    try:
                        mz = float(peak[0])
                        intensity = float(peak[1])
                        peaks.append((mz, intensity))
                    except (ValueError, TypeError):
                        continue
                        
        elif isinstance(peaks_data, dict):
            # Parse dictionary format {"mz": [...], "intensity": [...]}
            if "mz" in peaks_data and "intensity" in peaks_data:
                mz_list = peaks_data["mz"]
                intensity_list = peaks_data["intensity"]
                if len(mz_list) == len(intensity_list):
                    for mz, intensity in zip(mz_list, intensity_list):
                        try:
                            peaks.append((float(mz), float(intensity)))
                        except (ValueError, TypeError):
                            continue
                            
        return sorted(peaks)  # Sort by m/z
        
    async def get_spectrum(self, spectrum_id: str) -> Optional[MSSpectrum]:
        """
        Get detailed spectrum information.
        
        Args:
            spectrum_id: GlycoPOST spectrum identifier
            
        Returns:
            MSSpectrum object or None if not found
        """
        data = await self._make_request(f"spectrum/{spectrum_id}")
        
        if not data:
            return None
            
        try:
            peaks = self._parse_peaks(data.get("peaks"))
            
            return MSSpectrum(
                spectrum_id=spectrum_id,
                glytoucan_id=data.get("glytoucan_id"),
                precursor_mz=data.get("precursor_mz"),
                charge_state=data.get("charge_state"),
                collision_energy=data.get("collision_energy"),
                ionization_mode=data.get("ionization_mode"),
                instrument=data.get("instrument"),
                peaks=peaks,
                metadata=data.get("metadata", {}),
                experimental_conditions=data.get("conditions", {}),
                publication_pmid=data.get("pmid"),
                sample_type=data.get("sample_type"),
                organism_taxid=data.get("taxid")
            )
            
        except Exception as e:
            logger.error(f"Error parsing spectrum {spectrum_id}: {e}")
            return None
            
    async def search_spectra_by_glycan(self, 
                                     glytoucan_id: str) -> List[MSSpectrum]:
        """
        Search for spectra associated with a specific glycan structure.
        
        Args:
            glytoucan_id: GlyTouCan accession ID
            
        Returns:
            List of MSSpectrum objects
        """
        data = await self._make_request("search/spectra", glytoucan_id=glytoucan_id)
        
        if not data or "spectra" not in data:
            return []
            
        spectra = []
        
        for spectrum_data in data["spectra"]:
            try:
                spectrum_id = spectrum_data.get("spectrum_id")
                if spectrum_id:
                    spectrum = await self.get_spectrum(spectrum_id)
                    if spectrum:
                        spectra.append(spectrum)
                        
            except Exception as e:
                logger.warning(f"Error processing spectrum: {e}")
                continue
                
        return spectra
        
    async def search_spectra_by_mass(self, 
                                   precursor_mz: float,
                                   tolerance_ppm: float = 10.0,
                                   charge_state: Optional[int] = None) -> List[MSSpectrum]:
        """
        Search for spectra by precursor m/z.
        
        Args:
            precursor_mz: Target precursor m/z
            tolerance_ppm: Mass tolerance in ppm
            charge_state: Optional charge state filter
            
        Returns:
            List of matching MSSpectrum objects
        """
        params = {
            "precursor_mz": precursor_mz,
            "tolerance_ppm": tolerance_ppm
        }
        
        if charge_state:
            params["charge_state"] = charge_state
            
        data = await self._make_request("search/mass", **params)
        
        if not data or "spectra" not in data:
            return []
            
        spectra = []
        
        for spectrum_data in data["spectra"]:
            try:
                spectrum_id = spectrum_data.get("spectrum_id")
                if spectrum_id:
                    spectrum = await self.get_spectrum(spectrum_id)
                    if spectrum:
                        spectra.append(spectrum)
                        
            except Exception as e:
                logger.warning(f"Error processing spectrum: {e}")
                continue
                
        return spectra
        
    async def get_all_spectra(self, 
                            organism_taxid: Optional[int] = None,
                            limit: Optional[int] = None) -> AsyncIterator[List[MSSpectrum]]:
        """
        Get all spectra from GlycoPOST, optionally filtered by organism.
        
        Args:
            organism_taxid: Optional organism filter
            limit: Maximum number of spectra to return
            
        Yields:
            Batches of MSSpectrum objects
        """
        offset = 0
        total_fetched = 0
        
        while True:
            params = {
                "offset": offset,
                "limit": self.batch_size
            }
            
            if organism_taxid:
                params["taxid"] = organism_taxid
                
            data = await self._make_request("spectra", **params)
            
            if not data or "spectra" not in data or not data["spectra"]:
                break
                
            spectra_batch = []
            
            for spectrum_data in data["spectra"]:
                try:
                    spectrum_id = spectrum_data.get("spectrum_id")
                    if spectrum_id:
                        spectrum = await self.get_spectrum(spectrum_id)
                        if spectrum:
                            spectra_batch.append(spectrum)
                            
                except Exception as e:
                    logger.warning(f"Error processing spectrum: {e}")
                    continue
                    
            if spectra_batch:
                yield spectra_batch
                
            total_fetched += len(spectra_batch)
            offset += self.batch_size
            
            # Check limit
            if limit and total_fetched >= limit:
                break
                
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
    async def get_experimental_evidence(self, 
                                      spectrum_id: Optional[str] = None,
                                      glytoucan_id: Optional[str] = None) -> List[ExperimentalEvidence]:
        """
        Get experimental evidence linking spectra to glycan structures.
        
        Args:
            spectrum_id: Optional spectrum ID filter
            glytoucan_id: Optional glycan ID filter
            
        Returns:
            List of ExperimentalEvidence objects
        """
        params = {}
        if spectrum_id:
            params["spectrum_id"] = spectrum_id
        if glytoucan_id:
            params["glytoucan_id"] = glytoucan_id
            
        data = await self._make_request("evidence", **params)
        
        if not data or "evidence" not in data:
            return []
            
        evidence_list = []
        
        for evidence_data in data["evidence"]:
            try:
                evidence = ExperimentalEvidence(
                    spectrum_id=evidence_data.get("spectrum_id"),
                    glytoucan_id=evidence_data.get("glytoucan_id"),
                    evidence_type=evidence_data.get("evidence_type"),
                    confidence_score=evidence_data.get("confidence_score"),
                    annotation_method=evidence_data.get("annotation_method"),
                    validated=evidence_data.get("validated"),
                    curator=evidence_data.get("curator"),
                    publication_pmid=evidence_data.get("pmid"),
                    notes=evidence_data.get("notes")
                )
                evidence_list.append(evidence)
                
            except Exception as e:
                logger.warning(f"Error parsing evidence: {e}")
                continue
                
        return evidence_list
        
    def normalize_spectrum(self, 
                         spectrum: MSSpectrum,
                         method: str = "tic") -> MSSpectrum:
        """
        Normalize spectrum intensities.
        
        Args:
            spectrum: MSSpectrum to normalize
            method: Normalization method ("tic", "base_peak", "median")
            
        Returns:
            Normalized MSSpectrum
        """
        if not spectrum.peaks:
            return spectrum
            
        peaks = spectrum.peaks.copy()
        intensities = [peak[1] for peak in peaks]
        
        if method == "tic":
            # Total ion current normalization
            total_intensity = sum(intensities)
            if total_intensity > 0:
                normalized_peaks = [(mz, intensity / total_intensity * 100) 
                                  for mz, intensity in peaks]
        elif method == "base_peak":
            # Base peak normalization
            max_intensity = max(intensities) if intensities else 1
            normalized_peaks = [(mz, intensity / max_intensity * 100) 
                              for mz, intensity in peaks]
        elif method == "median":
            # Median normalization
            median_intensity = np.median(intensities) if intensities else 1
            normalized_peaks = [(mz, intensity / median_intensity * 100) 
                              for mz, intensity in peaks]
        else:
            normalized_peaks = peaks
            
        # Create new spectrum with normalized peaks
        normalized_spectrum = MSSpectrum(**asdict(spectrum))
        normalized_spectrum.peaks = normalized_peaks
        
        return normalized_spectrum
        
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            data = await self._make_request("statistics")
            
            if data:
                return {
                    "total_spectra": data.get("total_spectra", 0),
                    "unique_glycans": data.get("unique_glycans", 0),
                    "organisms_covered": data.get("organisms", []),
                    "instruments": data.get("instruments", []),
                    "ionization_modes": data.get("ionization_modes", []),
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            
        return {
            "error": "Failed to retrieve statistics",
            "last_updated": datetime.now().isoformat()
        }