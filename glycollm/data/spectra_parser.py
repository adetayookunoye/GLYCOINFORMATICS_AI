"""
MS/MS Spectra Parser for GlycoPOST MGF Files

Parses tandem mass spectrometry data from MGF (Mascot Generic Format) files,
extracts peaks, normalizes intensities, and prepares data for GlycoLLM training.

Author: Adetayo Research Team
Date: November 2025
"""

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
import numpy as np
from collections import defaultdict

try:
    from pyteomics import mgf, mzml, mzxml
    PYTEOMICS_AVAILABLE = True
except ImportError:
    PYTEOMICS_AVAILABLE = False
    logging.warning("pyteomics not available. Install with: pip install pyteomics")

logger = logging.getLogger(__name__)


@dataclass
class MSSpectrum:
    """Represents a single MS/MS spectrum"""
    spectrum_id: str
    precursor_mz: float
    precursor_charge: Optional[int]
    precursor_intensity: Optional[float]
    peaks: np.ndarray  # Shape: (n_peaks, 2) - [m/z, intensity]
    retention_time: Optional[float] = None
    collision_energy: Optional[float] = None
    glycan_structure: Optional[str] = None  # WURCS, GlycoCT, or IUPAC
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Ensure peaks are numpy array
        if not isinstance(self.peaks, np.ndarray):
            self.peaks = np.array(self.peaks)
        if len(self.peaks.shape) == 1 and len(self.peaks) > 0:
            # Reshape if flat array
            self.peaks = self.peaks.reshape(-1, 2)
    
    def normalize_intensities(self, method: str = "max"):
        """Normalize peak intensities"""
        if len(self.peaks) == 0:
            return
        
        intensities = self.peaks[:, 1]
        if method == "max":
            max_intensity = np.max(intensities)
            if max_intensity > 0:
                self.peaks[:, 1] = intensities / max_intensity
        elif method == "sum":
            total_intensity = np.sum(intensities)
            if total_intensity > 0:
                self.peaks[:, 1] = intensities / total_intensity
        elif method == "sqrt":
            self.peaks[:, 1] = np.sqrt(intensities)
            max_intensity = np.max(self.peaks[:, 1])
            if max_intensity > 0:
                self.peaks[:, 1] /= max_intensity
    
    def filter_peaks(self, min_intensity: float = 0.01, top_n: Optional[int] = None):
        """Filter low-intensity peaks and optionally keep only top N peaks"""
        if len(self.peaks) == 0:
            return
        
        # Filter by intensity threshold
        mask = self.peaks[:, 1] >= min_intensity
        self.peaks = self.peaks[mask]
        
        # Keep only top N peaks if specified
        if top_n is not None and len(self.peaks) > top_n:
            # Sort by intensity descending
            sorted_indices = np.argsort(self.peaks[:, 1])[::-1]
            self.peaks = self.peaks[sorted_indices[:top_n]]
            # Re-sort by m/z for consistency
            sorted_indices = np.argsort(self.peaks[:, 0])
            self.peaks = self.peaks[sorted_indices]
    
    def bin_spectrum(self, bin_size: float = 1.0, mz_range: Tuple[float, float] = (0, 2000)):
        """Bin spectrum into fixed m/z bins for neural network input"""
        min_mz, max_mz = mz_range
        n_bins = int((max_mz - min_mz) / bin_size)
        binned = np.zeros(n_bins)
        
        for mz, intensity in self.peaks:
            if min_mz <= mz < max_mz:
                bin_idx = int((mz - min_mz) / bin_size)
                binned[bin_idx] = max(binned[bin_idx], intensity)  # Take max if multiple peaks in bin
        
        return binned
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'spectrum_id': self.spectrum_id,
            'precursor_mz': float(self.precursor_mz),
            'precursor_charge': self.precursor_charge,
            'precursor_intensity': float(self.precursor_intensity) if self.precursor_intensity else None,
            'peaks': self.peaks.tolist(),
            'retention_time': float(self.retention_time) if self.retention_time else None,
            'collision_energy': float(self.collision_energy) if self.collision_energy else None,
            'glycan_structure': self.glycan_structure,
            'metadata': self.metadata
        }


class SpectraParser:
    """Parser for MS/MS spectra files (MGF, mzML, mzXML)"""
    
    def __init__(self, 
                 normalize: bool = True,
                 normalization_method: str = "max",
                 filter_peaks: bool = True,
                 min_intensity: float = 0.01,
                 top_n_peaks: Optional[int] = 150):
        """
        Initialize spectra parser
        
        Args:
            normalize: Whether to normalize intensities
            normalization_method: Method for normalization ('max', 'sum', 'sqrt')
            filter_peaks: Whether to filter low-intensity peaks
            min_intensity: Minimum relative intensity threshold
            top_n_peaks: Keep only top N most intense peaks (None = keep all)
        """
        if not PYTEOMICS_AVAILABLE:
            raise ImportError("pyteomics is required. Install with: pip install pyteomics")
        
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.filter_peaks = filter_peaks
        self.min_intensity = min_intensity
        self.top_n_peaks = top_n_peaks
        
        logger.info(f"SpectraParser initialized: normalize={normalize}, filter={filter_peaks}, top_n={top_n_peaks}")
    
    def parse_mgf(self, mgf_path: Path) -> Iterator[MSSpectrum]:
        """
        Parse MGF file and yield MSSpectrum objects
        
        Args:
            mgf_path: Path to MGF file
            
        Yields:
            MSSpectrum objects
        """
        logger.info(f"Parsing MGF file: {mgf_path}")
        
        with mgf.read(str(mgf_path)) as reader:
            for idx, spectrum_dict in enumerate(reader):
                try:
                    spectrum = self._parse_spectrum_dict(spectrum_dict, f"mgf_{idx}")
                    if spectrum:
                        yield spectrum
                except Exception as e:
                    logger.warning(f"Error parsing spectrum {idx}: {e}")
                    continue
    
    def parse_mzml(self, mzml_path: Path) -> Iterator[MSSpectrum]:
        """Parse mzML file and yield MSSpectrum objects"""
        logger.info(f"Parsing mzML file: {mzml_path}")
        
        with mzml.read(str(mzml_path)) as reader:
            for idx, spectrum_dict in enumerate(reader):
                try:
                    # Only process MS2 spectra
                    if spectrum_dict.get('ms level', 1) == 2:
                        spectrum = self._parse_mzml_spectrum(spectrum_dict, f"mzml_{idx}")
                        if spectrum:
                            yield spectrum
                except Exception as e:
                    logger.warning(f"Error parsing spectrum {idx}: {e}")
                    continue
    
    def parse_mzxml(self, mzxml_path: Path) -> Iterator[MSSpectrum]:
        """Parse mzXML file and yield MSSpectrum objects"""
        logger.info(f"Parsing mzXML file: {mzxml_path}")
        
        with mzxml.read(str(mzxml_path)) as reader:
            for idx, spectrum_dict in enumerate(reader):
                try:
                    # Only process MS2 spectra
                    if spectrum_dict.get('msLevel', 1) == 2:
                        spectrum = self._parse_mzxml_spectrum(spectrum_dict, f"mzxml_{idx}")
                        if spectrum:
                            yield spectrum
                except Exception as e:
                    logger.warning(f"Error parsing spectrum {idx}: {e}")
                    continue
    
    def parse_file(self, file_path: Path) -> Iterator[MSSpectrum]:
        """
        Auto-detect file format and parse
        
        Args:
            file_path: Path to spectrum file
            
        Yields:
            MSSpectrum objects
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.mgf':
            yield from self.parse_mgf(file_path)
        elif suffix == '.mzml':
            yield from self.parse_mzml(file_path)
        elif suffix in ['.mzxml', '.xml']:
            yield from self.parse_mzxml(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _parse_spectrum_dict(self, spectrum_dict: Dict, spectrum_id: str) -> Optional[MSSpectrum]:
        """Parse pyteomics spectrum dictionary (MGF format)"""
        try:
            # Extract precursor information
            precursor_mz = float(spectrum_dict.get('pepmass', [0])[0]) if 'pepmass' in spectrum_dict else float(spectrum_dict.get('precursor_mz', 0))
            precursor_charge = spectrum_dict.get('charge')
            if isinstance(precursor_charge, list):
                precursor_charge = precursor_charge[0] if precursor_charge else None
            precursor_intensity = float(spectrum_dict.get('pepmass', [0, 0])[1]) if 'pepmass' in spectrum_dict and len(spectrum_dict['pepmass']) > 1 else None
            
            # Extract peaks
            mz_array = spectrum_dict.get('m/z array', [])
            intensity_array = spectrum_dict.get('intensity array', [])
            
            if len(mz_array) == 0 or len(intensity_array) == 0:
                return None
            
            peaks = np.column_stack([mz_array, intensity_array])
            
            # Extract metadata
            retention_time = spectrum_dict.get('rtinseconds', spectrum_dict.get('retention_time'))
            collision_energy = spectrum_dict.get('collision_energy')
            
            # Try to extract glycan structure from title or params
            glycan_structure = None
            title = spectrum_dict.get('title', '')
            params = spectrum_dict.get('params', {})
            
            # Common patterns for glycan structures in titles
            for key in ['glycan', 'structure', 'composition', 'wurcs', 'glycoct']:
                if key in params:
                    glycan_structure = params[key]
                    break
            
            if not glycan_structure and title:
                # Try to extract from title
                glycan_structure = self._extract_glycan_from_title(title)
            
            # Create spectrum
            spectrum = MSSpectrum(
                spectrum_id=spectrum_id,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                precursor_intensity=precursor_intensity,
                peaks=peaks,
                retention_time=retention_time,
                collision_energy=collision_energy,
                glycan_structure=glycan_structure,
                metadata={k: v for k, v in spectrum_dict.items() if k not in ['m/z array', 'intensity array', 'pepmass']}
            )
            
            # Apply processing
            if self.normalize:
                spectrum.normalize_intensities(self.normalization_method)
            if self.filter_peaks:
                spectrum.filter_peaks(self.min_intensity, self.top_n_peaks)
            
            return spectrum
            
        except Exception as e:
            logger.error(f"Error parsing spectrum {spectrum_id}: {e}")
            return None
    
    def _parse_mzml_spectrum(self, spectrum_dict: Dict, spectrum_id: str) -> Optional[MSSpectrum]:
        """Parse mzML spectrum"""
        try:
            precursor_list = spectrum_dict.get('precursorList', {}).get('precursor', [])
            if not precursor_list:
                return None
            
            precursor = precursor_list[0]
            selected_ion = precursor.get('selectedIonList', {}).get('selectedIon', [{}])[0]
            
            precursor_mz = float(selected_ion.get('selected ion m/z', 0))
            precursor_charge = selected_ion.get('charge state')
            precursor_intensity = selected_ion.get('peak intensity')
            
            mz_array = spectrum_dict.get('m/z array', [])
            intensity_array = spectrum_dict.get('intensity array', [])
            
            if len(mz_array) == 0:
                return None
            
            peaks = np.column_stack([mz_array, intensity_array])
            
            scan_list = spectrum_dict.get('scanList', {}).get('scan', [])
            retention_time = scan_list[0].get('scan start time') if scan_list else None
            
            spectrum = MSSpectrum(
                spectrum_id=spectrum_id,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                precursor_intensity=precursor_intensity,
                peaks=peaks,
                retention_time=retention_time,
                metadata=spectrum_dict
            )
            
            if self.normalize:
                spectrum.normalize_intensities(self.normalization_method)
            if self.filter_peaks:
                spectrum.filter_peaks(self.min_intensity, self.top_n_peaks)
            
            return spectrum
            
        except Exception as e:
            logger.error(f"Error parsing mzML spectrum {spectrum_id}: {e}")
            return None
    
    def _parse_mzxml_spectrum(self, spectrum_dict: Dict, spectrum_id: str) -> Optional[MSSpectrum]:
        """Parse mzXML spectrum"""
        try:
            precursor_mz = float(spectrum_dict.get('precursorMz', [{}])[0].get('precursorMz', 0))
            precursor_charge = spectrum_dict.get('precursorMz', [{}])[0].get('precursorCharge')
            precursor_intensity = spectrum_dict.get('precursorMz', [{}])[0].get('precursorIntensity')
            
            mz_array = spectrum_dict.get('m/z array', [])
            intensity_array = spectrum_dict.get('intensity array', [])
            
            if len(mz_array) == 0:
                return None
            
            peaks = np.column_stack([mz_array, intensity_array])
            retention_time = spectrum_dict.get('retentionTime')
            
            spectrum = MSSpectrum(
                spectrum_id=spectrum_id,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                precursor_intensity=precursor_intensity,
                peaks=peaks,
                retention_time=retention_time,
                metadata=spectrum_dict
            )
            
            if self.normalize:
                spectrum.normalize_intensities(self.normalization_method)
            if self.filter_peaks:
                spectrum.filter_peaks(self.min_intensity, self.top_n_peaks)
            
            return spectrum
            
        except Exception as e:
            logger.error(f"Error parsing mzXML spectrum {spectrum_id}: {e}")
            return None
    
    def _extract_glycan_from_title(self, title: str) -> Optional[str]:
        """Try to extract glycan structure from spectrum title"""
        # Common patterns in GlycoPOST and other databases
        patterns = [
            r'WURCS=[^\s]+',  # WURCS format
            r'GlycoCT\s*=\s*[^\s]+',  # GlycoCT
            r'composition[:\s]+([A-Za-z0-9\-\(\)]+)',  # Composition
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None


class SpectraDatasetBuilder:
    """Build training datasets from parsed spectra"""
    
    def __init__(self, parser: SpectraParser):
        self.parser = parser
    
    def build_spec_to_struct_dataset(self, 
                                     spectra_files: List[Path],
                                     output_path: Optional[Path] = None) -> List[Dict]:
        """
        Build spec→struct training dataset
        
        Args:
            spectra_files: List of spectra file paths
            output_path: Optional path to save dataset
            
        Returns:
            List of training samples
        """
        dataset = []
        
        for file_path in spectra_files:
            logger.info(f"Processing {file_path}")
            
            for spectrum in self.parser.parse_file(file_path):
                if spectrum.glycan_structure:  # Only include if we have ground truth
                    sample = {
                        'spectrum_id': spectrum.spectrum_id,
                        'precursor_mz': spectrum.precursor_mz,
                        'precursor_charge': spectrum.precursor_charge,
                        'peaks': spectrum.peaks.tolist(),
                        'target_structure': spectrum.glycan_structure,
                        'metadata': spectrum.metadata
                    }
                    dataset.append(sample)
        
        logger.info(f"Built dataset with {len(dataset)} samples")
        
        if output_path:
            import json
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            logger.info(f"Saved dataset to {output_path}")
        
        return dataset
    
    def compute_statistics(self, spectra_files: List[Path]) -> Dict:
        """Compute dataset statistics"""
        stats = {
            'total_spectra': 0,
            'spectra_with_structure': 0,
            'precursor_mz': [],
            'num_peaks': [],
            'collision_energies': [],
            'charges': defaultdict(int)
        }
        
        for file_path in spectra_files:
            for spectrum in self.parser.parse_file(file_path):
                stats['total_spectra'] += 1
                if spectrum.glycan_structure:
                    stats['spectra_with_structure'] += 1
                
                stats['precursor_mz'].append(spectrum.precursor_mz)
                stats['num_peaks'].append(len(spectrum.peaks))
                
                if spectrum.collision_energy:
                    stats['collision_energies'].append(spectrum.collision_energy)
                if spectrum.precursor_charge:
                    stats['charges'][spectrum.precursor_charge] += 1
        
        # Compute summary statistics
        summary = {
            'total_spectra': stats['total_spectra'],
            'spectra_with_structure': stats['spectra_with_structure'],
            'coverage_rate': stats['spectra_with_structure'] / max(stats['total_spectra'], 1),
            'precursor_mz': {
                'mean': float(np.mean(stats['precursor_mz'])),
                'std': float(np.std(stats['precursor_mz'])),
                'min': float(np.min(stats['precursor_mz'])),
                'max': float(np.max(stats['precursor_mz']))
            },
            'num_peaks': {
                'mean': float(np.mean(stats['num_peaks'])),
                'std': float(np.std(stats['num_peaks'])),
                'min': int(np.min(stats['num_peaks'])),
                'max': int(np.max(stats['num_peaks']))
            },
            'charge_distribution': dict(stats['charges'])
        }
        
        if stats['collision_energies']:
            summary['collision_energy'] = {
                'mean': float(np.mean(stats['collision_energies'])),
                'std': float(np.std(stats['collision_energies']))
            }
        
        return summary


# Utility functions for integration with existing codebase

def load_glycopost_spectra(data_dir: Path, limit: Optional[int] = None) -> List[MSSpectrum]:
    """
    Load GlycoPOST spectra from directory
    
    Args:
        data_dir: Directory containing MGF files
        limit: Maximum number of spectra to load
        
    Returns:
        List of MSSpectrum objects
    """
    parser = SpectraParser()
    spectra = []
    
    mgf_files = list(Path(data_dir).glob("**/*.mgf"))
    logger.info(f"Found {len(mgf_files)} MGF files")
    
    for mgf_file in mgf_files:
        for spectrum in parser.parse_mgf(mgf_file):
            spectra.append(spectrum)
            if limit and len(spectra) >= limit:
                return spectra
    
    return spectra


def create_training_batch(spectra: List[MSSpectrum], 
                         bin_size: float = 1.0,
                         mz_range: Tuple[float, float] = (0, 2000)) -> Dict:
    """
    Create a training batch from spectra
    
    Args:
        spectra: List of MSSpectrum objects
        bin_size: m/z bin size for discretization
        mz_range: m/z range for binning
        
    Returns:
        Dictionary with batched data ready for model training
    """
    batch = {
        'spectrum_ids': [],
        'binned_spectra': [],
        'precursor_mz': [],
        'precursor_charge': [],
        'target_structures': [],
        'peak_lists': []
    }
    
    for spectrum in spectra:
        batch['spectrum_ids'].append(spectrum.spectrum_id)
        batch['binned_spectra'].append(spectrum.bin_spectrum(bin_size, mz_range))
        batch['precursor_mz'].append(spectrum.precursor_mz)
        batch['precursor_charge'].append(spectrum.precursor_charge if spectrum.precursor_charge else 0)
        batch['target_structures'].append(spectrum.glycan_structure if spectrum.glycan_structure else "")
        batch['peak_lists'].append(spectrum.peaks)
    
    # Convert to numpy arrays
    batch['binned_spectra'] = np.array(batch['binned_spectra'])
    batch['precursor_mz'] = np.array(batch['precursor_mz'])
    batch['precursor_charge'] = np.array(batch['precursor_charge'])
    
    return batch
