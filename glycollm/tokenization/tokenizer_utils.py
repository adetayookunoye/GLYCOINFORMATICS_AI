"""
Utility functions for glycoinformatics tokenization.

This module provides helper functions for preprocessing and analyzing
glycan-related data before tokenization.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class WURCSParseResult:
    """Result of WURCS parsing"""
    version: str
    unique_count: int
    residue_count: int
    linkage_count: int
    residues: List[str]
    connections: List[str]
    is_valid: bool
    error_message: Optional[str] = None


@dataclass 
class SpectrumAnalysis:
    """Analysis results for mass spectrum"""
    total_peaks: int
    base_peak_mz: float
    base_peak_intensity: float
    mass_range: Tuple[float, float]
    identified_fragments: List[str]
    neutral_losses: List[str]
    signal_to_noise: float


class WURCSValidator:
    """
    Validates and analyzes WURCS notation strings.
    """
    
    def __init__(self):
        # WURCS format patterns
        self.wurcs_pattern = re.compile(
            r'^WURCS=(\d+\.\d+)/(\d+),(\d+),(\d+)/(\[.*?\])+/(.*)$'
        )
        self.residue_pattern = re.compile(r'\[([^\]]+)\]')
        
        # Known monosaccharide patterns
        self.known_residues = {
            'a2122h-1b_1-5_2*NCC/3=O': 'GlcNAc',
            'a1122h-1b_1-5': 'Gal', 
            'a1221m-1a_1-5': 'Man',
            'a1221h-1b_1-5': 'Glc',
            'a112h-1b_1-5': 'Fuc',
            'a1122A-2a_2-6': 'Neu5Ac',
            'a1122G-2a_2-6': 'Neu5Gc',
        }
        
    def validate_wurcs(self, wurcs_string: str) -> WURCSParseResult:
        """
        Validate and parse WURCS notation.
        
        Args:
            wurcs_string: WURCS notation to validate
            
        Returns:
            Parsing result with validation status
        """
        if not wurcs_string or not isinstance(wurcs_string, str):
            return WURCSParseResult(
                version="", unique_count=0, residue_count=0, linkage_count=0,
                residues=[], connections=[], is_valid=False,
                error_message="Empty or invalid input"
            )
            
        wurcs_string = wurcs_string.strip()
        
        # Check basic format
        match = self.wurcs_pattern.match(wurcs_string)
        if not match:
            return WURCSParseResult(
                version="", unique_count=0, residue_count=0, linkage_count=0,
                residues=[], connections=[], is_valid=False,
                error_message="Invalid WURCS format"
            )
            
        try:
            version = match.group(1)
            unique_count = int(match.group(2))
            residue_count = int(match.group(3))
            linkage_count = int(match.group(4))
            residues_section = match.group(5)
            connections_section = match.group(6)
            
            # Parse residues
            residue_matches = self.residue_pattern.findall(residues_section)
            
            # Validate counts match actual content
            if len(residue_matches) != residue_count:
                return WURCSParseResult(
                    version=version, unique_count=unique_count,
                    residue_count=residue_count, linkage_count=linkage_count,
                    residues=residue_matches, connections=[connections_section],
                    is_valid=False,
                    error_message=f"Residue count mismatch: expected {residue_count}, found {len(residue_matches)}"
                )
                
            # Parse connections
            connections = []
            if connections_section:
                connection_parts = connections_section.split('/')
                connections = [part for part in connection_parts if part.strip()]
                
            return WURCSParseResult(
                version=version, unique_count=unique_count,
                residue_count=residue_count, linkage_count=linkage_count,
                residues=residue_matches, connections=connections,
                is_valid=True
            )
            
        except (ValueError, IndexError) as e:
            return WURCSParseResult(
                version="", unique_count=0, residue_count=0, linkage_count=0,
                residues=[], connections=[], is_valid=False,
                error_message=f"Parsing error: {str(e)}"
            )
            
    def identify_monosaccharides(self, residues: List[str]) -> Dict[str, int]:
        """
        Identify monosaccharide types from residue definitions.
        
        Args:
            residues: List of residue definition strings
            
        Returns:
            Dictionary mapping monosaccharide names to counts
        """
        monosaccharide_counts = {}
        
        for residue in residues:
            # Try exact matches first
            mono_name = self.known_residues.get(residue)
            
            if not mono_name:
                # Try pattern matching for variations
                mono_name = self._classify_residue_pattern(residue)
                
            if mono_name:
                monosaccharide_counts[mono_name] = monosaccharide_counts.get(mono_name, 0) + 1
            else:
                # Unknown residue
                monosaccharide_counts['Unknown'] = monosaccharide_counts.get('Unknown', 0) + 1
                
        return monosaccharide_counts
        
    def _classify_residue_pattern(self, residue: str) -> Optional[str]:
        """Classify residue by pattern matching"""
        
        # N-acetyl hexosamine patterns
        if 'NCC' in residue and '3=O' in residue:
            if 'a2122h' in residue:
                return 'GlcNAc'
            elif 'a2122A' in residue:
                return 'GalNAc'
                
        # Hexose patterns  
        elif 'h-1b_1-5' in residue:
            if 'a1122' in residue:
                return 'Gal'
            elif 'a1221' in residue:
                return 'Glc'
                
        # Mannose pattern
        elif 'h-1a_1-5' in residue and 'a1221m' in residue:
            return 'Man'
            
        # Deoxyhexose (Fucose) pattern
        elif 'h-1b_1-5' in residue and 'a112' in residue:
            return 'Fuc'
            
        # Sialic acid patterns
        elif 'A-2a_2-6' in residue:
            return 'Neu5Ac'
        elif 'G-2a_2-6' in residue:
            return 'Neu5Gc'
            
        return None


class SpectrumAnalyzer:
    """
    Analyzes mass spectrometry data for tokenization preprocessing.
    """
    
    def __init__(self):
        # Common glycan fragments (m/z values)
        self.glycan_fragments = {
            146.058: 'Fuc',
            163.060: 'Hex', 
            204.087: 'HexNAc',
            274.093: 'NeuAc',
            290.088: 'NeuGc',
            366.139: 'Hex-HexNAc',
            512.197: 'Hex2-HexNAc',
        }
        
        # Common neutral losses
        self.neutral_losses = {
            18.010: 'H2O',
            32.026: 'CH2O2', 
            60.021: 'C2H4O2',
            146.058: 'Fuc',
            162.053: 'Hex',
            203.079: 'HexNAc',
            274.093: 'NeuAc',
        }
        
    def analyze_spectrum(self, 
                        peaks: List[Tuple[float, float]],
                        precursor_mz: Optional[float] = None,
                        noise_threshold: float = 0.01) -> SpectrumAnalysis:
        """
        Analyze mass spectrum for preprocessing insights.
        
        Args:
            peaks: List of (m/z, intensity) tuples
            precursor_mz: Precursor ion m/z
            noise_threshold: Relative intensity threshold for noise
            
        Returns:
            Spectrum analysis results
        """
        if not peaks:
            return SpectrumAnalysis(
                total_peaks=0, base_peak_mz=0.0, base_peak_intensity=0.0,
                mass_range=(0.0, 0.0), identified_fragments=[],
                neutral_losses=[], signal_to_noise=0.0
            )
            
        # Sort peaks by intensity (descending)
        sorted_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
        
        # Find base peak
        base_peak = sorted_peaks[0]
        base_peak_mz, base_peak_intensity = base_peak
        
        # Calculate mass range
        mz_values = [peak[0] for peak in peaks]
        mass_range = (min(mz_values), max(mz_values))
        
        # Filter significant peaks (above noise threshold)
        max_intensity = base_peak_intensity
        significant_peaks = [
            peak for peak in peaks 
            if peak[1] / max_intensity >= noise_threshold
        ]
        
        # Identify fragments
        identified_fragments = self._identify_fragments(significant_peaks)
        
        # Identify neutral losses (if precursor known)
        neutral_losses = []
        if precursor_mz:
            neutral_losses = self._identify_losses(significant_peaks, precursor_mz)
            
        # Calculate signal-to-noise ratio
        if len(peaks) > 1:
            signal_intensities = [peak[1] for peak in significant_peaks]
            noise_intensities = [
                peak[1] for peak in peaks 
                if peak[1] / max_intensity < noise_threshold
            ]
            
            signal_mean = sum(signal_intensities) / len(signal_intensities) if signal_intensities else 0
            noise_mean = sum(noise_intensities) / len(noise_intensities) if noise_intensities else 0.001
            
            signal_to_noise = signal_mean / noise_mean if noise_mean > 0 else 1000.0
        else:
            signal_to_noise = 1000.0
            
        return SpectrumAnalysis(
            total_peaks=len(peaks),
            base_peak_mz=base_peak_mz,
            base_peak_intensity=base_peak_intensity,
            mass_range=mass_range,
            identified_fragments=identified_fragments,
            neutral_losses=neutral_losses,
            signal_to_noise=signal_to_noise
        )
        
    def _identify_fragments(self, 
                          peaks: List[Tuple[float, float]],
                          tolerance: float = 0.05) -> List[str]:
        """Identify known fragment ions"""
        fragments = []
        
        for mz, intensity in peaks:
            for ref_mz, fragment_name in self.glycan_fragments.items():
                if abs(mz - ref_mz) <= tolerance:
                    fragments.append(f"{fragment_name}@{mz:.3f}")
                    break
                    
        return fragments
        
    def _identify_losses(self, 
                        peaks: List[Tuple[float, float]],
                        precursor_mz: float,
                        tolerance: float = 0.05) -> List[str]:
        """Identify neutral loss patterns"""
        losses = []
        
        for mz, intensity in peaks:
            mass_diff = precursor_mz - mz
            
            for loss_mass, loss_name in self.neutral_losses.items():
                if abs(mass_diff - loss_mass) <= tolerance:
                    losses.append(f"-{loss_name}@{mz:.3f}")
                    break
                    
        return losses


class GlycanTextProcessor:
    """
    Preprocesses glycan-related scientific text for tokenization.
    """
    
    def __init__(self):
        # Glycomics terminology normalization
        self.term_normalizations = {
            # Monosaccharide name variants
            'n-acetylglucosamine': 'glcnac',
            'n-acetyl-glucosamine': 'glcnac',
            'glucnac': 'glcnac',
            'n-acetylgalactosamine': 'galnac',
            'n-acetyl-galactosamine': 'galnac',
            'galnac': 'galnac',
            'n-acetylneuraminic acid': 'neu5ac',
            'sialic acid': 'neu5ac',
            'neuraminic acid': 'neu5ac',
            
            # Structure type variants
            'n-glycan': 'n_glycan',
            'n-linked glycan': 'n_glycan',
            'o-glycan': 'o_glycan', 
            'o-linked glycan': 'o_glycan',
            'glycosaminoglycan': 'gag',
            'proteoglycan': 'proteoglycan',
            'glycolipid': 'glycolipid',
            
            # Method variants
            'mass spectrometry': 'ms',
            'liquid chromatography': 'lc',
            'lc-ms': 'lcms',
            'lc/ms': 'lcms',
            'maldi-tof': 'maldi_tof',
            'electrospray ionization': 'esi',
            
            # Linkage variants
            'alpha-1,2': 'α1-2',
            'alpha-1,3': 'α1-3',
            'alpha-1,4': 'α1-4',
            'alpha-1,6': 'α1-6',
            'beta-1,2': 'β1-2',
            'beta-1,3': 'β1-3',
            'beta-1,4': 'β1-4',
            'beta-1,6': 'β1-6',
        }
        
        # Chemical formula patterns
        self.formula_pattern = re.compile(r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b')
        
        # Mass value patterns (with units)
        self.mass_pattern = re.compile(r'\b\d+\.?\d*\s*(?:da|dalton|daltons|m/z|mz)\b', re.IGNORECASE)
        
        # Concentration patterns
        self.conc_pattern = re.compile(r'\b\d+\.?\d*\s*(?:m|mm|μm|um|nm|pm)\b', re.IGNORECASE)
        
    def preprocess_text(self, text: str, normalize_terms: bool = True) -> str:
        """
        Preprocess scientific text for glycomics analysis.
        
        Args:
            text: Input text to preprocess
            normalize_terms: Whether to normalize technical terms
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase and normalize whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Normalize technical terms
        if normalize_terms:
            for original, normalized in self.term_normalizations.items():
                text = re.sub(r'\b' + re.escape(original) + r'\b', normalized, text)
                
        # Standardize chemical formulas (keep original case)
        formulas = self.formula_pattern.findall(text)
        for formula in formulas:
            text = text.replace(formula.lower(), f"[FORMULA_{formula}]")
            
        # Standardize mass values
        masses = self.mass_pattern.findall(text)
        for mass in masses:
            normalized_mass = re.sub(r'\s+', '', mass).replace('dalton', 'da')
            text = text.replace(mass, f"[MASS_{normalized_mass}]")
            
        # Standardize concentrations
        concentrations = self.conc_pattern.findall(text)
        for conc in concentrations:
            normalized_conc = re.sub(r'\s+', '', conc)
            text = text.replace(conc, f"[CONC_{normalized_conc}]")
            
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\[\]-]', ' ', text)
        
        # Final whitespace normalization
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def extract_glycan_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract glycan-related entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted entities by category
        """
        entities = {
            'monosaccharides': [],
            'linkages': [],
            'structures': [],
            'methods': [],
            'modifications': [],
            'formulas': [],
            'masses': []
        }
        
        # Preprocess text
        processed_text = self.preprocess_text(text, normalize_terms=False)
        
        # Extract monosaccharides
        mono_patterns = [
            r'\b(?:glc|glucose)\b',
            r'\b(?:gal|galactose)\b', 
            r'\b(?:man|mannose)\b',
            r'\b(?:fuc|fucose)\b',
            r'\b(?:xyl|xylose)\b',
            r'\b(?:glcnac|n-acetylglucosamine)\b',
            r'\b(?:galnac|n-acetylgalactosamine)\b',
            r'\b(?:neu5ac|sialic acid)\b'
        ]
        
        for pattern in mono_patterns:
            matches = re.findall(pattern, processed_text, re.IGNORECASE)
            entities['monosaccharides'].extend(matches)
            
        # Extract linkages
        linkage_pattern = r'[αβ]1?[-,]\d+'
        entities['linkages'] = re.findall(linkage_pattern, processed_text)
        
        # Extract structure types
        struct_pattern = r'\b(?:n-?glycan|o-?glycan|glycoprotein|glycolipid|proteoglycan)\b'
        entities['structures'] = re.findall(struct_pattern, processed_text, re.IGNORECASE)
        
        # Extract methods
        method_pattern = r'\b(?:ms|lc|maldi|esi|tof|cid|hcd|etd)\b'
        entities['methods'] = re.findall(method_pattern, processed_text, re.IGNORECASE)
        
        # Extract modifications
        mod_pattern = r'\b(?:acetyl|sulfat|phosphat|methylat)\w*\b'
        entities['modifications'] = re.findall(mod_pattern, processed_text, re.IGNORECASE)
        
        # Extract formulas and masses
        entities['formulas'] = self.formula_pattern.findall(text)
        entities['masses'] = self.mass_pattern.findall(text)
        
        # Remove duplicates and empty strings
        for category in entities:
            entities[category] = list(set(filter(None, entities[category])))
            
        return entities


class TokenizerTester:
    """
    Testing utilities for glycoinformatics tokenizers.
    """
    
    def __init__(self):
        self.wurcs_validator = WURCSValidator()
        self.spectrum_analyzer = SpectrumAnalyzer()
        self.text_processor = GlycanTextProcessor()
        
    def test_wurcs_samples(self, wurcs_list: List[str]) -> Dict[str, any]:
        """
        Test WURCS tokenization with sample data.
        
        Args:
            wurcs_list: List of WURCS strings to test
            
        Returns:
            Test results summary
        """
        results = {
            'total_samples': len(wurcs_list),
            'valid_samples': 0,
            'invalid_samples': 0,
            'parsing_errors': [],
            'monosaccharide_distribution': {},
            'structure_complexity': []
        }
        
        all_monosaccharides = {}
        
        for i, wurcs in enumerate(wurcs_list):
            parse_result = self.wurcs_validator.validate_wurcs(wurcs)
            
            if parse_result.is_valid:
                results['valid_samples'] += 1
                
                # Analyze monosaccharide composition
                mono_counts = self.wurcs_validator.identify_monosaccharides(parse_result.residues)
                
                # Update global distribution
                for mono, count in mono_counts.items():
                    all_monosaccharides[mono] = all_monosaccharides.get(mono, 0) + count
                    
                # Record structure complexity
                complexity = {
                    'residue_count': parse_result.residue_count,
                    'linkage_count': parse_result.linkage_count,
                    'unique_monosaccharides': len(mono_counts)
                }
                results['structure_complexity'].append(complexity)
                
            else:
                results['invalid_samples'] += 1
                results['parsing_errors'].append({
                    'index': i,
                    'wurcs': wurcs,
                    'error': parse_result.error_message
                })
                
        results['monosaccharide_distribution'] = all_monosaccharides
        
        return results
        
    def test_spectrum_processing(self, 
                               spectra_list: List[List[Tuple[float, float]]]) -> Dict[str, any]:
        """
        Test spectrum tokenization with sample data.
        
        Args:
            spectra_list: List of peak lists to test
            
        Returns:
            Test results summary
        """
        results = {
            'total_spectra': len(spectra_list),
            'peak_statistics': {
                'min_peaks': float('inf'),
                'max_peaks': 0,
                'avg_peaks': 0,
                'total_peaks': 0
            },
            'fragment_identification': {},
            'mass_range_distribution': [],
            'signal_quality': []
        }
        
        fragment_counts = {}
        total_peaks = 0
        
        for spectrum in spectra_list:
            if not spectrum:
                continue
                
            analysis = self.spectrum_analyzer.analyze_spectrum(spectrum)
            
            # Update peak statistics
            peak_count = len(spectrum)
            total_peaks += peak_count
            
            results['peak_statistics']['min_peaks'] = min(
                results['peak_statistics']['min_peaks'], peak_count
            )
            results['peak_statistics']['max_peaks'] = max(
                results['peak_statistics']['max_peaks'], peak_count
            )
            
            # Record mass range
            results['mass_range_distribution'].append(analysis.mass_range)
            
            # Record signal quality
            results['signal_quality'].append(analysis.signal_to_noise)
            
            # Count identified fragments
            for fragment in analysis.identified_fragments:
                fragment_name = fragment.split('@')[0]  # Remove @m/z part
                fragment_counts[fragment_name] = fragment_counts.get(fragment_name, 0) + 1
                
        # Finalize statistics
        results['peak_statistics']['total_peaks'] = total_peaks
        if len(spectra_list) > 0:
            results['peak_statistics']['avg_peaks'] = total_peaks / len(spectra_list)
            
        if results['peak_statistics']['min_peaks'] == float('inf'):
            results['peak_statistics']['min_peaks'] = 0
            
        results['fragment_identification'] = fragment_counts
        
        return results
        
    def generate_test_report(self, 
                           wurcs_results: Optional[Dict] = None,
                           spectrum_results: Optional[Dict] = None,
                           text_samples: Optional[List[str]] = None) -> str:
        """
        Generate comprehensive test report.
        
        Args:
            wurcs_results: WURCS testing results
            spectrum_results: Spectrum testing results  
            text_samples: Text samples for entity extraction testing
            
        Returns:
            Formatted test report
        """
        report_lines = [
            "# Glycoinformatics Tokenizer Test Report",
            "=" * 50,
            ""
        ]
        
        # WURCS results
        if wurcs_results:
            report_lines.extend([
                "## WURCS Tokenization Results",
                f"- Total samples: {wurcs_results['total_samples']}",
                f"- Valid samples: {wurcs_results['valid_samples']}",
                f"- Invalid samples: {wurcs_results['invalid_samples']}",
                f"- Success rate: {wurcs_results['valid_samples']/wurcs_results['total_samples']*100:.1f}%",
                "",
                "### Monosaccharide Distribution:",
            ])
            
            for mono, count in wurcs_results['monosaccharide_distribution'].items():
                report_lines.append(f"- {mono}: {count}")
                
            if wurcs_results['parsing_errors']:
                report_lines.extend([
                    "", "### Parsing Errors:",
                ])
                for error in wurcs_results['parsing_errors'][:5]:  # Show first 5
                    report_lines.append(f"- Sample {error['index']}: {error['error']}")
                    
        # Spectrum results
        if spectrum_results:
            stats = spectrum_results['peak_statistics']
            report_lines.extend([
                "", "## Spectrum Processing Results",
                f"- Total spectra: {spectrum_results['total_spectra']}",
                f"- Peak count range: {stats['min_peaks']} - {stats['max_peaks']}",
                f"- Average peaks per spectrum: {stats['avg_peaks']:.1f}",
                f"- Total peaks processed: {stats['total_peaks']}",
                "",
                "### Fragment Identification:",
            ])
            
            for fragment, count in spectrum_results['fragment_identification'].items():
                report_lines.append(f"- {fragment}: {count} spectra")
                
        # Text processing results
        if text_samples:
            report_lines.extend([
                "", "## Text Processing Results",
                f"- Text samples analyzed: {len(text_samples)}",
                ""
            ])
            
            # Analyze sample text
            all_entities = {}
            for text in text_samples[:10]:  # Analyze first 10 samples
                entities = self.text_processor.extract_glycan_entities(text)
                for category, items in entities.items():
                    if category not in all_entities:
                        all_entities[category] = set()
                    all_entities[category].update(items)
                    
            report_lines.append("### Extracted Entity Types:")
            for category, items in all_entities.items():
                report_lines.append(f"- {category}: {len(items)} unique terms")
                
        return "\n".join(report_lines)


# Utility functions for data preparation
def load_sample_data(data_dir: str) -> Dict[str, List]:
    """
    Load sample data for tokenizer testing.
    
    Args:
        data_dir: Directory containing sample data files
        
    Returns:
        Dictionary with sample data by type
    """
    data_path = Path(data_dir)
    sample_data = {
        'wurcs_sequences': [],
        'spectra': [],
        'text_samples': []
    }
    
    # Load WURCS samples
    wurcs_file = data_path / "sample_wurcs.json"
    if wurcs_file.exists():
        with open(wurcs_file, 'r') as f:
            sample_data['wurcs_sequences'] = json.load(f)
            
    # Load spectrum samples
    spectra_file = data_path / "sample_spectra.json" 
    if spectra_file.exists():
        with open(spectra_file, 'r') as f:
            sample_data['spectra'] = json.load(f)
            
    # Load text samples
    text_file = data_path / "sample_texts.json"
    if text_file.exists():
        with open(text_file, 'r') as f:
            sample_data['text_samples'] = json.load(f)
            
    return sample_data


def create_sample_data_files(output_dir: str):
    """
    Create sample data files for testing.
    
    Args:
        output_dir: Directory to create sample files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample WURCS sequences
    sample_wurcs = [
        "WURCS=2.0/3,3,2/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5]/1-2-3/a4-b1_b4-c1",
        "WURCS=2.0/2,2,1/[a1122h-1b_1-5][a1221m-1a_1-5]/1-2/a4-b1",
        "WURCS=2.0/1,1,0/[a112h-1b_1-5]/1/",
        "WURCS=2.0/4,4,3/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1221m-1a_1-5][a112h-1b_1-5]/1-2-3-4/a4-b1_b3-c1_c6-d1"
    ]
    
    # Sample spectra (m/z, intensity pairs)
    sample_spectra = [
        [(163.060, 100.0), (204.087, 85.5), (366.139, 45.2), (512.197, 22.1)],
        [(146.058, 100.0), (292.116, 78.3), (438.174, 34.7)],
        [(274.093, 100.0), (657.235, 67.8), (819.288, 23.4), (981.341, 12.1)]
    ]
    
    # Sample text
    sample_texts = [
        "N-linked glycans were analyzed by MALDI-TOF mass spectrometry showing α1-6 linked mannose residues.",
        "The core structure contains GlcNAc and galactose with β1-4 linkages characteristic of complex N-glycans.",
        "Sialic acid residues were identified at m/z 274.093 indicating terminal Neu5Ac modifications.",
        "LC-MS analysis revealed fucosylated structures with loss of 146.058 Da corresponding to fucose neutral loss."
    ]
    
    # Save sample files
    with open(output_path / "sample_wurcs.json", 'w') as f:
        json.dump(sample_wurcs, f, indent=2)
        
    with open(output_path / "sample_spectra.json", 'w') as f:
        json.dump(sample_spectra, f, indent=2)
        
    with open(output_path / "sample_texts.json", 'w') as f:
        json.dump(sample_texts, f, indent=2)
        
    logger.info(f"Sample data files created in {output_dir}")