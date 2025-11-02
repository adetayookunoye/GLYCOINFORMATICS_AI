"""
Multimodal Dataset Creation for GlycoLLM

This module provides dataset loaders and processors for creating
multimodal training datasets from glycoinformatics sources.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Iterator, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class MultimodalSample:
    """Data class for multimodal training samples"""
    # Identifiers
    sample_id: str
    glytoucan_id: Optional[str] = None
    uniprot_id: Optional[str] = None
    spectrum_id: Optional[str] = None
    
    # Text modality
    text: Optional[str] = None  # Scientific text, annotations, descriptions
    text_type: Optional[str] = None  # "abstract", "annotation", "protocol", etc.
    
    # Structure modality  
    wurcs_sequence: Optional[str] = None
    glycoct_sequence: Optional[str] = None
    iupac_name: Optional[str] = None
    structure_graph: Optional[Dict[str, Any]] = None  # Graph representation
    
    # Spectra modality
    spectra_peaks: Optional[List[Tuple[float, float]]] = None  # [(mz, intensity)]
    precursor_mz: Optional[float] = None
    charge_state: Optional[int] = None
    collision_energy: Optional[float] = None
    
    # Metadata
    organism_taxid: Optional[int] = None
    tissue: Optional[str] = None
    disease: Optional[str] = None
    experimental_method: Optional[str] = None
    confidence_score: Optional[float] = None
    
    # Labels for supervised tasks
    labels: Optional[Dict[str, Any]] = None


class TextCorpusLoader:
    """
    Loads and processes glycomics literature and annotations.
    
    Handles PubMed abstracts, database annotations, experimental protocols,
    and other textual glycomics resources.
    """
    
    def __init__(self, data_dir: str = "data/raw/text"):
        """
        Initialize text corpus loader.
        
        Args:
            data_dir: Directory containing text data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Text preprocessing patterns
        self.glycan_patterns = {
            # Common glycan names and abbreviations
            'monosaccharides': r'\b(glucose|galactose|mannose|fucose|xylose|GlcNAc|GalNAc|ManNAc|Neu5Ac|Neu5Gc)\b',
            'linkages': r'\b([αβ]1?-[1-6])\b',
            'structures': r'\b(N-glycan|O-glycan|glycosaminoglycan|proteoglycan|glycolipid)\b',
            'modifications': r'\b(acetyl|sulfat|phosphat|methylat)\w*\b'
        }
        
    def load_pubmed_abstracts(self, 
                            query_terms: List[str] = None,
                            max_abstracts: int = 10000) -> List[Dict[str, Any]]:
        """
        Load PubMed abstracts related to glycomics.
        
        Args:
            query_terms: Search terms for glycomics literature
            max_abstracts: Maximum number of abstracts to retrieve
            
        Returns:
            List of abstract data dictionaries
        """
        if query_terms is None:
            query_terms = [
                "glycan", "glycosylation", "oligosaccharide", 
                "N-glycan", "O-glycan", "glycoprotein",
                "lectin", "carbohydrate", "mass spectrometry glycan"
            ]
            
        abstracts = []
        
        # In a real implementation, this would use Entrez API
        # For now, simulate with sample data
        sample_abstracts = [
            {
                "pmid": "12345678",
                "title": "Mass spectrometric analysis of N-linked glycans from human serum proteins",
                "abstract": "N-linked glycosylation is a crucial post-translational modification affecting protein function. We analyzed serum glycoproteins using LC-MS/MS to identify disease-associated glycan alterations. Complex-type N-glycans with terminal sialic acid residues were predominantly observed.",
                "authors": ["Smith, J.", "Johnson, A."],
                "journal": "Analytical Chemistry",
                "year": 2023,
                "keywords": ["glycomics", "mass spectrometry", "N-glycan", "serum"]
            },
            {
                "pmid": "87654321", 
                "title": "Structural characterization of O-glycans in cancer biomarker discovery",
                "abstract": "O-linked glycosylation patterns differ significantly between normal and cancerous tissues. We employed MALDI-TOF MS to profile mucin-type O-glycans, revealing truncated glycan structures in tumor samples with elevated Tn and STn antigen expression.",
                "authors": ["Brown, K.", "Wilson, M."],
                "journal": "Journal of Proteome Research", 
                "year": 2023,
                "keywords": ["O-glycan", "cancer", "biomarker", "MALDI-TOF"]
            }
        ]
        
        abstracts.extend(sample_abstracts)
        logger.info(f"Loaded {len(abstracts)} PubMed abstracts")
        
        return abstracts[:max_abstracts]
        
    def load_database_annotations(self) -> List[Dict[str, Any]]:
        """
        Load glycan annotations from databases.
        
        Returns:
            List of annotation data
        """
        annotations = []
        
        # Sample database annotations
        sample_annotations = [
            {
                "glytoucan_id": "G00001MO",
                "annotation": "Tri-antennary complex-type N-glycan with terminal sialic acid residues. Commonly found on immunoglobulins and acute-phase proteins.",
                "source": "GlyTouCan",
                "biological_function": "Protein stability, immune recognition",
                "tissue_specificity": "Serum, liver"
            },
            {
                "glytoucan_id": "G00002MO", 
                "annotation": "Core GlcNAc-Gal disaccharide unit. Basic building block of complex N-glycan structures.",
                "source": "GlyTouCan",
                "biological_function": "Glycan extension, protein folding",
                "tissue_specificity": "Ubiquitous"
            }
        ]
        
        annotations.extend(sample_annotations)
        logger.info(f"Loaded {len(annotations)} database annotations")
        
        return annotations
        
    def load_experimental_protocols(self) -> List[Dict[str, Any]]:
        """
        Load experimental protocols and methods.
        
        Returns:
            List of protocol descriptions
        """
        protocols = []
        
        # Sample protocols
        sample_protocols = [
            {
                "protocol_id": "PROT001",
                "title": "LC-MS/MS analysis of N-glycans",
                "description": "Proteins are denatured and N-glycans are released using PNGase F digestion. Released glycans are permethylated and analyzed by MALDI-TOF MS in positive ion mode. Collision-induced dissociation (CID) is used for structural elucidation.",
                "steps": [
                    "Protein denaturation with DTT and iodoacetamide",
                    "PNGase F digestion to release N-glycans", 
                    "Solid-phase permethylation",
                    "MALDI-TOF MS analysis",
                    "CID fragmentation for structure confirmation"
                ],
                "applications": ["N-glycan profiling", "disease biomarker discovery"]
            }
        ]
        
        protocols.extend(sample_protocols)
        logger.info(f"Loaded {len(protocols)} experimental protocols")
        
        return protocols
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess and clean text data.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
            
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Normalize glycan terminology
        replacements = {
            'N-acetylglucosamine': 'GlcNAc',
            'N-acetylgalactosamine': 'GalNAc',
            'N-acetylneuraminic acid': 'Neu5Ac',
            'sialic acid': 'Neu5Ac'
        }
        
        for old, new in replacements.items():
            text = re.sub(old, new, text, flags=re.IGNORECASE)
            
        return text
        
    def extract_glycan_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract glycan-related entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted entities by category
        """
        entities = {}
        
        for category, pattern in self.glycan_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[category] = list(set(matches))  # Remove duplicates
            
        return entities


class SpectraDataLoader:
    """
    Loads and processes mass spectrometry data.
    
    Handles various MS formats, normalization, and peak processing
    for glycan structure elucidation.
    """
    
    def __init__(self, data_dir: str = "data/raw/spectra"):
        """
        Initialize spectra data loader.
        
        Args:
            data_dir: Directory containing spectra files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_maldi_spectra(self, file_paths: List[str] = None) -> List[Dict[str, Any]]:
        """
        Load MALDI-TOF MS spectra.
        
        Args:
            file_paths: List of spectrum file paths
            
        Returns:
            List of spectrum data
        """
        spectra = []
        
        # Sample MALDI spectra
        sample_spectra = [
            {
                "spectrum_id": "MALDI001",
                "glytoucan_id": "G00001MO",
                "instrument": "MALDI-TOF MS",
                "ionization_mode": "ESI+",
                "precursor_mz": 911.334,
                "peaks": [
                    [163.060, 1000], [204.087, 2500], [274.093, 800],
                    [366.140, 5000], [528.195, 3000], [690.251, 1500],
                    [852.307, 2000], [911.334, 10000]  # [M+Na]+
                ],
                "metadata": {
                    "acquisition_date": "2023-10-15",
                    "sample_type": "serum",
                    "preparation": "permethylated"
                }
            },
            {
                "spectrum_id": "MALDI002", 
                "glytoucan_id": "G00002MO",
                "instrument": "MALDI-TOF MS",
                "ionization_mode": "ESI+",
                "precursor_mz": 366.139,
                "peaks": [
                    [126.055, 800], [163.060, 1200], [204.087, 1500],
                    [274.093, 600], [366.139, 8000]  # [M+Na]+
                ],
                "metadata": {
                    "acquisition_date": "2023-10-15",
                    "sample_type": "tissue",
                    "preparation": "permethylated"
                }
            }
        ]
        
        spectra.extend(sample_spectra)
        logger.info(f"Loaded {len(spectra)} MALDI spectra")
        
        return spectra
        
    def load_esi_spectra(self, file_paths: List[str] = None) -> List[Dict[str, Any]]:
        """
        Load ESI-MS/MS spectra.
        
        Args:
            file_paths: List of spectrum file paths
            
        Returns:
            List of spectrum data
        """
        spectra = []
        
        # Sample ESI spectra with CID fragmentation
        sample_spectra = [
            {
                "spectrum_id": "ESI001",
                "glytoucan_id": "G00001MO", 
                "instrument": "ESI-Q-TOF MS",
                "ionization_mode": "ESI-",
                "precursor_mz": 887.308,  # [M-H]-
                "charge_state": -1,
                "collision_energy": 20.0,
                "fragmentation": "CID",
                "peaks": [
                    [145.050, 500], [161.045, 800], [179.056, 600],
                    [221.066, 400], [290.087, 1200], [383.114, 2000],
                    [545.170, 1500], [707.225, 800], [887.308, 3000]
                ],
                "metadata": {
                    "acquisition_date": "2023-10-16", 
                    "sample_type": "cell culture",
                    "preparation": "native"
                }
            }
        ]
        
        spectra.extend(sample_spectra)
        logger.info(f"Loaded {len(spectra)} ESI spectra")
        
        return spectra
        
    def normalize_spectrum(self, 
                         peaks: List[Tuple[float, float]],
                         method: str = "tic") -> List[Tuple[float, float]]:
        """
        Normalize spectrum intensities.
        
        Args:
            peaks: List of (m/z, intensity) tuples
            method: Normalization method ('tic', 'base_peak', 'median')
            
        Returns:
            Normalized peak list
        """
        if not peaks:
            return []
            
        intensities = [peak[1] for peak in peaks]
        
        if method == "tic":
            # Total ion current normalization
            total = sum(intensities)
            factor = 100.0 / total if total > 0 else 1.0
        elif method == "base_peak":
            # Base peak normalization
            max_intensity = max(intensities) if intensities else 1.0
            factor = 100.0 / max_intensity
        elif method == "median":
            # Median normalization  
            median_intensity = np.median(intensities) if intensities else 1.0
            factor = 100.0 / median_intensity
        else:
            factor = 1.0
            
        normalized_peaks = [(mz, intensity * factor) for mz, intensity in peaks]
        return normalized_peaks
        
    def bin_spectrum(self,
                   peaks: List[Tuple[float, float]], 
                   bin_width: float = 0.1,
                   mz_range: Tuple[float, float] = (50, 2000)) -> np.ndarray:
        """
        Bin spectrum into fixed m/z intervals.
        
        Args:
            peaks: List of (m/z, intensity) tuples
            bin_width: Width of m/z bins
            mz_range: (min_mz, max_mz) range
            
        Returns:
            Binned spectrum as numpy array
        """
        min_mz, max_mz = mz_range
        n_bins = int((max_mz - min_mz) / bin_width)
        
        binned_spectrum = np.zeros(n_bins)
        
        for mz, intensity in peaks:
            if min_mz <= mz <= max_mz:
                bin_idx = int((mz - min_mz) / bin_width)
                if bin_idx < n_bins:
                    binned_spectrum[bin_idx] += intensity
                    
        return binned_spectrum


class StructuralDataLoader:
    """
    Loads and processes glycan structural data.
    
    Handles various structure representations (WURCS, GlycoCT, IUPAC)
    and converts them to graph representations for ML models.
    """
    
    def __init__(self, data_dir: str = "data/raw/structures"):
        """
        Initialize structural data loader.
        
        Args:
            data_dir: Directory containing structure files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Common monosaccharide codes
        self.monosaccharide_codes = {
            'Glc': 'Glucose',
            'Gal': 'Galactose', 
            'Man': 'Mannose',
            'Fuc': 'Fucose',
            'Xyl': 'Xylose',
            'GlcNAc': 'N-acetylglucosamine',
            'GalNAc': 'N-acetylgalactosamine',
            'ManNAc': 'N-acetylmannosamine',
            'Neu5Ac': 'N-acetylneuraminic acid',
            'Neu5Gc': 'N-glycolylneuraminic acid'
        }
        
    def load_wurcs_structures(self) -> List[Dict[str, Any]]:
        """
        Load glycan structures in WURCS format.
        
        Returns:
            List of structure data
        """
        structures = []
        
        # Sample WURCS structures
        sample_structures = [
            {
                "glytoucan_id": "G00001MO",
                "wurcs": "WURCS=2.0/3,3,2/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5]/1-2-3/a4-b1_b4-c1",
                "mass_mono": 910.327,
                "composition": {"Hex": 3, "HexNAc": 2},
                "description": "Tri-antennary N-glycan core"
            },
            {
                "glytoucan_id": "G00002MO",
                "wurcs": "WURCS=2.0/2,2,1/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1a_1-5]/1-2/a4-b1", 
                "mass_mono": 365.132,
                "composition": {"Hex": 1, "HexNAc": 1},
                "description": "Core GlcNAc-Gal disaccharide"
            }
        ]
        
        structures.extend(sample_structures)
        logger.info(f"Loaded {len(structures)} WURCS structures")
        
        return structures
        
    def parse_wurcs_to_graph(self, wurcs: str) -> Dict[str, Any]:
        """
        Parse WURCS notation into graph representation.
        
        Args:
            wurcs: WURCS sequence string
            
        Returns:
            Graph representation with nodes and edges
        """
        # Simplified WURCS parsing (full implementation would be more complex)
        graph = {
            "nodes": [],
            "edges": [],
            "features": {}
        }
        
        try:
            # Extract basic components from WURCS
            if wurcs.startswith("WURCS="):
                # Parse version and basic structure info
                parts = wurcs.split("/")
                if len(parts) >= 4:
                    version = parts[0].split("=")[1]
                    counts = parts[1].split(",")  # [unique_count, residue_count, linkage_count]
                    residues = parts[2] if len(parts) > 2 else ""
                    connections = parts[3] if len(parts) > 3 else ""
                    
                    # Create simplified nodes (one per residue)
                    residue_count = int(counts[1]) if len(counts) > 1 else 1
                    
                    for i in range(residue_count):
                        node = {
                            "id": i,
                            "type": "monosaccharide",
                            "position": i,
                            "features": {
                                "residue_index": i,
                                "is_terminal": i == residue_count - 1
                            }
                        }
                        graph["nodes"].append(node)
                        
                    # Create edges based on connections
                    # This is a simplified version
                    for i in range(residue_count - 1):
                        edge = {
                            "source": i,
                            "target": i + 1,
                            "type": "glycosidic_bond",
                            "features": {
                                "linkage": "1-4",  # Default assumption
                                "anomeric": "beta"  # Default assumption
                            }
                        }
                        graph["edges"].append(edge)
                        
        except Exception as e:
            logger.warning(f"Error parsing WURCS {wurcs}: {e}")
            
        return graph
        
    def calculate_structure_features(self, graph: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate structural features from graph representation.
        
        Args:
            graph: Graph representation of glycan
            
        Returns:
            Dictionary of structural features
        """
        features = {}
        
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        # Basic topological features
        features["num_residues"] = len(nodes)
        features["num_bonds"] = len(edges)
        features["branching_factor"] = len(edges) - len(nodes) + 1 if nodes else 0
        
        # Degree distribution
        degree_count = defaultdict(int)
        for edge in edges:
            degree_count[edge["source"]] += 1
            degree_count[edge["target"]] += 1
            
        degrees = list(degree_count.values())
        features["max_degree"] = max(degrees) if degrees else 0
        features["avg_degree"] = np.mean(degrees) if degrees else 0
        
        # Terminal residues (degree 1)
        features["terminal_residues"] = sum(1 for d in degrees if d == 1)
        
        return features


class MultimodalDatasetBuilder:
    """
    Builds integrated multimodal datasets for GlycoLLM training.
    
    Combines text, spectra, and structural data into unified training samples
    with appropriate labels for various tasks.
    """
    
    def __init__(self, 
                 output_dir: str = "data/processed/multimodal",
                 max_samples: int = 100000):
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Directory for processed datasets
            max_samples: Maximum number of samples to create
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_samples = max_samples
        
        # Initialize loaders
        self.text_loader = TextCorpusLoader()
        self.spectra_loader = SpectraDataLoader()
        self.structure_loader = StructuralDataLoader()
        
        # Sample tracking
        self.samples = []
        self.sample_counter = 0
        
    def build_dataset(self, tasks: List[str] = None) -> Dict[str, Any]:
        """
        Build complete multimodal dataset.
        
        Args:
            tasks: List of tasks to prepare data for
            
        Returns:
            Dataset statistics and metadata
        """
        if tasks is None:
            tasks = [
                "structure_elucidation",
                "mass_prediction", 
                "text_classification",
                "cross_modal_retrieval"
            ]
            
        logger.info(f"Building multimodal dataset for tasks: {tasks}")
        
        # Load all modalities
        text_data = self._load_text_data()
        spectra_data = self._load_spectra_data()
        structure_data = self._load_structure_data()
        
        # Create aligned samples
        self._create_multimodal_samples(text_data, spectra_data, structure_data)
        
        # Generate task-specific labels
        for task in tasks:
            self._generate_task_labels(task)
            
        # Split datasets
        splits = self._create_dataset_splits()
        
        # Save processed datasets
        self._save_datasets(splits)
        
        # Generate statistics
        stats = self._generate_statistics()
        
        logger.info(f"Dataset built: {len(self.samples)} samples across {len(tasks)} tasks")
        
        return stats
        
    def _load_text_data(self) -> List[Dict[str, Any]]:
        """Load and consolidate text data"""
        text_data = []
        
        # Load PubMed abstracts
        abstracts = self.text_loader.load_pubmed_abstracts(max_abstracts=1000)
        for abstract in abstracts:
            text_data.append({
                "id": f"pubmed_{abstract['pmid']}",
                "text": f"{abstract['title']} {abstract['abstract']}",
                "type": "abstract",
                "metadata": abstract
            })
            
        # Load database annotations
        annotations = self.text_loader.load_database_annotations()
        for annotation in annotations:
            text_data.append({
                "id": f"annotation_{annotation['glytoucan_id']}",
                "text": annotation['annotation'],
                "type": "annotation", 
                "glytoucan_id": annotation['glytoucan_id'],
                "metadata": annotation
            })
            
        logger.info(f"Loaded {len(text_data)} text documents")
        return text_data
        
    def _load_spectra_data(self) -> List[Dict[str, Any]]:
        """Load and consolidate spectra data"""
        spectra_data = []
        
        # Load MALDI spectra
        maldi_spectra = self.spectra_loader.load_maldi_spectra()
        spectra_data.extend(maldi_spectra)
        
        # Load ESI spectra 
        esi_spectra = self.spectra_loader.load_esi_spectra()
        spectra_data.extend(esi_spectra)
        
        # Normalize spectra
        for spectrum in spectra_data:
            spectrum['normalized_peaks'] = self.spectra_loader.normalize_spectrum(
                spectrum['peaks'], method="tic"
            )
            
        logger.info(f"Loaded {len(spectra_data)} spectra")
        return spectra_data
        
    def _load_structure_data(self) -> List[Dict[str, Any]]:
        """Load and consolidate structure data"""
        structure_data = []
        
        # Load WURCS structures
        wurcs_structures = self.structure_loader.load_wurcs_structures()
        
        for structure in wurcs_structures:
            # Parse to graph representation
            graph = self.structure_loader.parse_wurcs_to_graph(structure['wurcs'])
            structure['graph'] = graph
            
            # Calculate features
            features = self.structure_loader.calculate_structure_features(graph)
            structure['features'] = features
            
            structure_data.append(structure)
            
        logger.info(f"Loaded {len(structure_data)} structures")
        return structure_data
        
    def _create_multimodal_samples(self, 
                                 text_data: List[Dict], 
                                 spectra_data: List[Dict],
                                 structure_data: List[Dict]]):
        """Create aligned multimodal samples"""
        
        # Index data by identifiers
        structures_by_id = {s['glytoucan_id']: s for s in structure_data}
        spectra_by_glycan = defaultdict(list)
        text_by_glycan = defaultdict(list)
        
        for spectrum in spectra_data:
            if spectrum.get('glytoucan_id'):
                spectra_by_glycan[spectrum['glytoucan_id']].append(spectrum)
                
        for text in text_data:
            if text.get('glytoucan_id'):
                text_by_glycan[text['glytoucan_id']].append(text)
                
        # Create multimodal samples
        for glytoucan_id, structure in structures_by_id.items():
            # Get associated spectra and text
            related_spectra = spectra_by_glycan.get(glytoucan_id, [])
            related_text = text_by_glycan.get(glytoucan_id, [])
            
            # Create samples for each spectrum
            for spectrum in related_spectra:
                sample = MultimodalSample(
                    sample_id=f"sample_{self.sample_counter}",
                    glytoucan_id=glytoucan_id,
                    spectrum_id=spectrum['spectrum_id'],
                    
                    # Structure data
                    wurcs_sequence=structure['wurcs'],
                    structure_graph=structure['graph'],
                    
                    # Spectra data
                    spectra_peaks=spectrum['normalized_peaks'],
                    precursor_mz=spectrum.get('precursor_mz'),
                    charge_state=spectrum.get('charge_state'),
                    collision_energy=spectrum.get('collision_energy'),
                    
                    # Text data (combine available text)
                    text=' '.join([t['text'] for t in related_text]) if related_text else None,
                    text_type='combined' if related_text else None,
                    
                    # Metadata
                    experimental_method=spectrum.get('instrument')
                )
                
                self.samples.append(sample)
                self.sample_counter += 1
                
        logger.info(f"Created {len(self.samples)} multimodal samples")
        
    def _generate_task_labels(self, task: str):
        """Generate labels for specific tasks"""
        
        if task == "structure_elucidation":
            # Label: predict WURCS from spectrum
            for sample in self.samples:
                if sample.wurcs_sequence and sample.spectra_peaks:
                    if sample.labels is None:
                        sample.labels = {}
                    sample.labels["target_wurcs"] = sample.wurcs_sequence
                    
        elif task == "mass_prediction":
            # Label: predict mass from structure
            for sample in self.samples:
                if sample.structure_graph and sample.precursor_mz:
                    if sample.labels is None:
                        sample.labels = {}
                    sample.labels["target_mass"] = sample.precursor_mz
                    
        elif task == "text_classification":
            # Label: classify text type/relevance
            for sample in self.samples:
                if sample.text:
                    if sample.labels is None:
                        sample.labels = {}
                    sample.labels["text_type"] = sample.text_type or "unknown"
                    
        logger.info(f"Generated labels for task: {task}")
        
    def _create_dataset_splits(self) -> Dict[str, List[MultimodalSample]]:
        """Create train/validation/test splits"""
        
        np.random.shuffle(self.samples)
        
        n_samples = len(self.samples)
        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)
        
        splits = {
            "train": self.samples[:n_train],
            "validation": self.samples[n_train:n_train + n_val],
            "test": self.samples[n_train + n_val:]
        }
        
        logger.info(f"Dataset splits: train={len(splits['train'])}, "
                   f"val={len(splits['validation'])}, test={len(splits['test'])}")
        
        return splits
        
    def _save_datasets(self, splits: Dict[str, List[MultimodalSample]]):
        """Save processed datasets to files"""
        
        for split_name, samples in splits.items():
            # Save as pickle for efficient loading
            pkl_path = self.output_dir / f"{split_name}_dataset.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump([asdict(sample) for sample in samples], f)
                
            # Save as JSON for inspection
            json_path = self.output_dir / f"{split_name}_dataset.json"
            with open(json_path, 'w') as f:
                json.dump([asdict(sample) for sample in samples[:100]], f, indent=2, default=str)
                
            logger.info(f"Saved {split_name} dataset: {len(samples)} samples")
            
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate dataset statistics"""
        
        stats = {
            "total_samples": len(self.samples),
            "modality_coverage": {
                "text_only": sum(1 for s in self.samples if s.text and not s.spectra_peaks and not s.wurcs_sequence),
                "spectra_only": sum(1 for s in self.samples if s.spectra_peaks and not s.text and not s.wurcs_sequence), 
                "structure_only": sum(1 for s in self.samples if s.wurcs_sequence and not s.text and not s.spectra_peaks),
                "multimodal": sum(1 for s in self.samples if sum([bool(s.text), bool(s.spectra_peaks), bool(s.wurcs_sequence)]) >= 2)
            },
            "unique_glycans": len(set(s.glytoucan_id for s in self.samples if s.glytoucan_id)),
            "unique_spectra": len(set(s.spectrum_id for s in self.samples if s.spectrum_id)),
            "creation_date": datetime.now().isoformat()
        }
        
        # Save statistics
        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        return stats