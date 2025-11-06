#!/usr/bin/env python3
"""
Production-Grade GlycoWorks Data Processing Pipeline
====================================================

Converts GlycoWorks experimental CSV files (44 files, 46K+ measurements)
into multimodal training samples for GlycoLLM.

Features:
- Robust CSV loading with error handling
- Sample metadata parsing from column names
- Multimodal sample creation with experimental context
- Production-ready logging and validation
- Memory-efficient processing for large datasets

Author: Glycoinformatics AI Team
Date: November 5, 2025
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
from collections import defaultdict, Counter
import hashlib
import pickle

# GlycoLLM imports
from glycollm.data.multimodal_dataset import MultimodalSample
# Tokenizer imports are conditional to avoid torch dependencies when not needed
# from glycollm.tokenization.glyco_tokenizer import GlycoTokenizer
# from glycollm.models.glyco_tokenizer import GlycoTokenizer as GlycoTokenizerModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('glycoworks_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentalSample:
    """Represents a single experimental sample with metadata"""
    sample_id: str
    tissue_type: str
    condition: str
    replicate: int
    species: str
    glycan_type: str  # N-glycan, O-glycan, GSL, etc.
    pmid: str
    raw_name: str
    abundances: Dict[str, float]  # glycan -> abundance


@dataclass
class GlycoWorksDataset:
    """Complete GlycoWorks dataset with all experimental data"""
    samples: List[ExperimentalSample]
    glycans: List[str]
    metadata: Dict[str, Any]
    statistics: Dict[str, Any]


class GlycoWorksProcessor:
    """
    Production-grade processor for GlycoWorks experimental data.

    Converts 44 CSV files containing 46K+ experimental measurements
    into multimodal training samples compatible with GlycoLLM.
    """

    def __init__(self,
                 data_dir: str = "dataset/glycoworks_glycan_data",
                 output_dir: str = "data/processed/glycoworks",
                 tokenizer_path: Optional[str] = None):
        """
        Initialize GlycoWorks processor.

        Args:
            data_dir: Directory containing GlycoWorks CSV files
            output_dir: Directory for processed output
            tokenizer_path: Path to pre-trained GlycoTokenizer
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer (conditional import to avoid torch dependencies)
        self.tokenizer = None
        if tokenizer_path:
            try:
                # Import tokenizer only when needed
                from glycollm.tokenization.glyco_tokenizer import GlycoTokenizer
                self.tokenizer = GlycoTokenizer.from_pretrained(tokenizer_path)
                logger.info(f"Loaded GlycoTokenizer from {tokenizer_path}")
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")
                logger.info("Continuing without tokenizer - some features may be limited")

        # Processing state
        self.dataset = None
        self.multimodal_samples = []
        self.processing_stats = defaultdict(int)

        # Sample name parsing patterns
        self.sample_patterns = {
            # Pattern: tissue_condition_replicate_glycan_type
            'standard': re.compile(r'^([A-Z]+)_(\d+)_([A-Z]+)_(\d+)_([A-Z]+)$'),
            # Pattern: tissue_date_condition_replicate_glycan_type
            'detailed': re.compile(r'^([A-Z]+)_(\d+)_([A-Z]+)_(\d+)_([A-Z]+)$'),
            # Pattern: simple tissue_condition_replicate
            'simple': re.compile(r'^([A-Z]+)_([A-Z]+)_(\d+)$'),
        }

        # Tissue type mappings
        self.tissue_mappings = {
            'NG': 'N-glycan',
            'OG': 'O-glycan',
            'GSL': 'Glycosphingolipid',
            'GP': 'Glycoprotein',
            'GL': 'Glycolipid',
        }

        # Condition mappings (expand as needed)
        self.condition_mappings = {
            'Control': 'control',
            'CTRL': 'control',
            'WT': 'wild_type',
            'KO': 'knockout',
            'OE': 'overexpression',
            'SSSMuG': 'SSSMuG',  # Specific experimental condition
            'EC': 'E_coli_infection',
            'PA': 'P_aeruginosa_infection',
            'SA': 'S_aureus_infection',
            'HV': 'healthy_volunteer',
            'SV': 'severe_case',
            'SC': 'serum_control',
        }

    def process_all_csv_files(self) -> GlycoWorksDataset:
        """
        Process all GlycoWorks CSV files and create unified dataset.

        Returns:
            Complete GlycoWorks dataset
        """
        logger.info("🔬 Starting GlycoWorks data processing pipeline...")

        start_time = datetime.now()
        all_samples = []
        all_glycans = set()

        # Get all CSV files
        csv_files = list(self.data_dir.glob("glycomics_*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to process")

        for csv_file in csv_files:
            try:
                logger.info(f"Processing {csv_file.name}...")
                file_samples, file_glycans = self._process_single_csv(csv_file)
                all_samples.extend(file_samples)
                all_glycans.update(file_glycans)

                self.processing_stats['files_processed'] += 1
                self.processing_stats['samples_loaded'] += len(file_samples)

            except Exception as e:
                logger.error(f"Failed to process {csv_file.name}: {e}")
                self.processing_stats['files_failed'] += 1
                continue

        # Create unified dataset
        dataset = GlycoWorksDataset(
            samples=all_samples,
            glycans=list(all_glycans),
            metadata=self._extract_metadata(all_samples),
            statistics=self._calculate_statistics(all_samples)
        )

        processing_time = datetime.now() - start_time
        logger.info(f"✅ Processing complete in {processing_time}")
        logger.info(f"   📊 Loaded {len(all_samples)} samples, {len(all_glycans)} unique glycans")

        self.dataset = dataset
        return dataset

    def _process_single_csv(self, csv_path: Path) -> Tuple[List[ExperimentalSample], set]:
        """
        Process a single GlycoWorks CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            Tuple of (samples, unique_glycans)
        """
        # Extract metadata from filename
        file_metadata = self._parse_filename_metadata(csv_path.name)

        # Load CSV with robust error handling
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin1')
        except Exception as e:
            raise ValueError(f"Could not read CSV {csv_path}: {e}")

        if df.empty:
            logger.warning(f"Empty CSV file: {csv_path}")
            return [], set()

        # Validate required columns
        if 'glycan' not in df.columns:
            raise ValueError(f"CSV {csv_path} missing 'glycan' column")

        # Get sample columns (all columns except 'glycan')
        sample_columns = [col for col in df.columns if col != 'glycan']

        if not sample_columns:
            logger.warning(f"No sample columns found in {csv_path}")
            return [], set()

        samples = []
        unique_glycans = set()

        # Process each glycan row
        for idx, row in df.iterrows():
            glycan = str(row['glycan']).strip()
            if not glycan or glycan.lower() == 'nan':
                continue

            unique_glycans.add(glycan)

            # Process each sample column
            for sample_col in sample_columns:
                abundance = row[sample_col]

                # Handle missing/zero values
                if pd.isna(abundance) or abundance == 0:
                    continue

                try:
                    abundance = float(abundance)
                    if abundance <= 0:
                        continue
                except (ValueError, TypeError):
                    continue

                # Parse sample metadata
                sample_metadata = self._parse_sample_metadata(
                    sample_col, file_metadata
                )

                # Create experimental sample
                sample = ExperimentalSample(
                    sample_id=f"{file_metadata['pmid']}_{sample_col}_{idx}",
                    tissue_type=sample_metadata['tissue_type'],
                    condition=sample_metadata['condition'],
                    replicate=sample_metadata['replicate'],
                    species=file_metadata['species'],
                    glycan_type=file_metadata['glycan_type'],
                    pmid=file_metadata['pmid'],
                    raw_name=sample_col,
                    abundances={glycan: abundance}
                )

                samples.append(sample)

        logger.info(f"   📊 Processed {len(samples)} measurements from {len(unique_glycans)} glycans")
        return samples, unique_glycans

    def _parse_filename_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Parse metadata from GlycoWorks filename.

        Args:
            filename: CSV filename

        Returns:
            Metadata dictionary
        """
        # Example: glycomics_human_brain_N_PMID38343116.csv
        # Pattern: glycomics_species_tissue_glycan_type_PMIDxxxxx.csv

        metadata = {
            'species': 'unknown',
            'tissue': 'unknown',
            'glycan_type': 'unknown',
            'pmid': 'unknown'
        }

        # Extract PMID
        pmid_match = re.search(r'PMID(\d+)', filename)
        if pmid_match:
            metadata['pmid'] = pmid_match.group(1)

        # Parse components
        parts = filename.replace('.csv', '').split('_')

        if len(parts) >= 4:
            if 'human' in parts:
                metadata['species'] = 'human'
            elif 'mouse' in parts:
                metadata['species'] = 'mouse'
            elif 'macaque' in parts:
                metadata['species'] = 'macaque'

            # Tissue type
            tissue_keywords = ['brain', 'liver', 'serum', 'plasma', 'skin', 'prostate',
                             'ovarian', 'leukemia', 'colorectal', 'gastric', 'retina',
                             'platelets', 'leukemia', 'keratinocytes', 'lipoproteins', 'milk']

            for tissue in tissue_keywords:
                if tissue in ' '.join(parts):
                    metadata['tissue'] = tissue
                    break

            # Glycan type
            if 'N' in parts:
                metadata['glycan_type'] = 'N-glycan'
            elif 'O' in parts:
                metadata['glycan_type'] = 'O-glycan'
            elif 'GSL' in parts:
                metadata['glycan_type'] = 'glycosphingolipid'
            elif 'GP' in parts:
                metadata['glycan_type'] = 'glycoprotein'

        return metadata

    def _parse_sample_metadata(self, sample_name: str, file_metadata: Dict) -> Dict[str, Any]:
        """
        Parse detailed metadata from sample column name.

        Args:
            sample_name: Sample column name
            file_metadata: File-level metadata

        Returns:
            Sample metadata dictionary
        """
        metadata = {
            'tissue_type': file_metadata.get('tissue', 'unknown'),
            'condition': 'unknown',
            'replicate': 1
        }

        # Try different parsing patterns
        for pattern_name, pattern in self.sample_patterns.items():
            match = pattern.match(sample_name)
            if match:
                if pattern_name == 'standard':
                    # tissue_condition_replicate_glycan_type
                    tissue, condition, replicate, glycan_type = match.groups()
                    metadata.update({
                        'tissue_type': tissue,
                        'condition': self.condition_mappings.get(condition, condition.lower()),
                        'replicate': int(replicate),
                        'glycan_type': self.tissue_mappings.get(glycan_type, glycan_type)
                    })
                elif pattern_name == 'detailed':
                    # tissue_date_condition_replicate_glycan_type
                    tissue, date, condition, replicate, glycan_type = match.groups()
                    metadata.update({
                        'tissue_type': tissue,
                        'condition': self.condition_mappings.get(condition, condition.lower()),
                        'replicate': int(replicate),
                        'glycan_type': self.tissue_mappings.get(glycan_type, glycan_type),
                        'date': date
                    })
                elif pattern_name == 'simple':
                    # tissue_condition_replicate
                    tissue, condition, replicate = match.groups()
                    metadata.update({
                        'tissue_type': tissue,
                        'condition': self.condition_mappings.get(condition, condition.lower()),
                        'replicate': int(replicate)
                    })
                break

        # Fallback: extract meaningful parts
        if metadata['condition'] == 'unknown':
            # Look for known condition keywords
            for condition_key, condition_value in self.condition_mappings.items():
                if condition_key in sample_name:
                    metadata['condition'] = condition_value
                    break

        return metadata

    def _extract_metadata(self, samples: List[ExperimentalSample]) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from processed samples.

        Args:
            samples: List of experimental samples

        Returns:
            Metadata dictionary
        """
        # Get all unique glycans from sample abundances
        all_glycans = set()
        for sample in samples:
            all_glycans.update(sample.abundances.keys())

        metadata = {
            'processing_date': datetime.now().isoformat(),
            'total_samples': len(samples),
            'unique_glycans': len(all_glycans),
            'species_distribution': Counter(s.species for s in samples),
            'tissue_distribution': Counter(s.tissue_type for s in samples),
            'condition_distribution': Counter(s.condition for s in samples),
            'glycan_type_distribution': Counter(s.glycan_type for s in samples),
            'pmid_distribution': Counter(s.pmid for s in samples),
        }

        return metadata

    def _calculate_statistics(self, samples: List[ExperimentalSample]) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for the dataset.

        Args:
            samples: List of experimental samples

        Returns:
            Statistics dictionary
        """
        if not samples:
            return {}

        # Abundance statistics
        all_abundances = []
        for sample in samples:
            all_abundances.extend(sample.abundances.values())

        abundances = np.array(all_abundances)

        stats = {
            'total_measurements': len(all_abundances),
            'abundance_stats': {
                'mean': float(np.mean(abundances)),
                'median': float(np.median(abundances)),
                'std': float(np.std(abundances)),
                'min': float(np.min(abundances)),
                'max': float(np.max(abundances)),
                'q25': float(np.percentile(abundances, 25)),
                'q75': float(np.percentile(abundances, 75)),
            },
            'data_completeness': {
                'samples_with_data': len([s for s in samples if s.abundances]),
                'avg_glycans_per_sample': np.mean([len(s.abundances) for s in samples]),
                'max_glycans_per_sample': max(len(s.abundances) for s in samples),
                'min_glycans_per_sample': min(len(s.abundances) for s in samples),
            }
        }

        return stats

    def create_multimodal_samples(self,
                                task_types: List[str] = None) -> List[MultimodalSample]:
        """
        Convert experimental data to multimodal training samples.

        Args:
            task_types: Types of tasks to create samples for

        Returns:
            List of multimodal samples
        """
        if not self.dataset:
            raise ValueError("Must process CSV files first")

        if task_types is None:
            task_types = [
                'abundance_prediction',
                'tissue_classification',
                'biomarker_discovery',
                'quantitative_structure_elucidation'
            ]

        logger.info(f"Creating multimodal samples for tasks: {task_types}")

        multimodal_samples = []

        # Group samples by experimental context
        context_groups = self._group_samples_by_context()

        for context_key, context_samples in context_groups.items():
            # Create samples for each task type
            for task_type in task_types:
                task_samples = self._create_task_samples(
                    context_samples, task_type
                )
                multimodal_samples.extend(task_samples)

        self.multimodal_samples = multimodal_samples

        logger.info(f"Created {len(multimodal_samples)} multimodal training samples")
        return multimodal_samples

    def _group_samples_by_context(self) -> Dict[str, List[ExperimentalSample]]:
        """
        Group samples by experimental context (tissue, condition, species).

        Returns:
            Dictionary of context groups
        """
        context_groups = defaultdict(list)

        for sample in self.dataset.samples:
            # Create context key
            context_key = f"{sample.species}_{sample.tissue_type}_{sample.condition}_{sample.pmid}"
            context_groups[context_key].append(sample)

        logger.info(f"Grouped {len(self.dataset.samples)} samples into {len(context_groups)} contexts")
        return context_groups

    def _create_task_samples(self,
                           context_samples: List[ExperimentalSample],
                           task_type: str) -> List[MultimodalSample]:
        """
        Create multimodal samples for a specific task type.

        Args:
            context_samples: Samples from same experimental context
            task_type: Type of task to create samples for

        Returns:
            List of multimodal samples
        """
        samples = []

        if task_type == 'abundance_prediction':
            samples.extend(self._create_abundance_prediction_samples(context_samples))
        elif task_type == 'tissue_classification':
            samples.extend(self._create_tissue_classification_samples(context_samples))
        elif task_type == 'biomarker_discovery':
            samples.extend(self._create_biomarker_samples(context_samples))
        elif task_type == 'quantitative_structure_elucidation':
            samples.extend(self._create_quantitative_structure_samples(context_samples))

        return samples

    def _create_abundance_prediction_samples(self,
                                          context_samples: List[ExperimentalSample]) -> List[MultimodalSample]:
        """
        Create samples for abundance prediction task.

        Predicts glycan abundances from experimental context.
        """
        samples = []

        # Get all unique glycans in this context
        all_glycans = set()
        for sample in context_samples:
            all_glycans.update(sample.abundances.keys())

        for glycan in all_glycans:
            # Get abundances across replicates/conditions
            abundances = []
            contexts = []

            for sample in context_samples:
                if glycan in sample.abundances:
                    abundances.append(sample.abundances[glycan])
                    contexts.append({
                        'tissue': sample.tissue_type,
                        'condition': sample.condition,
                        'replicate': sample.replicate,
                        'species': sample.species
                    })

            if len(abundances) >= 2:  # Need at least 2 measurements
                # Create multimodal sample
                sample_id = f"abundance_{hashlib.md5(glycan.encode()).hexdigest()[:8]}"

                multimodal_sample = MultimodalSample(
                    sample_id=sample_id,
                    glytoucan_id=None,  # Experimental data, no GlyTouCan IDs
                    spectrum_id=None,   # No spectra in GlycoWorks
                    wurcs_sequence=None,  # Will be predicted
                    structure_graph=None,
                    spectra_peaks=None,
                    precursor_mz=None,
                    charge_state=None,
                    collision_energy=None,
                    text=f"Predict abundance of glycan {glycan} in {contexts[0]['tissue']} under {contexts[0]['condition']} conditions",
                    text_type='experimental_context',
                    tissue=contexts[0]['tissue'],
                    experimental_method='quantitative_glycomics',
                    confidence_score=np.mean(abundances) / (np.mean(abundances) + np.std(abundances)) if np.mean(abundances) > 0 else 0
                )

                # Set task-specific labels
                multimodal_sample.labels = {
                    'task_type': 'abundance_prediction',
                    'target_abundance': np.mean(abundances),
                    'abundance_std': np.std(abundances),
                    'glycan_sequence': glycan
                }

                samples.append(multimodal_sample)

        return samples

    def _create_tissue_classification_samples(self,
                                            context_samples: List[ExperimentalSample]) -> List[MultimodalSample]:
        """
        Create samples for tissue classification task.

        Classifies tissue type from glycan abundance profiles.
        """
        samples = []

        # Group by replicate to create profile samples
        replicate_groups = defaultdict(list)
        for sample in context_samples:
            key = f"{sample.condition}_{sample.replicate}_{sample.pmid}"
            replicate_groups[key].append(sample)

        for replicate_key, replicate_samples in replicate_groups.items():
            if len(replicate_samples) < 3:  # Need minimum glycan coverage
                continue

            # Create glycan abundance profile
            profile = {}
            for sample in replicate_samples:
                profile.update(sample.abundances)

            if len(profile) >= 5:  # Minimum glycans for classification
                sample_id = f"tissue_{hashlib.md5(replicate_key.encode()).hexdigest()[:8]}"

                # Create text description of profile
                top_glycans = sorted(profile.items(), key=lambda x: x[1], reverse=True)[:10]
                profile_text = f"Glycan abundance profile: {', '.join([f'{g}:{a:.3f}' for g, a in top_glycans])}"

                multimodal_sample = MultimodalSample(
                    sample_id=sample_id,
                    glytoucan_id=None,
                    spectrum_id=None,
                    wurcs_sequence=None,
                    structure_graph=None,
                    spectra_peaks=None,
                    precursor_mz=None,
                    charge_state=None,
                    collision_energy=None,
                    text=profile_text,
                    text_type='glycan_profile',
                    tissue=replicate_samples[0].tissue_type,
                    experimental_method='glycan_profiling',
                    confidence_score=len(profile) / 20.0  # Normalize by expected profile size
                )

                # Set classification labels
                multimodal_sample.labels = {
                    'task_type': 'tissue_classification',
                    'target_tissue': replicate_samples[0].tissue_type,
                    'target_condition': replicate_samples[0].condition,
                    'glycan_profile': profile
                }

                samples.append(multimodal_sample)

        return samples

    def _create_biomarker_samples(self,
                                context_samples: List[ExperimentalSample]) -> List[MultimodalSample]:
        """
        Create samples for biomarker discovery task.

        Identifies discriminatory glycans between conditions.
        """
        samples = []

        # Group by condition
        condition_groups = defaultdict(list)
        for sample in context_samples:
            condition_groups[sample.condition].append(sample)

        if len(condition_groups) < 2:
            return samples  # Need at least 2 conditions

        # Find differentially abundant glycans
        diff_glycans = self._find_differential_glycans(condition_groups)

        for glycan, stats in diff_glycans.items():
            sample_id = f"biomarker_{hashlib.md5(glycan.encode()).hexdigest()[:8]}"

            # Create context description
            context_text = f"Evaluate glycan {glycan} as potential biomarker between conditions: {list(condition_groups.keys())}"

            multimodal_sample = MultimodalSample(
                sample_id=sample_id,
                glytoucan_id=None,
                spectrum_id=None,
                wurcs_sequence=None,
                structure_graph=None,
                spectra_peaks=None,
                precursor_mz=None,
                charge_state=None,
                collision_energy=None,
                text=context_text,
                text_type='biomarker_analysis',
                tissue=context_samples[0].tissue_type,
                experimental_method='biomarker_discovery',
                confidence_score=stats.get('significant', False)
            )

            # Set biomarker labels
            multimodal_sample.labels = {
                'task_type': 'biomarker_discovery',
                'glycan': glycan,
                'fold_change': stats.get('fold_change', 1.0),
                'p_value': stats.get('p_value', 1.0),
                'significant': stats.get('significant', False)
            }

            samples.append(multimodal_sample)

        return samples

    def _create_quantitative_structure_samples(self,
                                            context_samples: List[ExperimentalSample]) -> List[MultimodalSample]:
        """
        Create samples for quantitative structure elucidation.

        Uses experimental abundances to guide structure prediction.
        """
        samples = []

        # Get glycans with quantitative data
        glycan_abundances = defaultdict(list)
        for sample in context_samples:
            for glycan, abundance in sample.abundances.items():
                glycan_abundances[glycan].append(abundance)

        for glycan, abundances in glycan_abundances.items():
            if len(abundances) >= 3:  # Need sufficient measurements
                sample_id = f"quantitative_{hashlib.md5(glycan.encode()).hexdigest()[:8]}"

                # Create quantitative context
                quant_text = f"Elucidate structure of glycan {glycan} with experimental abundances: mean={np.mean(abundances):.3f}, std={np.std(abundances):.3f}"

                multimodal_sample = MultimodalSample(
                    sample_id=sample_id,
                    glytoucan_id=None,
                    spectrum_id=None,
                    wurcs_sequence=None,  # To be predicted
                    structure_graph=None,
                    spectra_peaks=None,
                    precursor_mz=None,
                    charge_state=None,
                    collision_energy=None,
                    text=quant_text,
                    text_type='quantitative_elucidation',
                    tissue=context_samples[0].tissue_type,
                    experimental_method='quantitative_glycomics',
                    confidence_score=1.0 - (np.std(abundances) / np.mean(abundances)) if np.mean(abundances) > 0 else 0
                )

                # Set quantitative labels
                multimodal_sample.labels = {
                    'task_type': 'quantitative_structure_elucidation',
                    'glycan_sequence': glycan,
                    'experimental_abundances': abundances,
                    'target_structure': None  # To be annotated with known structures
                }

                samples.append(multimodal_sample)

        return samples

    def _find_differential_glycans(self, condition_groups: Dict[str, List[ExperimentalSample]]) -> Dict[str, Dict]:
        """
        Find differentially abundant glycans between conditions.

        Args:
            condition_groups: Samples grouped by condition

        Returns:
            Dictionary of glycan -> differential statistics
        """
        diff_glycans = {}

        # Get all glycans across conditions
        all_glycans = set()
        for samples in condition_groups.values():
            for sample in samples:
                all_glycans.update(sample.abundances.keys())

        for glycan in all_glycans:
            condition_abundances = {}

            for condition, samples in condition_groups.items():
                abundances = []
                for sample in samples:
                    if glycan in sample.abundances:
                        abundances.append(sample.abundances[glycan])

                if abundances:
                    condition_abundances[condition] = abundances

            if len(condition_abundances) >= 2:
                # Simple fold change calculation (can be enhanced with statistical tests)
                conditions = list(condition_abundances.keys())
                mean1 = np.mean(condition_abundances[conditions[0]])
                mean2 = np.mean(condition_abundances[conditions[1]])

                if mean1 > 0 and mean2 > 0:
                    fold_change = mean2 / mean1 if mean1 > mean2 else mean1 / mean2
                    direction = "up" if mean2 > mean1 else "down"

                    # Simple significance check (can be enhanced)
                    std1 = np.std(condition_abundances[conditions[0]])
                    std2 = np.std(condition_abundances[conditions[1]])
                    significant = fold_change > 2.0  # Arbitrary threshold

                    diff_glycans[glycan] = {
                        'fold_change': fold_change,
                        'direction': direction,
                        'condition1_mean': mean1,
                        'condition2_mean': mean2,
                        'condition1_std': std1,
                        'condition2_std': std2,
                        'significant': significant,
                        'p_value': 0.05 if significant else 0.5  # Placeholder
                    }

        return diff_glycans

    def save_processed_data(self, output_format: str = 'pickle'):
        """
        Save processed GlycoWorks data and multimodal samples.

        Args:
            output_format: 'pickle', 'json', or 'both'
        """
        if not self.dataset or not self.multimodal_samples:
            raise ValueError("Must process data and create samples first")

        logger.info(f"Saving processed data in {output_format} format...")

        # Save dataset
        if output_format in ['pickle', 'both']:
            dataset_path = self.output_dir / 'glycoworks_dataset.pkl'
            with open(dataset_path, 'wb') as f:
                pickle.dump(self.dataset, f)
            logger.info(f"Saved dataset to {dataset_path}")

        if output_format in ['json', 'both']:
            dataset_json = self.output_dir / 'glycoworks_dataset.json'
            with open(dataset_json, 'w') as f:
                json.dump({
                    'samples': [asdict(s) for s in self.dataset.samples[:100]],  # Sample first 100
                    'metadata': self.dataset.metadata,
                    'statistics': self.dataset.statistics,
                    'total_samples': len(self.dataset.samples)
                }, f, indent=2, default=str)
            logger.info(f"Saved dataset summary to {dataset_json}")

        # Save multimodal samples
        if output_format in ['pickle', 'both']:
            samples_path = self.output_dir / 'multimodal_samples.pkl'
            with open(samples_path, 'wb') as f:
                pickle.dump(self.multimodal_samples, f)
            logger.info(f"Saved {len(self.multimodal_samples)} multimodal samples to {samples_path}")

        # Save statistics
        stats_path = self.output_dir / 'processing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump({
                'processing_stats': dict(self.processing_stats),
                'dataset_stats': self.dataset.statistics,
                'multimodal_stats': {
                    'total_samples': len(self.multimodal_samples),
                    'task_distribution': Counter(
                        s.labels.get('task_type', 'unknown') for s in self.multimodal_samples
                    )
                }
            }, f, indent=2)
        logger.info(f"Saved processing statistics to {stats_path}")

    def validate_processing(self) -> Dict[str, Any]:
        """
        Validate the processing pipeline and data quality.

        Returns:
            Validation results
        """
        validation_results = {
            'dataset_loaded': self.dataset is not None,
            'multimodal_samples_created': len(self.multimodal_samples) > 0,
            'data_quality_checks': {},
            'processing_integrity': {}
        }

        if self.dataset:
            samples = self.dataset.samples

            # Data quality checks
            validation_results['data_quality_checks'] = {
                'total_samples': len(samples),
                'samples_with_abundances': len([s for s in samples if s.abundances]),
                'unique_glycans': len(self.dataset.glycans),
                'species_coverage': len(set(s.species for s in samples)),
                'tissue_coverage': len(set(s.tissue_type for s in samples)),
                'condition_coverage': len(set(s.condition for s in samples)),
                'abundance_range_check': all(
                    all(a > 0 for a in s.abundances.values()) for s in samples if s.abundances
                )
            }

        if self.multimodal_samples:
            # Processing integrity checks
            validation_results['processing_integrity'] = {
                'total_multimodal_samples': len(self.multimodal_samples),
                'task_types': list(set(
                    s.labels.get('task_type', 'unknown') for s in self.multimodal_samples
                )),
                'samples_with_labels': len([s for s in self.multimodal_samples if s.labels]),
                'samples_with_text': len([s for s in self.multimodal_samples if s.text]),
                'unique_sample_ids': len(set(s.sample_id for s in self.multimodal_samples))
            }

        # Log validation results
        logger.info("🔍 Processing Validation Results:")
        for category, checks in validation_results.items():
            if isinstance(checks, dict):
                logger.info(f"   {category}:")
                for check, value in checks.items():
                    logger.info(f"     {check}: {value}")

        return validation_results


def main():
    """Main execution function for GlycoWorks processing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Process GlycoWorks experimental data')
    parser.add_argument('--data-dir', default='dataset/glycoworks_glycan_data',
                       help='Directory containing GlycoWorks CSV files')
    parser.add_argument('--output-dir', default='data/processed/glycoworks',
                       help='Output directory for processed data')
    parser.add_argument('--tokenizer-path', default=None,
                       help='Path to pre-trained GlycoTokenizer')
    parser.add_argument('--task-types', nargs='+',
                       default=['abundance_prediction', 'tissue_classification',
                               'biomarker_discovery', 'quantitative_structure_elucidation'],
                       help='Task types to create samples for')
    parser.add_argument('--output-format', choices=['pickle', 'json', 'both'],
                       default='both', help='Output format for saved data')

    args = parser.parse_args()

    # Initialize processor
    processor = GlycoWorksProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path
    )

    try:
        # Process all CSV files
        logger.info("🚀 Starting GlycoWorks processing pipeline...")
        dataset = processor.process_all_csv_files()

        # Create multimodal samples
        multimodal_samples = processor.create_multimodal_samples(args.task_types)

        # Save processed data
        processor.save_processed_data(args.output_format)

        # Validate processing
        validation = processor.validate_processing()

        logger.info("✅ GlycoWorks processing pipeline completed successfully!")
        logger.info(f"   📊 Processed {len(dataset.samples)} experimental samples")
        logger.info(f"   🎯 Created {len(multimodal_samples)} multimodal training samples")

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise


if __name__ == '__main__':
    main()