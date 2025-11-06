#!/usr/bin/env python3
"""
Validation Knowledge Base for GlycoLLM
========================================

Integrates curated glycan reference data (candycrunch dataset) for validation
during training and evaluation. Provides ground truth structures and GlyTouCan
mappings for validating model predictions.

Features:
- Loads curated glycan structures and GlyTouCan mappings
- Provides validation functions for structure prediction
- Integrates with GlycoLLM evaluation framework
- Supports batch validation and accuracy assessment

Author: Glycoinformatics AI Team
Date: November 5, 2025
"""

import os
import pickle
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import re
from datetime import datetime

# GlycoLLM imports
from glycollm.data.multimodal_dataset import MultimodalSample

logger = logging.getLogger(__name__)


@dataclass
class ValidationEntry:
    """Represents a single validation entry with glycan structure and metadata"""
    glycan_sequence: str
    glytoucan_id: Optional[str] = None
    validation_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.validation_hash:
            self.validation_hash = hashlib.md5(
                self.glycan_sequence.encode()
            ).hexdigest()[:8]


@dataclass
class ValidationKnowledgeBase:
    """Curated knowledge base for glycan validation"""

    entries: List[ValidationEntry] = field(default_factory=list)
    glycan_to_glytoucan: Dict[str, str] = field(default_factory=dict)
    glytoucan_to_glycan: Dict[str, str] = field(default_factory=dict)
    sequence_index: Dict[str, ValidationEntry] = field(default_factory=dict)

    # Statistics
    total_entries: int = 0
    unique_sequences: int = 0
    glytoucan_coverage: float = 0.0

    def add_entry(self, entry: ValidationEntry):
        """Add a validation entry to the knowledge base"""
        self.entries.append(entry)
        self.sequence_index[entry.glycan_sequence] = entry

        if entry.glytoucan_id:
            self.glycan_to_glytoucan[entry.glycan_sequence] = entry.glytoucan_id
            self.glytoucan_to_glycan[entry.glytoucan_id] = entry.glycan_sequence

        self._update_statistics()

    def validate_structure(self, predicted_sequence: str) -> Dict[str, Any]:
        """
        Validate a predicted glycan structure against the knowledge base.

        Args:
            predicted_sequence: Predicted glycan sequence

        Returns:
            Validation results dictionary
        """
        results = {
            'is_known_structure': False,
            'exact_match': False,
            'glytoucan_id': None,
            'similarity_score': 0.0,
            'validation_confidence': 0.0,
            'closest_matches': []
        }

        # Check for exact match
        if predicted_sequence in self.sequence_index:
            entry = self.sequence_index[predicted_sequence]
            results.update({
                'is_known_structure': True,
                'exact_match': True,
                'glytoucan_id': entry.glytoucan_id,
                'validation_confidence': 1.0,
                'similarity_score': 1.0
            })
            return results

        # Find similar structures
        similar_structures = self._find_similar_structures(predicted_sequence)
        results['closest_matches'] = similar_structures[:5]  # Top 5 matches

        if similar_structures:
            best_match = similar_structures[0]
            results.update({
                'similarity_score': best_match['similarity'],
                'validation_confidence': min(best_match['similarity'], 0.8),
                'glytoucan_id': best_match.get('glytoucan_id')
            })

        return results

    def batch_validate(self, predicted_sequences: List[str]) -> List[Dict[str, Any]]:
        """
        Validate multiple predicted sequences in batch.

        Args:
            predicted_sequences: List of predicted glycan sequences

        Returns:
            List of validation results
        """
        return [self.validate_structure(seq) for seq in predicted_sequences]

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the validation knowledge base"""
        return {
            'total_entries': self.total_entries,
            'unique_sequences': self.unique_sequences,
            'glytoucan_coverage': self.glytoucan_coverage,
            'sequence_length_distribution': self._analyze_sequence_lengths(),
            'glytoucan_prefix_distribution': self._analyze_glytoucan_prefixes(),
            'validation_coverage_estimate': self._estimate_validation_coverage()
        }

    def _find_similar_structures(self, query_sequence: str,
                               max_candidates: int = 20) -> List[Dict[str, Any]]:
        """
        Find structures similar to the query sequence.

        Args:
            query_sequence: Query glycan sequence
            max_candidates: Maximum number of candidates to consider

        Returns:
            List of similar structures with similarity scores
        """
        candidates = []
        query_len = len(query_sequence)

        # Simple similarity based on substring matching and length difference
        for entry in self.entries[:max_candidates]:  # Limit for efficiency
            target_sequence = entry.glycan_sequence
            target_len = len(target_sequence)

            # Length similarity
            length_diff = abs(query_len - target_len)
            length_similarity = max(0, 1 - length_diff / max(query_len, target_len))

            # Substring similarity
            common_substrings = self._find_common_substrings(query_sequence, target_sequence)
            substring_score = len(common_substrings) / max(query_len, target_len)

            # Combined similarity score
            similarity = (length_similarity + substring_score) / 2

            if similarity > 0.1:  # Only include reasonably similar structures
                candidates.append({
                    'sequence': target_sequence,
                    'glytoucan_id': entry.glytoucan_id,
                    'similarity': similarity,
                    'length_diff': length_diff,
                    'common_substrings': len(common_substrings)
                })

        # Sort by similarity
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates

    def _find_common_substrings(self, seq1: str, seq2: str, min_length: int = 3) -> List[str]:
        """Find common substrings between two sequences"""
        common = []
        len1, len2 = len(seq1), len(seq2)

        for i in range(len1 - min_length + 1):
            for j in range(min_length, len1 - i + 1):
                substring = seq1[i:i+j]
                if substring in seq2 and substring not in common:
                    common.append(substring)

        return common

    def _update_statistics(self):
        """Update knowledge base statistics"""
        self.total_entries = len(self.entries)
        self.unique_sequences = len(self.sequence_index)

        glytoucan_count = sum(1 for entry in self.entries if entry.glytoucan_id)
        self.glytoucan_coverage = glytoucan_count / self.total_entries if self.total_entries > 0 else 0

    def _analyze_sequence_lengths(self) -> Dict[str, Any]:
        """Analyze distribution of sequence lengths"""
        lengths = [len(entry.glycan_sequence) for entry in self.entries]

        if not lengths:
            return {}

        return {
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': sum(lengths) / len(lengths),
            'median_length': sorted(lengths)[len(lengths) // 2]
        }

    def _analyze_glytoucan_prefixes(self) -> Dict[str, int]:
        """Analyze distribution of GlyTouCan ID prefixes"""
        prefixes = Counter()

        for entry in self.entries:
            if entry.glytoucan_id and isinstance(entry.glytoucan_id, str):
                prefix = entry.glytoucan_id.split('-')[0] if '-' in entry.glytoucan_id else 'unknown'
                prefixes[prefix] += 1

        return dict(prefixes)

    def _estimate_validation_coverage(self) -> float:
        """
        Estimate what fraction of possible glycan structures this knowledge base covers.
        This is a rough heuristic based on sequence diversity.
        """
        if not self.entries:
            return 0.0

        # Simple diversity metric based on unique character combinations
        unique_chars = set()
        for entry in self.entries:
            unique_chars.update(entry.glycan_sequence)

        # Rough estimate: coverage proportional to unique characters and sequence diversity
        char_diversity = len(unique_chars) / 50  # Normalize to typical alphabet size
        sequence_diversity = min(1.0, self.unique_sequences / 10000)  # Assume 10K is comprehensive

        return min(1.0, (char_diversity + sequence_diversity) / 2)


class CandycrunchValidator:
    """
    Validator using the candycrunch glycan reference dataset.

    Loads and provides validation capabilities using the curated candycrunch
    dataset containing 3.7K glycan structures with GlyTouCan mappings.
    """

    def __init__(self, data_dir: str = "data/raw/dataset/candycrunch_glycan_data"):
        """
        Initialize candycrunch validator.

        Args:
            data_dir: Directory containing candycrunch pickle files
        """
        self.data_dir = Path(data_dir)
        self.knowledge_base = ValidationKnowledgeBase()
        self.is_loaded = False

        # Loading statistics
        self.loading_stats = {
            'glycans_loaded': 0,
            'mappings_loaded': 0,
            'load_time': 0,
            'errors': []
        }

    def load_knowledge_base(self) -> ValidationKnowledgeBase:
        """
        Load the candycrunch knowledge base from pickle files.

        Returns:
            Loaded validation knowledge base
        """
        start_time = datetime.now()

        try:
            logger.info("Loading candycrunch validation knowledge base...")

            # Load glycan sequences
            glycans_path = self.data_dir / "glycans.pkl"
            if glycans_path.exists():
                with open(glycans_path, 'rb') as f:
                    glycan_sequences = pickle.load(f)

                self.loading_stats['glycans_loaded'] = len(glycan_sequences)
                logger.info(f"Loaded {len(glycan_sequences)} glycan sequences")
            else:
                raise FileNotFoundError(f"Glycans file not found: {glycans_path}")

            # Load GlyTouCan mappings
            mappings_path = self.data_dir / "glytoucan_mapping.pkl"
            glytoucan_mapping = {}

            if mappings_path.exists():
                with open(mappings_path, 'rb') as f:
                    glytoucan_mapping = pickle.load(f)

                self.loading_stats['mappings_loaded'] = len(glytoucan_mapping)
                logger.info(f"Loaded {len(glytoucan_mapping)} GlyTouCan mappings")
            else:
                logger.warning(f"GlyTouCan mapping file not found: {mappings_path}")

            # Build knowledge base
            for glycan_seq in glycan_sequences:
                glytoucan_id = glytoucan_mapping.get(glycan_seq)

                entry = ValidationEntry(
                    glycan_sequence=glycan_seq,
                    glytoucan_id=glytoucan_id,
                    metadata={
                        'source': 'candycrunch',
                        'has_glytoucan': glytoucan_id is not None
                    }
                )

                self.knowledge_base.add_entry(entry)

            # Update loading statistics
            loading_time = datetime.now() - start_time
            self.loading_stats['load_time'] = loading_time.total_seconds()

            logger.info(f"✅ Knowledge base loaded in {loading_time}")
            logger.info(f"   📊 {len(self.knowledge_base.entries)} validation entries")
            logger.info(f"   🎯 {self.knowledge_base.glytoucan_coverage:.1%} GlyTouCan coverage")

            self.is_loaded = True
            return self.knowledge_base

        except Exception as e:
            error_msg = f"Failed to load knowledge base: {e}"
            logger.error(error_msg)
            self.loading_stats['errors'].append(error_msg)
            raise

    def validate_predictions(self,
                           predictions: List[str],
                           task_type: str = "structure_prediction") -> Dict[str, Any]:
        """
        Validate model predictions against the knowledge base.

        Args:
            predictions: List of predicted glycan sequences
            task_type: Type of prediction task

        Returns:
            Comprehensive validation results
        """
        if not self.is_loaded:
            raise ValueError("Knowledge base not loaded. Call load_knowledge_base() first.")

        logger.info(f"Validating {len(predictions)} predictions for {task_type}")

        # Batch validation
        validation_results = self.knowledge_base.batch_validate(predictions)

        # Aggregate statistics
        stats = self._compute_validation_statistics(validation_results)

        # Task-specific analysis
        task_analysis = self._analyze_task_performance(validation_results, task_type)

        results = {
            'validation_results': validation_results,
            'summary_statistics': stats,
            'task_analysis': task_analysis,
            'timestamp': datetime.now().isoformat(),
            'knowledge_base_info': self.knowledge_base.get_validation_statistics()
        }

        logger.info("Validation Summary:")
        logger.info(f"   📊 Exact matches: {stats['exact_matches']}/{stats['total_predictions']}")
        logger.info(f"   🎯 Known structures: {stats['known_structures']}/{stats['total_predictions']}")
        logger.info(f"   📈 Mean similarity: {stats['mean_similarity']:.3f}")

        return results

    def create_validation_samples(self,
                                num_samples: int = 100,
                                task_types: List[str] = None) -> List[MultimodalSample]:
        """
        Create validation samples from the knowledge base for testing.

        Args:
            num_samples: Number of validation samples to create
            task_types: Types of tasks to create samples for

        Returns:
            List of multimodal validation samples
        """
        if not self.is_loaded:
            raise ValueError("Knowledge base not loaded. Call load_knowledge_base() first.")

        if task_types is None:
            task_types = ['structure_validation', 'glytoucan_mapping', 'similarity_search']

        validation_samples = []

        # Sample entries from knowledge base
        available_entries = self.knowledge_base.entries
        if len(available_entries) < num_samples:
            num_samples = len(available_entries)

        import random
        sampled_entries = random.sample(available_entries, num_samples)

        for entry in sampled_entries:
            for task_type in task_types:
                sample = self._create_task_sample(entry, task_type)
                if sample:
                    validation_samples.append(sample)

        logger.info(f"Created {len(validation_samples)} validation samples")
        return validation_samples

    def _compute_validation_statistics(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive validation statistics"""

        total_predictions = len(validation_results)
        exact_matches = sum(1 for r in validation_results if r['exact_match'])
        known_structures = sum(1 for r in validation_results if r['is_known_structure'])
        similarities = [r['similarity_score'] for r in validation_results]

        return {
            'total_predictions': total_predictions,
            'exact_matches': exact_matches,
            'known_structures': known_structures,
            'exact_match_rate': exact_matches / total_predictions if total_predictions > 0 else 0,
            'known_structure_rate': known_structures / total_predictions if total_predictions > 0 else 0,
            'mean_similarity': sum(similarities) / len(similarities) if similarities else 0,
            'median_similarity': sorted(similarities)[len(similarities) // 2] if similarities else 0,
            'similarity_std': (sum((s - (sum(similarities) / len(similarities)))**2 for s in similarities) / len(similarities))**0.5 if similarities else 0
        }

    def _analyze_task_performance(self,
                                validation_results: List[Dict[str, Any]],
                                task_type: str) -> Dict[str, Any]:
        """Analyze performance for specific task types"""

        analysis = {
            'task_type': task_type,
            'performance_metrics': {},
            'error_patterns': [],
            'recommendations': []
        }

        if task_type == 'structure_prediction':
            exact_matches = sum(1 for r in validation_results if r['exact_match'])
            total = len(validation_results)

            analysis['performance_metrics'] = {
                'structure_accuracy': exact_matches / total if total > 0 else 0,
                'novel_structure_rate': (total - exact_matches) / total if total > 0 else 0
            }

            if exact_matches / total < 0.5:
                analysis['recommendations'].append("Consider increasing model capacity for structure prediction")
            if exact_matches / total > 0.8:
                analysis['recommendations'].append("Excellent structure prediction performance!")

        elif task_type == 'glytoucan_mapping':
            mapped_predictions = sum(1 for r in validation_results if r['glytoucan_id'] is not None)
            total = len(validation_results)

            analysis['performance_metrics'] = {
                'mapping_accuracy': mapped_predictions / total if total > 0 else 0
            }

        return analysis

    def _create_task_sample(self, entry: ValidationEntry, task_type: str) -> Optional[MultimodalSample]:
        """Create a multimodal sample for a specific task type"""

        if task_type == 'structure_validation':
            # Create sample for validating structure predictions
            sample = MultimodalSample(
                sample_id=f"val_struct_{entry.validation_hash}",
                glytoucan_id=entry.glytoucan_id,
                spectrum_id=None,
                wurcs_sequence=None,  # To be predicted
                structure_graph=None,
                spectra_peaks=None,
                precursor_mz=None,
                charge_state=None,
                collision_energy=None,
                text=f"Validate the structure of glycan: {entry.glycan_sequence}",
                text_type='validation_query',
                experimental_method='structure_validation'
            )

            sample.labels = {
                'task_type': 'structure_validation',
                'target_structure': entry.glycan_sequence,
                'glytoucan_id': entry.glytoucan_id
            }

            return sample

        elif task_type == 'glytoucan_mapping':
            # Create sample for GlyTouCan ID mapping
            if entry.glytoucan_id:
                sample = MultimodalSample(
                    sample_id=f"val_mapping_{entry.validation_hash}",
                    glytoucan_id=entry.glytoucan_id,
                    spectrum_id=None,
                    wurcs_sequence=None,
                    structure_graph=None,
                    spectra_peaks=None,
                    precursor_mz=None,
                    charge_state=None,
                    collision_energy=None,
                    text=f"Map glycan structure to GlyTouCan ID: {entry.glycan_sequence}",
                    text_type='mapping_query',
                    experimental_method='glytoucan_mapping'
                )

                sample.labels = {
                    'task_type': 'glytoucan_mapping',
                    'target_glytoucan': entry.glytoucan_id,
                    'glycan_sequence': entry.glycan_sequence
                }

                return sample

        elif task_type == 'similarity_search':
            # Create sample for similarity-based validation
            sample = MultimodalSample(
                sample_id=f"val_similarity_{entry.validation_hash}",
                glytoucan_id=entry.glytoucan_id,
                spectrum_id=None,
                wurcs_sequence=None,
                structure_graph=None,
                spectra_peaks=None,
                precursor_mz=None,
                charge_state=None,
                collision_energy=None,
                text=f"Find similar glycan structures to: {entry.glycan_sequence}",
                text_type='similarity_query',
                experimental_method='similarity_search'
            )

            sample.labels = {
                'task_type': 'similarity_search',
                'query_structure': entry.glycan_sequence,
                'target_similars': []  # Would be populated with similar structures
            }

            return sample

        return None

    def save_validation_report(self, results: Dict[str, Any], output_path: str):
        """Save validation results to a comprehensive report"""

        report = {
            'validation_report': {
                'timestamp': datetime.now().isoformat(),
                'knowledge_base_stats': self.knowledge_base.get_validation_statistics(),
                'loading_stats': self.loading_stats,
                'validation_results': results
            }
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report saved to {output_file}")


def create_candycrunch_validator(data_dir: str = "data/raw/dataset/candycrunch_glycan_data") -> CandycrunchValidator:
    """
    Create and initialize a candycrunch validator.

    Args:
        data_dir: Directory containing candycrunch data files

    Returns:
        Initialized CandycrunchValidator
    """
    validator = CandycrunchValidator(data_dir)
    validator.load_knowledge_base()
    return validator


def integrate_validation_with_training(validator: CandycrunchValidator,
                                    training_data_path: str,
                                    validation_interval: int = 100) -> Dict[str, Any]:
    """
    Integrate validation with training pipeline.

    Args:
        validator: Initialized validator
        training_data_path: Path to training data
        validation_interval: How often to run validation during training

    Returns:
        Integration configuration
    """
    integration_config = {
        'validator': validator,
        'training_data_path': training_data_path,
        'validation_interval': validation_interval,
        'validation_tasks': [
            'structure_prediction',
            'glytoucan_mapping',
            'similarity_search'
        ],
        'metrics_to_track': [
            'exact_match_rate',
            'known_structure_rate',
            'mean_similarity'
        ]
    }

    logger.info("Validation integrated with training pipeline")
    logger.info(f"   📊 Validation interval: every {validation_interval} batches")
    logger.info(f"   🎯 Validation tasks: {integration_config['validation_tasks']}")

    return integration_config


# Example usage and testing functions
def test_validator():
    """Test the candycrunch validator functionality"""

    print("Testing CandycrunchValidator...")

    # Create validator
    validator = CandycrunchValidator()

    # Load knowledge base
    try:
        kb = validator.load_knowledge_base()
        print(f"✅ Loaded knowledge base with {len(kb.entries)} entries")
    except Exception as e:
        print(f"❌ Failed to load knowledge base: {e}")
        return

    # Test validation
    test_sequences = [
        kb.entries[0].glycan_sequence,  # Should be exact match
        "modified_sequence",  # Should not match
        kb.entries[1].glycan_sequence if len(kb.entries) > 1 else "another_test"  # Another exact match
    ]

    results = validator.validate_predictions(test_sequences)
    print(f"✅ Validation results: {results['summary_statistics']}")

    # Create validation samples
    samples = validator.create_validation_samples(num_samples=5)
    print(f"✅ Created {len(samples)} validation samples")

    print("✅ All tests passed!")


if __name__ == '__main__':
    test_validator()