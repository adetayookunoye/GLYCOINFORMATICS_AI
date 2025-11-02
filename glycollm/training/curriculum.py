"""
Curriculum learning strategies for GlycoLLM training.

This module implements progressive training curricula that help the model
learn glycoinformatics concepts in a structured manner, from simple to complex.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import random
import math

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class CurriculumStage:
    """Configuration for a curriculum learning stage."""
    
    name: str
    difficulty: DifficultyLevel
    duration_epochs: int
    data_fraction: float = 1.0
    max_sequence_length: Optional[int] = None
    complexity_filters: Dict[str, Any] = None
    loss_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.complexity_filters is None:
            self.complexity_filters = {}
        if self.loss_weights is None:
            self.loss_weights = {}


class StructureComplexityAnalyzer:
    """
    Analyzes glycan structure complexity for curriculum learning.
    
    Determines difficulty based on:
    - Number of monosaccharides
    - Branching complexity
    - Linkage diversity
    - Presence of rare modifications
    """
    
    def __init__(self):
        # Simple monosaccharides (easier to learn)
        self.simple_monosaccharides = {
            'Glc', 'Gal', 'Man', 'GlcNAc', 'GalNAc', 'Fuc'
        }
        
        # Complex monosaccharides (harder to learn)
        self.complex_monosaccharides = {
            'Neu5Ac', 'Neu5Gc', 'KDN', 'IdoA', 'GlcA'
        }
        
        # Standard linkages (easier)
        self.standard_linkages = {
            'α1-2', 'α1-3', 'α1-4', 'α1-6',
            'β1-2', 'β1-3', 'β1-4', 'β1-6'
        }
        
        # Complex linkages (harder)
        self.complex_linkages = {
            'α2-3', 'α2-6', 'α2-8',
            'β2-3', 'β2-6', 'β2-8'
        }
        
    def analyze_wurcs_complexity(self, wurcs: str) -> Dict[str, Union[int, float]]:
        """
        Analyze complexity of a WURCS structure.
        
        Args:
            wurcs: WURCS notation string
            
        Returns:
            Complexity metrics dictionary
        """
        metrics = {
            'monosaccharide_count': 0,
            'branching_points': 0,
            'complex_residue_ratio': 0.0,
            'linkage_diversity': 0,
            'has_modifications': False,
            'overall_complexity': 0.0
        }
        
        if not wurcs or not isinstance(wurcs, str):
            return metrics
            
        try:
            # Parse WURCS sections
            parts = wurcs.split('/')
            if len(parts) < 4:
                return metrics
                
            # Count monosaccharides from uniqueRES section
            unique_res = parts[1] if len(parts) > 1 else ""
            res_count = unique_res.count('[')
            metrics['monosaccharide_count'] = res_count
            
            # Analyze connectivity for branching
            if len(parts) > 2:
                connectivity = parts[2]
                # Count branching points (residues with multiple connections)
                connection_counts = {}
                for connection in connectivity.split('-'):
                    if connection and connection.isdigit():
                        res_id = int(connection)
                        connection_counts[res_id] = connection_counts.get(res_id, 0) + 1
                        
                metrics['branching_points'] = sum(1 for count in connection_counts.values() if count > 2)
                
            # Analyze residue complexity
            if len(parts) > 1:
                residues = parts[1]
                complex_count = 0
                total_count = res_count
                
                for complex_mono in self.complex_monosaccharides:
                    complex_count += residues.count(complex_mono)
                    
                if total_count > 0:
                    metrics['complex_residue_ratio'] = complex_count / total_count
                    
            # Estimate linkage diversity (simplified)
            if len(parts) > 2:
                linkages = parts[2]
                unique_linkages = set()
                
                # Extract linkage patterns (simplified heuristic)
                for i in range(len(linkages) - 1):
                    if linkages[i].isdigit():
                        unique_linkages.add(linkages[i])
                        
                metrics['linkage_diversity'] = len(unique_linkages)
                
            # Check for modifications
            metrics['has_modifications'] = 'MOD' in wurcs or 'SO3' in wurcs or 'PO3' in wurcs
            
            # Calculate overall complexity score
            complexity = (
                min(metrics['monosaccharide_count'] / 10.0, 1.0) * 0.3 +
                min(metrics['branching_points'] / 5.0, 1.0) * 0.25 +
                metrics['complex_residue_ratio'] * 0.2 +
                min(metrics['linkage_diversity'] / 8.0, 1.0) * 0.15 +
                (0.1 if metrics['has_modifications'] else 0.0)
            )
            metrics['overall_complexity'] = complexity
            
        except Exception as e:
            logger.warning(f"Error analyzing WURCS complexity: {e}")
            
        return metrics
        
    def get_difficulty_level(self, complexity_metrics: Dict[str, Union[int, float]]) -> DifficultyLevel:
        """
        Determine difficulty level based on complexity metrics.
        
        Args:
            complexity_metrics: Output from analyze_wurcs_complexity
            
        Returns:
            Difficulty level classification
        """
        complexity_score = complexity_metrics.get('overall_complexity', 0.0)
        
        if complexity_score < 0.25:
            return DifficultyLevel.BEGINNER
        elif complexity_score < 0.5:
            return DifficultyLevel.INTERMEDIATE
        elif complexity_score < 0.75:
            return DifficultyLevel.ADVANCED
        else:
            return DifficultyLevel.EXPERT


class SpectraComplexityAnalyzer:
    """
    Analyzes mass spectra complexity for curriculum learning.
    
    Determines difficulty based on:
    - Number of peaks
    - Intensity distribution
    - Fragment diversity
    - Signal-to-noise ratio
    """
    
    def __init__(self):
        pass
        
    def analyze_spectra_complexity(self, peaks: List[Tuple[float, float]]) -> Dict[str, Union[int, float]]:
        """
        Analyze complexity of mass spectra.
        
        Args:
            peaks: List of (m/z, intensity) tuples
            
        Returns:
            Complexity metrics dictionary
        """
        metrics = {
            'peak_count': 0,
            'intensity_range': 0.0,
            'mass_range': 0.0,
            'peak_density': 0.0,
            'signal_complexity': 0.0,
            'overall_complexity': 0.0
        }
        
        if not peaks:
            return metrics
            
        try:
            mz_values = [peak[0] for peak in peaks]
            intensities = [peak[1] for peak in peaks]
            
            # Basic statistics
            metrics['peak_count'] = len(peaks)
            metrics['mass_range'] = max(mz_values) - min(mz_values) if mz_values else 0.0
            metrics['intensity_range'] = max(intensities) - min(intensities) if intensities else 0.0
            
            # Peak density (peaks per m/z unit)
            if metrics['mass_range'] > 0:
                metrics['peak_density'] = metrics['peak_count'] / metrics['mass_range']
                
            # Signal complexity (coefficient of variation of intensities)
            if len(intensities) > 1:
                mean_intensity = sum(intensities) / len(intensities)
                variance = sum((x - mean_intensity) ** 2 for x in intensities) / len(intensities)
                std_dev = math.sqrt(variance)
                metrics['signal_complexity'] = std_dev / mean_intensity if mean_intensity > 0 else 0.0
                
            # Overall complexity score
            complexity = (
                min(metrics['peak_count'] / 100.0, 1.0) * 0.4 +
                min(metrics['peak_density'] / 1.0, 1.0) * 0.3 +
                min(metrics['signal_complexity'] / 2.0, 1.0) * 0.3
            )
            metrics['overall_complexity'] = complexity
            
        except Exception as e:
            logger.warning(f"Error analyzing spectra complexity: {e}")
            
        return metrics
        
    def get_difficulty_level(self, complexity_metrics: Dict[str, Union[int, float]]) -> DifficultyLevel:
        """
        Determine difficulty level based on complexity metrics.
        
        Args:
            complexity_metrics: Output from analyze_spectra_complexity
            
        Returns:
            Difficulty level classification
        """
        complexity_score = complexity_metrics.get('overall_complexity', 0.0)
        
        if complexity_score < 0.3:
            return DifficultyLevel.BEGINNER
        elif complexity_score < 0.6:
            return DifficultyLevel.INTERMEDIATE
        elif complexity_score < 0.8:
            return DifficultyLevel.ADVANCED
        else:
            return DifficultyLevel.EXPERT


class CurriculumManager:
    """
    Manages curriculum learning progression for GlycoLLM training.
    
    Implements progressive difficulty increase and adaptive scheduling.
    """
    
    def __init__(self):
        self.structure_analyzer = StructureComplexityAnalyzer()
        self.spectra_analyzer = SpectraComplexityAnalyzer()
        self.current_stage = 0
        self.stages = []
        self.performance_history = []
        
    def create_default_curriculum(self) -> List[CurriculumStage]:
        """
        Create a default curriculum for glycoinformatics learning.
        
        Returns:
            List of curriculum stages
        """
        stages = [
            # Stage 1: Simple monosaccharides and basic linkages
            CurriculumStage(
                name="Basic Monosaccharides",
                difficulty=DifficultyLevel.BEGINNER,
                duration_epochs=5,
                data_fraction=0.3,
                max_sequence_length=128,
                complexity_filters={
                    'max_monosaccharides': 3,
                    'max_branching': 0,
                    'allowed_residues': ['Glc', 'Gal', 'Man', 'GlcNAc'],
                    'max_peaks': 20
                },
                loss_weights={
                    'structure': 1.0,
                    'spectra': 0.5,
                    'text': 0.3
                }
            ),
            
            # Stage 2: Linear chains and standard linkages
            CurriculumStage(
                name="Linear Glycans",
                difficulty=DifficultyLevel.INTERMEDIATE,
                duration_epochs=8,
                data_fraction=0.5,
                max_sequence_length=256,
                complexity_filters={
                    'max_monosaccharides': 6,
                    'max_branching': 1,
                    'standard_linkages_only': True,
                    'max_peaks': 50
                },
                loss_weights={
                    'structure': 1.0,
                    'spectra': 0.7,
                    'text': 0.5
                }
            ),
            
            # Stage 3: Branched structures and complex residues
            CurriculumStage(
                name="Branched N-Glycans",
                difficulty=DifficultyLevel.ADVANCED,
                duration_epochs=10,
                data_fraction=0.7,
                max_sequence_length=512,
                complexity_filters={
                    'max_monosaccharides': 12,
                    'max_branching': 3,
                    'include_complex_residues': True,
                    'max_peaks': 100
                },
                loss_weights={
                    'structure': 1.0,
                    'spectra': 1.0,
                    'text': 0.8
                }
            ),
            
            # Stage 4: Complex glycans with modifications
            CurriculumStage(
                name="Complex Glycans",
                difficulty=DifficultyLevel.EXPERT,
                duration_epochs=15,
                data_fraction=1.0,
                max_sequence_length=1024,
                complexity_filters={
                    'no_limits': True,
                    'include_modifications': True
                },
                loss_weights={
                    'structure': 1.0,
                    'spectra': 1.0,
                    'text': 1.0
                }
            )
        ]
        
        return stages
        
    def initialize_curriculum(self, custom_stages: Optional[List[CurriculumStage]] = None):
        """
        Initialize curriculum with stages.
        
        Args:
            custom_stages: Optional custom curriculum stages
        """
        if custom_stages:
            self.stages = custom_stages
        else:
            self.stages = self.create_default_curriculum()
            
        self.current_stage = 0
        self.performance_history = []
        
        logger.info(f"Initialized curriculum with {len(self.stages)} stages")
        for i, stage in enumerate(self.stages):
            logger.info(f"  Stage {i+1}: {stage.name} ({stage.difficulty.value})")
            
    def get_current_stage(self) -> Optional[CurriculumStage]:
        """
        Get the current curriculum stage.
        
        Returns:
            Current curriculum stage or None if finished
        """
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return None
        
    def should_advance_stage(self, epoch: int, performance_metrics: Dict[str, float]) -> bool:
        """
        Determine if curriculum should advance to next stage.
        
        Args:
            epoch: Current training epoch
            performance_metrics: Current model performance
            
        Returns:
            True if should advance to next stage
        """
        current = self.get_current_stage()
        if not current:
            return False
            
        # Record performance
        self.performance_history.append({
            'epoch': epoch,
            'stage': self.current_stage,
            'metrics': performance_metrics.copy()
        })
        
        # Check duration-based advancement
        stage_epochs = sum(1 for record in self.performance_history 
                          if record['stage'] == self.current_stage)
        
        if stage_epochs >= current.duration_epochs:
            logger.info(f"Advancing curriculum: completed {stage_epochs} epochs for stage {current.name}")
            return True
            
        # Check performance-based advancement (optional early advancement)
        if stage_epochs >= current.duration_epochs // 2:
            recent_performance = [
                record['metrics'] for record in self.performance_history[-5:]
                if record['stage'] == self.current_stage
            ]
            
            if len(recent_performance) >= 3:
                # Check if performance is consistently high
                avg_accuracy = sum(
                    metrics.get('structure_accuracy', 0.0) + 
                    metrics.get('spectra_accuracy', 0.0) + 
                    metrics.get('text_accuracy', 0.0)
                    for metrics in recent_performance
                ) / (len(recent_performance) * 3)
                
                if avg_accuracy > 0.85:
                    logger.info(f"Early advancement: high performance ({avg_accuracy:.3f}) in stage {current.name}")
                    return True
                    
        return False
        
    def advance_stage(self) -> bool:
        """
        Advance to the next curriculum stage.
        
        Returns:
            True if advanced successfully, False if at end
        """
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            new_stage = self.get_current_stage()
            logger.info(f"Advanced to curriculum stage {self.current_stage + 1}: {new_stage.name}")
            return True
        else:
            logger.info("Curriculum completed - using full dataset")
            return False
            
    def filter_dataset_by_stage(self, dataset: List[Dict[str, Any]], 
                               stage: CurriculumStage) -> List[Dict[str, Any]]:
        """
        Filter dataset according to current curriculum stage.
        
        Args:
            dataset: Full dataset
            stage: Current curriculum stage
            
        Returns:
            Filtered dataset for current stage
        """
        if not stage.complexity_filters:
            # No filtering - use data fraction only
            sample_size = int(len(dataset) * stage.data_fraction)
            return random.sample(dataset, sample_size) if sample_size < len(dataset) else dataset
            
        filtered_data = []
        
        for sample in dataset:
            if self._sample_meets_criteria(sample, stage.complexity_filters):
                filtered_data.append(sample)
                
        # Apply data fraction
        target_size = int(len(dataset) * stage.data_fraction)
        if len(filtered_data) > target_size:
            filtered_data = random.sample(filtered_data, target_size)
            
        logger.info(f"Filtered dataset: {len(filtered_data)} samples for stage {stage.name}")
        return filtered_data
        
    def _sample_meets_criteria(self, sample: Dict[str, Any], 
                              criteria: Dict[str, Any]) -> bool:
        """
        Check if a sample meets curriculum stage criteria.
        
        Args:
            sample: Data sample
            criteria: Filtering criteria
            
        Returns:
            True if sample meets criteria
        """
        if criteria.get('no_limits', False):
            return True
            
        # Check structure complexity
        if 'structure' in sample or 'wurcs' in sample:
            wurcs = sample.get('structure', sample.get('wurcs', ''))
            if wurcs:
                struct_metrics = self.structure_analyzer.analyze_wurcs_complexity(wurcs)
                
                # Check monosaccharide count
                if 'max_monosaccharides' in criteria:
                    if struct_metrics['monosaccharide_count'] > criteria['max_monosaccharides']:
                        return False
                        
                # Check branching
                if 'max_branching' in criteria:
                    if struct_metrics['branching_points'] > criteria['max_branching']:
                        return False
                        
                # Check allowed residues
                if 'allowed_residues' in criteria:
                    # Simplified check - would need more sophisticated WURCS parsing
                    allowed = criteria['allowed_residues']
                    for residue in ['Neu5Ac', 'IdoA', 'GlcA']:
                        if residue not in allowed and residue in wurcs:
                            return False
                            
        # Check spectra complexity  
        if 'spectra' in sample:
            spectra = sample['spectra']
            if isinstance(spectra, list) and spectra:
                if 'max_peaks' in criteria:
                    if len(spectra) > criteria['max_peaks']:
                        return False
                        
        return True
        
    def get_loss_weights(self) -> Dict[str, float]:
        """
        Get loss weights for current curriculum stage.
        
        Returns:
            Dictionary of loss weights
        """
        current = self.get_current_stage()
        if current and current.loss_weights:
            return current.loss_weights
        return {'structure': 1.0, 'spectra': 1.0, 'text': 1.0}
        
    def get_max_sequence_length(self) -> Optional[int]:
        """
        Get maximum sequence length for current stage.
        
        Returns:
            Maximum sequence length or None for no limit
        """
        current = self.get_current_stage()
        return current.max_sequence_length if current else None
        
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """
        Get current curriculum progress information.
        
        Returns:
            Progress information dictionary
        """
        current = self.get_current_stage()
        
        progress = {
            'current_stage': self.current_stage,
            'total_stages': len(self.stages),
            'stage_name': current.name if current else "Completed",
            'stage_difficulty': current.difficulty.value if current else "complete",
            'progress_percent': (self.current_stage / len(self.stages)) * 100
        }
        
        if self.performance_history:
            stage_history = [r for r in self.performance_history if r['stage'] == self.current_stage]
            if stage_history:
                progress['stage_epochs'] = len(stage_history)
                progress['target_epochs'] = current.duration_epochs if current else 0
                
        return progress


def create_curriculum_manager() -> CurriculumManager:
    """Create a curriculum manager with default settings."""
    manager = CurriculumManager()
    manager.initialize_curriculum()
    return manager