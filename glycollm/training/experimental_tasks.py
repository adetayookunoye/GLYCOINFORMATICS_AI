#!/usr/bin/env python3
"""
Experimental Task Adaptation for GlycoLLM
==========================================

Adapts existing GlycoLLM training tasks for quantitative experimental data
from GlycoWorks, enabling authentic experimental training.

Features:
- Abundance-aware structure elucidation
- Tissue-specific glycan prediction
- Biomarker discovery tasks
- Quantitative validation metrics

Author: Glycoinformatics AI Team
Date: November 5, 2025
"""

import logging
import numpy as np
# import torch  # Lazy import to avoid segmentation faults
# import torch.nn as nn  # Lazy import to avoid segmentation faults
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
from collections import defaultdict

from glycollm.data.multimodal_dataset import MultimodalSample
from glycollm.training.trainer import TrainingConfig, GlycoLLMTrainer
from glycollm.models.glycollm import GlycoLLM, GlycoLLMConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentalTaskConfig:
    """Configuration for experimental training tasks"""
    task_type: str
    abundance_weighting: bool = True
    tissue_specific: bool = False
    biomarker_focus: bool = False
    quantitative_validation: bool = True
    min_abundance_threshold: float = 0.01
    max_abundance_percentile: float = 95.0
    differential_expression_threshold: float = 2.0


class ExperimentalTaskAdapter:
    """
    Adapts GlycoLLM training tasks for experimental GlycoWorks data.

    Transforms synthetic training tasks to leverage authentic experimental
    abundances, tissue contexts, and biological conditions.
    """

    def _import_torch(self):
        """Lazy import of torch to avoid segmentation faults."""
        if not hasattr(self, '_torch_imported'):
            try:
                import torch
                import torch.nn as nn
                self.torch = torch
                self.nn = nn
                self._torch_imported = True
            except ImportError as e:
                raise ImportError(f"Failed to import torch: {e}")
        return self.torch, self.nn

    def __init__(self,
                 base_config: TrainingConfig,
                 experimental_config: ExperimentalTaskConfig):
        """
        Initialize experimental task adapter.

        Args:
            base_config: Base GlycoLLM training configuration
            experimental_config: Experimental task configuration
        """
        self.base_config = base_config
        self.experimental_config = experimental_config

        # Task-specific loss functions
        self.task_losses = {
            'abundance_prediction': self._abundance_prediction_loss,
            'tissue_classification': self._tissue_classification_loss,
            'biomarker_discovery': self._biomarker_discovery_loss,
            'quantitative_structure_elucidation': self._quantitative_structure_loss
        }

        # Task-specific metrics
        self.task_metrics = {
            'abundance_prediction': self._abundance_prediction_metrics,
            'tissue_classification': self._tissue_classification_metrics,
            'biomarker_discovery': self._biomarker_discovery_metrics,
            'quantitative_structure_elucidation': self._quantitative_structure_metrics
        }

    def adapt_sample_for_task(self, sample: MultimodalSample) -> MultimodalSample:
        """
        Adapt a multimodal sample for experimental task training.

        Args:
            sample: Original multimodal sample

        Returns:
            Adapted sample with experimental task labels
        """
        task_type = sample.labels.get('task_type', 'unknown')

        if task_type == 'abundance_prediction':
            return self._adapt_abundance_prediction_sample(sample)
        elif task_type == 'tissue_classification':
            return self._adapt_tissue_classification_sample(sample)
        elif task_type == 'biomarker_discovery':
            return self._adapt_biomarker_sample(sample)
        elif task_type == 'quantitative_structure_elucidation':
            return self._adapt_quantitative_structure_sample(sample)
        else:
            # Return original sample for unknown tasks
            return sample

    def _adapt_abundance_prediction_sample(self, sample: MultimodalSample) -> MultimodalSample:
        """
        Adapt sample for abundance prediction task.

        Predicts glycan abundances from experimental context.
        """
        # Extract experimental context
        exp_context = sample.experimental_context or {}

        # Create abundance prediction target
        target_abundance = sample.labels.get('target_abundance', 0.0)
        abundance_std = sample.labels.get('abundance_std', 0.0)

        # Apply abundance filtering
        if target_abundance < self.experimental_config.min_abundance_threshold:
            # Mark as low abundance (can be handled differently in loss)
            sample.labels['low_abundance'] = True

        # Create quantitative text input
        glycan = exp_context.get('glycan', 'unknown')
        tissue = exp_context.get('tissue_type', 'unknown')
        condition = exp_context.get('condition', 'unknown')

        quantitative_text = (
            f"Predict the abundance of glycan {glycan} in {tissue} tissue "
            f"under {condition} conditions. "
            f"Expected abundance: {target_abundance:.3f} ± {abundance_std:.3f}"
        )

        # Update sample
        sample.text = quantitative_text
        sample.text_type = 'abundance_prediction'

        # Set regression target (normalized abundance)
        if self.experimental_config.abundance_weighting:
            # Apply percentile-based normalization
            sample.labels['normalized_abundance'] = self._normalize_abundance(target_abundance)

        return sample

    def _adapt_tissue_classification_sample(self, sample: MultimodalSample) -> MultimodalSample:
        """
        Adapt sample for tissue classification task.

        Classifies tissue type from glycan abundance profiles.
        """
        exp_context = sample.experimental_context or {}
        profile = exp_context.get('profile', {})

        # Create classification text
        num_glycans = exp_context.get('num_glycans', 0)
        tissue_type = exp_context.get('tissue_type', 'unknown')

        classification_text = (
            f"Classify the tissue type from this glycan abundance profile "
            f"containing {num_glycans} glycans. "
            f"Profile: {self._format_glycan_profile(profile)}"
        )

        # Update sample
        sample.text = classification_text
        sample.text_type = 'tissue_classification'

        # Set classification target
        sample.labels['target_tissue'] = tissue_type
        sample.labels['num_classes'] = self._get_tissue_class_count()

        return sample

    def _adapt_biomarker_sample(self, sample: MultimodalSample) -> MultimodalSample:
        """
        Adapt sample for biomarker discovery task.

        Identifies discriminatory glycans between conditions.
        """
        exp_context = sample.experimental_context or {}
        diff_stats = exp_context.get('differential_stats', {})

        glycan = exp_context.get('glycan', 'unknown')
        conditions = exp_context.get('conditions', [])
        fold_change = diff_stats.get('fold_change', 1.0)
        significant = diff_stats.get('significant', False)

        biomarker_text = (
            f"Evaluate glycan {glycan} as a potential biomarker. "
            f"Comparing conditions: {conditions[0]} vs {conditions[1]}. "
            f"Fold change: {fold_change:.2f}. "
            f"Significant: {significant}. "
            f"Determine if this glycan discriminates between conditions."
        )

        # Update sample
        sample.text = biomarker_text
        sample.text_type = 'biomarker_discovery'

        # Set biomarker classification target
        sample.labels['is_biomarker'] = significant
        sample.labels['fold_change'] = fold_change
        sample.labels['biomarker_score'] = self._calculate_biomarker_score(diff_stats)

        return sample

    def _adapt_quantitative_structure_sample(self, sample: MultimodalSample) -> MultimodalSample:
        """
        Adapt sample for quantitative structure elucidation.

        Uses experimental abundances to guide structure prediction.
        """
        exp_context = sample.experimental_context or {}

        glycan = exp_context.get('glycan', 'unknown')
        abundances = exp_context.get('abundances', [])
        mean_abundance = exp_context.get('mean_abundance', 0.0)
        cv = exp_context.get('cv', 0.0)  # Coefficient of variation

        quantitative_text = (
            f"Elucidate the structure of glycan {glycan} using quantitative data. "
            f"Mean abundance: {mean_abundance:.3f}, "
            f"Coefficient of variation: {cv:.3f}, "
            f"Measurements: {len(abundances)}. "
            f"Use experimental abundance information to guide structure prediction."
        )

        # Update sample
        sample.text = quantitative_text
        sample.text_type = 'quantitative_structure_elucidation'

        # Set quantitative structure target
        sample.labels['glycan_sequence'] = glycan
        sample.labels['quantitative_confidence'] = self._calculate_quantitative_confidence(abundances)

        return sample

    def _normalize_abundance(self, abundance: float) -> float:
        """
        Normalize abundance values for training.

        Uses percentile-based normalization to handle skewed distributions.
        """
        # This would be calibrated on the full dataset
        # For now, use log transformation
        if abundance > 0:
            return np.log1p(abundance)
        return 0.0

    def _format_glycan_profile(self, profile: Dict[str, float], max_glycans: int = 20) -> str:
        """
        Format glycan abundance profile for text input.

        Args:
            profile: Dictionary of glycan -> abundance
            max_glycans: Maximum glycans to include

        Returns:
            Formatted profile string
        """
        # Sort by abundance
        sorted_glycans = sorted(profile.items(), key=lambda x: x[1], reverse=True)

        # Take top glycans
        top_glycans = sorted_glycans[:max_glycans]

        # Format as string
        profile_parts = []
        for glycan, abundance in top_glycans:
            # Truncate long glycan names
            short_glycan = glycan[:50] + "..." if len(glycan) > 50 else glycan
            profile_parts.append(f"{short_glycan}:{abundance:.3f}")

        return ", ".join(profile_parts)

    def _get_tissue_class_count(self) -> int:
        """
        Get number of tissue classes for classification.

        This would be determined from the dataset.
        """
        # Common tissue types in GlycoWorks
        return 15  # brain, liver, serum, skin, prostate, etc.

    def _calculate_biomarker_score(self, diff_stats: Dict[str, Any]) -> float:
        """
        Calculate biomarker discrimination score.

        Args:
            diff_stats: Differential expression statistics

        Returns:
            Biomarker score (0-1)
        """
        fold_change = diff_stats.get('fold_change', 1.0)
        p_value = diff_stats.get('p_value', 1.0)
        significant = diff_stats.get('significant', False)

        if not significant:
            return 0.0

        # Simple score based on fold change and significance
        fold_score = min(fold_change / 5.0, 1.0)  # Cap at 5-fold
        p_score = 1.0 - p_value  # Higher for more significant

        return (fold_score + p_score) / 2.0

    def _calculate_quantitative_confidence(self, abundances: List[float]) -> float:
        """
        Calculate confidence score based on quantitative measurements.

        Args:
            abundances: List of abundance measurements

        Returns:
            Confidence score (0-1)
        """
        if len(abundances) < 2:
            return 0.0

        # Higher confidence with more measurements and lower variability
        n_measurements = len(abundances)
        cv = np.std(abundances) / np.mean(abundances) if np.mean(abundances) > 0 else 1.0

        measurement_score = min(n_measurements / 10.0, 1.0)  # Cap at 10 measurements
        variability_score = max(0, 1.0 - cv)  # Lower CV = higher confidence

        return (measurement_score + variability_score) / 2.0

    def get_task_loss_function(self, task_type: str) -> callable:
        """
        Get the appropriate loss function for a task type.

        Args:
            task_type: Type of experimental task

        Returns:
            Loss function
        """
        return self.task_losses.get(task_type, self._default_loss)

    def get_task_metrics_function(self, task_type: str) -> callable:
        """
        Get the appropriate metrics function for a task type.

        Args:
            task_type: Type of experimental task

        Returns:
            Metrics function
        """
        return self.task_metrics.get(task_type, self._default_metrics)

    def _abundance_prediction_loss(self, predictions, targets, sample_weights=None):
        """
        Loss function for abundance prediction (regression task).
        """
        torch, nn = self._import_torch()
        # Use Huber loss for robustness to outliers
        loss_fn = nn.HuberLoss(delta=1.0)

        if sample_weights is not None:
            # Weight samples by abundance level or experimental confidence
            loss = loss_fn(predictions, targets)
            return torch.mean(loss * sample_weights)
        else:
            return loss_fn(predictions, targets)

    def _tissue_classification_loss(self, predictions, targets, class_weights=None):
        """
        Loss function for tissue classification.
        """
        torch, nn = self._import_torch()
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        return loss_fn(predictions, targets)

    def _biomarker_discovery_loss(self, predictions, targets, pos_weight=None):
        """
        Loss function for biomarker discovery (binary classification).
        """
        torch, nn = self._import_torch()
        if pos_weight is not None:
            # Handle class imbalance (few true biomarkers)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        return loss_fn(predictions.squeeze(), targets.float())

    def _quantitative_structure_loss(self, predictions, targets, quantitative_weights=None):
        """
        Loss function for quantitative structure elucidation.
        """
        torch, nn = self._import_torch()
        # Combine structure prediction loss with quantitative constraints
        structure_loss = nn.CrossEntropyLoss()(predictions, targets)

        if quantitative_weights is not None:
            # Modulate loss by quantitative confidence
            structure_loss = structure_loss * quantitative_weights

        return structure_loss

    def _default_loss(self, predictions, targets, **kwargs):
        """Default loss function fallback."""
        torch, nn = self._import_torch()
        return nn.MSELoss()(predictions, targets)

    def _abundance_prediction_metrics(self, predictions, targets):
        """
        Calculate metrics for abundance prediction.
        """
        torch, nn = self._import_torch()
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        # Calculate regression metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))

        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def _tissue_classification_metrics(self, predictions, targets):
        """
        Calculate metrics for tissue classification.
        """
        torch, nn = self._import_torch()
        predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        # Accuracy
        accuracy = np.mean(predictions == targets)

        # Could add F1, precision, recall per class
        return {
            'accuracy': accuracy
        }

    def _biomarker_discovery_metrics(self, predictions, targets):
        """
        Calculate metrics for biomarker discovery.
        """
        torch, nn = self._import_torch()
        predictions = torch.sigmoid(predictions).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        # Binary classification metrics
        predictions_binary = (predictions > 0.5).astype(int)

        tp = np.sum((predictions_binary == 1) & (targets == 1))
        fp = np.sum((predictions_binary == 1) & (targets == 0))
        fn = np.sum((predictions_binary == 0) & (targets == 1))
        tn = np.sum((predictions_binary == 0) & (targets == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

    def _quantitative_structure_metrics(self, predictions, targets):
        """
        Calculate metrics for quantitative structure elucidation.
        """
        torch, nn = self._import_torch()
        # Use standard structure prediction metrics
        predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        accuracy = np.mean(predictions == targets)

        return {
            'structure_accuracy': accuracy
        }

    def _default_metrics(self, predictions, targets):
        """Default metrics fallback."""
        torch, nn = self._import_torch()
        return {'default_accuracy': torch.mean((predictions == targets).float()).item()}


class ExperimentalTrainer(GlycoLLMTrainer):
    """
    Extended GlycoLLM trainer with experimental task support.

    Integrates experimental task adaptation into the training pipeline.
    """

    def __init__(self,
                 model: GlycoLLM,
                 config: TrainingConfig,
                 experimental_config: ExperimentalTaskConfig,
                 task_adapter: ExperimentalTaskAdapter):
        """
        Initialize experimental trainer.

        Args:
            model: GlycoLLM model
            config: Training configuration
            experimental_config: Experimental task configuration
            task_adapter: Task adapter for experimental data
        """
        super().__init__(model, config)
        self.experimental_config = experimental_config
        self.task_adapter = task_adapter

        # Task-specific components
        self.task_losses = {}
        self.task_metrics = {}

    def prepare_experimental_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare batch for experimental task training.

        Args:
            batch: Raw batch from dataloader

        Returns:
            Prepared batch with experimental adaptations
        """
        # Adapt samples for their specific tasks
        adapted_samples = []
        for sample in batch.get('samples', []):
            if hasattr(sample, 'labels') and 'task_type' in sample.labels:
                adapted_sample = self.task_adapter.adapt_sample_for_task(sample)
                adapted_samples.append(adapted_sample)
            else:
                adapted_samples.append(sample)

        batch['adapted_samples'] = adapted_samples

        # Group by task type for task-specific processing
        task_groups = defaultdict(list)
        for sample in adapted_samples:
            task_type = sample.labels.get('task_type', 'unknown')
            task_groups[task_type].append(sample)

        batch['task_groups'] = task_groups

        return batch

    def compute_experimental_loss(self, batch):
        """
        Compute loss for experimental task batch.

        Args:
            batch: Prepared experimental batch

        Returns:
            Combined loss across all tasks
        """
        torch, nn = self._import_torch()
        total_loss = 0.0
        task_losses = {}

        for task_type, samples in batch['task_groups'].items():
            if not samples:
                continue

            # Get task-specific loss function
            loss_fn = self.task_adapter.get_task_loss_function(task_type)

            # Prepare predictions and targets for this task
            predictions, targets, weights = self._prepare_task_inputs(samples, task_type)

            if predictions is not None and targets is not None:
                # Compute task-specific loss
                task_loss = loss_fn(predictions, targets, weights)
                task_losses[task_type] = task_loss.item()

                # Weight task loss (could be configurable)
                task_weight = self._get_task_weight(task_type)
                total_loss += task_weight * task_loss

        # Store for logging
        self.current_task_losses = task_losses

        return total_loss

    def compute_experimental_metrics(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute metrics for experimental task batch.

        Args:
            batch: Prepared experimental batch

        Returns:
            Dictionary of metrics
        """
        all_metrics = {}

        for task_type, samples in batch['task_groups'].items():
            if not samples:
                continue

            # Get task-specific metrics function
            metrics_fn = self.task_adapter.get_task_metrics_function(task_type)

            # Prepare predictions and targets
            predictions, targets, _ = self._prepare_task_inputs(samples, task_type)

            if predictions is not None and targets is not None:
                # Compute task-specific metrics
                task_metrics = metrics_fn(predictions, targets)

                # Prefix metrics with task type
                prefixed_metrics = {f"{task_type}_{k}": v for k, v in task_metrics.items()}
                all_metrics.update(prefixed_metrics)

        return all_metrics

    def _prepare_task_inputs(self, samples, task_type):
        """
        Prepare model inputs and targets for a specific task.

        Args:
            samples: Samples for this task
            task_type: Type of task

        Returns:
            Tuple of (predictions, targets, weights)
        """
        if task_type == 'abundance_prediction':
            return self._prepare_abundance_inputs(samples)
        elif task_type == 'tissue_classification':
            return self._prepare_classification_inputs(samples)
        elif task_type == 'biomarker_discovery':
            return self._prepare_biomarker_inputs(samples)
        elif task_type == 'quantitative_structure_elucidation':
            return self._prepare_structure_inputs(samples)
        else:
            return None, None, None

    def _prepare_abundance_inputs(self, samples):
        """Prepare inputs for abundance prediction."""
        torch, nn = self._import_torch()
        # This would integrate with the model's forward pass
        # Simplified placeholder
        targets = torch.tensor([s.labels.get('normalized_abundance', 0.0) for s in samples])
        predictions = torch.randn(len(samples), 1)  # Placeholder
        weights = torch.ones(len(samples))

        return predictions, targets, weights

    def _prepare_classification_inputs(self, samples):
        """Prepare inputs for tissue classification."""
        torch, nn = self._import_torch()
        # Simplified placeholder
        targets = torch.tensor([0 for s in samples])  # Would map tissue names to indices
        predictions = torch.randn(len(samples), self.task_adapter._get_tissue_class_count())
        weights = torch.ones(len(samples))

        return predictions, targets, weights

    def _prepare_biomarker_inputs(self, samples):
        """Prepare inputs for biomarker discovery."""
        torch, nn = self._import_torch()
        targets = torch.tensor([1.0 if s.labels.get('is_biomarker', False) else 0.0 for s in samples])
        predictions = torch.randn(len(samples), 1)
        weights = torch.ones(len(samples))

        return predictions, targets, weights

    def _prepare_structure_inputs(self, samples):
        """Prepare inputs for quantitative structure elucidation."""
        torch, nn = self._import_torch()
        # Would prepare structure prediction targets
        targets = torch.tensor([0 for s in samples])  # Placeholder
        predictions = torch.randn(len(samples), 1000)  # Placeholder vocab size
        weights = torch.tensor([s.labels.get('quantitative_confidence', 1.0) for s in samples])

        return predictions, targets, weights

    def _get_task_weight(self, task_type: str) -> float:
        """
        Get loss weight for a task type.

        Args:
            task_type: Type of task

        Returns:
            Loss weight
        """
        # Configurable task weights
        default_weights = {
            'abundance_prediction': 1.0,
            'tissue_classification': 1.0,
            'biomarker_discovery': 2.0,  # Higher weight for biomarker tasks
            'quantitative_structure_elucidation': 1.5
        }

        return default_weights.get(task_type, 1.0)


def create_experimental_training_config(base_config_path: Optional[str] = None) -> Tuple[TrainingConfig, ExperimentalTaskConfig]:
    """
    Create training configuration for experimental tasks.

    Args:
        base_config_path: Path to base training config

    Returns:
        Tuple of (training_config, experimental_config)
    """
    # Load or create base config
    if base_config_path:
        with open(base_config_path, 'r') as f:
            base_dict = json.load(f)
        base_config = TrainingConfig(**base_dict)
    else:
        base_config = TrainingConfig(
            experiment_name="glycoworks_experimental_training",
            output_dir="outputs/glycoworks",
            max_epochs=50,
            batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=1000,
            save_steps=5000,
            eval_steps=1000,
            logging_steps=100,
            gradient_accumulation_steps=4,
            fp16=True,
            dataloader_num_workers=4,
            seed=42
        )

    # Create experimental config
    experimental_config = ExperimentalTaskConfig(
        task_type="mixed_experimental",
        abundance_weighting=True,
        tissue_specific=True,
        biomarker_focus=True,
        quantitative_validation=True,
        min_abundance_threshold=0.01,
        max_abundance_percentile=95.0,
        differential_expression_threshold=2.0
    )

    return base_config, experimental_config


def main():
    """Main function for experimental task adaptation."""
    import argparse

    parser = argparse.ArgumentParser(description='Adapt GlycoLLM for experimental tasks')
    parser.add_argument('--base-config', help='Path to base training config')
    parser.add_argument('--task-types', nargs='+',
                       default=['abundance_prediction', 'tissue_classification'],
                       help='Experimental task types to adapt')
    parser.add_argument('--output-config', default='experimental_training_config.json',
                       help='Output path for adapted config')

    args = parser.parse_args()

    # Create experimental training configuration
    base_config, experimental_config = create_experimental_training_config(args.base_config)

    # Create task adapter
    task_adapter = ExperimentalTaskAdapter(base_config, experimental_config)

    # Save configuration
    config_dict = {
        'base_config': base_config.__dict__,
        'experimental_config': experimental_config.__dict__,
        'task_types': args.task_types
    }

    with open(args.output_config, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"✅ Experimental training configuration saved to {args.output_config}")
    logger.info(f"   📊 Task types: {args.task_types}")
    logger.info(f"   🔧 Abundance weighting: {experimental_config.abundance_weighting}")
    logger.info(f"   🎯 Biomarker focus: {experimental_config.biomarker_focus}")


if __name__ == '__main__':
    main()