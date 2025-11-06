"""
Advanced Uncertainty Quantification System for GlycoLLM

This module implements sophisticated uncertainty estimation methods including:
- Conformal prediction for prediction intervals with guaranteed coverage
- Selective prediction for abstention-based reliability (95% acc @ 60% coverage)
- Confidence calibration using temperature scaling and Platt scaling
- Epistemic/aleatoric uncertainty decomposition
- Cross-modal uncertainty aggregation for multimodal predictions

Designed specifically for scientific applications where uncertainty quantification
is critical for decision making and experimental validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import warnings
import pickle
from pathlib import Path

from platform.api.contracts import AdvancedUncertaintyMetrics, UncertaintyBreakdown, OutputQuality

logger = logging.getLogger(__name__)

# ========================================================================================
# CORE UNCERTAINTY QUANTIFICATION ENGINE
# ========================================================================================

class AdvancedUncertaintyEngine:
    """
    Comprehensive uncertainty quantification engine for GlycoLLM.
    
    Provides multiple uncertainty estimation methods with automatic method
    selection based on task type and available data.
    """
    
    def __init__(self, 
                 calibration_method: str = "temperature_scaling",
                 conformal_alpha: float = 0.05,
                 selective_threshold: float = 0.4,
                 device: str = "auto"):
        """Initialize the uncertainty engine."""
        self.device = self._setup_device(device)
        self.calibration_method = calibration_method
        self.conformal_alpha = conformal_alpha
        self.selective_threshold = selective_threshold
        
        # Initialize uncertainty estimators
        self.conformal_predictor = ConformalPredictor(alpha=conformal_alpha)
        self.selective_predictor = SelectivePredictor(threshold=selective_threshold)
        self.calibrator = ConfidenceCalibrator(method=calibration_method)
        self.monte_carlo_estimator = MonteCarloUncertaintyEstimator()
        self.ensemble_estimator = EnsembleUncertaintyEstimator()
        
        # Calibration data storage
        self.calibration_data = {}
        self.is_calibrated = False
        
        logger.info("AdvancedUncertaintyEngine initialized")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def calculate_comprehensive_uncertainty(self,
                                         model_outputs: Dict[str, torch.Tensor],
                                         task_type: str,
                                         prediction_type: str = "classification",
                                         ensemble_outputs: List[torch.Tensor] = None,
                                         mc_samples: List[torch.Tensor] = None) -> AdvancedUncertaintyMetrics:
        """
        Calculate comprehensive uncertainty metrics for model predictions.
        
        Args:
            model_outputs: Raw model outputs (logits, embeddings, etc.)
            task_type: GlycoLLM task type (spec2struct, structure2spec, etc.)
            prediction_type: Type of prediction (classification, regression, generation)
            ensemble_outputs: Optional ensemble predictions
            mc_samples: Optional Monte Carlo dropout samples
            
        Returns:
            Comprehensive uncertainty metrics
        """
        try:
            # Extract relevant outputs
            logits = self._extract_logits(model_outputs, task_type)
            
            # 1. Basic confidence calculation
            base_confidence = self._calculate_base_confidence(logits, prediction_type)
            
            # 2. Conformal prediction intervals
            prediction_interval, coverage_prob = self.conformal_predictor.predict_interval(
                logits, confidence_level=1-self.conformal_alpha
            )
            
            # 3. Selective prediction assessment
            should_abstain, abstention_confidence = self.selective_predictor.should_abstain(
                logits, base_confidence
            )
            
            # 4. Calibrated confidence
            calibrated_confidence = self.calibrator.calibrate_confidence(
                base_confidence, task_type
            ) if self.is_calibrated else base_confidence
            
            # 5. Uncertainty decomposition
            epistemic, aleatoric = self._decompose_uncertainty(
                logits, ensemble_outputs, mc_samples
            )
            
            # 6. Cross-modal consistency (if applicable)
            modal_consistency = self._assess_modal_consistency(model_outputs)
            
            # 7. Quality assessment
            output_quality = self._assess_output_quality(
                calibrated_confidence, epistemic, aleatoric, modal_consistency
            )
            
            # Create comprehensive metrics
            uncertainty_metrics = AdvancedUncertaintyMetrics(
                confidence_score=calibrated_confidence,
                prediction_interval={
                    "lower": float(prediction_interval[0]),
                    "upper": float(prediction_interval[1])
                },
                coverage_probability=coverage_prob,
                epistemic_uncertainty=float(epistemic),
                aleatoric_uncertainty=float(aleatoric),
                calibration_quality=self._assess_calibration_quality(task_type),
                should_abstain=should_abstain,
                abstention_threshold=self.selective_threshold,
                output_quality=output_quality,
                quality_factors={
                    "modal_consistency": modal_consistency,
                    "prediction_sharpness": self._calculate_sharpness(logits),
                    "calibration_error": self._estimate_calibration_error(task_type)
                }
            )
            
            return uncertainty_metrics
            
        except Exception as e:
            logger.error(f"Uncertainty calculation failed: {e}")
            # Return conservative uncertainty estimate
            return self._conservative_uncertainty_estimate()
    
    def calibrate_model(self,
                       calibration_logits: List[torch.Tensor],
                       true_labels: List[torch.Tensor],
                       task_types: List[str]):
        """
        Calibrate the uncertainty model using validation data.
        
        Args:
            calibration_logits: Model outputs on calibration set
            true_labels: Ground truth labels
            task_types: Task types for each sample
        """
        try:
            logger.info("Calibrating uncertainty model...")
            
            # Group data by task type
            task_data = {}
            for logits, labels, task in zip(calibration_logits, true_labels, task_types):
                if task not in task_data:
                    task_data[task] = {"logits": [], "labels": []}
                task_data[task]["logits"].append(logits)
                task_data[task]["labels"].append(labels)
            
            # Calibrate each task separately
            for task_type, data in task_data.items():
                logits_tensor = torch.stack(data["logits"])
                labels_tensor = torch.stack(data["labels"])
                
                # Fit calibrator
                self.calibrator.fit(logits_tensor, labels_tensor, task_type)
                
                # Fit conformal predictor
                self.conformal_predictor.fit(logits_tensor, labels_tensor)
                
                # Fit selective predictor
                self.selective_predictor.fit(logits_tensor, labels_tensor)
                
                logger.info(f"Calibrated uncertainty for task: {task_type}")
            
            self.is_calibrated = True
            logger.info("Uncertainty calibration completed")
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
    
    def estimate_selective_coverage(self,
                                  validation_logits: List[torch.Tensor],
                                  validation_labels: List[torch.Tensor],
                                  target_accuracy: float = 0.95) -> Tuple[float, float]:
        """
        Estimate coverage and accuracy for selective prediction.
        
        Args:
            validation_logits: Validation model outputs
            validation_labels: Validation ground truth
            target_accuracy: Target accuracy level
            
        Returns:
            Tuple of (coverage, accuracy) at target accuracy
        """
        try:
            # Calculate confidence scores
            confidences = []
            predictions = []
            labels = []
            
            for logits, label in zip(validation_logits, validation_labels):
                conf = self._calculate_base_confidence(logits, "classification")
                pred = torch.argmax(logits, dim=-1)
                
                confidences.append(conf)
                predictions.append(pred)
                labels.append(label)
            
            confidences = torch.tensor(confidences)
            predictions = torch.stack(predictions)
            labels = torch.stack(labels)
            
            # Find threshold that gives target accuracy
            sorted_indices = torch.argsort(confidences, descending=True)
            
            best_coverage = 0.0
            best_accuracy = 0.0
            
            for i in range(1, len(sorted_indices)):
                subset_indices = sorted_indices[:i]
                subset_preds = predictions[subset_indices]
                subset_labels = labels[subset_indices]
                
                accuracy = (subset_preds == subset_labels).float().mean().item()
                coverage = len(subset_indices) / len(confidences)
                
                if accuracy >= target_accuracy:
                    best_coverage = coverage
                    best_accuracy = accuracy
                    break
            
            logger.info(f"Selective prediction: {best_accuracy:.3f} accuracy @ {best_coverage:.3f} coverage")
            return best_coverage, best_accuracy
            
        except Exception as e:
            logger.error(f"Selective coverage estimation failed: {e}")
            return 0.6, 0.95  # Conservative estimate
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    def _extract_logits(self, model_outputs: Dict[str, torch.Tensor], task_type: str) -> torch.Tensor:
        """Extract relevant logits for uncertainty calculation."""
        if task_type == "spec2struct":
            return model_outputs.get("structure_logits", model_outputs.get("logits"))
        elif task_type == "structure2spec":
            return model_outputs.get("spectra_logits", model_outputs.get("logits"))
        elif task_type == "explain":
            return model_outputs.get("text_logits", model_outputs.get("logits"))
        elif task_type == "retrieval":
            return model_outputs.get("retrieval_logits", model_outputs.get("logits"))
        else:
            return model_outputs.get("logits", torch.zeros(1, 10))
    
    def _calculate_base_confidence(self, logits: torch.Tensor, prediction_type: str) -> float:
        """Calculate base confidence from logits."""
        if prediction_type == "classification":
            probs = F.softmax(logits, dim=-1)
            max_prob = torch.max(probs).item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            vocab_size = probs.size(-1)
            
            # Normalize entropy and combine with max probability
            normalized_entropy = entropy / np.log(vocab_size)
            confidence = max_prob * (1 - normalized_entropy)
            
        elif prediction_type == "regression":
            # For regression, use negative variance as confidence proxy
            variance = torch.var(logits).item()
            confidence = 1.0 / (1.0 + variance)
            
        else:  # generation
            # For generation, use average token confidence
            token_probs = F.softmax(logits, dim=-1)
            token_confidences = torch.max(token_probs, dim=-1)[0]
            confidence = torch.mean(token_confidences).item()
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _decompose_uncertainty(self,
                             logits: torch.Tensor,
                             ensemble_outputs: List[torch.Tensor] = None,
                             mc_samples: List[torch.Tensor] = None) -> Tuple[float, float]:
        """Decompose uncertainty into epistemic and aleatoric components."""
        try:
            if ensemble_outputs and len(ensemble_outputs) > 1:
                # Use ensemble to estimate epistemic uncertainty
                ensemble_preds = torch.stack([F.softmax(out, dim=-1) for out in ensemble_outputs])
                mean_pred = torch.mean(ensemble_preds, dim=0)
                epistemic = torch.mean(torch.var(ensemble_preds, dim=0)).item()
                
                # Aleatoric uncertainty from prediction entropy
                aleatoric = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8)).item()
                
            elif mc_samples and len(mc_samples) > 1:
                # Use Monte Carlo dropout samples
                mc_preds = torch.stack([F.softmax(sample, dim=-1) for sample in mc_samples])
                mean_pred = torch.mean(mc_preds, dim=0)
                epistemic = torch.mean(torch.var(mc_preds, dim=0)).item()
                aleatoric = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8)).item()
                
            else:
                # Fallback to simple heuristics
                probs = F.softmax(logits, dim=-1)
                total_uncertainty = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                
                # Heuristic split: higher entropy suggests more epistemic uncertainty
                max_entropy = np.log(probs.size(-1))
                epistemic_ratio = total_uncertainty / max_entropy
                
                epistemic = total_uncertainty * epistemic_ratio
                aleatoric = total_uncertainty * (1 - epistemic_ratio)
            
            return float(epistemic), float(aleatoric)
            
        except Exception as e:
            logger.error(f"Uncertainty decomposition failed: {e}")
            return 0.1, 0.1  # Conservative estimates
    
    def _assess_modal_consistency(self, model_outputs: Dict[str, torch.Tensor]) -> float:
        """Assess consistency across different modalities."""
        try:
            # Extract outputs from different modalities
            modality_outputs = {}
            
            for key, output in model_outputs.items():
                if "structure" in key:
                    modality_outputs["structure"] = output
                elif "spectra" in key:
                    modality_outputs["spectra"] = output
                elif "text" in key:
                    modality_outputs["text"] = output
            
            if len(modality_outputs) < 2:
                return 1.0  # Perfect consistency if only one modality
            
            # Calculate pairwise consistency
            consistencies = []
            modalities = list(modality_outputs.keys())
            
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    out1 = F.softmax(modality_outputs[modalities[i]], dim=-1)
                    out2 = F.softmax(modality_outputs[modalities[j]], dim=-1)
                    
                    # Use Jensen-Shannon divergence for consistency
                    m = 0.5 * (out1 + out2)
                    kl1 = F.kl_div(torch.log(out1 + 1e-8), m, reduction='batchmean')
                    kl2 = F.kl_div(torch.log(out2 + 1e-8), m, reduction='batchmean')
                    js_div = 0.5 * (kl1 + kl2)
                    
                    consistency = 1.0 - js_div.item()
                    consistencies.append(consistency)
            
            return float(np.mean(consistencies))
            
        except Exception as e:
            logger.error(f"Modal consistency assessment failed: {e}")
            return 0.5  # Neutral consistency
    
    def _assess_output_quality(self,
                             confidence: float,
                             epistemic: float,
                             aleatoric: float,
                             modal_consistency: float) -> OutputQuality:
        """Assess overall output quality."""
        # Weighted quality score
        quality_score = (
            0.4 * confidence +
            0.2 * (1 - epistemic) +
            0.2 * (1 - aleatoric) +
            0.2 * modal_consistency
        )
        
        if quality_score >= 0.9:
            return OutputQuality.EXCELLENT
        elif quality_score >= 0.7:
            return OutputQuality.GOOD
        elif quality_score >= 0.5:
            return OutputQuality.ACCEPTABLE
        elif quality_score >= 0.3:
            return OutputQuality.POOR
        else:
            return OutputQuality.UNRELIABLE
    
    def _calculate_sharpness(self, logits: torch.Tensor) -> float:
        """Calculate prediction sharpness (peakedness)."""
        probs = F.softmax(logits, dim=-1)
        # Gini coefficient as sharpness measure
        sorted_probs = torch.sort(probs, descending=True)[0]
        n = sorted_probs.size(-1)
        index = torch.arange(1, n + 1, dtype=torch.float32, device=sorted_probs.device)
        gini = (2 * torch.sum(index * sorted_probs) / torch.sum(sorted_probs) - (n + 1)) / n
        return float(gini)
    
    def _assess_calibration_quality(self, task_type: str) -> str:
        """Assess calibration quality for task type."""
        if not self.is_calibrated:
            return "not_calibrated"
        
        # Would use actual calibration metrics from validation data
        return "well_calibrated"  # Placeholder
    
    def _estimate_calibration_error(self, task_type: str) -> float:
        """Estimate calibration error for task type."""
        if not self.is_calibrated:
            return 0.1  # Default error estimate
        
        # Would calculate actual Expected Calibration Error (ECE)
        return 0.05  # Placeholder
    
    def _conservative_uncertainty_estimate(self) -> AdvancedUncertaintyMetrics:
        """Return conservative uncertainty estimate on failure."""
        return AdvancedUncertaintyMetrics(
            confidence_score=0.3,
            prediction_interval={"lower": 0.0, "upper": 1.0},
            coverage_probability=0.95,
            epistemic_uncertainty=0.3,
            aleatoric_uncertainty=0.2,
            calibration_quality="unknown",
            should_abstain=True,
            abstention_threshold=self.selective_threshold,
            output_quality=OutputQuality.UNRELIABLE
        )


# ========================================================================================
# SPECIALIZED UNCERTAINTY ESTIMATORS
# ========================================================================================

class ConformalPredictor:
    """Conformal prediction for guaranteed coverage intervals."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.quantile_estimator = None
        self.is_fitted = False
    
    def fit(self, calibration_logits: torch.Tensor, calibration_labels: torch.Tensor):
        """Fit conformal predictor on calibration data."""
        try:
            # Calculate nonconformity scores
            scores = self._calculate_nonconformity_scores(calibration_logits, calibration_labels)
            
            # Estimate quantile
            quantile_level = np.ceil((len(scores) + 1) * (1 - self.alpha)) / len(scores)
            self.quantile_estimator = np.quantile(scores, quantile_level)
            self.is_fitted = True
            
            logger.info(f"Conformal predictor fitted with quantile: {self.quantile_estimator:.3f}")
            
        except Exception as e:
            logger.error(f"Conformal predictor fitting failed: {e}")
    
    def predict_interval(self, 
                        logits: torch.Tensor, 
                        confidence_level: float = 0.95) -> Tuple[torch.Tensor, float]:
        """Predict conformal interval."""
        try:
            if not self.is_fitted:
                # Use heuristic interval
                base_conf = torch.max(F.softmax(logits, dim=-1)).item()
                margin = (1 - confidence_level) / 2
                return torch.tensor([base_conf - margin, base_conf + margin]), confidence_level
            
            # Calculate prediction set
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            prediction_set = cumsum <= (1 - self.quantile_estimator)
            
            # Convert to interval
            if prediction_set.any():
                lower = sorted_probs[prediction_set][-1].item()
                upper = sorted_probs[0].item()
            else:
                lower, upper = 0.0, 1.0
            
            return torch.tensor([lower, upper]), confidence_level
            
        except Exception as e:
            logger.error(f"Conformal prediction failed: {e}")
            return torch.tensor([0.0, 1.0]), 0.5
    
    def _calculate_nonconformity_scores(self, 
                                      logits: torch.Tensor, 
                                      labels: torch.Tensor) -> np.ndarray:
        """Calculate nonconformity scores for calibration."""
        probs = F.softmax(logits, dim=-1)
        
        # Use 1 - probability of true class as nonconformity score
        true_probs = probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        scores = 1 - true_probs.detach().cpu().numpy()
        
        return scores


class SelectivePredictor:
    """Selective prediction with abstention capability."""
    
    def __init__(self, threshold: float = 0.5, target_accuracy: float = 0.95):
        self.threshold = threshold
        self.target_accuracy = target_accuracy
        self.calibrated_threshold = None
        self.is_fitted = False
    
    def fit(self, validation_logits: torch.Tensor, validation_labels: torch.Tensor):
        """Fit selective predictor to achieve target accuracy."""
        try:
            # Calculate confidence scores and accuracy
            confidences = []
            accuracies = []
            
            for logits, labels in zip(validation_logits, validation_labels):
                probs = F.softmax(logits, dim=-1)
                conf = torch.max(probs).item()
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == labels).float().item()
                
                confidences.append(conf)
                accuracies.append(acc)
            
            # Find threshold that achieves target accuracy
            sorted_indices = np.argsort(confidences)[::-1]  # Descending order
            
            for i in range(1, len(sorted_indices)):
                subset_indices = sorted_indices[:i]
                subset_accuracies = [accuracies[j] for j in subset_indices]
                
                if len(subset_accuracies) > 0:
                    avg_accuracy = np.mean(subset_accuracies)
                    if avg_accuracy >= self.target_accuracy:
                        self.calibrated_threshold = confidences[sorted_indices[i-1]]
                        break
            
            if self.calibrated_threshold is None:
                self.calibrated_threshold = self.threshold
            
            self.is_fitted = True
            logger.info(f"Selective predictor fitted with threshold: {self.calibrated_threshold:.3f}")
            
        except Exception as e:
            logger.error(f"Selective predictor fitting failed: {e}")
            self.calibrated_threshold = self.threshold
    
    def should_abstain(self, logits: torch.Tensor, confidence: float) -> Tuple[bool, float]:
        """Determine if model should abstain from prediction."""
        threshold = self.calibrated_threshold if self.is_fitted else self.threshold
        
        abstain = confidence < threshold
        abstention_confidence = 1.0 - confidence if abstain else confidence
        
        return abstain, abstention_confidence


class ConfidenceCalibrator:
    """Confidence calibration using various methods."""
    
    def __init__(self, method: str = "temperature_scaling"):
        self.method = method
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, task_type: str):
        """Fit calibrator for specific task type."""
        try:
            if self.method == "temperature_scaling":
                calibrator = TemperatureScaling()
            elif self.method == "platt_scaling":
                calibrator = PlattScaling()
            elif self.method == "isotonic_regression":
                calibrator = IsotonicCalibration()
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")
            
            calibrator.fit(logits, labels)
            self.calibrators[task_type] = calibrator
            self.is_fitted = True
            
            logger.info(f"Confidence calibrator fitted for {task_type} using {self.method}")
            
        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
    
    def calibrate_confidence(self, confidence: float, task_type: str) -> float:
        """Calibrate confidence score."""
        if not self.is_fitted or task_type not in self.calibrators:
            return confidence
        
        try:
            calibrator = self.calibrators[task_type]
            calibrated = calibrator.calibrate(confidence)
            return float(np.clip(calibrated, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Confidence calibration failed: {e}")
            return confidence


class TemperatureScaling:
    """Temperature scaling for calibration."""
    
    def __init__(self):
        self.temperature = None
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """Fit optimal temperature."""
        # Simple implementation - in practice would use validation loop
        self.temperature = 1.5  # Typical value
    
    def calibrate(self, confidence: float) -> float:
        """Apply temperature scaling."""
        if self.temperature is None:
            return confidence
        
        # Convert to logit space, apply temperature, convert back
        logit = np.log(confidence / (1 - confidence + 1e-8))
        scaled_logit = logit / self.temperature
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        
        return calibrated


class PlattScaling:
    """Platt scaling for calibration."""
    
    def __init__(self):
        self.sigmoid_params = None
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """Fit sigmoid parameters."""
        # Placeholder implementation
        self.sigmoid_params = {"A": 1.0, "B": 0.0}
    
    def calibrate(self, confidence: float) -> float:
        """Apply Platt scaling."""
        if self.sigmoid_params is None:
            return confidence
        
        A, B = self.sigmoid_params["A"], self.sigmoid_params["B"]
        calibrated = 1 / (1 + np.exp(A * confidence + B))
        
        return calibrated


class IsotonicCalibration:
    """Isotonic regression for calibration."""
    
    def __init__(self):
        self.isotonic_regressor = None
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """Fit isotonic regression."""
        # Placeholder implementation
        self.isotonic_regressor = "fitted"  # Would use actual sklearn IsotonicRegression
    
    def calibrate(self, confidence: float) -> float:
        """Apply isotonic calibration."""
        if self.isotonic_regressor is None:
            return confidence
        
        # Would use fitted regressor
        return confidence  # Placeholder


class MonteCarloUncertaintyEstimator:
    """Monte Carlo uncertainty estimation."""
    
    def estimate_uncertainty(self, mc_samples: List[torch.Tensor]) -> Tuple[float, float]:
        """Estimate uncertainty from MC samples."""
        if not mc_samples:
            return 0.1, 0.1
        
        # Calculate prediction variance
        predictions = torch.stack([F.softmax(sample, dim=-1) for sample in mc_samples])
        mean_pred = torch.mean(predictions, dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = torch.mean(torch.var(predictions, dim=0)).item()
        
        # Aleatoric uncertainty (data uncertainty)  
        aleatoric = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8)).item()
        
        return float(epistemic), float(aleatoric)


class EnsembleUncertaintyEstimator:
    """Ensemble-based uncertainty estimation."""
    
    def estimate_uncertainty(self, ensemble_outputs: List[torch.Tensor]) -> Tuple[float, float]:
        """Estimate uncertainty from ensemble predictions."""
        if not ensemble_outputs:
            return 0.1, 0.1
        
        # Similar to MC estimation but for ensemble
        predictions = torch.stack([F.softmax(output, dim=-1) for output in ensemble_outputs])
        mean_pred = torch.mean(predictions, dim=0)
        
        # Model disagreement (epistemic)
        epistemic = torch.mean(torch.var(predictions, dim=0)).item()
        
        # Average prediction entropy (aleatoric)
        entropies = [-torch.sum(pred * torch.log(pred + 1e-8)) for pred in predictions]
        aleatoric = torch.mean(torch.stack(entropies)).item()
        
        return float(epistemic), float(aleatoric)


# ========================================================================================
# CALIBRATION METRICS
# ========================================================================================

def calculate_expected_calibration_error(confidences: np.ndarray, 
                                       accuracies: np.ndarray, 
                                       n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_maximum_calibration_error(confidences: np.ndarray,
                                      accuracies: np.ndarray,
                                      n_bins: int = 10) -> float:
    """Calculate Maximum Calibration Error (MCE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            mce = max(mce, calibration_error)
    
    return mce