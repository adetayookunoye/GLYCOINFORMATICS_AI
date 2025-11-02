"""
Evaluation metrics for GlycoLLM training and validation.

This module implements specialized evaluation metrics for glycoinformatics
tasks including structure prediction accuracy, spectra similarity, and
cross-modal retrieval performance.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import json
import re

# Optional imports
try:
    import torch
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    
    # Overall metrics
    total_samples: int = 0
    average_loss: float = 0.0
    
    # Structure prediction metrics
    structure_accuracy: float = 0.0
    structure_bleu: float = 0.0
    structure_exact_match: float = 0.0
    wurcs_validity: float = 0.0
    
    # Spectra prediction metrics
    spectra_accuracy: float = 0.0
    spectra_mse: float = 0.0
    peak_detection_f1: float = 0.0
    intensity_correlation: float = 0.0
    
    # Text generation metrics
    text_accuracy: float = 0.0
    text_bleu: float = 0.0
    text_rouge: Dict[str, float] = None
    perplexity: float = 0.0
    
    # Cross-modal retrieval metrics
    retrieval_recall_at_1: float = 0.0
    retrieval_recall_at_5: float = 0.0
    retrieval_recall_at_10: float = 0.0
    retrieval_mrr: float = 0.0  # Mean reciprocal rank
    
    # Task-specific metrics
    monosaccharide_accuracy: float = 0.0
    linkage_accuracy: float = 0.0
    fragment_identification_f1: float = 0.0
    
    def __post_init__(self):
        if self.text_rouge is None:
            self.text_rouge = {}


class StructureEvaluator:
    """
    Evaluator for glycan structure prediction tasks.
    
    Measures WURCS prediction accuracy, validity, and structural correctness.
    """
    
    def __init__(self):
        # WURCS validation patterns
        self.wurcs_pattern = re.compile(r'^WURCS=\d+\.\d+/\d+,\d+,\d+/(\[.*?\])+/.*$')
        
        # Monosaccharide patterns for detailed analysis
        self.monosaccharide_patterns = {
            'GlcNAc': r'a2122h.*NCC',
            'Gal': r'a1122h-1b',
            'Man': r'a1221m-1a',
            'Fuc': r'a112h-1b',
            'Neu5Ac': r'a1122A-2a'
        }
        
        # Linkage patterns
        self.linkage_patterns = {
            'α1-2': r'a1-.*2',
            'α1-3': r'a1-.*3', 
            'α1-4': r'a1-.*4',
            'α1-6': r'a1-.*6',
            'β1-2': r'b1-.*2',
            'β1-3': r'b1-.*3',
            'β1-4': r'b1-.*4',
            'β1-6': r'b1-.*6'
        }
        
    def evaluate_batch(self, 
                      predictions: List[str],
                      targets: List[str]) -> Dict[str, float]:
        """
        Evaluate a batch of structure predictions.
        
        Args:
            predictions: Predicted WURCS sequences
            targets: Target WURCS sequences
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            'exact_match': 0.0,
            'wurcs_validity': 0.0,
            'bleu_score': 0.0,
            'monosaccharide_accuracy': 0.0,
            'linkage_accuracy': 0.0,
            'structure_similarity': 0.0
        }
        
        if not predictions or not targets:
            return results
            
        total_samples = len(predictions)
        exact_matches = 0
        valid_wurcs = 0
        mono_correct = 0
        linkage_correct = 0
        bleu_scores = []
        
        for pred, target in zip(predictions, targets):
            # Exact match
            if pred.strip() == target.strip():
                exact_matches += 1
                
            # WURCS validity
            if self._is_valid_wurcs(pred):
                valid_wurcs += 1
                
            # BLEU score (character-level)
            bleu = self._compute_bleu(pred, target)
            bleu_scores.append(bleu)
            
            # Monosaccharide accuracy
            if self._compare_monosaccharides(pred, target):
                mono_correct += 1
                
            # Linkage accuracy
            if self._compare_linkages(pred, target):
                linkage_correct += 1
                
        results['exact_match'] = exact_matches / total_samples
        results['wurcs_validity'] = valid_wurcs / total_samples
        results['bleu_score'] = np.mean(bleu_scores) if bleu_scores else 0.0
        results['monosaccharide_accuracy'] = mono_correct / total_samples
        results['linkage_accuracy'] = linkage_correct / total_samples
        
        return results
        
    def _is_valid_wurcs(self, wurcs: str) -> bool:
        """Check if WURCS string is syntactically valid."""
        if not wurcs or not isinstance(wurcs, str):
            return False
            
        return bool(self.wurcs_pattern.match(wurcs.strip()))
        
    def _compute_bleu(self, prediction: str, target: str) -> float:
        """Compute character-level BLEU score."""
        if not prediction or not target:
            return 0.0
            
        # Simple character-level BLEU approximation
        pred_chars = set(prediction)
        target_chars = set(target)
        
        if not target_chars:
            return 1.0 if not pred_chars else 0.0
            
        overlap = len(pred_chars & target_chars)
        precision = overlap / len(pred_chars) if pred_chars else 0.0
        recall = overlap / len(target_chars)
        
        if precision == 0 or recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)
        
    def _compare_monosaccharides(self, pred: str, target: str) -> bool:
        """Compare monosaccharide composition."""
        pred_monos = self._extract_monosaccharides(pred)
        target_monos = self._extract_monosaccharides(target)
        
        return pred_monos == target_monos
        
    def _extract_monosaccharides(self, wurcs: str) -> Dict[str, int]:
        """Extract monosaccharide counts from WURCS."""
        mono_counts = defaultdict(int)
        
        for mono_type, pattern in self.monosaccharide_patterns.items():
            matches = re.findall(pattern, wurcs)
            mono_counts[mono_type] = len(matches)
            
        return dict(mono_counts)
        
    def _compare_linkages(self, pred: str, target: str) -> bool:
        """Compare linkage patterns."""
        pred_linkages = self._extract_linkages(pred)
        target_linkages = self._extract_linkages(target)
        
        return pred_linkages == target_linkages
        
    def _extract_linkages(self, wurcs: str) -> Set[str]:
        """Extract linkage types from WURCS."""
        linkages = set()
        
        for linkage_type, pattern in self.linkage_patterns.items():
            if re.search(pattern, wurcs):
                linkages.add(linkage_type)
                
        return linkages


class SpectraEvaluator:
    """
    Evaluator for mass spectra prediction and analysis tasks.
    
    Measures peak prediction accuracy, intensity correlation, and
    fragment identification performance.
    """
    
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance  # m/z tolerance for peak matching
        
    def evaluate_batch(self,
                      predicted_spectra: List[List[Tuple[float, float]]],
                      target_spectra: List[List[Tuple[float, float]]]) -> Dict[str, float]:
        """
        Evaluate a batch of spectra predictions.
        
        Args:
            predicted_spectra: Predicted peak lists [(m/z, intensity), ...]
            target_spectra: Target peak lists
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            'peak_detection_f1': 0.0,
            'intensity_correlation': 0.0,
            'mse': 0.0,
            'coverage': 0.0
        }
        
        if not predicted_spectra or not target_spectra:
            return results
            
        f1_scores = []
        correlations = []
        mse_values = []
        coverage_values = []
        
        for pred_peaks, target_peaks in zip(predicted_spectra, target_spectra):
            # Peak detection F1
            f1 = self._compute_peak_f1(pred_peaks, target_peaks)
            f1_scores.append(f1)
            
            # Intensity correlation
            corr = self._compute_intensity_correlation(pred_peaks, target_peaks)
            correlations.append(corr)
            
            # MSE for matched peaks
            mse = self._compute_peak_mse(pred_peaks, target_peaks)
            mse_values.append(mse)
            
            # Coverage (fraction of target peaks detected)
            coverage = self._compute_coverage(pred_peaks, target_peaks)
            coverage_values.append(coverage)
            
        results['peak_detection_f1'] = np.mean(f1_scores)
        results['intensity_correlation'] = np.mean([c for c in correlations if c is not None])
        results['mse'] = np.mean(mse_values)
        results['coverage'] = np.mean(coverage_values)
        
        return results
        
    def _compute_peak_f1(self, pred_peaks: List[Tuple[float, float]],
                        target_peaks: List[Tuple[float, float]]) -> float:
        """Compute F1 score for peak detection."""
        if not target_peaks:
            return 1.0 if not pred_peaks else 0.0
            
        if not pred_peaks:
            return 0.0
            
        # Match peaks within tolerance
        matched_pred = set()
        matched_target = set()
        
        for i, (pred_mz, _) in enumerate(pred_peaks):
            for j, (target_mz, _) in enumerate(target_peaks):
                if abs(pred_mz - target_mz) <= self.tolerance:
                    matched_pred.add(i)
                    matched_target.add(j)
                    
        tp = len(matched_pred)
        fp = len(pred_peaks) - tp
        fn = len(target_peaks) - len(matched_target)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)
        
    def _compute_intensity_correlation(self, pred_peaks: List[Tuple[float, float]],
                                     target_peaks: List[Tuple[float, float]]) -> Optional[float]:
        """Compute correlation between predicted and target intensities."""
        if not HAS_DEPENDENCIES or not pred_peaks or not target_peaks:
            return None
            
        # Match peaks and collect intensities
        pred_intensities = []
        target_intensities = []
        
        for pred_mz, pred_int in pred_peaks:
            best_match = None
            best_diff = float('inf')
            
            for target_mz, target_int in target_peaks:
                diff = abs(pred_mz - target_mz)
                if diff <= self.tolerance and diff < best_diff:
                    best_match = target_int
                    best_diff = diff
                    
            if best_match is not None:
                pred_intensities.append(pred_int)
                target_intensities.append(best_match)
                
        if len(pred_intensities) < 2:
            return None
            
        try:
            correlation = np.corrcoef(pred_intensities, target_intensities)[0, 1]
            return correlation if not np.isnan(correlation) else None
        except:
            return None
            
    def _compute_peak_mse(self, pred_peaks: List[Tuple[float, float]],
                         target_peaks: List[Tuple[float, float]]) -> float:
        """Compute MSE for matched peaks."""
        if not pred_peaks or not target_peaks:
            return 0.0
            
        squared_errors = []
        
        for pred_mz, pred_int in pred_peaks:
            best_match = None
            best_diff = float('inf')
            
            for target_mz, target_int in target_peaks:
                diff = abs(pred_mz - target_mz)
                if diff <= self.tolerance and diff < best_diff:
                    best_match = (target_mz, target_int)
                    best_diff = diff
                    
            if best_match is not None:
                target_mz, target_int = best_match
                mz_error = (pred_mz - target_mz) ** 2
                int_error = (pred_int - target_int) ** 2
                squared_errors.append(mz_error + int_error)
                
        return np.mean(squared_errors) if squared_errors else 0.0
        
    def _compute_coverage(self, pred_peaks: List[Tuple[float, float]],
                         target_peaks: List[Tuple[float, float]]) -> float:
        """Compute coverage (fraction of target peaks detected)."""
        if not target_peaks:
            return 1.0
            
        if not pred_peaks:
            return 0.0
            
        detected = 0
        
        for target_mz, _ in target_peaks:
            for pred_mz, _ in pred_peaks:
                if abs(pred_mz - target_mz) <= self.tolerance:
                    detected += 1
                    break
                    
        return detected / len(target_peaks)


class TextEvaluator:
    """
    Evaluator for text generation tasks in glycoinformatics.
    
    Measures text quality, scientific accuracy, and terminology usage.
    """
    
    def __init__(self):
        # Glycomics terminology for domain-specific evaluation
        self.glycomics_terms = {
            'monosaccharides': ['glucose', 'galactose', 'mannose', 'fucose', 'glcnac', 'galnac', 'neu5ac'],
            'linkages': ['α1-2', 'α1-3', 'α1-4', 'α1-6', 'β1-2', 'β1-3', 'β1-4', 'β1-6'],
            'structures': ['n-glycan', 'o-glycan', 'glycoprotein', 'glycolipid'],
            'methods': ['ms', 'lc-ms', 'maldi', 'esi', 'cid', 'hcd']
        }
        
    def evaluate_batch(self,
                      predictions: List[str],
                      targets: List[str]) -> Dict[str, float]:
        """
        Evaluate a batch of text predictions.
        
        Args:
            predictions: Generated text
            targets: Target text
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            'bleu_score': 0.0,
            'rouge_l': 0.0,
            'exact_match': 0.0,
            'terminology_accuracy': 0.0,
            'semantic_similarity': 0.0
        }
        
        if not predictions or not targets:
            return results
            
        exact_matches = 0
        bleu_scores = []
        rouge_scores = []
        term_accuracies = []
        
        for pred, target in zip(predictions, targets):
            # Exact match
            if pred.strip().lower() == target.strip().lower():
                exact_matches += 1
                
            # BLEU score
            bleu = self._compute_text_bleu(pred, target)
            bleu_scores.append(bleu)
            
            # ROUGE-L score
            rouge = self._compute_rouge_l(pred, target)
            rouge_scores.append(rouge)
            
            # Terminology accuracy
            term_acc = self._compute_terminology_accuracy(pred, target)
            term_accuracies.append(term_acc)
            
        results['exact_match'] = exact_matches / len(predictions)
        results['bleu_score'] = np.mean(bleu_scores)
        results['rouge_l'] = np.mean(rouge_scores)
        results['terminology_accuracy'] = np.mean(term_accuracies)
        
        return results
        
    def _compute_text_bleu(self, prediction: str, target: str) -> float:
        """Compute BLEU score for text."""
        pred_words = prediction.lower().split()
        target_words = target.lower().split()
        
        if not target_words:
            return 1.0 if not pred_words else 0.0
            
        if not pred_words:
            return 0.0
            
        # Simple word-level BLEU approximation
        pred_set = set(pred_words)
        target_set = set(target_words)
        
        overlap = len(pred_set & target_set)
        precision = overlap / len(pred_set)
        recall = overlap / len(target_set)
        
        if precision == 0 or recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)
        
    def _compute_rouge_l(self, prediction: str, target: str) -> float:
        """Compute ROUGE-L score."""
        pred_words = prediction.lower().split()
        target_words = target.lower().split()
        
        if not target_words:
            return 1.0 if not pred_words else 0.0
            
        if not pred_words:
            return 0.0
            
        # Longest common subsequence
        lcs_length = self._lcs_length(pred_words, target_words)
        
        if lcs_length == 0:
            return 0.0
            
        precision = lcs_length / len(pred_words)
        recall = lcs_length / len(target_words)
        
        return 2 * precision * recall / (precision + recall)
        
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    
        return dp[m][n]
        
    def _compute_terminology_accuracy(self, prediction: str, target: str) -> float:
        """Compute accuracy of glycomics terminology usage."""
        pred_lower = prediction.lower()
        target_lower = target.lower()
        
        total_terms = 0
        correct_terms = 0
        
        for category, terms in self.glycomics_terms.items():
            for term in terms:
                if term in target_lower:
                    total_terms += 1
                    if term in pred_lower:
                        correct_terms += 1
                        
        if total_terms == 0:
            return 1.0
            
        return correct_terms / total_terms


class CrossModalEvaluator:
    """
    Evaluator for cross-modal retrieval and alignment tasks.
    
    Measures how well the model can match structures with spectra and text.
    """
    
    def __init__(self):
        pass
        
    def evaluate_retrieval(self,
                          query_embeddings: torch.Tensor,
                          candidate_embeddings: torch.Tensor,
                          ground_truth_indices: List[int]) -> Dict[str, float]:
        """
        Evaluate cross-modal retrieval performance.
        
        Args:
            query_embeddings: Query embeddings [num_queries, dim]
            candidate_embeddings: Candidate embeddings [num_candidates, dim]  
            ground_truth_indices: Correct candidate index for each query
            
        Returns:
            Retrieval metrics
        """
        if not HAS_DEPENDENCIES:
            return {}
            
        results = {
            'recall_at_1': 0.0,
            'recall_at_5': 0.0,
            'recall_at_10': 0.0,
            'mrr': 0.0,
            'map': 0.0
        }
        
        if query_embeddings.size(0) == 0:
            return results
            
        # Compute similarity matrix
        similarities = torch.matmul(query_embeddings, candidate_embeddings.T)
        
        # Get top-k indices for each query
        _, top_indices = torch.topk(similarities, k=min(10, similarities.size(1)), dim=1)
        
        recall_1 = 0
        recall_5 = 0
        recall_10 = 0
        reciprocal_ranks = []
        
        for i, gt_idx in enumerate(ground_truth_indices):
            top_k = top_indices[i].cpu().numpy()
            
            # Check recalls
            if gt_idx in top_k[:1]:
                recall_1 += 1
            if gt_idx in top_k[:5]:
                recall_5 += 1
            if gt_idx in top_k[:10]:
                recall_10 += 1
                
            # Compute reciprocal rank
            try:
                rank = np.where(top_k == gt_idx)[0][0] + 1
                reciprocal_ranks.append(1.0 / rank)
            except:
                reciprocal_ranks.append(0.0)
                
        num_queries = len(ground_truth_indices)
        results['recall_at_1'] = recall_1 / num_queries
        results['recall_at_5'] = recall_5 / num_queries
        results['recall_at_10'] = recall_10 / num_queries
        results['mrr'] = np.mean(reciprocal_ranks)
        
        return results


class GlycoLLMEvaluator:
    """
    Comprehensive evaluator for GlycoLLM model.
    
    Combines all evaluation components for multimodal assessment.
    """
    
    def __init__(self):
        self.structure_evaluator = StructureEvaluator()
        self.spectra_evaluator = SpectraEvaluator()
        self.text_evaluator = TextEvaluator()
        self.crossmodal_evaluator = CrossModalEvaluator()
        
    def evaluate_model_outputs(self,
                             model_outputs: Dict[str, Any],
                             targets: Dict[str, Any]) -> EvaluationResults:
        """
        Evaluate complete model outputs against targets.
        
        Args:
            model_outputs: Model predictions
            targets: Ground truth targets
            
        Returns:
            Comprehensive evaluation results
        """
        results = EvaluationResults()
        
        # Structure prediction evaluation
        if 'structure_predictions' in model_outputs and 'structure_targets' in targets:
            struct_metrics = self.structure_evaluator.evaluate_batch(
                model_outputs['structure_predictions'],
                targets['structure_targets']
            )
            results.structure_accuracy = struct_metrics.get('exact_match', 0.0)
            results.structure_bleu = struct_metrics.get('bleu_score', 0.0)
            results.wurcs_validity = struct_metrics.get('wurcs_validity', 0.0)
            results.monosaccharide_accuracy = struct_metrics.get('monosaccharide_accuracy', 0.0)
            results.linkage_accuracy = struct_metrics.get('linkage_accuracy', 0.0)
            
        # Spectra prediction evaluation
        if 'spectra_predictions' in model_outputs and 'spectra_targets' in targets:
            spectra_metrics = self.spectra_evaluator.evaluate_batch(
                model_outputs['spectra_predictions'],
                targets['spectra_targets']
            )
            results.spectra_accuracy = spectra_metrics.get('coverage', 0.0)
            results.spectra_mse = spectra_metrics.get('mse', 0.0)
            results.peak_detection_f1 = spectra_metrics.get('peak_detection_f1', 0.0)
            results.intensity_correlation = spectra_metrics.get('intensity_correlation', 0.0)
            
        # Text generation evaluation
        if 'text_predictions' in model_outputs and 'text_targets' in targets:
            text_metrics = self.text_evaluator.evaluate_batch(
                model_outputs['text_predictions'],
                targets['text_targets']
            )
            results.text_accuracy = text_metrics.get('exact_match', 0.0)
            results.text_bleu = text_metrics.get('bleu_score', 0.0)
            results.text_rouge = {'rouge_l': text_metrics.get('rouge_l', 0.0)}
            
        # Cross-modal retrieval evaluation
        if ('query_embeddings' in model_outputs and 
            'candidate_embeddings' in model_outputs and
            'retrieval_targets' in targets):
            
            retrieval_metrics = self.crossmodal_evaluator.evaluate_retrieval(
                model_outputs['query_embeddings'],
                model_outputs['candidate_embeddings'],
                targets['retrieval_targets']
            )
            results.retrieval_recall_at_1 = retrieval_metrics.get('recall_at_1', 0.0)
            results.retrieval_recall_at_5 = retrieval_metrics.get('recall_at_5', 0.0)
            results.retrieval_recall_at_10 = retrieval_metrics.get('recall_at_10', 0.0)
            results.retrieval_mrr = retrieval_metrics.get('mrr', 0.0)
            
        return results
        
    def print_evaluation_summary(self, results: EvaluationResults):
        """Print formatted evaluation summary."""
        
        print("\n" + "="*60)
        print("GlycoLLM Evaluation Summary")
        print("="*60)
        
        print(f"\n📊 Overall Metrics:")
        print(f"   Total Samples: {results.total_samples}")
        print(f"   Average Loss: {results.average_loss:.6f}")
        
        print(f"\n🧬 Structure Prediction:")
        print(f"   Accuracy: {results.structure_accuracy:.3f}")
        print(f"   BLEU Score: {results.structure_bleu:.3f}")
        print(f"   WURCS Validity: {results.wurcs_validity:.3f}")
        print(f"   Monosaccharide Acc: {results.monosaccharide_accuracy:.3f}")
        print(f"   Linkage Accuracy: {results.linkage_accuracy:.3f}")
        
        print(f"\n📈 Spectra Prediction:")
        print(f"   Coverage: {results.spectra_accuracy:.3f}")
        print(f"   Peak Detection F1: {results.peak_detection_f1:.3f}")
        print(f"   Intensity Correlation: {results.intensity_correlation:.3f}")
        print(f"   MSE: {results.spectra_mse:.6f}")
        
        print(f"\n📝 Text Generation:")
        print(f"   Accuracy: {results.text_accuracy:.3f}")
        print(f"   BLEU Score: {results.text_bleu:.3f}")
        if results.text_rouge:
            print(f"   ROUGE-L: {results.text_rouge.get('rouge_l', 0.0):.3f}")
        print(f"   Perplexity: {results.perplexity:.3f}")
        
        print(f"\n🔄 Cross-Modal Retrieval:")
        print(f"   Recall@1: {results.retrieval_recall_at_1:.3f}")
        print(f"   Recall@5: {results.retrieval_recall_at_5:.3f}")
        print(f"   Recall@10: {results.retrieval_recall_at_10:.3f}")
        print(f"   MRR: {results.retrieval_mrr:.3f}")
        
        print("\n" + "="*60)


def create_evaluator() -> GlycoLLMEvaluator:
    """Create a comprehensive GlycoLLM evaluator."""
    return GlycoLLMEvaluator()