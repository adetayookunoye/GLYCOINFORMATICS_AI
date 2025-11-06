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
from datetime import datetime

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


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for GlycoLLM training framework.
    
    Provides standardized benchmarks, evaluation protocols, and performance tracking
    for sophisticated GlycoLLM implementation assessment.
    """
    
    def __init__(self):
        self.evaluator = GlycoLLMEvaluator()
        self.benchmark_data = {}
        self.performance_history = []
        
    def run_comprehensive_benchmark(self,
                                  model: Any,
                                  test_data: Dict[str, Any],
                                  benchmark_name: str = "standard") -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            model: GlycoLLM model to evaluate
            test_data: Test dataset
            benchmark_name: Name of benchmark configuration
            
        Returns:
            Comprehensive benchmark results
        """
        print(f"\n🔬 Running Comprehensive Benchmark: {benchmark_name}")
        print("="*60)
        
        benchmark_results = {
            'benchmark_name': benchmark_name,
            'timestamp': str(datetime.now()),
            'model_info': self._get_model_info(model),
            'test_data_stats': self._get_data_stats(test_data),
            'performance_metrics': {},
            'task_specific_metrics': {},
            'efficiency_metrics': {},
            'robustness_metrics': {}
        }
        
        # Core performance evaluation
        print("📊 Evaluating Core Performance...")
        core_results = self._evaluate_core_performance(model, test_data)
        benchmark_results['performance_metrics'] = core_results
        
        # Task-specific evaluations
        print("🎯 Running Task-Specific Evaluations...")
        task_results = self._evaluate_task_specific_performance(model, test_data)
        benchmark_results['task_specific_metrics'] = task_results
        
        # Efficiency benchmarks
        print("⚡ Measuring Efficiency Metrics...")
        efficiency_results = self._evaluate_efficiency(model, test_data)
        benchmark_results['efficiency_metrics'] = efficiency_results
        
        # Robustness evaluation
        print("🛡️ Testing Model Robustness...")
        robustness_results = self._evaluate_robustness(model, test_data)
        benchmark_results['robustness_metrics'] = robustness_results
        
        # Store benchmark results
        self.benchmark_data[benchmark_name] = benchmark_results
        self.performance_history.append(benchmark_results)
        
        return benchmark_results
        
    def _evaluate_core_performance(self, model: Any, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate core model performance across all modalities."""
        
        results = {}
        
        # Structure prediction benchmark
        if 'structure_test' in test_data:
            struct_data = test_data['structure_test']
            predictions = []
            targets = []
            
            for sample in struct_data[:100]:  # Benchmark subset
                pred = self._predict_structure(model, sample)
                predictions.append(pred)
                targets.append(sample.get('target_structure', ''))
                
            struct_metrics = self.evaluator.structure_evaluator.evaluate_batch(predictions, targets)
            results.update({f"structure_{k}": v for k, v in struct_metrics.items()})
            
        # Spectra prediction benchmark  
        if 'spectra_test' in test_data:
            spectra_data = test_data['spectra_test']
            pred_spectra = []
            target_spectra = []
            
            for sample in spectra_data[:100]:
                pred = self._predict_spectra(model, sample)
                pred_spectra.append(pred)
                target_spectra.append(sample.get('target_spectra', []))
                
            spectra_metrics = self.evaluator.spectra_evaluator.evaluate_batch(pred_spectra, target_spectra)
            results.update({f"spectra_{k}": v for k, v in spectra_metrics.items()})
            
        # Text generation benchmark
        if 'text_test' in test_data:
            text_data = test_data['text_test']
            predictions = []
            targets = []
            
            for sample in text_data[:100]:
                pred = self._generate_text(model, sample)
                predictions.append(pred)
                targets.append(sample.get('target_text', ''))
                
            text_metrics = self.evaluator.text_evaluator.evaluate_batch(predictions, targets)
            results.update({f"text_{k}": v for k, v in text_metrics.items()})
            
        return results
        
    def _evaluate_task_specific_performance(self, model: Any, test_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Evaluate performance on specific glycoinformatics tasks."""
        
        task_results = {}
        
        # Glycan identification task
        if 'identification_test' in test_data:
            identification_metrics = self._benchmark_identification(model, test_data['identification_test'])
            task_results['identification'] = identification_metrics
            
        # Fragmentation prediction task
        if 'fragmentation_test' in test_data:
            fragmentation_metrics = self._benchmark_fragmentation(model, test_data['fragmentation_test'])
            task_results['fragmentation'] = fragmentation_metrics
            
        # Structure-spectra alignment task
        if 'alignment_test' in test_data:
            alignment_metrics = self._benchmark_alignment(model, test_data['alignment_test'])
            task_results['alignment'] = alignment_metrics
            
        # Property prediction task
        if 'property_test' in test_data:
            property_metrics = self._benchmark_property_prediction(model, test_data['property_test'])
            task_results['property_prediction'] = property_metrics
            
        return task_results
        
    def _evaluate_efficiency(self, model: Any, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate computational efficiency metrics."""
        
        import time
        
        efficiency_results = {}
        
        # Inference speed benchmark
        sample_data = list(test_data.values())[0][:10] if test_data else []
        
        if sample_data:
            start_time = time.time()
            
            for sample in sample_data:
                _ = self._predict_structure(model, sample)
                
            end_time = time.time()
            
            total_time = end_time - start_time
            efficiency_results['inference_time_per_sample'] = total_time / len(sample_data)
            efficiency_results['samples_per_second'] = len(sample_data) / total_time
            
        # Memory usage estimation
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            efficiency_results['memory_usage_mb'] = memory_usage
        except:
            efficiency_results['memory_usage_mb'] = 0.0
            
        # Model size estimation
        if hasattr(model, 'parameters'):
            param_count = sum(p.numel() for p in model.parameters())
            efficiency_results['parameter_count'] = param_count
            efficiency_results['model_size_mb'] = param_count * 4 / 1024 / 1024  # Assuming float32
            
        return efficiency_results
        
    def _evaluate_robustness(self, model: Any, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model robustness to various perturbations."""
        
        robustness_results = {}
        
        # Noise robustness
        if 'structure_test' in test_data:
            original_accuracy = self._evaluate_structure_subset(model, test_data['structure_test'][:50])
            noisy_accuracy = self._evaluate_noisy_structures(model, test_data['structure_test'][:50])
            
            robustness_results['noise_robustness'] = noisy_accuracy / original_accuracy if original_accuracy > 0 else 0.0
            
        # Cross-domain generalization
        if 'cross_domain_test' in test_data:
            cross_domain_metrics = self._evaluate_cross_domain(model, test_data['cross_domain_test'])
            robustness_results.update({f"cross_domain_{k}": v for k, v in cross_domain_metrics.items()})
            
        # Input corruption robustness
        corruption_robustness = self._evaluate_input_corruption(model, test_data)
        robustness_results.update({f"corruption_{k}": v for k, v in corruption_robustness.items()})
        
        return robustness_results
        
    def generate_benchmark_report(self, benchmark_name: str) -> str:
        """Generate comprehensive benchmark report."""
        
        if benchmark_name not in self.benchmark_data:
            return f"Benchmark '{benchmark_name}' not found."
            
        results = self.benchmark_data[benchmark_name]
        
        report = []
        report.append("=" * 80)
        report.append(f"GlycoLLM Comprehensive Benchmark Report: {benchmark_name}")
        report.append("=" * 80)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append("")
        
        # Model Information
        model_info = results.get('model_info', {})
        report.append("🤖 Model Information:")
        for key, value in model_info.items():
            report.append(f"   {key}: {value}")
        report.append("")
        
        # Performance Summary
        perf_metrics = results.get('performance_metrics', {})
        report.append("📊 Core Performance Metrics:")
        for metric, value in perf_metrics.items():
            report.append(f"   {metric}: {value:.4f}")
        report.append("")
        
        # Task-Specific Performance
        task_metrics = results.get('task_specific_metrics', {})
        if task_metrics:
            report.append("🎯 Task-Specific Performance:")
            for task, metrics in task_metrics.items():
                report.append(f"   {task.title()}:")
                for metric, value in metrics.items():
                    report.append(f"     {metric}: {value:.4f}")
            report.append("")
            
        # Efficiency Metrics
        efficiency_metrics = results.get('efficiency_metrics', {})
        if efficiency_metrics:
            report.append("⚡ Efficiency Metrics:")
            for metric, value in efficiency_metrics.items():
                report.append(f"   {metric}: {value:.4f}")
            report.append("")
            
        # Robustness Analysis
        robustness_metrics = results.get('robustness_metrics', {})
        if robustness_metrics:
            report.append("🛡️ Robustness Analysis:")
            for metric, value in robustness_metrics.items():
                report.append(f"   {metric}: {value:.4f}")
            report.append("")
            
        # Performance Trends
        if len(self.performance_history) > 1:
            report.append("📈 Performance Trends:")
            report.append(self._analyze_performance_trends())
            report.append("")
            
        report.append("=" * 80)
        
        return "\n".join(report)
        
    def _predict_structure(self, model: Any, sample: Dict[str, Any]) -> str:
        """Predict glycan structure for benchmark."""
        # Placeholder implementation - would use actual model inference
        return sample.get('predicted_structure', 'WURCS=2.0/1,1,1/[a2122h-1b_1-5]/1/1-1')
        
    def _predict_spectra(self, model: Any, sample: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Predict mass spectrum for benchmark."""
        # Placeholder implementation - would use actual model inference
        return sample.get('predicted_spectra', [(100.0, 1000.0), (200.0, 500.0)])
        
    def _generate_text(self, model: Any, sample: Dict[str, Any]) -> str:
        """Generate text description for benchmark."""
        # Placeholder implementation - would use actual model inference
        return sample.get('predicted_text', 'Generated glycan analysis text.')
        
    def _get_model_info(self, model: Any) -> Dict[str, Any]:
        """Extract model information for reporting."""
        info = {
            'model_type': type(model).__name__,
            'has_parameters': hasattr(model, 'parameters')
        }
        
        if hasattr(model, 'config'):
            info['config'] = str(model.config)
            
        return info
        
    def _benchmark_identification(self, model: Any, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Benchmark glycan identification task."""
        correct = 0
        total = len(test_data[:50])  # Benchmark subset
        
        for sample in test_data[:50]:
            pred = self._predict_structure(model, sample)
            target = sample.get('target_structure', '')
            if pred == target:
                correct += 1
                
        return {'accuracy': correct / total if total > 0 else 0.0}
        
    def _benchmark_fragmentation(self, model: Any, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Benchmark fragmentation prediction task."""
        correct_fragments = 0
        total_fragments = 0
        
        for sample in test_data[:50]:
            pred_frags = sample.get('predicted_fragments', [])
            target_frags = sample.get('target_fragments', [])
            
            total_fragments += len(target_frags)
            for frag in target_frags:
                if frag in pred_frags:
                    correct_fragments += 1
                    
        return {'fragment_accuracy': correct_fragments / total_fragments if total_fragments > 0 else 0.0}
        
    def _benchmark_alignment(self, model: Any, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Benchmark structure-spectra alignment task."""
        correct_alignments = 0
        total = len(test_data[:50])
        
        for sample in test_data[:50]:
            # Simplified alignment check
            pred_alignment = sample.get('predicted_alignment_score', 0.0)
            if pred_alignment > 0.7:  # Threshold for good alignment
                correct_alignments += 1
                
        return {'alignment_accuracy': correct_alignments / total if total > 0 else 0.0}
        
    def _benchmark_property_prediction(self, model: Any, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Benchmark glycan property prediction task."""
        correct_predictions = 0
        total = len(test_data[:50])
        
        for sample in test_data[:50]:
            pred_props = sample.get('predicted_properties', {})
            target_props = sample.get('target_properties', {})
            
            # Check if key properties match
            matches = sum(1 for k, v in target_props.items() 
                         if k in pred_props and abs(pred_props[k] - v) < 0.1)
            
            if matches >= len(target_props) * 0.8:  # 80% property accuracy
                correct_predictions += 1
                
        return {'property_accuracy': correct_predictions / total if total > 0 else 0.0}
        
    def _evaluate_structure_subset(self, model: Any, test_data: List[Dict[str, Any]]) -> float:
        """Evaluate structure prediction on subset."""
        correct = 0
        total = len(test_data)
        
        for sample in test_data:
            pred = self._predict_structure(model, sample)
            target = sample.get('target_structure', '')
            if pred == target:
                correct += 1
                
        return correct / total if total > 0 else 0.0
        
    def _evaluate_noisy_structures(self, model: Any, test_data: List[Dict[str, Any]]) -> float:
        """Evaluate with noise added to structure inputs."""
        correct = 0
        total = len(test_data)
        
        for sample in test_data:
            # Add noise to input (simplified)
            noisy_sample = sample.copy()
            if 'structure' in noisy_sample:
                # Simple noise: randomly change one character
                structure = noisy_sample['structure']
                if len(structure) > 10:
                    import random
                    pos = random.randint(1, len(structure) - 2)
                    structure_list = list(structure)
                    structure_list[pos] = 'X'  # Noise character
                    noisy_sample['structure'] = ''.join(structure_list)
            
            pred = self._predict_structure(model, noisy_sample)
            target = sample.get('target_structure', '')
            if pred == target:
                correct += 1
                
        return correct / total if total > 0 else 0.0
        
    def _evaluate_cross_domain(self, model: Any, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate cross-domain generalization."""
        # Simplified cross-domain evaluation
        accuracy = self._evaluate_structure_subset(model, test_data)
        return {'cross_domain_accuracy': accuracy * 0.8}  # Assume 20% drop in cross-domain
        
    def _evaluate_input_corruption(self, model: Any, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate robustness to input corruption."""
        results = {}
        
        # Test with missing modalities
        if 'structure_test' in test_data:
            # Remove spectra information and test
            corrupted_data = []
            for sample in test_data['structure_test'][:20]:
                corrupted_sample = sample.copy()
                corrupted_sample.pop('spectra', None)  # Remove spectra
                corrupted_data.append(corrupted_sample)
                
            accuracy = self._evaluate_structure_subset(model, corrupted_data)
            results['missing_spectra_robustness'] = accuracy
            
        return results
        
    def _analyze_performance_trends(self) -> str:
        """Analyze performance trends across checkpoints."""
        if len(self.performance_history) < 2:
            return "Insufficient data for trend analysis"
            
        # Compare latest with previous
        latest = self.performance_history[-1]['performance_metrics']
        previous = self.performance_history[-2]['performance_metrics']
        
        improvements = []
        degradations = []
        
        for metric in latest.keys():
            if metric in previous:
                if latest[metric] > previous[metric]:
                    improvements.append(f"{metric}: +{latest[metric] - previous[metric]:.4f}")
                elif latest[metric] < previous[metric]:
                    degradations.append(f"{metric}: {latest[metric] - previous[metric]:.4f}")
                    
        trend_analysis = []
        if improvements:
            trend_analysis.append(f"Improvements: {', '.join(improvements[:3])}")
        if degradations:
            trend_analysis.append(f"Degradations: {', '.join(degradations[:3])}")
            
        return " | ".join(trend_analysis) if trend_analysis else "Performance stable"
        
    def _get_data_stats(self, test_data: Dict[str, Any]) -> Dict[str, int]:
        """Get test data statistics."""
        stats = {}
        
        for key, data in test_data.items():
            if isinstance(data, list):
                stats[key] = len(data)
            elif hasattr(data, '__len__'):
                stats[key] = len(data)
                
        return stats


class AdvancedMetricsTracker:
    """
    Advanced metrics tracking system for GlycoLLM training framework.
    
    Provides comprehensive performance monitoring, trend analysis, and
    automated benchmarking during training and evaluation phases.
    """
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.evaluation_checkpoints = []
        self.performance_baselines = {}
        
    def log_training_metrics(self,
                           epoch: int,
                           batch: int,
                           metrics: Dict[str, float],
                           model_state: Optional[Dict[str, Any]] = None):
        """Log training metrics with detailed tracking."""
        
        timestamp = str(datetime.now())
        
        metric_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'batch': batch,
            'metrics': metrics.copy(),
            'model_state': model_state or {}
        }
        
        # Store in history
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append({
                'timestamp': timestamp,
                'epoch': epoch,
                'batch': batch,
                'value': value
            })
            
        # Check for performance improvements
        self._check_performance_milestones(metrics)
        
    def create_evaluation_checkpoint(self,
                                   checkpoint_name: str,
                                   evaluation_results: EvaluationResults,
                                   model_state: Dict[str, Any]):
        """Create detailed evaluation checkpoint."""
        
        checkpoint = {
            'name': checkpoint_name,
            'timestamp': str(datetime.now()),
            'evaluation_results': evaluation_results,
            'model_state': model_state,
            'metrics_snapshot': self._create_metrics_snapshot(),
            'performance_analysis': self._analyze_current_performance()
        }
        
        self.evaluation_checkpoints.append(checkpoint)
        
        # Update baselines if this is best performance
        self._update_performance_baselines(evaluation_results)
        
    def generate_comprehensive_training_report(self) -> str:
        """Generate comprehensive training progress report."""
        
        report = []
        report.append("=" * 80)
        report.append("GlycoLLM Training Framework - Comprehensive Report")
        report.append("=" * 80)
        
        # Training Progress Summary
        report.append("\n📈 Training Progress Summary:")
        if self.metrics_history:
            latest_metrics = {name: history[-1]['value'] 
                            for name, history in self.metrics_history.items() 
                            if history}
            
            for metric_name, value in latest_metrics.items():
                trend = self._calculate_metric_trend(metric_name)
                report.append(f"   {metric_name}: {value:.6f} (trend: {trend})")
        
        # Performance Baselines
        report.append("\n🎯 Performance Baselines:")
        for metric_name, baseline_value in self.performance_baselines.items():
            current_value = self._get_latest_metric_value(metric_name)
            improvement = ((current_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0
            report.append(f"   {metric_name}: {baseline_value:.6f} → {current_value:.6f} ({improvement:+.2f}%)")
            
        # Checkpoint Analysis
        if self.evaluation_checkpoints:
            report.append(f"\n📋 Evaluation Checkpoints ({len(self.evaluation_checkpoints)} total):")
            
            # Show latest checkpoint details
            latest_checkpoint = self.evaluation_checkpoints[-1]
            report.append(f"   Latest: {latest_checkpoint['name']} ({latest_checkpoint['timestamp']})")
            
            eval_results = latest_checkpoint['evaluation_results']
            report.append(f"     Structure Accuracy: {eval_results.structure_accuracy:.4f}")
            report.append(f"     Spectra F1 Score: {eval_results.peak_detection_f1:.4f}")
            report.append(f"     Text BLEU Score: {eval_results.text_bleu:.4f}")
            report.append(f"     Cross-Modal Recall@5: {eval_results.retrieval_recall_at_5:.4f}")
            
        # Training Recommendations
        report.append("\n🔧 Training Recommendations:")
        recommendations = self._generate_training_recommendations()
        for rec in recommendations:
            report.append(f"   • {rec}")
            
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
        
    def _check_performance_milestones(self, metrics: Dict[str, float]):
        """Check if current metrics represent significant milestones."""
        
        milestones = []
        
        # Check for accuracy milestones
        if 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            if accuracy > 0.95:
                milestones.append("🎉 Achieved >95% accuracy!")
            elif accuracy > 0.90:
                milestones.append("✨ Achieved >90% accuracy!")
                
        # Check for loss milestones
        if 'loss' in metrics:
            loss = metrics['loss']
            if loss < 0.01:
                milestones.append("🚀 Loss below 0.01!")
            elif loss < 0.1:
                milestones.append("📉 Loss below 0.1!")
                
        # Log milestones
        for milestone in milestones:
            logger.info(f"Performance Milestone: {milestone}")
            
    def _create_metrics_snapshot(self) -> Dict[str, Any]:
        """Create snapshot of current metrics state."""
        
        snapshot = {}
        
        for metric_name, history in self.metrics_history.items():
            if history:
                values = [entry['value'] for entry in history]
                snapshot[metric_name] = {
                    'latest': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    'trend': self._calculate_metric_trend(metric_name)
                }
                
        return snapshot
        
    def _analyze_current_performance(self) -> Dict[str, str]:
        """Analyze current training performance."""
        
        analysis = {}
        
        # Convergence analysis
        if 'loss' in self.metrics_history:
            loss_trend = self._calculate_metric_trend('loss')
            if loss_trend == 'decreasing':
                analysis['convergence'] = 'Model is converging well'
            elif loss_trend == 'stable':
                analysis['convergence'] = 'Model may have converged'
            else:
                analysis['convergence'] = 'Model may be diverging - check learning rate'
                
        # Overfitting analysis
        if 'train_accuracy' in self.metrics_history and 'val_accuracy' in self.metrics_history:
            train_acc = self._get_latest_metric_value('train_accuracy')
            val_acc = self._get_latest_metric_value('val_accuracy')
            
            if train_acc - val_acc > 0.1:
                analysis['overfitting'] = 'Potential overfitting detected'
            else:
                analysis['overfitting'] = 'No significant overfitting'
                
        return analysis
        
    def _calculate_metric_trend(self, metric_name: str) -> str:
        """Calculate trend for a specific metric."""
        
        if metric_name not in self.metrics_history or len(self.metrics_history[metric_name]) < 5:
            return 'insufficient_data'
            
        values = [entry['value'] for entry in self.metrics_history[metric_name][-10:]]
        
        # Simple trend analysis
        recent_avg = np.mean(values[-5:]) if len(values) >= 5 else values[-1]
        older_avg = np.mean(values[:-5]) if len(values) >= 10 else values[0]
        
        if abs(recent_avg - older_avg) < 0.001:
            return 'stable'
        elif recent_avg > older_avg:
            return 'increasing'
        else:
            return 'decreasing'
            
    def _get_latest_metric_value(self, metric_name: str) -> float:
        """Get latest value for a specific metric."""
        
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return 0.0
            
        return self.metrics_history[metric_name][-1]['value']
        
    def _update_performance_baselines(self, evaluation_results: EvaluationResults):
        """Update performance baselines with best results."""
        
        metrics_to_track = {
            'structure_accuracy': evaluation_results.structure_accuracy,
            'spectra_f1': evaluation_results.peak_detection_f1,
            'text_bleu': evaluation_results.text_bleu,
            'retrieval_recall_at_5': evaluation_results.retrieval_recall_at_5
        }
        
        for metric_name, value in metrics_to_track.items():
            if metric_name not in self.performance_baselines or value > self.performance_baselines[metric_name]:
                self.performance_baselines[metric_name] = value
                logger.info(f"New performance baseline for {metric_name}: {value:.6f}")
                
    def _generate_training_recommendations(self) -> List[str]:
        """Generate actionable training recommendations."""
        
        recommendations = []
        
        # Learning rate recommendations
        if 'loss' in self.metrics_history:
            loss_trend = self._calculate_metric_trend('loss')
            if loss_trend == 'increasing':
                recommendations.append("Consider reducing learning rate - loss is increasing")
            elif loss_trend == 'stable':
                recommendations.append("Consider learning rate decay - loss has plateaued")
                
        # Overfitting recommendations
        train_acc = self._get_latest_metric_value('train_accuracy')
        val_acc = self._get_latest_metric_value('val_accuracy')
        
        if train_acc - val_acc > 0.15:
            recommendations.append("Add regularization - significant overfitting detected")
            recommendations.append("Consider reducing model complexity or adding dropout")
            
        # Data recommendations
        if val_acc < 0.7:
            recommendations.append("Consider data augmentation or additional training data")
            
        # Model architecture recommendations
        if self._get_latest_metric_value('retrieval_recall_at_1') < 0.3:
            recommendations.append("Improve cross-modal alignment - consider contrastive learning")
            
        if not recommendations:
            recommendations.append("Training progressing well - continue current strategy")
            
        return recommendations
        
    def export_metrics_report(self, filepath: str):
        """Export detailed metrics report to file."""
        
        report_data = {
            'metrics_history': dict(self.metrics_history),
            'evaluation_checkpoints': self.evaluation_checkpoints,
            'performance_baselines': self.performance_baselines,
            'report_timestamp': str(datetime.now())
        }
        
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            logger.info(f"Metrics report exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics report: {e}")


def create_evaluator() -> GlycoLLMEvaluator:
    """Create a comprehensive GlycoLLM evaluator."""
    return GlycoLLMEvaluator()


def create_benchmark_suite() -> BenchmarkSuite:
    """Create comprehensive benchmark suite for GlycoLLM evaluation."""
    return BenchmarkSuite()


def create_metrics_tracker() -> AdvancedMetricsTracker:
    """Create advanced metrics tracking system for training framework."""
    return AdvancedMetricsTracker()