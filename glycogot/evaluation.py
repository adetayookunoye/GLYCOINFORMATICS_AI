"""
Evaluation Metrics and Assessment for Graph of Thoughts Implementation
=====================================================================

This module provides comprehensive evaluation and assessment tools
for Graph of Thoughts reasoning performance, path quality, and
overall reasoning effectiveness.
"""

import time
import statistics
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .graph_reasoning import ThoughtNode, ThoughtGraph, ThoughtType
from .search_algorithms import SearchResult

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Types of evaluation metrics"""
    PATH_QUALITY = "path_quality"
    REASONING_DEPTH = "reasoning_depth"
    REASONING_BREADTH = "reasoning_breadth"
    CONSISTENCY = "consistency"
    NOVELTY = "novelty"
    EFFICIENCY = "efficiency"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"


@dataclass
class PathEvaluation:
    """Evaluation results for a single reasoning path"""
    path_id: str
    path: List[str]
    
    # Quality metrics
    overall_score: float = 0.0
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    evidence_score: float = 0.0
    novelty_score: float = 0.0
    
    # Structural metrics
    path_length: int = 0
    reasoning_depth: int = 0
    thought_diversity: float = 0.0
    
    # Consistency metrics
    consistency_score: float = 0.0
    contradiction_count: int = 0
    
    # Metadata
    evaluation_time: float = 0.0
    evaluator_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'path_id': self.path_id,
            'path': self.path,
            'overall_score': self.overall_score,
            'confidence_score': self.confidence_score,
            'relevance_score': self.relevance_score,
            'evidence_score': self.evidence_score,
            'novelty_score': self.novelty_score,
            'path_length': self.path_length,
            'reasoning_depth': self.reasoning_depth,
            'thought_diversity': self.thought_diversity,
            'consistency_score': self.consistency_score,
            'contradiction_count': self.contradiction_count,
            'evaluation_time': self.evaluation_time,
            'evaluator_version': self.evaluator_version
        }


@dataclass
class GraphEvaluation:
    """Evaluation results for an entire thought graph"""
    graph_id: str
    
    # Graph structure metrics
    total_thoughts: int = 0
    total_connections: int = 0
    graph_density: float = 0.0
    average_degree: float = 0.0
    
    # Quality metrics
    average_confidence: float = 0.0
    average_relevance: float = 0.0
    thought_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Reasoning metrics
    reasoning_depth: int = 0
    reasoning_breadth: int = 0
    cycles_detected: int = 0
    
    # Performance metrics
    evaluation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'graph_id': self.graph_id,
            'total_thoughts': self.total_thoughts,
            'total_connections': self.total_connections,
            'graph_density': self.graph_density,
            'average_degree': self.average_degree,
            'average_confidence': self.average_confidence,
            'average_relevance': self.average_relevance,
            'thought_type_distribution': self.thought_type_distribution,
            'reasoning_depth': self.reasoning_depth,
            'reasoning_breadth': self.reasoning_breadth,
            'cycles_detected': self.cycles_detected,
            'evaluation_time': self.evaluation_time
        }


@dataclass
class ComparisonResult:
    """Results comparing multiple reasoning approaches"""
    approaches: List[str]
    metrics: Dict[str, List[float]]
    winner: str
    statistical_significance: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'approaches': self.approaches,
            'metrics': self.metrics,
            'winner': self.winner,
            'statistical_significance': self.statistical_significance
        }


class PathEvaluator:
    """
    Evaluates the quality and effectiveness of reasoning paths.
    
    This class provides comprehensive evaluation of individual reasoning
    paths through multiple quality dimensions.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize path evaluator.
        
        Args:
            weights: Weights for combining different evaluation metrics
        """
        self.weights = weights or {
            'confidence': 0.25,
            'relevance': 0.25,
            'evidence': 0.20,
            'novelty': 0.15,
            'consistency': 0.15
        }
    
    def evaluate_path(self, path: List[str], graph: ThoughtGraph, 
                     path_id: Optional[str] = None) -> PathEvaluation:
        """
        Comprehensively evaluate a reasoning path.
        
        Args:
            path: List of thought IDs representing the reasoning path
            graph: The thought graph containing the thoughts
            path_id: Optional identifier for this path
            
        Returns:
            PathEvaluation object with comprehensive metrics
        """
        start_time = time.time()
        
        if path_id is None:
            path_id = f"path_{int(time.time() * 1000)}"
        
        evaluation = PathEvaluation(path_id=path_id, path=path)
        
        # Get thought objects
        thoughts = []
        for thought_id in path:
            thought = graph.get_thought(thought_id)
            if thought:
                thoughts.append(thought)
        
        if not thoughts:
            evaluation.evaluation_time = time.time() - start_time
            return evaluation
        
        # Calculate quality metrics
        evaluation.confidence_score = self._calculate_confidence_score(thoughts)
        evaluation.relevance_score = self._calculate_relevance_score(thoughts)
        evaluation.evidence_score = self._calculate_evidence_score(thoughts)
        evaluation.novelty_score = self._calculate_novelty_score(thoughts)
        
        # Calculate structural metrics
        evaluation.path_length = len(thoughts)
        evaluation.reasoning_depth = self._calculate_reasoning_depth(thoughts)
        evaluation.thought_diversity = self._calculate_thought_diversity(thoughts)
        
        # Calculate consistency metrics
        evaluation.consistency_score, evaluation.contradiction_count = self._calculate_consistency(thoughts)
        
        # Calculate overall score
        evaluation.overall_score = self._calculate_overall_score(evaluation)
        
        evaluation.evaluation_time = time.time() - start_time
        return evaluation
    
    def _calculate_confidence_score(self, thoughts: List[ThoughtNode]) -> float:
        """Calculate average confidence score for the path"""
        if not thoughts:
            return 0.0
        
        confidences = [thought.confidence for thought in thoughts]
        
        # Use weighted average with more weight on later thoughts
        weights = [i + 1 for i in range(len(thoughts))]
        total_weight = sum(weights)
        
        weighted_sum = sum(conf * weight for conf, weight in zip(confidences, weights))
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_relevance_score(self, thoughts: List[ThoughtNode]) -> float:
        """Calculate average relevance score for the path"""
        if not thoughts:
            return 0.0
            
        return sum(thought.relevance for thought in thoughts) / len(thoughts)
    
    def _calculate_evidence_score(self, thoughts: List[ThoughtNode]) -> float:
        """Calculate evidence strength score for the path"""
        if not thoughts:
            return 0.0
        
        # Combine evidence strength and evidence count
        evidence_strengths = [thought.evidence_strength for thought in thoughts]
        evidence_counts = [len(thought.evidence) for thought in thoughts]
        
        # Normalize evidence counts (assume max of 5 pieces of evidence is excellent)
        normalized_counts = [min(count / 5.0, 1.0) for count in evidence_counts]
        
        # Average both components
        avg_strength = sum(evidence_strengths) / len(evidence_strengths)
        avg_count = sum(normalized_counts) / len(normalized_counts)
        
        return (avg_strength + avg_count) / 2.0
    
    def _calculate_novelty_score(self, thoughts: List[ThoughtNode]) -> float:
        """Calculate novelty score for the path"""
        if not thoughts:
            return 0.0
            
        novelty_scores = [thought.novelty for thought in thoughts]
        
        # Weight novelty higher for synthesis and hypothesis thoughts
        weighted_novelties = []
        for thought in thoughts:
            weight = 1.0
            if thought.thought_type in [ThoughtType.SYNTHESIS, ThoughtType.HYPOTHESIS]:
                weight = 1.5
            elif thought.thought_type == ThoughtType.ANALOGY:
                weight = 1.2
                
            weighted_novelties.append(thought.novelty * weight)
        
        return sum(weighted_novelties) / len(thoughts)
    
    def _calculate_reasoning_depth(self, thoughts: List[ThoughtNode]) -> int:
        """Calculate the reasoning depth (number of reasoning steps)"""
        reasoning_types = {
            ThoughtType.DEDUCTION, ThoughtType.INDUCTION,
            ThoughtType.ANALOGY, ThoughtType.CAUSAL,
            ThoughtType.SYNTHESIS, ThoughtType.HYPOTHESIS
        }
        
        depth = sum(1 for thought in thoughts if thought.thought_type in reasoning_types)
        return depth
    
    def _calculate_thought_diversity(self, thoughts: List[ThoughtNode]) -> float:
        """Calculate diversity of thought types in the path"""
        if not thoughts:
            return 0.0
        
        thought_types = set(thought.thought_type for thought in thoughts)
        max_possible_types = len(ThoughtType)
        
        return len(thought_types) / max_possible_types
    
    def _calculate_consistency(self, thoughts: List[ThoughtNode]) -> Tuple[float, int]:
        """Calculate consistency score and count contradictions"""
        if len(thoughts) < 2:
            return 1.0, 0
        
        contradictions = 0
        total_pairs = 0
        consistency_sum = 0.0
        
        # Compare all pairs of thoughts
        for i in range(len(thoughts)):
            for j in range(i + 1, len(thoughts)):
                thought1, thought2 = thoughts[i], thoughts[j]
                
                # Simple contradiction detection based on content
                consistency = self._check_thought_consistency(thought1, thought2)
                consistency_sum += consistency
                total_pairs += 1
                
                if consistency < 0.3:  # Threshold for contradiction
                    contradictions += 1
        
        average_consistency = consistency_sum / max(total_pairs, 1)
        return average_consistency, contradictions
    
    def _check_thought_consistency(self, thought1: ThoughtNode, thought2: ThoughtNode) -> float:
        """Check consistency between two thoughts"""
        content1 = thought1.content.lower()
        content2 = thought2.content.lower()
        
        # Simple keyword-based consistency check
        contradiction_indicators = [
            ('not', ''), ('never', 'always'), ('impossible', 'possible'),
            ('cannot', 'can'), ('wrong', 'correct'), ('false', 'true')
        ]
        
        for neg_word, pos_word in contradiction_indicators:
            if neg_word in content1 and pos_word in content2:
                return 0.2  # Low consistency
            if pos_word in content1 and neg_word in content2:
                return 0.2  # Low consistency
        
        # Check for overlapping concepts (higher consistency)
        words1 = set(content1.split())
        words2 = set(content2.split())
        overlap = len(words1.intersection(words2))
        
        if overlap > 0:
            return 0.8  # High consistency
        else:
            return 0.6  # Neutral consistency
    
    def _calculate_overall_score(self, evaluation: PathEvaluation) -> float:
        """Calculate weighted overall score"""
        return (
            evaluation.confidence_score * self.weights.get('confidence', 0.25) +
            evaluation.relevance_score * self.weights.get('relevance', 0.25) +
            evaluation.evidence_score * self.weights.get('evidence', 0.20) +
            evaluation.novelty_score * self.weights.get('novelty', 0.15) +
            evaluation.consistency_score * self.weights.get('consistency', 0.15)
        )


class GraphEvaluator:
    """
    Evaluates the overall structure and quality of thought graphs.
    
    This class provides comprehensive evaluation of the entire reasoning
    graph structure and content quality.
    """
    
    def evaluate_graph(self, graph: ThoughtGraph, graph_id: Optional[str] = None) -> GraphEvaluation:
        """
        Comprehensively evaluate a thought graph.
        
        Args:
            graph: The thought graph to evaluate
            graph_id: Optional identifier for this graph
            
        Returns:
            GraphEvaluation object with comprehensive metrics
        """
        start_time = time.time()
        
        if graph_id is None:
            graph_id = f"graph_{int(time.time() * 1000)}"
        
        evaluation = GraphEvaluation(graph_id=graph_id)
        
        # Basic structure metrics
        evaluation.total_thoughts = len(graph.nodes)
        evaluation.total_connections = len(graph.edges)
        
        if evaluation.total_thoughts > 1:
            max_connections = evaluation.total_thoughts * (evaluation.total_thoughts - 1)
            evaluation.graph_density = evaluation.total_connections / max_connections
            evaluation.average_degree = (2 * evaluation.total_connections) / evaluation.total_thoughts
        
        # Quality metrics
        if graph.nodes:
            confidences = [thought.confidence for thought in graph.nodes.values()]
            relevances = [thought.relevance for thought in graph.nodes.values()]
            
            evaluation.average_confidence = sum(confidences) / len(confidences)
            evaluation.average_relevance = sum(relevances) / len(relevances)
        
        # Thought type distribution
        type_counts = {}
        for thought in graph.nodes.values():
            thought_type = thought.thought_type.value
            type_counts[thought_type] = type_counts.get(thought_type, 0) + 1
        
        evaluation.thought_type_distribution = type_counts
        
        # Reasoning metrics
        evaluation.reasoning_depth = self._calculate_graph_reasoning_depth(graph)
        evaluation.reasoning_breadth = self._calculate_graph_reasoning_breadth(graph)
        evaluation.cycles_detected = len(graph.detect_cycles())
        
        evaluation.evaluation_time = time.time() - start_time
        return evaluation
    
    def _calculate_graph_reasoning_depth(self, graph: ThoughtGraph) -> int:
        """Calculate maximum reasoning depth in the graph"""
        max_depth = 0
        
        for root in graph.get_root_thoughts():
            depth = self._calculate_depth_from_node(graph, root.id, set())
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_depth_from_node(self, graph: ThoughtGraph, node_id: str, visited: Set[str]) -> int:
        """Calculate depth from a specific node"""
        if node_id in visited:
            return 0  # Avoid cycles
        
        visited.add(node_id)
        thought = graph.get_thought(node_id)
        
        if not thought or not thought.children:
            return 1
        
        max_child_depth = 0
        for child_id in thought.children:
            child_depth = self._calculate_depth_from_node(graph, child_id, visited.copy())
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth
    
    def _calculate_graph_reasoning_breadth(self, graph: ThoughtGraph) -> int:
        """Calculate reasoning breadth (maximum thoughts at any level)"""
        # Simple approximation: count thoughts with no parents (roots)
        roots = graph.get_root_thoughts()
        return len(roots)


class ReasoningComparator:
    """
    Compares different reasoning approaches and Graph of Thoughts configurations.
    
    This class enables systematic comparison of different reasoning strategies,
    search algorithms, and thought generation approaches.
    """
    
    def __init__(self):
        self.path_evaluator = PathEvaluator()
        self.graph_evaluator = GraphEvaluator()
    
    def compare_reasoning_paths(self, paths_data: List[Tuple[str, List[str], ThoughtGraph]],
                               metrics: List[EvaluationMetric] = None) -> ComparisonResult:
        """
        Compare multiple reasoning paths.
        
        Args:
            paths_data: List of tuples (approach_name, path, graph)
            metrics: Metrics to compare (default: all)
            
        Returns:
            ComparisonResult with detailed comparison
        """
        if metrics is None:
            metrics = list(EvaluationMetric)
        
        approaches = [data[0] for data in paths_data]
        metric_values = {metric.value: [] for metric in metrics}
        
        # Evaluate each path
        evaluations = []
        for approach_name, path, graph in paths_data:
            evaluation = self.path_evaluator.evaluate_path(path, graph, approach_name)
            evaluations.append((approach_name, evaluation))
            
            # Extract metric values
            for metric in metrics:
                if metric == EvaluationMetric.PATH_QUALITY:
                    metric_values[metric.value].append(evaluation.overall_score)
                elif metric == EvaluationMetric.REASONING_DEPTH:
                    metric_values[metric.value].append(evaluation.reasoning_depth)
                elif metric == EvaluationMetric.CONSISTENCY:
                    metric_values[metric.value].append(evaluation.consistency_score)
                elif metric == EvaluationMetric.NOVELTY:
                    metric_values[metric.value].append(evaluation.novelty_score)
                elif metric == EvaluationMetric.EFFICIENCY:
                    metric_values[metric.value].append(1.0 / max(evaluation.path_length, 1))
                # Add more metrics as needed
        
        # Determine winner
        overall_scores = metric_values.get('path_quality', [0] * len(approaches))
        if overall_scores:
            best_idx = overall_scores.index(max(overall_scores))
            winner = approaches[best_idx]
        else:
            winner = approaches[0] if approaches else "none"
        
        # Calculate statistical significance (simplified)
        significance = self._calculate_statistical_significance(metric_values)
        
        return ComparisonResult(
            approaches=approaches,
            metrics=metric_values,
            winner=winner,
            statistical_significance=significance
        )
    
    def compare_search_algorithms(self, search_results: List[Tuple[str, SearchResult]]) -> Dict[str, Any]:
        """
        Compare search algorithm performance.
        
        Args:
            search_results: List of (algorithm_name, search_result) tuples
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            'algorithms': [],
            'success_rates': [],
            'search_times': [],
            'nodes_explored': [],
            'path_lengths': []
        }
        
        for algo_name, result in search_results:
            comparison['algorithms'].append(algo_name)
            comparison['success_rates'].append(1.0 if result.success else 0.0)
            comparison['search_times'].append(result.search_time)
            comparison['nodes_explored'].append(result.nodes_explored)
            comparison['path_lengths'].append(len(result.path) if result.success else 0)
        
        # Calculate summary statistics
        if comparison['search_times']:
            comparison['avg_search_time'] = statistics.mean(comparison['search_times'])
            comparison['avg_nodes_explored'] = statistics.mean(comparison['nodes_explored'])
            comparison['overall_success_rate'] = statistics.mean(comparison['success_rates'])
        
        return comparison
    
    def _calculate_statistical_significance(self, metric_values: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate statistical significance of differences (simplified)"""
        significance = {}
        
        for metric_name, values in metric_values.items():
            if len(values) >= 2:
                # Simple variance-based significance measure
                mean_val = statistics.mean(values)
                if mean_val > 0:
                    std_val = statistics.stdev(values) if len(values) > 1 else 0
                    significance[metric_name] = 1.0 - min(std_val / mean_val, 1.0)
                else:
                    significance[metric_name] = 0.0
            else:
                significance[metric_name] = 0.0
        
        return significance


class PerformanceAnalyzer:
    """
    Analyzes performance characteristics of Graph of Thoughts implementations.
    
    This class provides detailed performance analysis including timing,
    memory usage, and scalability metrics.
    """
    
    def __init__(self):
        self.performance_logs = []
    
    def analyze_reasoning_performance(self, 
                                   thought_counts: List[int],
                                   reasoning_function: Callable[[int], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance characteristics across different problem sizes.
        
        Args:
            thought_counts: List of thought counts to test
            reasoning_function: Function that performs reasoning given thought count
            
        Returns:
            Performance analysis results
        """
        results = {
            'thought_counts': thought_counts,
            'execution_times': [],
            'memory_usage': [],
            'success_rates': [],
            'quality_scores': []
        }
        
        for count in thought_counts:
            start_time = time.time()
            
            try:
                result = reasoning_function(count)
                execution_time = time.time() - start_time
                
                results['execution_times'].append(execution_time)
                results['success_rates'].append(1.0 if result.get('success', False) else 0.0)
                results['quality_scores'].append(result.get('quality_score', 0.0))
                
                # Simplified memory usage (would need actual memory profiling)
                results['memory_usage'].append(count * 0.001)  # Placeholder
                
            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")
                results['execution_times'].append(float('inf'))
                results['success_rates'].append(0.0)
                results['quality_scores'].append(0.0)
                results['memory_usage'].append(0.0)
        
        # Calculate performance metrics
        if results['execution_times']:
            results['avg_execution_time'] = statistics.mean([t for t in results['execution_times'] if t != float('inf')])
            results['scalability_factor'] = self._calculate_scalability_factor(
                thought_counts, results['execution_times']
            )
        
        return results
    
    def _calculate_scalability_factor(self, sizes: List[int], times: List[float]) -> float:
        """Calculate scalability factor (how execution time grows with size)"""
        if len(sizes) < 2:
            return 1.0
        
        # Simple linear regression to estimate growth rate
        valid_pairs = [(s, t) for s, t in zip(sizes, times) if t != float('inf')]
        if len(valid_pairs) < 2:
            return 1.0
        
        # Calculate approximate growth factor
        time_ratios = []
        size_ratios = []
        
        for i in range(1, len(valid_pairs)):
            size1, time1 = valid_pairs[i-1]
            size2, time2 = valid_pairs[i]
            
            if size1 > 0 and time1 > 0:
                size_ratios.append(size2 / size1)
                time_ratios.append(time2 / time1)
        
        if time_ratios and size_ratios:
            # Average ratio of time growth to size growth
            avg_time_ratio = statistics.mean(time_ratios)
            avg_size_ratio = statistics.mean(size_ratios)
            return avg_time_ratio / avg_size_ratio if avg_size_ratio > 0 else 1.0
        
        return 1.0


# Utility functions for evaluation
def create_evaluation_report(path_evaluations: List[PathEvaluation],
                           graph_evaluation: GraphEvaluation,
                           comparison_results: Optional[ComparisonResult] = None) -> Dict[str, Any]:
    """Create comprehensive evaluation report"""
    
    report = {
        'timestamp': time.time(),
        'graph_evaluation': graph_evaluation.to_dict(),
        'path_evaluations': [eval.to_dict() for eval in path_evaluations],
        'summary_statistics': {}
    }
    
    # Calculate summary statistics
    if path_evaluations:
        overall_scores = [eval.overall_score for eval in path_evaluations]
        confidence_scores = [eval.confidence_score for eval in path_evaluations]
        
        report['summary_statistics'] = {
            'num_paths_evaluated': len(path_evaluations),
            'avg_overall_score': statistics.mean(overall_scores),
            'max_overall_score': max(overall_scores),
            'min_overall_score': min(overall_scores),
            'avg_confidence_score': statistics.mean(confidence_scores),
            'avg_path_length': statistics.mean([eval.path_length for eval in path_evaluations])
        }
    
    if comparison_results:
        report['comparison_results'] = comparison_results.to_dict()
    
    return report


# Example usage and testing
if __name__ == "__main__":
    from .graph_reasoning import ThoughtGraph, ThoughtNode, ThoughtType
    
    # Create test graph
    graph = ThoughtGraph()
    
    # Add test thoughts
    obs = ThoughtNode(content="Observation", thought_type=ThoughtType.OBSERVATION, confidence=0.8, relevance=0.9)
    hyp = ThoughtNode(content="Hypothesis", thought_type=ThoughtType.HYPOTHESIS, confidence=0.6, relevance=0.8)
    ded = ThoughtNode(content="Deduction", thought_type=ThoughtType.DEDUCTION, confidence=0.7, relevance=0.9)
    
    graph.add_thought(obs)
    graph.add_thought(hyp)
    graph.add_thought(ded)
    
    # Create test path
    test_path = [obs.id, hyp.id, ded.id]
    
    # Test evaluators
    path_evaluator = PathEvaluator()
    graph_evaluator = GraphEvaluator()
    
    path_eval = path_evaluator.evaluate_path(test_path, graph)
    graph_eval = graph_evaluator.evaluate_graph(graph)
    
    print(f"Path evaluation score: {path_eval.overall_score:.3f}")
    print(f"Graph total thoughts: {graph_eval.total_thoughts}")
    print(f"Graph average confidence: {graph_eval.average_confidence:.3f}")
    
    # Create evaluation report
    report = create_evaluation_report([path_eval], graph_eval)
    print(f"Evaluation report created with {len(report)} sections")