"""
Graph of Thoughts (GoT) Implementation for Glycoinformatics
===========================================================

This package implements a true Graph of Thoughts reasoning system specifically
designed for glycoinformatics applications. It provides graph-based reasoning
capabilities with nodes representing atomic thoughts and edges representing
logical relationships.

Key Components:
- ThoughtNode: Individual atomic thoughts with confidence and evidence
- ThoughtGraph: Graph structure managing thought relationships
- GraphOfThoughts: Main reasoning engine orchestrating the process
- Search Algorithms: BFS, DFS, Best-First, A*, Monte Carlo Tree Search
- Thought Generators: Deductive, Inductive, Analogical, Causal reasoning
- Evaluation: Comprehensive metrics for reasoning quality assessment
- Integration: Platform integration with LLM and knowledge graph support

This implementation follows established Graph of Thoughts research paradigms
and provides proper graph data structures and search algorithms rather than
sequential reasoning chains.
"""

from .graph_reasoning import (
    # Core classes
    ThoughtNode,
    ThoughtGraph, 
    GraphOfThoughts,
    
    # Enums and types
    ThoughtType,
    ThoughtStatus,
    SearchStrategy
)

from .thought_generators import (
    # Generator classes
    ThoughtGenerator,
    DeductiveReasoningGenerator,
    InductiveReasoningGenerator,
    AnalogicalReasoningGenerator,
    CausalReasoningGenerator,
    SynthesisGenerator,
    EvaluationGenerator,
    HypothesisGenerator,
    
    # Factory functions
    create_standard_generators,
    generators_to_functions
)

from .search_algorithms import (
    # Algorithm classes
    SearchAlgorithm,
    BreadthFirstSearch,
    DepthFirstSearch,
    BestFirstSearch,
    AStarSearch,
    MonteCarloTreeSearch,
    RandomWalkSearch,
    
    # Result classes
    SearchResult,
    
    # Factory and utilities
    SearchAlgorithmFactory,
    compare_search_algorithms,
    create_goal_condition_from_content,
    create_goal_condition_from_type,
    create_goal_condition_high_confidence
)

from .evaluation import (
    # Evaluation classes
    PathEvaluator,
    GraphEvaluator,
    ReasoningComparator,
    PerformanceAnalyzer,
    
    # Result classes
    PathEvaluation,
    GraphEvaluation,
    ComparisonResult,
    
    # Enums
    EvaluationMetric,
    
    # Utilities
    create_evaluation_report
)

from .integration import (
    # Integration classes
    GlycoGoTIntegrator,
    
    # Request/Response classes
    ReasoningRequest,
    ReasoningResponse,
    
    # Enums
    IntegrationMode,
    ReasoningTask,
    
    # Utilities
    create_mass_spec_reasoning_request,
    quick_structure_analysis
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Glycoinformatics AI Research Team"
__email__ = "research@glycoinformatics.ai"
__description__ = "Graph of Thoughts reasoning engine for glycoinformatics applications"

# Export all public APIs
__all__ = [
    # Core graph reasoning
    'ThoughtNode',
    'ThoughtGraph',
    'GraphOfThoughts',
    'ThoughtType',
    'ThoughtStatus',
    'SearchStrategy',
    
    # Thought generation
    'ThoughtGenerator',
    'DeductiveReasoningGenerator',
    'InductiveReasoningGenerator', 
    'AnalogicalReasoningGenerator',
    'CausalReasoningGenerator',
    'SynthesisGenerator',
    'EvaluationGenerator',
    'HypothesisGenerator',
    'create_standard_generators',
    'generators_to_functions',
    
    # Search algorithms
    'SearchAlgorithm',
    'BreadthFirstSearch',
    'DepthFirstSearch',
    'BestFirstSearch',
    'AStarSearch', 
    'MonteCarloTreeSearch',
    'RandomWalkSearch',
    'SearchResult',
    'SearchAlgorithmFactory',
    'compare_search_algorithms',
    'create_goal_condition_from_content',
    'create_goal_condition_from_type',
    'create_goal_condition_high_confidence',
    
    # Evaluation and metrics
    'PathEvaluator',
    'GraphEvaluator',
    'ReasoningComparator',
    'PerformanceAnalyzer',
    'PathEvaluation',
    'GraphEvaluation',
    'ComparisonResult',
    'EvaluationMetric',
    'create_evaluation_report',
    
    # Platform integration
    'GlycoGoTIntegrator',
    'ReasoningRequest',
    'ReasoningResponse',
    'IntegrationMode',
    'ReasoningTask',
    'create_mass_spec_reasoning_request',
    'quick_structure_analysis',
]


def create_got_engine(max_thoughts: int = 1000, max_depth: int = 20) -> GraphOfThoughts:
    """
    Create a configured Graph of Thoughts engine.
    
    Args:
        max_thoughts: Maximum number of thoughts to generate
        max_depth: Maximum reasoning depth
        
    Returns:
        Configured GraphOfThoughts instance
    """
    return GraphOfThoughts(max_thoughts=max_thoughts, max_depth=max_depth)


def create_glyco_integrator(integration_mode: IntegrationMode = IntegrationMode.STANDALONE,
                           **kwargs) -> GlycoGoTIntegrator:
    """
    Create a configured glycoinformatics integrator.
    
    Args:
        integration_mode: How to integrate with the platform
        **kwargs: Additional configuration options
        
    Returns:
        Configured GlycoGoTIntegrator instance
    """
    return GlycoGoTIntegrator(integration_mode=integration_mode, **kwargs)


def quick_example_usage():
    """
    Demonstrate quick usage of the Graph of Thoughts engine.
    
    This function shows how to use the basic functionality for
    glycoinformatics reasoning tasks.
    """
    # Create GoT engine
    got = create_got_engine()
    
    # Set reasoning goal
    goal_id = got.set_goal("Identify glycan structure from mass spectrum")
    
    # Add observations
    observations = [
        "Precursor ion at m/z 1234.56",
        "Fragment at m/z 666.33 suggests disaccharide loss",
        "Fragment at m/z 204.09 indicates HexNAc residue"
    ]
    obs_ids = got.add_initial_observations(observations)
    
    # Generate thoughts using standard generators
    generators = generators_to_functions(create_standard_generators())
    new_thoughts = got.generate_thoughts(obs_ids, generators, max_new_thoughts=10)
    
    # Search for solution paths
    paths = got.search_solution_paths(SearchStrategy.BEST_FIRST, max_paths=3)
    
    # Evaluate solutions
    if paths:
        evaluations = got.evaluate_solutions(paths)
        best_path = evaluations[0]['path'] if evaluations else []
        explanation = got.explain_reasoning(best_path)
        
        return {
            'success': len(best_path) > 0,
            'reasoning_path': best_path,
            'explanation': explanation,
            'graph_stats': got.get_reasoning_statistics()
        }
    
    return {'success': False, 'message': 'No reasoning paths found'}


# Module-level configuration
import logging

# Set up logging for the package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info(f"GlycoGoT package initialized - version {__version__}")