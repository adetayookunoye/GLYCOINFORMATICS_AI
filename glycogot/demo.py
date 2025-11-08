"""
Graph of Thoughts (GoT) Demo and Testing Suite
==============================================

This module provides demonstrations and comprehensive tests for the
Graph of Thoughts implementation in glycoinformatics.
"""

import asyncio
import json
import time
from typing import Dict, List, Any

from .graph_reasoning import (
    GraphOfThoughts, ThoughtNode, ThoughtType, SearchStrategy
)
from .thought_generators import create_standard_generators, generators_to_functions
from .search_algorithms import (
    SearchAlgorithmFactory, compare_search_algorithms,
    create_goal_condition_from_content
)
from .evaluation import (
    PathEvaluator, GraphEvaluator, create_evaluation_report
)
from .integration import (
    GlycoGoTIntegrator, IntegrationMode, ReasoningTask,
    create_mass_spec_reasoning_request
)


def demo_basic_graph_reasoning():
    """Demonstrate basic graph reasoning functionality"""
    print("🧠 Basic Graph of Thoughts Demo")
    print("=" * 50)
    
    # Create GoT engine
    got = GraphOfThoughts(max_thoughts=50, max_depth=10)
    
    # Set glycoinformatics reasoning goal
    goal_id = got.set_goal(
        "Determine glycan structure from mass spectrometry data",
        {"analysis_type": "MS/MS", "confidence_required": 0.8}
    )
    print(f"✓ Goal set: {goal_id}")
    
    # Add initial observations
    observations = [
        "Precursor ion observed at m/z 1234.567",
        "Prominent fragment at m/z 666.33 with high intensity",
        "Diagnostic fragment at m/z 204.09 indicating HexNAc",
        "Fragment at m/z 342.11 suggesting neutral loss pattern",
        "MS2 fragmentation shows cross-ring cleavages"
    ]
    
    obs_ids = got.add_initial_observations(observations)
    print(f"✓ Added {len(obs_ids)} initial observations")
    
    # Generate thoughts using all standard generators
    generators = create_standard_generators()
    generator_functions = generators_to_functions(generators)
    
    print(f"✓ Using {len(generator_functions)} thought generators")
    
    # Iterative thought generation
    current_thoughts = obs_ids
    for iteration in range(3):
        new_thoughts = got.generate_thoughts(
            source_thoughts=current_thoughts,
            thought_generators=generator_functions,
            max_new_thoughts=10
        )
        print(f"  Iteration {iteration + 1}: Generated {len(new_thoughts)} new thoughts")
        current_thoughts = new_thoughts
        
        if not new_thoughts:
            break
    
    # Search for solution paths
    print("\n🔍 Searching for reasoning paths...")
    paths = got.search_solution_paths(
        strategy=SearchStrategy.BEST_FIRST,
        max_paths=3
    )
    print(f"✓ Found {len(paths)} reasoning paths")
    
    # Evaluate solutions
    if paths:
        evaluations = got.evaluate_solutions(paths)
        best_evaluation = evaluations[0]
        
        print(f"\n🏆 Best reasoning path (score: {best_evaluation['overall_score']:.3f}):")
        explanation = got.explain_reasoning(best_evaluation['path'])
        print(explanation)
    
    # Get comprehensive statistics
    stats = got.get_reasoning_statistics()
    print(f"\n📊 Graph Statistics:")
    print(f"  Total thoughts: {stats['num_nodes']}")
    print(f"  Connections: {stats['num_edges']}")
    print(f"  Average confidence: {stats['avg_confidence']:.3f}")
    print(f"  Graph density: {stats['density']:.3f}")
    
    return got


def demo_search_algorithm_comparison():
    """Demonstrate comparison of different search algorithms"""
    print("\n🔍 Search Algorithm Comparison Demo")
    print("=" * 50)
    
    # Create test graph
    got = GraphOfThoughts(max_thoughts=30)
    goal_id = got.set_goal("Find optimal reasoning path")
    
    # Add structured test data
    obs_ids = got.add_initial_observations([
        "Initial observation A",
        "Initial observation B", 
        "Initial observation C"
    ])
    
    # Generate some thoughts
    generators = generators_to_functions(create_standard_generators()[:3])  # Use first 3
    got.generate_thoughts(obs_ids, generators, max_new_thoughts=15)
    
    # Create goal condition
    goal_condition = create_goal_condition_from_content("optimal")
    
    # Test all search algorithms
    algorithms = SearchAlgorithmFactory.get_all_algorithms()
    results = compare_search_algorithms(
        got.graph, obs_ids, goal_condition, algorithms
    )
    
    print("Algorithm Comparison Results:")
    print("-" * 40)
    for algo_name, result in results.items():
        status = "✓ Success" if result.success else "✗ Failed"
        print(f"{algo_name:25} | {status} | Time: {result.search_time:.4f}s | Nodes: {result.nodes_explored}")
    
    return results


def demo_thought_generators():
    """Demonstrate different thought generators"""
    print("\n🧩 Thought Generator Demo")
    print("=" * 50)
    
    # Create simple graph
    graph = got.graph if 'got' in locals() else GraphOfThoughts().graph
    
    # Create test thought
    test_thought = ThoughtNode(
        content="Fragment ion observed at m/z 204.09 with high intensity",
        thought_type=ThoughtType.OBSERVATION,
        confidence=0.8,
        relevance=0.9,
        evidence_strength=0.7
    )
    graph.add_thought(test_thought)
    
    # Test each generator
    generators = create_standard_generators()
    
    for generator in generators:
        print(f"\n{generator.get_generator_type().upper().replace('_', ' ')}:")
        print("-" * 30)
        
        generated_thoughts = generator.generate(test_thought, graph)
        
        for i, thought in enumerate(generated_thoughts, 1):
            print(f"  {i}. {thought.content[:80]}...")
            print(f"     Type: {thought.thought_type.value}, Confidence: {thought.confidence:.2f}")
            
            # Add to graph for next generators
            graph.add_thought(thought)
    
    print(f"\n✓ Total thoughts generated: {len(graph.nodes)}")
    return generators


async def demo_full_integration():
    """Demonstrate full platform integration"""
    print("\n🔗 Full Integration Demo")
    print("=" * 50)
    
    # Create integrator
    integrator = GlycoGoTIntegrator(
        integration_mode=IntegrationMode.STANDALONE,
        max_thoughts=100,
        max_reasoning_time=30.0
    )
    
    # Create structure elucidation request
    mass_spectrum_data = {
        'precursor_mass': 1647.58,
        'fragments': [
            {'mz': 1485.52, 'intensity': 45, 'annotation': 'loss_of_hexose'},
            {'mz': 1444.50, 'intensity': 30, 'annotation': 'unknown'},
            {'mz': 1282.44, 'intensity': 85, 'annotation': 'disaccharide_loss'},
            {'mz': 1120.38, 'intensity': 60, 'annotation': 'unknown'},
            {'mz': 958.32, 'intensity': 40, 'annotation': 'trisaccharide_loss'},
            {'mz': 666.23, 'intensity': 95, 'annotation': 'reducing_end'},
            {'mz': 504.17, 'intensity': 70, 'annotation': 'unknown'},
            {'mz': 366.14, 'intensity': 80, 'annotation': 'hex_signature'},
            {'mz': 204.09, 'intensity': 100, 'annotation': 'hexnac_signature'}
        ],
        'sample_info': {
            'source': 'human_serum',
            'preparation': 'permethylated',
            'ionization': 'positive_mode'
        }
    }
    
    request = integrator.create_structure_elucidation_request(
        mass_spectrum_data,
        {'analysis_confidence_required': 0.7}
    )
    
    print(f"✓ Created reasoning request: {request.task_id}")
    print(f"  Task type: {request.task_type.value}")
    print(f"  Goal: {request.goal}")
    
    # Process the request
    print("\n🚀 Processing reasoning request...")
    start_time = time.time()
    
    response = await integrator.process_reasoning_request(request)
    
    processing_time = time.time() - start_time
    
    # Display results
    print(f"\n📊 Results (processed in {processing_time:.2f}s):")
    print(f"  Success: {'✓' if response.success else '✗'}")
    print(f"  Confidence: {response.confidence:.3f}")
    print(f"  Reasoning paths found: {len(response.reasoning_paths)}")
    print(f"  Thoughts generated: {response.metadata.get('thoughts_generated', 0)}")
    
    print(f"\n📝 Explanation:")
    print(response.explanation[:300] + "..." if len(response.explanation) > 300 else response.explanation)
    
    # Get integrator statistics
    stats = integrator.get_reasoning_statistics()
    print(f"\n📈 Integration Statistics:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Average execution time: {stats['average_execution_time']:.3f}s")
    
    return response


def demo_evaluation_metrics():
    """Demonstrate evaluation and metrics"""
    print("\n📊 Evaluation Metrics Demo")
    print("=" * 50)
    
    # Create test scenario
    got = GraphOfThoughts()
    goal_id = got.set_goal("Evaluate reasoning quality")
    
    obs_ids = got.add_initial_observations([
        "High-quality observation with strong evidence",
        "Medium-quality observation with some uncertainty", 
        "Lower-quality observation requiring validation"
    ])
    
    # Generate reasoning paths
    generators = generators_to_functions(create_standard_generators()[:4])
    got.generate_thoughts(obs_ids, generators, max_new_thoughts=20)
    
    paths = got.search_solution_paths(SearchStrategy.BEST_FIRST, max_paths=3)
    
    # Evaluate paths
    path_evaluator = PathEvaluator()
    graph_evaluator = GraphEvaluator()
    
    print("Path Evaluations:")
    print("-" * 30)
    
    path_evaluations = []
    for i, path in enumerate(paths):
        evaluation = path_evaluator.evaluate_path(path, got.graph, f"path_{i+1}")
        path_evaluations.append(evaluation)
        
        print(f"Path {i+1}:")
        print(f"  Overall Score: {evaluation.overall_score:.3f}")
        print(f"  Confidence: {evaluation.confidence_score:.3f}")
        print(f"  Evidence: {evaluation.evidence_score:.3f}")
        print(f"  Consistency: {evaluation.consistency_score:.3f}")
        print(f"  Length: {evaluation.path_length}")
        print()
    
    # Evaluate entire graph
    graph_evaluation = graph_evaluator.evaluate_graph(got.graph)
    
    print("Graph Evaluation:")
    print("-" * 20)
    print(f"Total thoughts: {graph_evaluation.total_thoughts}")
    print(f"Total connections: {graph_evaluation.total_connections}")
    print(f"Graph density: {graph_evaluation.graph_density:.3f}")
    print(f"Average confidence: {graph_evaluation.average_confidence:.3f}")
    print(f"Reasoning depth: {graph_evaluation.reasoning_depth}")
    print(f"Cycles detected: {graph_evaluation.cycles_detected}")
    
    # Create comprehensive report
    report = create_evaluation_report(path_evaluations, graph_evaluation)
    
    print(f"\n📋 Evaluation Report Summary:")
    summary = report['summary_statistics']
    print(f"  Paths evaluated: {summary['num_paths_evaluated']}")
    print(f"  Max score: {summary['max_overall_score']:.3f}")
    print(f"  Average score: {summary['avg_overall_score']:.3f}")
    print(f"  Average path length: {summary['avg_path_length']:.1f}")
    
    return report


def run_comprehensive_demo():
    """Run comprehensive demonstration of all features"""
    print("🚀 Comprehensive Graph of Thoughts Demo")
    print("=" * 60)
    print("This demo showcases the complete GoT implementation")
    print("for glycoinformatics reasoning tasks.\n")
    
    try:
        # Run all demos
        demo1_result = demo_basic_graph_reasoning()
        demo2_result = demo_search_algorithm_comparison()
        demo3_result = demo_thought_generators()
        
        # Async demo
        print("\n⏳ Running integration demo...")
        demo4_result = asyncio.run(demo_full_integration())
        
        demo5_result = demo_evaluation_metrics()
        
        # Summary
        print("\n🎯 Demo Summary")
        print("=" * 30)
        print("✓ Basic graph reasoning: Complete")
        print("✓ Search algorithms: Complete")
        print("✓ Thought generators: Complete") 
        print("✓ Full integration: Complete")
        print("✓ Evaluation metrics: Complete")
        
        print(f"\n🏆 All demonstrations completed successfully!")
        print(f"   Total thoughts across demos: ~{demo1_result.graph.calculate_graph_metrics()['num_nodes']}")
        print(f"   Integration success rate: {demo4_result.success}")
        
        return {
            'basic_reasoning': demo1_result,
            'search_comparison': demo2_result,
            'thought_generation': demo3_result,
            'integration': demo4_result,
            'evaluation': demo5_result
        }
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_performance():
    """Benchmark GoT performance across different problem sizes"""
    print("\n⚡ Performance Benchmark")
    print("=" * 40)
    
    thought_counts = [10, 25, 50, 100]
    results = {
        'sizes': thought_counts,
        'times': [],
        'memory': [],
        'success_rates': []
    }
    
    for size in thought_counts:
        print(f"Testing with {size} max thoughts...")
        start_time = time.time()
        
        try:
            got = GraphOfThoughts(max_thoughts=size)
            goal_id = got.set_goal(f"Benchmark test for {size} thoughts")
            
            obs_ids = got.add_initial_observations([
                f"Benchmark observation {i}" for i in range(min(5, size//5))
            ])
            
            generators = generators_to_functions(create_standard_generators()[:3])
            got.generate_thoughts(obs_ids, generators, max_new_thoughts=size//2)
            
            paths = got.search_solution_paths(SearchStrategy.BEST_FIRST, max_paths=2)
            
            execution_time = time.time() - start_time
            success = len(paths) > 0
            
            results['times'].append(execution_time)
            results['success_rates'].append(1.0 if success else 0.0)
            results['memory'].append(len(got.graph.nodes) * 0.001)  # Approximation
            
            print(f"  ✓ Completed in {execution_time:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results['times'].append(float('inf'))
            results['success_rates'].append(0.0)
            results['memory'].append(0.0)
    
    # Print summary
    print(f"\nBenchmark Summary:")
    print("-" * 20)
    for i, size in enumerate(thought_counts):
        time_val = results['times'][i]
        success_val = results['success_rates'][i]
        time_str = f"{time_val:.3f}s" if time_val != float('inf') else "Failed"
        
        print(f"Size {size:3d}: {time_str:>8} | Success: {'✓' if success_val else '✗'}")
    
    return results


if __name__ == "__main__":
    # Run the comprehensive demo
    results = run_comprehensive_demo()
    
    if results:
        print("\n🔬 Running performance benchmark...")
        benchmark_results = benchmark_performance()
        
        print(f"\n🎉 All demos and benchmarks completed!")
        print(f"   Check the results for detailed analysis.")
    else:
        print("\n❌ Demo suite failed. Check error messages above.")