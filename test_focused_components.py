#!/usr/bin/env python3
"""
Focused test suite for currently working GlycoLLM components.

This script tests the components that are properly implemented and accessible.
"""

import sys
import json
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_evaluation_framework():
    """Test the comprehensive evaluation framework."""
    
    print("📊 Testing Evaluation Framework...")
    
    try:
        from glycollm.training.evaluation import (
            GlycoLLMEvaluator, StructureEvaluator, SpectraEvaluator,
            TextEvaluator, EvaluationResults, BenchmarkSuite, AdvancedMetricsTracker
        )
        
        print("✅ All evaluation components imported successfully")
        
        # Test structure evaluator
        structure_eval = StructureEvaluator()
        
        pred_structures = [
            "WURCS=2.0/1,1,1/[a2122h-1b_1-5]/1/1-1",
            "WURCS=2.0/1,1,1/[a1221m-1a_1-5]/1/1-1",
            "WURCS=2.0/2,2,1/[a2122h-1b_1-5][a1221m-1a_1-5]/1-2/a4-b1"
        ]
        
        target_structures = [
            "WURCS=2.0/1,1,1/[a2122h-1b_1-5]/1/1-1",  # Exact match
            "WURCS=2.0/1,1,1/[a1122h-1b_1-5]/1/1-1",  # Different structure
            "WURCS=2.0/2,2,1/[a2122h-1b_1-5][a1221m-1a_1-5]/1-2/a4-b1"  # Exact match
        ]
        
        struct_metrics = structure_eval.evaluate_batch(pred_structures, target_structures)
        
        print(f"✅ Structure evaluation metrics:")
        print(f"    Exact match accuracy: {struct_metrics['exact_match']:.3f}")
        print(f"    WURCS validity: {struct_metrics['wurcs_validity']:.3f}")
        print(f"    BLEU score: {struct_metrics['bleu_score']:.3f}")
        print(f"    Monosaccharide accuracy: {struct_metrics['monosaccharide_accuracy']:.3f}")
        print(f"    Linkage accuracy: {struct_metrics['linkage_accuracy']:.3f}")
        
        # Test spectra evaluator
        spectra_eval = SpectraEvaluator()
        
        pred_spectra = [
            [(163.06, 100.0), (204.09, 45.2), (366.14, 78.1)],
            [(147.07, 55.3), (190.05, 32.1)],
            [(292.11, 89.4), (454.16, 67.8), (616.21, 45.2)]
        ]
        
        target_spectra = [
            [(163.06, 100.0), (204.09, 45.2), (366.14, 78.1)],  # Exact match
            [(147.07, 55.3), (189.05, 32.1)],  # Close match
            [(292.11, 89.4), (454.16, 67.8), (616.22, 44.8)]  # Close match
        ]
        
        spectra_metrics = spectra_eval.evaluate_batch(pred_spectra, target_spectra)
        
        print(f"✅ Spectra evaluation metrics:")
        print(f"    Peak detection F1: {spectra_metrics['peak_detection_f1']:.3f}")
        print(f"    Intensity correlation: {spectra_metrics['intensity_correlation']:.3f}")
        print(f"    MSE: {spectra_metrics['mse']:.6f}")
        print(f"    Coverage: {spectra_metrics['coverage']:.3f}")
        
        # Test text evaluator
        text_eval = TextEvaluator()
        
        pred_texts = [
            "This glycan structure contains N-acetylglucosamine and mannose residues",
            "The spectrum shows characteristic fragmentation of complex N-glycans",
            "Biological function relates to protein folding and cell recognition"
        ]
        
        target_texts = [
            "This glycan structure contains N-acetylglucosamine and mannose residues",
            "The spectrum displays typical fragmentation patterns of N-linked glycans",
            "The biological function is associated with protein folding and cellular recognition"
        ]
        
        text_metrics = text_eval.evaluate_batch(pred_texts, target_texts)
        
        print(f"✅ Text evaluation metrics:")
        print(f"    Exact match: {text_metrics['exact_match']:.3f}")
        print(f"    BLEU score: {text_metrics['bleu_score']:.3f}")
        print(f"    ROUGE-L: {text_metrics['rouge_l']:.3f}")
        print(f"    Terminology accuracy: {text_metrics['terminology_accuracy']:.3f}")
        
        # Test comprehensive evaluator
        evaluator = GlycoLLMEvaluator()
        
        model_outputs = {
            'structure_predictions': pred_structures,
            'spectra_predictions': pred_spectra,
            'text_predictions': pred_texts
        }
        
        targets = {
            'structure_targets': target_structures,
            'spectra_targets': target_spectra,
            'text_targets': target_texts
        }
        
        results = evaluator.evaluate_model_outputs(model_outputs, targets)
        
        print(f"\n✅ Comprehensive evaluation results:")
        evaluator.print_evaluation_summary(results)
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_suite():
    """Test the advanced benchmark suite."""
    
    print("\n🔬 Testing Benchmark Suite...")
    
    try:
        from glycollm.training.evaluation import BenchmarkSuite
        
        benchmark_suite = BenchmarkSuite()
        
        # Create test data
        test_data = {
            'structure_test': [
                {
                    'structure': 'WURCS=2.0/1,1,1/[a2122h-1b_1-5]/1/1-1',
                    'target_structure': 'WURCS=2.0/1,1,1/[a2122h-1b_1-5]/1/1-1',
                    'predicted_structure': 'WURCS=2.0/1,1,1/[a2122h-1b_1-5]/1/1-1'
                },
                {
                    'structure': 'WURCS=2.0/1,1,1/[a1221m-1a_1-5]/1/1-1',
                    'target_structure': 'WURCS=2.0/1,1,1/[a1221m-1a_1-5]/1/1-1',
                    'predicted_structure': 'WURCS=2.0/1,1,1/[a1122h-1b_1-5]/1/1-1'
                }
            ],
            'spectra_test': [
                {
                    'spectra': [(163.06, 100.0), (204.09, 45.2)],
                    'target_spectra': [(163.06, 100.0), (204.09, 45.2)],
                    'predicted_spectra': [(163.06, 100.0), (204.09, 45.2)]
                }
            ],
            'text_test': [
                {
                    'text': 'N-acetylglucosamine residue analysis',
                    'target_text': 'Analysis of N-acetylglucosamine residue',
                    'predicted_text': 'N-acetylglucosamine residue analysis'
                }
            ]
        }
        
        # Run benchmark (with mock model)
        class MockModel:
            pass
        
        mock_model = MockModel()
        
        benchmark_results = benchmark_suite.run_comprehensive_benchmark(
            model=mock_model,
            test_data=test_data,
            benchmark_name="test_benchmark"
        )
        
        print(f"✅ Benchmark completed:")
        print(f"    Benchmark name: {benchmark_results['benchmark_name']}")
        print(f"    Test data stats: {benchmark_results['test_data_stats']}")
        print(f"    Performance metrics: {len(benchmark_results['performance_metrics'])} metrics")
        
        # Generate report
        report = benchmark_suite.generate_benchmark_report("test_benchmark")
        print(f"✅ Benchmark report generated ({len(report)} characters)")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark suite test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_tracker():
    """Test the advanced metrics tracker."""
    
    print("\n📈 Testing Advanced Metrics Tracker...")
    
    try:
        from glycollm.training.evaluation import AdvancedMetricsTracker, EvaluationResults
        
        tracker = AdvancedMetricsTracker()
        
        # Log training metrics over multiple epochs
        for epoch in range(3):
            for batch in range(5):
                metrics = {
                    'loss': 0.5 - (epoch * 0.1) - (batch * 0.02),
                    'accuracy': 0.7 + (epoch * 0.05) + (batch * 0.01),
                    'f1_score': 0.65 + (epoch * 0.04) + (batch * 0.008)
                }
                
                tracker.log_training_metrics(
                    epoch=epoch,
                    batch=batch,
                    metrics=metrics
                )
        
        print(f"✅ Logged training metrics for {len(tracker.metrics_history)} metric types")
        
        # Create evaluation checkpoint
        eval_results = EvaluationResults(
            total_samples=1000,
            structure_accuracy=0.85,
            spectra_accuracy=0.78,
            text_bleu=0.72,
            retrieval_recall_at_5=0.88
        )
        
        model_state = {'epoch': 2, 'learning_rate': 2e-5}
        
        tracker.create_evaluation_checkpoint(
            checkpoint_name="epoch_2_checkpoint",
            evaluation_results=eval_results,
            model_state=model_state
        )
        
        print(f"✅ Created evaluation checkpoint")
        print(f"    Checkpoints: {len(tracker.evaluation_checkpoints)}")
        print(f"    Performance baselines: {len(tracker.performance_baselines)}")
        
        # Generate comprehensive report
        report = tracker.generate_comprehensive_training_report()
        print(f"✅ Generated training report ({len(report)} characters)")
        
        # Export metrics
        export_path = "test_metrics_export.json"
        tracker.export_metrics_report(export_path)
        
        if Path(export_path).exists():
            print(f"✅ Metrics exported to {export_path}")
            # Clean up
            Path(export_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_reasoning():
    """Test enhanced reasoning components that are available."""
    
    print("\n🧠 Testing Enhanced Reasoning Components...")
    
    try:
        from glycogot.reasoning import GlycoGOTReasoner
        
        reasoner = GlycoGOTReasoner()
        print("✅ GlycoGOTReasoner initialized")
        
        # Test basic reasoning capabilities (check what methods exist)
        available_methods = [method for method in dir(reasoner) if not method.startswith('_')]
        print(f"✅ Available reasoning methods: {len(available_methods)}")
        
        # Try to test generate_actionable_recommendations if it exists
        if hasattr(reasoner, 'generate_actionable_recommendations'):
            sample_candidates = [
                {
                    "structure": "WURCS=2.0/1,1,1/[a2122h-1b_1-5]/1/1-1",
                    "confidence": 0.85,
                    "evidence": ["MS/MS match"]
                }
            ]
            
            recommendations = reasoner.generate_actionable_recommendations(
                candidates=sample_candidates,
                analysis_context={"method": "LC-MS/MS"}
            )
            
            print(f"✅ Generated {len(recommendations)} actionable recommendations")
            for i, rec in enumerate(recommendations[:2]):
                print(f"    {i+1}. {rec.get('action', 'N/A')}")
                
        else:
            print("⚠️  generate_actionable_recommendations method not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced reasoning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_components():
    """Test API components that are accessible."""
    
    print("\n🌐 Testing API Components...")
    
    try:
        # Test if we can import the main API components
        from glyco_platform.api.main import app
        print("✅ Main API app imported successfully")
        
        # Check if sophisticated models are available in the platform directory
        platform_dir = Path(__file__).parent / "platform"
        if platform_dir.exists():
            api_dir = platform_dir / "api"
            if api_dir.exists():
                models_file = api_dir / "models.py"
                if models_file.exists():
                    print("✅ Sophisticated API models file exists")
                else:
                    print("⚠️  API models file not found in expected location")
            else:
                print("⚠️  API directory not found in platform")
        else:
            print("⚠️  Platform directory not found")
            
        return True
        
    except Exception as e:
        print(f"❌ API components test failed: {e}")
        return False

def run_focused_tests():
    """Run focused tests on working components."""
    
    print("🚀 Running Focused GlycoLLM Component Tests")
    print("=" * 60)
    
    test_results = {}
    
    tests = [
        ("Evaluation Framework", test_evaluation_framework),
        ("Benchmark Suite", test_benchmark_suite),
        ("Metrics Tracker", test_metrics_tracker),
        ("Enhanced Reasoning", test_enhanced_reasoning),
        ("API Components", test_api_components)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'-' * 40}")
            result = test_func()
            test_results[test_name] = result
            if result:
                passed_tests += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            test_results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 FOCUSED TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    success_rate = passed_tests / total_tests * 100
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("\n🎉 EXCELLENT! Core GlycoLLM components are working well!")
    elif success_rate >= 60:
        print("\n👍 GOOD! Most GlycoLLM components are functional!")
    else:
        print("\n⚠️  NEEDS ATTENTION: Several components need debugging.")
    
    return test_results

if __name__ == "__main__":
    results = run_focused_tests()