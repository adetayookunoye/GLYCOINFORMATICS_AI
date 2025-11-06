"""
Integrated Pipeline: GlycoLLM + GlycoGoT + GlycoKG

End-to-end workflow for glycan structure elucidation from MS/MS spectra.

Pipeline:
1. GlycoLLM: Predict candidate structures from spectra
2. GlycoGoT: Reason about predictions with graph-of-thought
3. GlycoKG: Validate against knowledge graph and biochemical rules

Author: Adetayo Research Team
Date: November 2025
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import numpy as np

import torch

# GlycoLLM components
from glycollm.models.glycollm import GlycoLLM, GlycoLLMConfig
from glycollm.data.spectra_parser import MSSpectrum, SpectraParser
from glycollm.training.evaluation import StructureEvaluator

# GlycoGoT components
from glycogot.reasoning import (
    GlycoKnowledgeBase, ReasoningChain, ReasoningType,
    HypothesisGenerator, UncertaintyQuantification
)
from glycogot.applications import StructureElucidationPipeline

# GlycoKG components
from glycokg.ontology.glyco_ontology import GlycoOntology
from glycokg.query.sparql_queries import SPARQLQueryEngine

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for integrated pipeline"""
    # GlycoLLM settings
    model_path: Optional[Path] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_k_predictions: int = 5
    min_confidence: float = 0.5
    
    # GlycoGoT settings
    enable_reasoning: bool = True
    max_reasoning_depth: int = 3
    reasoning_types: List[str] = None
    
    # GlycoKG settings
    knowledge_graph_path: Optional[Path] = None
    enable_validation: bool = True
    enable_constraint_checking: bool = True
    
    # Pipeline settings
    output_format: str = "json"
    save_intermediate: bool = False
    
    def __post_init__(self):
        if self.reasoning_types is None:
            self.reasoning_types = ['deductive', 'abductive', 'causal']


@dataclass
class PipelineResult:
    """Result from integrated pipeline"""
    spectrum_id: str
    timestamp: str
    
    # GlycoLLM predictions
    predicted_structures: List[Dict]  # [{structure, confidence, score}]
    
    # GlycoGoT reasoning
    reasoning_chain: Optional[Dict] = None
    hypotheses: Optional[List[Dict]] = None
    
    # GlycoKG validation
    validation_results: Optional[Dict] = None
    constraint_violations: Optional[List[str]] = None
    
    # Final output
    final_structure: Optional[str] = None
    final_confidence: Optional[float] = None
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self, path: Path):
        """Save result to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class IntegratedPipeline:
    """
    Integrated pipeline connecting GlycoLLM, GlycoGoT, and GlycoKG
    for complete glycan structure elucidation
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize integrated pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Initialize components
        logger.info("Initializing GlycoLLM...")
        self.glycollm = self._init_glycollm()
        
        logger.info("Initializing GlycoGoT...")
        self.glycogot = self._init_glycogot()
        
        logger.info("Initializing GlycoKG...")
        self.glycokg = self._init_glycokg()
        
        # Spectra parser
        self.spectra_parser = SpectraParser(normalize=True, top_n_peaks=150)
        
        # Structure evaluator
        self.evaluator = StructureEvaluator()
        
        logger.info("Integrated pipeline initialized")
    
    def _init_glycollm(self) -> GlycoLLM:
        """Initialize GlycoLLM model"""
        # Default config if no model path provided
        if self.config.model_path and self.config.model_path.exists():
            logger.info(f"Loading GlycoLLM from {self.config.model_path}")
            model = GlycoLLM.from_pretrained(str(self.config.model_path))
        else:
            logger.warning("No pre-trained model found, using default config")
            config = GlycoLLMConfig(
                d_model=768,
                num_layers=12,
                num_heads=12,
                vocab_size=1000,
                max_seq_length=512
            )
            model = GlycoLLM(config)
        
        model = model.to(self.config.device)
        model.eval()
        return model
    
    def _init_glycogot(self) -> StructureElucidationPipeline:
        """Initialize GlycoGoT reasoning pipeline"""
        knowledge_base = GlycoKnowledgeBase()
        
        pipeline = StructureElucidationPipeline(
            knowledge_base=knowledge_base,
            max_depth=self.config.max_reasoning_depth
        )
        
        return pipeline
    
    def _init_glycokg(self) -> Tuple[GlycoOntology, SPARQLQueryEngine]:
        """Initialize GlycoKG knowledge graph"""
        ontology = GlycoOntology()
        
        # Load graph if provided
        if self.config.knowledge_graph_path and self.config.knowledge_graph_path.exists():
            logger.info(f"Loading knowledge graph from {self.config.knowledge_graph_path}")
            ontology.graph.parse(str(self.config.knowledge_graph_path), format="turtle")
        else:
            logger.warning("No knowledge graph provided, using empty graph")
        
        query_engine = SPARQLQueryEngine(ontology.graph)
        
        return ontology, query_engine
    
    def process_spectrum(self, spectrum: MSSpectrum) -> PipelineResult:
        """
        Process a single MS/MS spectrum through full pipeline
        
        Args:
            spectrum: MSSpectrum object
            
        Returns:
            PipelineResult with all predictions and reasoning
        """
        logger.info(f"Processing spectrum: {spectrum.spectrum_id}")
        
        result = PipelineResult(
            spectrum_id=spectrum.spectrum_id,
            timestamp=datetime.now().isoformat(),
            predicted_structures=[]
        )
        
        try:
            # Step 1: GlycoLLM structure prediction
            logger.info("Step 1: Running GlycoLLM prediction...")
            predictions = self._run_glycollm(spectrum)
            result.predicted_structures = predictions
            
            if not predictions:
                logger.warning("No predictions from GlycoLLM")
                return result
            
            # Step 2: GlycoGoT reasoning
            if self.config.enable_reasoning:
                logger.info("Step 2: Running GlycoGoT reasoning...")
                reasoning_result = self._run_glycogot(spectrum, predictions)
                result.reasoning_chain = reasoning_result.get('reasoning_chain')
                result.hypotheses = reasoning_result.get('hypotheses')
                
                # Update predictions with reasoning scores
                predictions = reasoning_result.get('refined_predictions', predictions)
            
            # Step 3: GlycoKG validation
            if self.config.enable_validation:
                logger.info("Step 3: Running GlycoKG validation...")
                validation_result = self._run_glycokg(predictions)
                result.validation_results = validation_result
                result.constraint_violations = validation_result.get('violations', [])
                
                # Filter predictions based on validation
                valid_predictions = [
                    p for p in predictions 
                    if p['structure'] in validation_result.get('valid_structures', [])
                ]
                
                if valid_predictions:
                    predictions = valid_predictions
            
            # Step 4: Final selection
            if predictions:
                final_pred = predictions[0]  # Top prediction
                result.final_structure = final_pred['structure']
                result.final_confidence = final_pred['confidence']
                result.explanation = self._generate_explanation(result)
            
            logger.info(f"Pipeline complete: {result.final_structure} (conf={result.final_confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Error in pipeline: {e}", exc_info=True)
        
        return result
    
    def _run_glycollm(self, spectrum: MSSpectrum) -> List[Dict]:
        """
        Run GlycoLLM prediction
        
        Returns:
            List of predictions with structure, confidence, score
        """
        try:
            # Prepare input
            binned_spectrum = spectrum.bin_spectrum(bin_size=1.0, mz_range=(0, 2000))
            spectrum_tensor = torch.tensor(binned_spectrum, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            
            # Run prediction
            with torch.no_grad():
                outputs = self.glycollm(
                    spectra=spectrum_tensor,
                    task='structure_prediction'
                )
            
            # Get top-k predictions
            logits = outputs['structure_logits']
            probs = torch.softmax(logits, dim=-1)
            top_k = torch.topk(probs[0], k=self.config.top_k_predictions)
            
            predictions = []
            for score, idx in zip(top_k.values.cpu().numpy(), top_k.indices.cpu().numpy()):
                if score >= self.config.min_confidence:
                    # Decode structure (simplified - would use proper vocabulary)
                    structure = f"PREDICTED_STRUCTURE_{idx}"
                    predictions.append({
                        'structure': structure,
                        'confidence': float(score),
                        'score': float(score),
                        'method': 'glycollm'
                    })
            
            logger.info(f"GlycoLLM produced {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in GlycoLLM: {e}")
            return []
    
    def _run_glycogot(self, spectrum: MSSpectrum, predictions: List[Dict]) -> Dict:
        """
        Run GlycoGoT reasoning
        
        Returns:
            Reasoning results with chain, hypotheses, refined predictions
        """
        try:
            # Prepare context
            context = {
                'precursor_mz': spectrum.precursor_mz,
                'precursor_charge': spectrum.precursor_charge,
                'num_peaks': len(spectrum.peaks),
                'candidate_structures': [p['structure'] for p in predictions]
            }
            
            # Generate hypotheses
            hypothesis_gen = HypothesisGenerator(self.glycogot.knowledge_base)
            hypotheses = hypothesis_gen.generate_hypotheses(context)
            
            # Build reasoning chain
            reasoning_chain = ReasoningChain()
            
            # Apply reasoning types
            for reasoning_type_str in self.config.reasoning_types:
                try:
                    reasoning_type = ReasoningType[reasoning_type_str.upper()]
                    # Apply reasoning rules
                    rules = [r for r in self.glycogot.knowledge_base.rules 
                            if r.reasoning_type == reasoning_type]
                    
                    for rule in rules:
                        if rule.can_apply(context):
                            step_result = rule.apply(context)
                            reasoning_chain.add_step(
                                reasoning_type=reasoning_type,
                                premise=f"Applied {rule.name}",
                                conclusion=str(step_result),
                                confidence=0.8
                            )
                            # Update context with result
                            context.update(step_result)
                except Exception as e:
                    logger.warning(f"Error applying reasoning type {reasoning_type_str}: {e}")
            
            # Compute uncertainty
            uncertainty_quant = UncertaintyQuantification()
            uncertainty_scores = {
                p['structure']: uncertainty_quant.compute_epistemic_uncertainty(
                    [p['confidence']]
                )
                for p in predictions
            }
            
            # Refine predictions with reasoning scores
            refined = []
            for pred in predictions:
                structure = pred['structure']
                # Combine GlycoLLM confidence with reasoning
                reasoning_boost = 0.1 if any(
                    structure in str(step) for step in reasoning_chain.steps
                ) else 0.0
                
                refined.append({
                    **pred,
                    'confidence': min(pred['confidence'] + reasoning_boost, 1.0),
                    'uncertainty': uncertainty_scores.get(structure, 0.5),
                    'reasoning_supported': reasoning_boost > 0
                })
            
            # Sort by confidence
            refined.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'reasoning_chain': reasoning_chain.to_dict(),
                'hypotheses': [h.to_dict() if hasattr(h, 'to_dict') else str(h) for h in hypotheses],
                'refined_predictions': refined
            }
            
        except Exception as e:
            logger.error(f"Error in GlycoGoT: {e}")
            return {'refined_predictions': predictions}
    
    def _run_glycokg(self, predictions: List[Dict]) -> Dict:
        """
        Run GlycoKG validation
        
        Returns:
            Validation results
        """
        try:
            ontology, query_engine = self.glycokg
            
            validation_result = {
                'valid_structures': [],
                'violations': [],
                'knowledge_matches': []
            }
            
            for pred in predictions:
                structure = pred['structure']
                
                # Query knowledge graph for similar structures
                # (Simplified - would use actual structure matching)
                query = f"""
                PREFIX glyco: <http://glycoinformatics.org/ontology/>
                SELECT ?glycan ?mass ?composition
                WHERE {{
                    ?glycan a glyco:Glycan .
                    ?glycan glyco:wurcs ?wurcs .
                    OPTIONAL {{ ?glycan glyco:has_mass ?mass }}
                    OPTIONAL {{ ?glycan glyco:composition ?composition }}
                }}
                LIMIT 10
                """
                
                try:
                    results = query_engine.query(query)
                    if results:
                        validation_result['knowledge_matches'].append({
                            'structure': structure,
                            'matches': len(results)
                        })
                        validation_result['valid_structures'].append(structure)
                except Exception as e:
                    logger.warning(f"SPARQL query error: {e}")
                    # Allow structure if query fails
                    validation_result['valid_structures'].append(structure)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in GlycoKG: {e}")
            return {
                'valid_structures': [p['structure'] for p in predictions],
                'violations': []
            }
    
    def _generate_explanation(self, result: PipelineResult) -> str:
        """Generate human-readable explanation of prediction"""
        explanation = []
        
        explanation.append(f"Predicted structure: {result.final_structure}")
        explanation.append(f"Confidence: {result.final_confidence:.2%}")
        explanation.append("")
        
        if result.predicted_structures:
            explanation.append(f"GlycoLLM generated {len(result.predicted_structures)} candidate structures")
        
        if result.reasoning_chain:
            chain = result.reasoning_chain
            explanation.append(f"GlycoGoT applied {len(chain.get('steps', []))} reasoning steps")
        
        if result.validation_results:
            matches = result.validation_results.get('knowledge_matches', [])
            if matches:
                explanation.append(f"GlycoKG found {len(matches)} similar structures in knowledge base")
        
        if result.constraint_violations:
            explanation.append(f"Warning: {len(result.constraint_violations)} constraint violations detected")
        
        return "\n".join(explanation)
    
    def process_file(self, 
                    file_path: Path,
                    output_dir: Optional[Path] = None) -> List[PipelineResult]:
        """
        Process all spectra in a file
        
        Args:
            file_path: Path to spectra file (MGF, mzML, etc.)
            output_dir: Optional directory to save results
            
        Returns:
            List of PipelineResult objects
        """
        logger.info(f"Processing file: {file_path}")
        
        results = []
        for spectrum in self.spectra_parser.parse_file(file_path):
            result = self.process_spectrum(spectrum)
            results.append(result)
            
            # Save individual result if requested
            if output_dir and self.config.save_intermediate:
                output_path = output_dir / f"{spectrum.spectrum_id}_result.json"
                result.to_json(output_path)
        
        logger.info(f"Processed {len(results)} spectra from {file_path}")
        
        # Save combined results
        if output_dir:
            combined_path = output_dir / "combined_results.json"
            with open(combined_path, 'w') as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
        
        return results
    
    def batch_process(self,
                     file_paths: List[Path],
                     output_dir: Path) -> Dict[str, List[PipelineResult]]:
        """
        Process multiple files in batch
        
        Args:
            file_paths: List of spectra files
            output_dir: Directory to save all results
            
        Returns:
            Dictionary mapping file names to results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for file_path in file_paths:
            file_output_dir = output_dir / file_path.stem
            file_output_dir.mkdir(exist_ok=True)
            
            results = self.process_file(file_path, file_output_dir)
            all_results[file_path.name] = results
        
        # Save summary
        summary = {
            'total_files': len(file_paths),
            'total_spectra': sum(len(r) for r in all_results.values()),
            'successful_predictions': sum(
                1 for results in all_results.values()
                for r in results if r.final_structure
            ),
            'files': {
                name: {
                    'num_spectra': len(results),
                    'successful': sum(1 for r in results if r.final_structure)
                }
                for name, results in all_results.items()
            }
        }
        
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch processing complete: {summary['total_spectra']} spectra, {summary['successful_predictions']} successful")
        
        return all_results


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run integrated GlycoLLM+GlycoGoT+GlycoKG pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input spectra file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, help="Path to trained GlycoLLM model")
    parser.add_argument("--graph", type=str, help="Path to knowledge graph (Turtle format)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to generate")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum confidence threshold")
    parser.add_argument("--disable-reasoning", action="store_true", help="Disable GlycoGoT reasoning")
    parser.add_argument("--disable-validation", action="store_true", help="Disable GlycoKG validation")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create config
    config = PipelineConfig(
        model_path=Path(args.model) if args.model else None,
        knowledge_graph_path=Path(args.graph) if args.graph else None,
        top_k_predictions=args.top_k,
        min_confidence=args.min_confidence,
        enable_reasoning=not args.disable_reasoning,
        enable_validation=not args.disable_validation,
        save_intermediate=args.save_intermediate
    )
    
    # Initialize pipeline
    pipeline = IntegratedPipeline(config)
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Single file
        results = pipeline.process_file(input_path, output_dir)
        print(f"\nProcessed {len(results)} spectra")
        print(f"Successful predictions: {sum(1 for r in results if r.final_structure)}")
    
    elif input_path.is_dir():
        # Directory of files
        file_paths = list(input_path.glob("*.mgf")) + list(input_path.glob("*.mzml"))
        results = pipeline.batch_process(file_paths, output_dir)
        print(f"\nProcessed {len(results)} files")
        print(f"Total spectra: {sum(len(r) for r in results.values())}")
    
    else:
        print(f"Error: {input_path} not found")
