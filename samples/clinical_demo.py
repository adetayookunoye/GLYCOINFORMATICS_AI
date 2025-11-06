"""
Clinical Demo: Cancer Biomarker Discovery

Demonstrates complete workflow for cancer biomarker discovery using
GlycoLLM + GlycoGoT + GlycoKG integrated system.

Example: Breast cancer glycan biomarker identification

Author: Adetayo Research Team
Date: November 2025
"""

import logging
from pathlib import Path
import json
from typing import List, Dict
import numpy as np

from scripts.integrated_pipeline import IntegratedPipeline, PipelineConfig, PipelineResult
from glycogot.applications.biomarker_discovery import (
    BiomarkerDiscoveryPipeline, GlycanExpression, DifferentialExpressionAnalyzer
)
from glycogot.applications.site_prediction import GlycosylationSitePredictor, GlycosylationAnnotator
from glycollm.data.spectra_parser import MSSpectrum, SpectraParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BreastCancerBiomarkerDemo:
    """
    Complete demo for breast cancer biomarker discovery
    
    Workflow:
    1. Load patient data (tumor vs normal tissue)
    2. MS/MS spectra analysis with GlycoLLM
    3. Differential expression analysis
    4. Biomarker candidate ranking
    5. Clinical interpretation with GlycoGoT
    6. Validation against GlycoKG
    """
    
    def __init__(self, 
                 model_path: Path,
                 knowledge_graph_path: Path):
        """Initialize demo"""
        logger.info("=" * 80)
        logger.info("CLINICAL DEMO: Breast Cancer Glycan Biomarker Discovery")
        logger.info("=" * 80)
        
        # Initialize integrated pipeline
        config = PipelineConfig(
            model_path=model_path,
            knowledge_graph_path=knowledge_graph_path,
            enable_reasoning=True,
            enable_validation=True
        )
        self.pipeline = IntegratedPipeline(config)
        
        # Initialize biomarker discovery
        self.biomarker_pipeline = BiomarkerDiscoveryPipeline()
        
        # Initialize site predictor
        self.site_predictor = GlycosylationSitePredictor()
        self.site_annotator = GlycosylationAnnotator(self.site_predictor)
        
        logger.info("Demo initialized")
    
    def load_synthetic_patient_data(self) -> Dict:
        """
        Load synthetic patient data for demonstration
        
        In real scenario, would load from clinical database
        """
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Loading Patient Data")
        logger.info("=" * 80)
        
        # Synthetic expression data
        np.random.seed(42)
        
        glycan_ids = [f"G{i:07d}MO" for i in range(1, 51)]
        patient_data = {
            'study': 'Breast Cancer Cohort 2025',
            'patients': {
                'tumor': [f'BC_T{i:03d}' for i in range(1, 21)],
                'normal': [f'BC_N{i:03d}' for i in range(1, 21)]
            },
            'expression_data': []
        }
        
        # Generate synthetic expression values
        for glycan_id in glycan_ids:
            # Simulate differential expression for some glycans
            is_differential = np.random.random() < 0.3
            
            for patient_id in patient_data['patients']['tumor']:
                base = 100 if is_differential else 50
                value = base + np.random.normal(0, 10)
                patient_data['expression_data'].append(
                    GlycanExpression(
                        glycan_id=glycan_id,
                        sample_id=f"{patient_id}_sample",
                        expression_level=max(0, value),
                        condition='tumor',
                        tissue='breast',
                        patient_id=patient_id
                    )
                )
            
            for patient_id in patient_data['patients']['normal']:
                base = 40 if is_differential else 50
                value = base + np.random.normal(0, 8)
                patient_data['expression_data'].append(
                    GlycanExpression(
                        glycan_id=glycan_id,
                        sample_id=f"{patient_id}_sample",
                        expression_level=max(0, value),
                        condition='normal',
                        tissue='breast',
                        patient_id=patient_id
                    )
                )
        
        logger.info(f"Loaded data from {len(patient_data['patients']['tumor'])} tumor samples")
        logger.info(f"Loaded data from {len(patient_data['patients']['normal'])} normal samples")
        logger.info(f"Total expression measurements: {len(patient_data['expression_data'])}")
        
        return patient_data
    
    def analyze_spectra(self, spectra_file: Path) -> List[PipelineResult]:
        """
        Analyze MS/MS spectra with integrated pipeline
        
        Step 2: Structure identification from spectra
        """
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: MS/MS Spectra Analysis with GlycoLLM")
        logger.info("=" * 80)
        
        if not spectra_file.exists():
            logger.warning(f"Spectra file not found: {spectra_file}")
            logger.info("Generating synthetic spectra for demo...")
            return self._generate_synthetic_spectra_results()
        
        results = self.pipeline.process_file(spectra_file)
        
        logger.info(f"Analyzed {len(results)} spectra")
        logger.info(f"Successful predictions: {sum(1 for r in results if r.final_structure)}")
        
        return results
    
    def _generate_synthetic_spectra_results(self) -> List[PipelineResult]:
        """Generate synthetic spectra results for demo"""
        from datetime import datetime
        
        results = []
        for i in range(10):
            result = PipelineResult(
                spectrum_id=f"DEMO_SPECTRUM_{i+1}",
                timestamp=datetime.now().isoformat(),
                predicted_structures=[
                    {
                        'structure': f'G{i+1:07d}MO',
                        'confidence': 0.85 - i * 0.05,
                        'method': 'glycollm'
                    }
                ],
                final_structure=f'G{i+1:07d}MO',
                final_confidence=0.85 - i * 0.05,
                explanation="Demo prediction"
            )
            results.append(result)
        
        logger.info(f"Generated {len(results)} synthetic spectra results")
        return results
    
    def discover_biomarkers(self, patient_data: Dict) -> List:
        """
        Discover biomarker candidates
        
        Step 3: Differential expression and biomarker ranking
        """
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Biomarker Discovery")
        logger.info("=" * 80)
        
        candidates = self.biomarker_pipeline.discover_biomarkers(
            expression_data=patient_data['expression_data'],
            disease='Breast Cancer',
            case_condition='tumor',
            control_condition='normal',
            top_n=10
        )
        
        logger.info(f"\nTop 5 Biomarker Candidates:")
        for i, candidate in enumerate(candidates[:5], 1):
            logger.info(f"{i}. {candidate.glycan_id}")
            logger.info(f"   Score: {candidate.biomarker_score:.3f}")
            logger.info(f"   Log2FC: {candidate.differential_expression.log2_fold_change:.2f}")
            logger.info(f"   P-value: {candidate.differential_expression.adjusted_p_value:.2e}")
            logger.info(f"   Associations: {len(candidate.clinical_associations)}")
        
        return candidates
    
    def analyze_glycosylation_sites(self, protein_sequence: str) -> Dict:
        """
        Analyze glycosylation sites
        
        Step 4: Site prediction for top biomarker
        """
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Glycosylation Site Analysis")
        logger.info("=" * 80)
        
        annotation = self.site_annotator.annotate_sequence(
            sequence=protein_sequence,
            protein_id="MUC1"  # Example: MUC1 in breast cancer
        )
        
        logger.info(f"\nProtein: MUC1 (Mucin-1)")
        logger.info(f"Sequence length: {annotation['sequence_length']} residues")
        logger.info(f"N-linked sites: {annotation['n_linked_count']}")
        logger.info(f"O-linked sites: {annotation['o_linked_count']}")
        
        if annotation['n_linked_count'] > 0:
            logger.info("\nTop N-linked sites:")
            for site in annotation['n_linked_sites'][:3]:
                logger.info(f"  Position {site['position']}: {site['motif']} (confidence: {site['confidence']:.2f})")
        
        return annotation
    
    def generate_clinical_report(self,
                                 biomarkers: List,
                                 spectra_results: List[PipelineResult],
                                 site_analysis: Dict) -> Dict:
        """
        Generate comprehensive clinical report
        
        Step 5: Clinical interpretation
        """
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Clinical Report Generation")
        logger.info("=" * 80)
        
        report = {
            'study': 'Breast Cancer Glycan Biomarker Analysis',
            'date': '2025-11-03',
            'summary': {
                'total_biomarkers_identified': len(biomarkers),
                'high_confidence_biomarkers': sum(1 for b in biomarkers if b.biomarker_score > 0.7),
                'spectra_analyzed': len(spectra_results),
                'successful_identifications': sum(1 for r in spectra_results if r.final_structure)
            },
            'top_biomarkers': [
                {
                    'rank': b.rank,
                    'glycan_id': b.glycan_id,
                    'score': b.biomarker_score,
                    'fold_change': b.differential_expression.fold_change,
                    'p_value': b.differential_expression.adjusted_p_value,
                    'clinical_significance': self._interpret_biomarker(b)
                }
                for b in biomarkers[:5]
            ],
            'glycosylation_analysis': {
                'protein': 'MUC1',
                'n_linked_sites': site_analysis['n_linked_count'],
                'o_linked_sites': site_analysis['o_linked_count'],
                'interpretation': 'Increased O-glycosylation consistent with breast cancer phenotype'
            },
            'recommendations': [
                'Further validation in independent cohort recommended',
                'Consider MUC1 glycosylation as therapeutic target',
                'Develop diagnostic assay for top 3 biomarkers',
                'Investigate biosynthetic pathways for identified glycans'
            ]
        }
        
        logger.info("\n=== CLINICAL REPORT SUMMARY ===")
        logger.info(json.dumps(report['summary'], indent=2))
        logger.info("\n=== TOP BIOMARKERS ===")
        for bm in report['top_biomarkers']:
            logger.info(f"{bm['rank']}. {bm['glycan_id']} (score: {bm['score']:.3f})")
            logger.info(f"   {bm['clinical_significance']}")
        
        return report
    
    def _interpret_biomarker(self, candidate) -> str:
        """Interpret biomarker clinical significance"""
        fc = candidate.differential_expression.fold_change
        
        if fc > 3:
            return "Strongly upregulated in tumor tissue - excellent diagnostic potential"
        elif fc > 2:
            return "Moderately upregulated - potential diagnostic marker"
        elif fc < 0.5:
            return "Downregulated in tumor - may indicate protective role"
        else:
            return "Minor changes - requires further investigation"
    
    def run_complete_demo(self, output_dir: Path) -> Dict:
        """Run complete demo workflow"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING COMPLETE CLINICAL DEMO")
        logger.info("=" * 80)
        
        # Step 1: Load data
        patient_data = self.load_synthetic_patient_data()
        
        # Step 2: Analyze spectra
        spectra_file = Path("data/raw/glycopost/breast_cancer_samples.mgf")
        spectra_results = self.analyze_spectra(spectra_file)
        
        # Step 3: Discover biomarkers
        biomarkers = self.discover_biomarkers(patient_data)
        
        # Step 4: Analyze glycosylation sites (example MUC1 sequence)
        muc1_sequence = "MTPGTQSPFFLLLLLTVLTV" + "GSTAPPAHGVTSAPDTRPAPGSTAPPAHGVTSAPDTRPAP" * 5
        site_analysis = self.analyze_glycosylation_sites(muc1_sequence)
        
        # Step 5: Generate report
        report = self.generate_clinical_report(biomarkers, spectra_results, site_analysis)
        
        # Save all outputs
        report_path = output_dir / "clinical_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nReport saved to: {report_path}")
        
        biomarkers_path = output_dir / "biomarkers.json"
        with open(biomarkers_path, 'w') as f:
            json.dump([b.to_dict() for b in biomarkers], f, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info("DEMO COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"- Clinical report: {report_path}")
        logger.info(f"- Biomarkers: {biomarkers_path}")
        
        return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clinical demo for cancer biomarker discovery")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--graph", type=str, help="Path to knowledge graph")
    parser.add_argument("--output", type=str, default="results/clinical_demo", help="Output directory")
    
    args = parser.parse_args()
    
    # Use default paths if not provided
    model_path = Path(args.model) if args.model else Path("models/glycollm_final")
    graph_path = Path(args.graph) if args.graph else Path("data/processed/glycokg.ttl")
    output_dir = Path(args.output)
    
    # Run demo
    demo = BreastCancerBiomarkerDemo(
        model_path=model_path,
        knowledge_graph_path=graph_path
    )
    
    report = demo.run_complete_demo(output_dir)
    
    print("\n" + "=" * 80)
    print("DEMO RESULTS SUMMARY")
    print("=" * 80)
    print(json.dumps(report['summary'], indent=2))


if __name__ == "__main__":
    main()
