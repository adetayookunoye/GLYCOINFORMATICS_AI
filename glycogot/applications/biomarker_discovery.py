"""
Biomarker Discovery Pipeline for GlycoGoT

Differential expression analysis and clinical association discovery
for glycan biomarkers in cancer and other diseases.

Features:
- Differential expression analysis (case vs control)
- Statistical significance testing
- Clinical association mining
- Pathway enrichment analysis
- Biomarker candidate ranking

Author: Adetayo Research Team
Date: November 2025
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import json
from datetime import datetime
import numpy as np
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class GlycanExpression:
    """Glycan expression data"""
    glycan_id: str
    sample_id: str
    expression_level: float
    condition: str  # e.g., "cancer", "normal"
    tissue: Optional[str] = None
    patient_id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class DifferentialExpression:
    """Differential expression result"""
    glycan_id: str
    fold_change: float
    log2_fold_change: float
    p_value: float
    adjusted_p_value: float  # FDR-corrected
    mean_case: float
    mean_control: float
    std_case: float
    std_control: float
    n_case: int
    n_control: int
    significance: str  # "up", "down", "not_significant"
    
    def to_dict(self) -> Dict:
        return {
            'glycan_id': self.glycan_id,
            'fold_change': self.fold_change,
            'log2_fold_change': self.log2_fold_change,
            'p_value': self.p_value,
            'adjusted_p_value': self.adjusted_p_value,
            'mean_case': self.mean_case,
            'mean_control': self.mean_control,
            'std_case': self.std_case,
            'std_control': self.std_control,
            'n_case': self.n_case,
            'n_control': self.n_control,
            'significance': self.significance
        }


@dataclass
class BiomarkerCandidate:
    """Biomarker candidate with ranking"""
    glycan_id: str
    disease: str
    rank: int
    biomarker_score: float
    differential_expression: DifferentialExpression
    clinical_associations: List[str]
    pathway_enrichments: List[str]
    literature_support: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'glycan_id': self.glycan_id,
            'disease': self.disease,
            'rank': self.rank,
            'biomarker_score': self.biomarker_score,
            'differential_expression': self.differential_expression.to_dict(),
            'clinical_associations': self.clinical_associations,
            'pathway_enrichments': self.pathway_enrichments,
            'literature_support': self.literature_support
        }


class DifferentialExpressionAnalyzer:
    """Perform differential expression analysis"""
    
    def __init__(self,
                 p_value_threshold: float = 0.05,
                 fold_change_threshold: float = 2.0,
                 min_samples: int = 3):
        """
        Initialize analyzer
        
        Args:
            p_value_threshold: Significance threshold for adjusted p-values
            fold_change_threshold: Minimum fold change to consider significant
            min_samples: Minimum number of samples required per condition
        """
        self.p_value_threshold = p_value_threshold
        self.fold_change_threshold = fold_change_threshold
        self.min_samples = min_samples
        
        logger.info(f"DifferentialExpressionAnalyzer initialized (p={p_value_threshold}, fc={fold_change_threshold})")
    
    def analyze(self,
                expression_data: List[GlycanExpression],
                case_condition: str,
                control_condition: str) -> List[DifferentialExpression]:
        """
        Perform differential expression analysis
        
        Args:
            expression_data: List of expression measurements
            case_condition: Condition label for cases (e.g., "cancer")
            control_condition: Condition label for controls (e.g., "normal")
            
        Returns:
            List of differential expression results
        """
        logger.info(f"Analyzing differential expression: {case_condition} vs {control_condition}")
        
        # Group data by glycan and condition
        glycan_data = defaultdict(lambda: {'case': [], 'control': []})
        
        for expr in expression_data:
            if expr.condition == case_condition:
                glycan_data[expr.glycan_id]['case'].append(expr.expression_level)
            elif expr.condition == control_condition:
                glycan_data[expr.glycan_id]['control'].append(expr.expression_level)
        
        # Compute differential expression for each glycan
        results = []
        p_values = []
        
        for glycan_id, data in glycan_data.items():
            case_values = np.array(data['case'])
            control_values = np.array(data['control'])
            
            # Skip if insufficient samples
            if len(case_values) < self.min_samples or len(control_values) < self.min_samples:
                continue
            
            # Compute statistics
            mean_case = np.mean(case_values)
            mean_control = np.mean(control_values)
            std_case = np.std(case_values, ddof=1)
            std_control = np.std(control_values, ddof=1)
            
            # Compute fold change
            if mean_control > 0:
                fold_change = mean_case / mean_control
                log2_fc = np.log2(fold_change) if fold_change > 0 else -np.inf
            else:
                fold_change = np.inf if mean_case > 0 else 1.0
                log2_fc = np.inf if mean_case > 0 else 0.0
            
            # Perform t-test
            try:
                t_stat, p_value = stats.ttest_ind(case_values, control_values)
            except Exception as e:
                logger.warning(f"Error computing t-test for {glycan_id}: {e}")
                p_value = 1.0
            
            # Store result
            result = DifferentialExpression(
                glycan_id=glycan_id,
                fold_change=fold_change,
                log2_fold_change=log2_fc,
                p_value=p_value,
                adjusted_p_value=p_value,  # Will adjust later
                mean_case=mean_case,
                mean_control=mean_control,
                std_case=std_case,
                std_control=std_control,
                n_case=len(case_values),
                n_control=len(control_values),
                significance="not_significant"
            )
            
            results.append(result)
            p_values.append(p_value)
        
        # FDR correction (Benjamini-Hochberg)
        if p_values:
            adjusted_p_values = self._benjamini_hochberg(p_values)
            
            for result, adj_p in zip(results, adjusted_p_values):
                result.adjusted_p_value = adj_p
                
                # Determine significance
                if adj_p <= self.p_value_threshold:
                    if abs(result.log2_fold_change) >= np.log2(self.fold_change_threshold):
                        result.significance = "up" if result.log2_fold_change > 0 else "down"
        
        logger.info(f"Analyzed {len(results)} glycans, {sum(1 for r in results if r.significance != 'not_significant')} significant")
        
        return results
    
    def _benjamini_hochberg(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        # Compute adjusted p-values
        adjusted = np.zeros(n)
        prev_adjusted = 1.0
        
        for i in range(n - 1, -1, -1):
            rank = i + 1
            adjusted_p = min(sorted_p[i] * n / rank, prev_adjusted)
            adjusted[sorted_indices[i]] = adjusted_p
            prev_adjusted = adjusted_p
        
        return adjusted.tolist()
    
    def filter_significant(self, results: List[DifferentialExpression]) -> List[DifferentialExpression]:
        """Filter for significant results"""
        return [r for r in results if r.significance != "not_significant"]
    
    def rank_by_effect_size(self, results: List[DifferentialExpression]) -> List[DifferentialExpression]:
        """Rank results by effect size"""
        return sorted(results, key=lambda r: abs(r.log2_fold_change), reverse=True)


class ClinicalAssociationMiner:
    """Mine clinical associations for biomarker candidates"""
    
    def __init__(self, knowledge_base: Optional[Dict] = None):
        """
        Initialize miner
        
        Args:
            knowledge_base: Optional knowledge base with known associations
        """
        self.knowledge_base = knowledge_base or {}
        logger.info("ClinicalAssociationMiner initialized")
    
    def find_associations(self, 
                         glycan_id: str,
                         disease: str) -> List[str]:
        """
        Find clinical associations for glycan
        
        Args:
            glycan_id: Glycan identifier
            disease: Disease name
            
        Returns:
            List of clinical associations
        """
        associations = []
        
        # Check knowledge base
        if glycan_id in self.knowledge_base:
            kb_data = self.knowledge_base[glycan_id]
            
            if disease in kb_data.get('diseases', []):
                associations.append(f"Known association with {disease}")
            
            if 'functions' in kb_data:
                associations.extend([f"Function: {f}" for f in kb_data['functions'][:3]])
            
            if 'protein_interactions' in kb_data:
                associations.append(f"Interacts with {len(kb_data['protein_interactions'])} proteins")
        
        return associations
    
    def compute_literature_support(self, glycan_id: str, disease: str) -> int:
        """
        Estimate literature support (simplified)
        
        In real implementation, would query PubMed/literature databases
        """
        # Placeholder - would implement actual literature mining
        return np.random.randint(0, 50)


class PathwayEnrichmentAnalyzer:
    """Perform pathway enrichment analysis"""
    
    def __init__(self, pathway_database: Optional[Dict] = None):
        """
        Initialize analyzer
        
        Args:
            pathway_database: Database mapping glycans to pathways
        """
        self.pathway_database = pathway_database or {}
        logger.info("PathwayEnrichmentAnalyzer initialized")
    
    def analyze_enrichment(self,
                          glycan_ids: List[str],
                          background_glycans: Optional[List[str]] = None) -> List[Dict]:
        """
        Analyze pathway enrichment
        
        Args:
            glycan_ids: List of glycans of interest
            background_glycans: Background/universe of glycans
            
        Returns:
            List of enriched pathways with statistics
        """
        if not background_glycans:
            background_glycans = list(self.pathway_database.keys())
        
        # Count pathways
        pathway_counts = defaultdict(int)
        background_pathway_counts = defaultdict(int)
        
        for glycan_id in glycan_ids:
            if glycan_id in self.pathway_database:
                for pathway in self.pathway_database[glycan_id]:
                    pathway_counts[pathway] += 1
        
        for glycan_id in background_glycans:
            if glycan_id in self.pathway_database:
                for pathway in self.pathway_database[glycan_id]:
                    background_pathway_counts[pathway] += 1
        
        # Compute enrichment (Fisher's exact test)
        enrichments = []
        
        for pathway, count in pathway_counts.items():
            if count < 2:  # Skip pathways with few hits
                continue
            
            background_count = background_pathway_counts.get(pathway, 0)
            
            # 2x2 contingency table
            a = count  # In pathway, in set
            b = len(glycan_ids) - count  # Not in pathway, in set
            c = background_count - count  # In pathway, not in set
            d = len(background_glycans) - background_count - b  # Not in pathway, not in set
            
            try:
                odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])
            except Exception as e:
                logger.warning(f"Error computing Fisher's exact for {pathway}: {e}")
                continue
            
            enrichments.append({
                'pathway': pathway,
                'count': count,
                'total_in_set': len(glycan_ids),
                'background_count': background_count,
                'total_background': len(background_glycans),
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'enrichment_score': count / max(background_count / len(background_glycans), 0.01)
            })
        
        # Sort by p-value
        enrichments.sort(key=lambda x: x['p_value'])
        
        return enrichments


class BiomarkerDiscoveryPipeline:
    """Complete biomarker discovery pipeline"""
    
    def __init__(self,
                 de_analyzer: Optional[DifferentialExpressionAnalyzer] = None,
                 association_miner: Optional[ClinicalAssociationMiner] = None,
                 pathway_analyzer: Optional[PathwayEnrichmentAnalyzer] = None):
        """Initialize pipeline"""
        self.de_analyzer = de_analyzer or DifferentialExpressionAnalyzer()
        self.association_miner = association_miner or ClinicalAssociationMiner()
        self.pathway_analyzer = pathway_analyzer or PathwayEnrichmentAnalyzer()
        
        logger.info("BiomarkerDiscoveryPipeline initialized")
    
    def discover_biomarkers(self,
                           expression_data: List[GlycanExpression],
                           disease: str,
                           case_condition: str,
                           control_condition: str,
                           top_n: int = 50) -> List[BiomarkerCandidate]:
        """
        Complete biomarker discovery workflow
        
        Args:
            expression_data: Expression measurements
            disease: Disease name
            case_condition: Case condition label
            control_condition: Control condition label
            top_n: Number of top biomarkers to return
            
        Returns:
            List of ranked biomarker candidates
        """
        logger.info(f"Starting biomarker discovery for {disease}")
        
        # Step 1: Differential expression analysis
        de_results = self.de_analyzer.analyze(
            expression_data,
            case_condition,
            control_condition
        )
        
        significant_results = self.de_analyzer.filter_significant(de_results)
        ranked_results = self.de_analyzer.rank_by_effect_size(significant_results)
        
        logger.info(f"Found {len(significant_results)} significant glycans")
        
        # Step 2: Pathway enrichment
        significant_glycans = [r.glycan_id for r in significant_results]
        pathway_enrichments = self.pathway_analyzer.analyze_enrichment(significant_glycans)
        
        logger.info(f"Found {len(pathway_enrichments)} enriched pathways")
        
        # Step 3: Build biomarker candidates
        candidates = []
        
        for i, de_result in enumerate(ranked_results[:top_n]):
            glycan_id = de_result.glycan_id
            
            # Find associations
            associations = self.association_miner.find_associations(glycan_id, disease)
            
            # Find relevant pathways
            relevant_pathways = [
                p['pathway'] for p in pathway_enrichments
                if p['p_value'] < 0.05
            ][:5]
            
            # Get literature support
            lit_support = self.association_miner.compute_literature_support(glycan_id, disease)
            
            # Compute biomarker score
            biomarker_score = self._compute_biomarker_score(
                de_result,
                len(associations),
                len(relevant_pathways),
                lit_support
            )
            
            candidate = BiomarkerCandidate(
                glycan_id=glycan_id,
                disease=disease,
                rank=i + 1,
                biomarker_score=biomarker_score,
                differential_expression=de_result,
                clinical_associations=associations,
                pathway_enrichments=relevant_pathways,
                literature_support=lit_support
            )
            
            candidates.append(candidate)
        
        # Re-rank by biomarker score
        candidates.sort(key=lambda c: c.biomarker_score, reverse=True)
        for i, candidate in enumerate(candidates):
            candidate.rank = i + 1
        
        logger.info(f"Generated {len(candidates)} biomarker candidates")
        
        return candidates
    
    def _compute_biomarker_score(self,
                                de_result: DifferentialExpression,
                                n_associations: int,
                                n_pathways: int,
                                lit_support: int) -> float:
        """
        Compute integrated biomarker score
        
        Combines:
        - Effect size (log2 fold change)
        - Statistical significance (adjusted p-value)
        - Clinical associations
        - Pathway enrichment
        - Literature support
        """
        # Effect size component (0-1)
        effect_size_score = min(abs(de_result.log2_fold_change) / 5.0, 1.0)
        
        # Significance component (0-1)
        sig_score = 1.0 - min(de_result.adjusted_p_value, 1.0)
        
        # Association component (0-1)
        assoc_score = min(n_associations / 10.0, 1.0)
        
        # Pathway component (0-1)
        pathway_score = min(n_pathways / 5.0, 1.0)
        
        # Literature component (0-1)
        lit_score = min(lit_support / 20.0, 1.0)
        
        # Weighted combination
        score = (
            0.30 * effect_size_score +
            0.30 * sig_score +
            0.15 * assoc_score +
            0.15 * pathway_score +
            0.10 * lit_score
        )
        
        return score
    
    def export_results(self,
                      candidates: List[BiomarkerCandidate],
                      output_path: Path):
        """Export biomarker candidates to JSON"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'num_candidates': len(candidates),
            'candidates': [c.to_dict() for c in candidates]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results exported to {output_path}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Biomarker discovery pipeline")
    parser.add_argument("--expression-data", type=str, required=True, help="Expression data JSON file")
    parser.add_argument("--disease", type=str, required=True, help="Disease name")
    parser.add_argument("--case-condition", type=str, required=True, help="Case condition label")
    parser.add_argument("--control-condition", type=str, required=True, help="Control condition label")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top biomarkers")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="P-value threshold")
    parser.add_argument("--fc-threshold", type=float, default=2.0, help="Fold change threshold")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load expression data
    with open(args.expression_data) as f:
        data = json.load(f)
    
    expression_data = [
        GlycanExpression(**item) for item in data
    ]
    
    # Initialize pipeline
    de_analyzer = DifferentialExpressionAnalyzer(
        p_value_threshold=args.p_threshold,
        fold_change_threshold=args.fc_threshold
    )
    
    pipeline = BiomarkerDiscoveryPipeline(de_analyzer=de_analyzer)
    
    # Discover biomarkers
    candidates = pipeline.discover_biomarkers(
        expression_data=expression_data,
        disease=args.disease,
        case_condition=args.case_condition,
        control_condition=args.control_condition,
        top_n=args.top_n
    )
    
    # Export results
    pipeline.export_results(candidates, Path(args.output))
    
    # Print summary
    print(f"\n=== Top 10 Biomarker Candidates for {args.disease} ===")
    for candidate in candidates[:10]:
        print(f"{candidate.rank}. {candidate.glycan_id}")
        print(f"   Score: {candidate.biomarker_score:.3f}")
        print(f"   Log2FC: {candidate.differential_expression.log2_fold_change:.2f}")
        print(f"   P-value: {candidate.differential_expression.adjusted_p_value:.2e}")
        print()
