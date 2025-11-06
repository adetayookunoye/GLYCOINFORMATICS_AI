"""
Glycosylation Site Prediction for GlycoGoT

Predicts N-linked and O-linked glycosylation sites in protein sequences.
Implements motif-based and machine learning approaches.

N-linked: Asparagine (N) in N-X-S/T motif where X ≠ Proline
O-linked: Serine (S) or Threonine (T) sites (context-dependent)

Author: Adetayo Research Team
Date: November 2025
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class GlycosylationType(Enum):
    """Types of glycosylation"""
    N_LINKED = "N-linked"
    O_LINKED = "O-linked"
    C_LINKED = "C-linked"
    S_LINKED = "S-linked"
    UNKNOWN = "unknown"


@dataclass
class GlycosylationSite:
    """Represents a predicted glycosylation site"""
    position: int  # 0-indexed position in sequence
    residue: str  # Single letter amino acid code
    glycosylation_type: GlycosylationType
    motif: Optional[str] = None  # Sequence motif (e.g., "NXT")
    confidence: float = 0.0  # Prediction confidence [0-1]
    context_sequence: Optional[str] = None  # Surrounding sequence
    features: Optional[Dict] = None  # Additional features
    
    def __post_init__(self):
        if self.features is None:
            self.features = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'position': self.position,
            'residue': self.residue,
            'glycosylation_type': self.glycosylation_type.value,
            'motif': self.motif,
            'confidence': self.confidence,
            'context_sequence': self.context_sequence,
            'features': self.features
        }


class GlycosylationSitePredictor:
    """Predict glycosylation sites in protein sequences"""
    
    def __init__(self, 
                 context_window: int = 5,
                 min_confidence: float = 0.5):
        """
        Initialize predictor
        
        Args:
            context_window: Number of residues on each side for context
            min_confidence: Minimum confidence threshold for predictions
        """
        self.context_window = context_window
        self.min_confidence = min_confidence
        
        # N-linked motif patterns (N-X-S/T where X != P)
        self.n_linked_pattern = re.compile(r'N[^P][ST]')
        
        # O-linked prediction requires more sophisticated analysis
        # Common positions: regions rich in S/T, proline-rich regions
        
        logger.info(f"GlycosylationSitePredictor initialized (context={context_window}, min_conf={min_confidence})")
    
    def predict_n_linked_sites(self, sequence: str) -> List[GlycosylationSite]:
        """
        Predict N-linked glycosylation sites using N-X-S/T motif
        
        Args:
            sequence: Protein sequence (single-letter amino acid codes)
            
        Returns:
            List of predicted N-linked sites
        """
        sites = []
        sequence = sequence.upper()
        
        # Find all N-X-S/T motifs
        for match in self.n_linked_pattern.finditer(sequence):
            position = match.start()  # Position of N
            motif = match.group()
            
            # Extract context
            context_start = max(0, position - self.context_window)
            context_end = min(len(sequence), position + 3 + self.context_window)
            context = sequence[context_start:context_end]
            
            # Compute confidence based on biochemical rules
            confidence = self._compute_n_linked_confidence(sequence, position, motif)
            
            # Extract features
            features = self._extract_n_linked_features(sequence, position)
            
            site = GlycosylationSite(
                position=position,
                residue='N',
                glycosylation_type=GlycosylationType.N_LINKED,
                motif=motif,
                confidence=confidence,
                context_sequence=context,
                features=features
            )
            
            if confidence >= self.min_confidence:
                sites.append(site)
        
        logger.debug(f"Found {len(sites)} N-linked sites in sequence of length {len(sequence)}")
        return sites
    
    def predict_o_linked_sites(self, sequence: str) -> List[GlycosylationSite]:
        """
        Predict O-linked glycosylation sites
        
        O-linked sites are harder to predict as they lack a strict consensus motif.
        Uses heuristics:
        - Serine (S) or Threonine (T) residues
        - Proline-rich regions
        - Regions with clustered S/T
        - Surface accessibility (requires structure, not implemented here)
        
        Args:
            sequence: Protein sequence
            
        Returns:
            List of predicted O-linked sites
        """
        sites = []
        sequence = sequence.upper()
        
        # Find all S and T residues
        for i, residue in enumerate(sequence):
            if residue in ['S', 'T']:
                # Extract context
                context_start = max(0, i - self.context_window)
                context_end = min(len(sequence), i + 1 + self.context_window)
                context = sequence[context_start:context_end]
                
                # Compute confidence based on heuristics
                confidence = self._compute_o_linked_confidence(sequence, i)
                
                # Extract features
                features = self._extract_o_linked_features(sequence, i)
                
                site = GlycosylationSite(
                    position=i,
                    residue=residue,
                    glycosylation_type=GlycosylationType.O_LINKED,
                    motif=context,
                    confidence=confidence,
                    context_sequence=context,
                    features=features
                )
                
                if confidence >= self.min_confidence:
                    sites.append(site)
        
        logger.debug(f"Found {len(sites)} O-linked sites in sequence of length {len(sequence)}")
        return sites
    
    def predict_all_sites(self, sequence: str) -> Dict[str, List[GlycosylationSite]]:
        """
        Predict all glycosylation sites
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Dictionary with N-linked and O-linked predictions
        """
        return {
            'N-linked': self.predict_n_linked_sites(sequence),
            'O-linked': self.predict_o_linked_sites(sequence)
        }
    
    def _compute_n_linked_confidence(self, sequence: str, position: int, motif: str) -> float:
        """
        Compute confidence for N-linked site
        
        Factors:
        - Basic motif match: 0.6
        - Positive factors: hydrophilic region, not in transmembrane domain
        - Negative factors: proline nearby, hydrophobic region
        """
        confidence = 0.6  # Base confidence for motif match
        
        # Get surrounding context
        window = 10
        start = max(0, position - window)
        end = min(len(sequence), position + window)
        context = sequence[start:end]
        
        # Factor 1: Proline proximity (negative)
        if 'P' in context[:5] or 'P' in context[-5:]:
            confidence -= 0.1
        
        # Factor 2: Hydrophilicity of context (positive)
        hydrophilic = set('STNQKRHDE')
        hydrophilic_ratio = sum(1 for aa in context if aa in hydrophilic) / len(context)
        confidence += 0.2 * hydrophilic_ratio
        
        # Factor 3: Not in low complexity region (negative)
        if self._is_low_complexity(context):
            confidence -= 0.15
        
        # Factor 4: Distance from termini (positive if not too close)
        min_dist_from_terminus = min(position, len(sequence) - position - 1)
        if min_dist_from_terminus < 10:
            confidence -= 0.1
        
        # Factor 5: Accessibility proxy - charged residues nearby (positive)
        charged = set('KRHDE')
        charged_count = sum(1 for aa in context if aa in charged)
        confidence += 0.1 * min(charged_count / 5, 1.0)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _compute_o_linked_confidence(self, sequence: str, position: int) -> float:
        """
        Compute confidence for O-linked site
        
        Factors:
        - Proline proximity (positive for mucin-type)
        - S/T clustering (positive)
        - Proline-rich regions (positive)
        - Hydrophilic context (positive)
        """
        confidence = 0.4  # Lower base confidence (no strict motif)
        
        # Get surrounding context
        window = 10
        start = max(0, position - window)
        end = min(len(sequence), position + window)
        context = sequence[start:end]
        
        # Factor 1: Proline proximity (positive)
        proline_count = context.count('P')
        confidence += 0.15 * min(proline_count / 3, 1.0)
        
        # Factor 2: S/T clustering
        st_count = sum(1 for aa in context if aa in 'ST')
        st_ratio = st_count / len(context)
        if st_ratio > 0.3:
            confidence += 0.2
        
        # Factor 3: Mucin-like region (proline-rich + ST-rich)
        if proline_count >= 2 and st_ratio > 0.25:
            confidence += 0.15
        
        # Factor 4: Hydrophilic context
        hydrophilic = set('STNQKRHDE')
        hydrophilic_ratio = sum(1 for aa in context if aa in hydrophilic) / len(context)
        confidence += 0.1 * hydrophilic_ratio
        
        # Factor 5: Not in transmembrane-like region (negative)
        hydrophobic = set('AILMFWV')
        hydrophobic_ratio = sum(1 for aa in context if aa in hydrophobic) / len(context)
        if hydrophobic_ratio > 0.5:
            confidence -= 0.2
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _extract_n_linked_features(self, sequence: str, position: int) -> Dict:
        """Extract features for N-linked site"""
        window = 10
        start = max(0, position - window)
        end = min(len(sequence), position + window)
        context = sequence[start:end]
        
        return {
            'position_from_n_terminus': position,
            'position_from_c_terminus': len(sequence) - position - 1,
            'context_length': len(context),
            'hydrophilic_ratio': self._compute_hydrophilic_ratio(context),
            'charged_ratio': self._compute_charged_ratio(context),
            'proline_count': context.count('P'),
            'is_low_complexity': self._is_low_complexity(context)
        }
    
    def _extract_o_linked_features(self, sequence: str, position: int) -> Dict:
        """Extract features for O-linked site"""
        window = 10
        start = max(0, position - window)
        end = min(len(sequence), position + window)
        context = sequence[start:end]
        
        return {
            'position_from_n_terminus': position,
            'position_from_c_terminus': len(sequence) - position - 1,
            'context_length': len(context),
            'proline_ratio': context.count('P') / len(context),
            'st_ratio': sum(1 for aa in context if aa in 'ST') / len(context),
            'hydrophilic_ratio': self._compute_hydrophilic_ratio(context),
            'hydrophobic_ratio': self._compute_hydrophobic_ratio(context),
            'is_proline_rich': context.count('P') >= 2,
            'is_st_rich': sum(1 for aa in context if aa in 'ST') / len(context) > 0.3
        }
    
    def _compute_hydrophilic_ratio(self, sequence: str) -> float:
        """Compute ratio of hydrophilic residues"""
        hydrophilic = set('STNQKRHDE')
        return sum(1 for aa in sequence if aa in hydrophilic) / max(len(sequence), 1)
    
    def _compute_hydrophobic_ratio(self, sequence: str) -> float:
        """Compute ratio of hydrophobic residues"""
        hydrophobic = set('AILMFWV')
        return sum(1 for aa in sequence if aa in hydrophobic) / max(len(sequence), 1)
    
    def _compute_charged_ratio(self, sequence: str) -> float:
        """Compute ratio of charged residues"""
        charged = set('KRHDE')
        return sum(1 for aa in sequence if aa in charged) / max(len(sequence), 1)
    
    def _is_low_complexity(self, sequence: str) -> bool:
        """Check if sequence is low complexity"""
        if len(sequence) < 5:
            return False
        
        # Count unique residues
        unique_residues = len(set(sequence))
        
        # Low complexity if dominated by few residues
        return unique_residues / len(sequence) < 0.4


class GlycosylationAnnotator:
    """Annotate protein sequences with glycosylation information"""
    
    def __init__(self, predictor: Optional[GlycosylationSitePredictor] = None):
        """Initialize annotator"""
        self.predictor = predictor or GlycosylationSitePredictor()
    
    def annotate_sequence(self, 
                         sequence: str,
                         protein_id: Optional[str] = None,
                         known_sites: Optional[List[int]] = None) -> Dict:
        """
        Annotate protein sequence with glycosylation predictions
        
        Args:
            sequence: Protein sequence
            protein_id: Optional protein identifier
            known_sites: Optional list of known glycosylation sites (for validation)
            
        Returns:
            Annotation dictionary
        """
        predictions = self.predictor.predict_all_sites(sequence)
        
        annotation = {
            'protein_id': protein_id,
            'sequence_length': len(sequence),
            'n_linked_sites': [site.to_dict() for site in predictions['N-linked']],
            'o_linked_sites': [site.to_dict() for site in predictions['O-linked']],
            'n_linked_count': len(predictions['N-linked']),
            'o_linked_count': len(predictions['O-linked']),
            'total_predicted_sites': len(predictions['N-linked']) + len(predictions['O-linked'])
        }
        
        # Add statistics
        if predictions['N-linked']:
            n_confidences = [site.confidence for site in predictions['N-linked']]
            annotation['n_linked_stats'] = {
                'mean_confidence': float(np.mean(n_confidences)),
                'max_confidence': float(np.max(n_confidences)),
                'min_confidence': float(np.min(n_confidences))
            }
        
        if predictions['O-linked']:
            o_confidences = [site.confidence for site in predictions['O-linked']]
            annotation['o_linked_stats'] = {
                'mean_confidence': float(np.mean(o_confidences)),
                'max_confidence': float(np.max(o_confidences)),
                'min_confidence': float(np.min(o_confidences))
            }
        
        # Validation against known sites if provided
        if known_sites:
            annotation['validation'] = self._validate_predictions(
                predictions, known_sites
            )
        
        return annotation
    
    def _validate_predictions(self, 
                            predictions: Dict[str, List[GlycosylationSite]],
                            known_sites: List[int]) -> Dict:
        """Validate predictions against known sites"""
        predicted_positions = set()
        for sites in predictions.values():
            predicted_positions.update(site.position for site in sites)
        
        known_positions = set(known_sites)
        
        true_positives = predicted_positions & known_positions
        false_positives = predicted_positions - known_positions
        false_negatives = known_positions - predicted_positions
        
        precision = len(true_positives) / max(len(predicted_positions), 1)
        recall = len(true_positives) / max(len(known_positions), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        return {
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def batch_annotate(self, 
                      sequences: Dict[str, str],
                      known_sites_map: Optional[Dict[str, List[int]]] = None) -> Dict[str, Dict]:
        """
        Batch annotate multiple protein sequences
        
        Args:
            sequences: Dictionary mapping protein IDs to sequences
            known_sites_map: Optional dictionary mapping protein IDs to known sites
            
        Returns:
            Dictionary mapping protein IDs to annotations
        """
        annotations = {}
        
        for protein_id, sequence in sequences.items():
            known_sites = known_sites_map.get(protein_id) if known_sites_map else None
            annotations[protein_id] = self.annotate_sequence(
                sequence=sequence,
                protein_id=protein_id,
                known_sites=known_sites
            )
        
        logger.info(f"Annotated {len(sequences)} protein sequences")
        return annotations
    
    def generate_visualization(self, annotation: Dict) -> str:
        """
        Generate text visualization of glycosylation sites
        
        Args:
            annotation: Annotation dictionary
            
        Returns:
            Text visualization
        """
        lines = []
        lines.append(f"Protein: {annotation.get('protein_id', 'Unknown')}")
        lines.append(f"Length: {annotation['sequence_length']} residues")
        lines.append("")
        
        lines.append(f"N-linked sites: {annotation['n_linked_count']}")
        for site in annotation['n_linked_sites']:
            lines.append(f"  Position {site['position']}: {site['motif']} (confidence: {site['confidence']:.2f})")
        lines.append("")
        
        lines.append(f"O-linked sites: {annotation['o_linked_count']}")
        for site in annotation['o_linked_sites'][:10]:  # Show first 10
            lines.append(f"  Position {site['position']}: {site['residue']} (confidence: {site['confidence']:.2f})")
        if annotation['o_linked_count'] > 10:
            lines.append(f"  ... and {annotation['o_linked_count'] - 10} more")
        
        if 'validation' in annotation:
            lines.append("")
            lines.append("Validation metrics:")
            val = annotation['validation']
            lines.append(f"  Precision: {val['precision']:.3f}")
            lines.append(f"  Recall: {val['recall']:.3f}")
            lines.append(f"  F1-score: {val['f1_score']:.3f}")
        
        return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict glycosylation sites")
    parser.add_argument("--sequence", type=str, help="Protein sequence")
    parser.add_argument("--fasta", type=str, help="FASTA file with sequences")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum confidence")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    predictor = GlycosylationSitePredictor(min_confidence=args.min_confidence)
    annotator = GlycosylationAnnotator(predictor)
    
    if args.sequence:
        # Single sequence
        annotation = annotator.annotate_sequence(args.sequence)
        print(annotator.generate_visualization(annotation))
    
    elif args.fasta:
        # FASTA file
        from Bio import SeqIO
        sequences = {record.id: str(record.seq) for record in SeqIO.parse(args.fasta, "fasta")}
        annotations = annotator.batch_annotate(sequences)
        
        for protein_id, annotation in annotations.items():
            print("\n" + "="*80)
            print(annotator.generate_visualization(annotation))
    
    else:
        # Example
        example_sequence = "MQKITLWQRPLVTIKIGGQLKEALLDTGADDTVLEDMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNFNASTSPQLCRTYNSTKAGQTTTPTISQASAWFLGTTQPVNHKSTLKSLIPSLQRWQNGVLQRSPFLWMGYELHPDKWTVQPIMLPEKDSWTLKRFFRHPRHKRDVFLQLDRRTFFPYLRNIEISSHRHPVGLQYPGIHTLKNSQVTCSQSHHSNNPSLMPSPLQVVSSSVVHSNVTPSTPSPLRPLMTWDHGVLQQLNMDHNKTPNSNGGPGYAPDRNEQFGQLLAWFLQTNRTYGGSNHQKQYLSQNWFNTTVKEVLLTASKVAVKAATQNQLDFCPAGEFLAVKFKGAENNNNHESQLDLTLRVTVTLRSPTGCQSGSSGVKQDSPGSPPQGNSLTPSANSPVKHSGMLRSGSTLNFSPQSKPRWIQRGVPQPSTRMSSNLHLVSKLHQKLNKQIQRKKSICDHFTYFFQQLYPDRQQLFAVTPPEQQTWPKLPKAMRIQSGLLQHLLAYHLKSFANTRPGQQSINNVYQHYQLSVSLPSVQGVKSLPKLTPKQPQLQAQPAQRPVVPMPTLPTGTPTQTQQPQQPMVPPHNSMQHQQQQSTPHQHQTSPQPQQQQPQSSTVQPQHQPQPQPQTQHQQPPHPQQHLQSTVHQQQQPQPQHPQHQPHQHQQQPTQQQQPQQQLLGSHSFNCGGEFFYCNTTQLFNSTWNGTWNGTWNNTEGNNTTITLPCRIKQIINMWQEVGKAMYAPPIRGQIRCSSNITGLLLTRDGGNSDNETETFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTKAKRRVVQREKRAVGIGALFLGFLGAAGSTMGAASITLTVQARQLLSGIVQQQNNLLRAIEAQQHLLQLTVWGIKQLQARVLAVERYLKDQQLLGIWGCSGKLICTTTVPWNASWSNKSLNQIWDNMTWMEWDREINNYTDLIYTLIEESQNQQEKNEQELLELDKWASLWNWFSITNWLWYIKIFIMIVGGLVGLRIVFTVLSIVNRVRQGYSPLSFQTRLPAPRGPDRPEGIEEEGGERDRDRSGRLVDGFLALIWVDLRSLCLFSYHRLRDLLLIVTRIVELLGRRGWEALKYWWNLLQYWSQELKNSAVSLLNATAIAVAEGTDRVIEVVQRACRAILHIPRRIRQGLERALL"
        
        annotation = annotator.annotate_sequence(example_sequence, protein_id="Example")
        print(annotator.generate_visualization(annotation))
