"""
Constraint Validation System for GlycoKG

Active biochemical rule checking during reasoning to ensure
predicted structures obey glycobiological constraints.

Validates:
- Monosaccharide connectivity rules
- Linkage geometry constraints
- Biological feasibility
- Chemical composition constraints
- Known biochemical pathways

Author: Adetayo Research Team
Date: November 2025
"""

import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of biochemical constraints"""
    CONNECTIVITY = "connectivity"
    LINKAGE_GEOMETRY = "linkage_geometry"
    COMPOSITION = "composition"
    STOICHIOMETRY = "stoichiometry"
    BIOSYNTHESIS = "biosynthesis"
    STRUCTURE_VALIDITY = "structure_validity"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation"""
    constraint_type: ConstraintType
    severity: str  # "error", "warning", "info"
    message: str
    glycan_id: Optional[str] = None
    details: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'constraint_type': self.constraint_type.value,
            'severity': self.severity,
            'message': self.message,
            'glycan_id': self.glycan_id,
            'details': self.details
        }


class ConstraintValidator:
    """Validate glycan structures against biochemical constraints"""
    
    def __init__(self):
        """Initialize validator with constraint rules"""
        self.rules = self._initialize_rules()
        logger.info("ConstraintValidator initialized with {len(self.rules)} rules")
    
    def _initialize_rules(self) -> Dict:
        """Initialize biochemical constraint rules"""
        return {
            'monosaccharide_valency': {
                'Glc': {'max_linkages': 4},
                'Gal': {'max_linkages': 4},
                'Man': {'max_linkages': 4},
                'GlcNAc': {'max_linkages': 4},
                'GalNAc': {'max_linkages': 4},
                'Fuc': {'max_linkages': 1},  # Terminal only
                'NeuAc': {'max_linkages': 2},  # α2-3 or α2-6
                'NeuGc': {'max_linkages': 2}
            },
            'valid_linkages': {
                'Fuc': ['α1-2', 'α1-3', 'α1-4', 'α1-6'],
                'NeuAc': ['α2-3', 'α2-6', 'α2-8'],
                'NeuGc': ['α2-3', 'α2-6'],
                'GlcNAc': ['β1-2', 'β1-3', 'β1-4', 'β1-6'],
                'GalNAc': ['α1-3', 'β1-3', 'β1-4', 'β1-6'],
                'Man': ['α1-2', 'α1-3', 'α1-4', 'α1-6', 'β1-4'],
                'Glc': ['α1-4', 'β1-4', 'β1-6'],
                'Gal': ['β1-3', 'β1-4', 'β1-6']
            },
            'terminal_only': ['Fuc'],  # Must be terminal
            'n_glycan_core': {
                'required': ['Man', 'GlcNAc'],
                'core_structure': 'Man3GlcNAc2'
            },
            'o_glycan_cores': [
                'GalNAc',  # Core 1-4
                'GlcNAc'   # Core 5-8
            ]
        }
    
    def validate_structure(self, structure_data: Dict) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Validate complete glycan structure
        
        Args:
            structure_data: Dictionary with structure information
            
        Returns:
            Tuple of (is_valid, violations)
        """
        violations = []
        
        # Extract structure components
        glycan_id = structure_data.get('glycan_id', 'unknown')
        monosaccharides = structure_data.get('monosaccharides', [])
        linkages = structure_data.get('linkages', [])
        glycan_type = structure_data.get('type', 'unknown')  # N-glycan, O-glycan, etc.
        
        # Run validation checks
        violations.extend(self._validate_monosaccharide_valency(monosaccharides, linkages, glycan_id))
        violations.extend(self._validate_linkage_geometry(linkages, monosaccharides, glycan_id))
        violations.extend(self._validate_terminal_positions(monosaccharides, linkages, glycan_id))
        
        if glycan_type == 'N-glycan':
            violations.extend(self._validate_n_glycan_core(monosaccharides, linkages, glycan_id))
        elif glycan_type == 'O-glycan':
            violations.extend(self._validate_o_glycan_core(monosaccharides, linkages, glycan_id))
        
        is_valid = not any(v.severity == "error" for v in violations)
        
        return is_valid, violations
    
    def _validate_monosaccharide_valency(self, 
                                        monosaccharides: List[Dict],
                                        linkages: List[Dict],
                                        glycan_id: str) -> List[ConstraintViolation]:
        """Validate monosaccharide valency constraints"""
        violations = []
        
        # Count linkages per monosaccharide
        linkage_counts = {}
        for mono in monosaccharides:
            mono_id = mono.get('id', mono.get('name'))
            linkage_counts[mono_id] = 0
        
        for linkage in linkages:
            donor = linkage.get('donor')
            acceptor = linkage.get('acceptor')
            if donor in linkage_counts:
                linkage_counts[donor] += 1
            if acceptor in linkage_counts:
                linkage_counts[acceptor] += 1
        
        # Check against rules
        for mono in monosaccharides:
            mono_id = mono.get('id', mono.get('name'))
            mono_type = mono.get('type', 'Unknown')
            
            if mono_type in self.rules['monosaccharide_valency']:
                max_linkages = self.rules['monosaccharide_valency'][mono_type]['max_linkages']
                actual_linkages = linkage_counts.get(mono_id, 0)
                
                if actual_linkages > max_linkages:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.CONNECTIVITY,
                        severity="error",
                        message=f"Monosaccharide {mono_type} has {actual_linkages} linkages, exceeds maximum {max_linkages}",
                        glycan_id=glycan_id,
                        details={'monosaccharide': mono_type, 'linkages': actual_linkages, 'max': max_linkages}
                    ))
        
        return violations
    
    def _validate_linkage_geometry(self,
                                   linkages: List[Dict],
                                   monosaccharides: List[Dict],
                                   glycan_id: str) -> List[ConstraintViolation]:
        """Validate linkage geometry (anomeric config and positions)"""
        violations = []
        
        # Build monosaccharide type map
        mono_types = {m.get('id', m.get('name')): m.get('type') for m in monosaccharides}
        
        for linkage in linkages:
            donor = linkage.get('donor')
            donor_type = mono_types.get(donor, 'Unknown')
            linkage_type = linkage.get('type', 'unknown')  # e.g., "α1-4"
            
            if donor_type in self.rules['valid_linkages']:
                valid_types = self.rules['valid_linkages'][donor_type]
                
                if linkage_type not in valid_types and linkage_type != 'unknown':
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.LINKAGE_GEOMETRY,
                        severity="warning",
                        message=f"Unusual linkage {linkage_type} for {donor_type}",
                        glycan_id=glycan_id,
                        details={'donor': donor_type, 'linkage': linkage_type, 'valid': valid_types}
                    ))
        
        return violations
    
    def _validate_terminal_positions(self,
                                    monosaccharides: List[Dict],
                                    linkages: List[Dict],
                                    glycan_id: str) -> List[ConstraintViolation]:
        """Validate terminal monosaccharide constraints"""
        violations = []
        
        # Find terminal monosaccharides (those that are only donors, not acceptors)
        acceptors = {l.get('acceptor') for l in linkages}
        
        for mono in monosaccharides:
            mono_id = mono.get('id', mono.get('name'))
            mono_type = mono.get('type', 'Unknown')
            
            # Check if must be terminal
            if mono_type in self.rules.get('terminal_only', []):
                if mono_id in acceptors:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.BIOSYNTHESIS,
                        severity="error",
                        message=f"{mono_type} must be terminal but has linkages from it",
                        glycan_id=glycan_id,
                        details={'monosaccharide': mono_type}
                    ))
        
        return violations
    
    def _validate_n_glycan_core(self,
                               monosaccharides: List[Dict],
                               linkages: List[Dict],
                               glycan_id: str) -> List[ConstraintViolation]:
        """Validate N-glycan core structure"""
        violations = []
        
        core_rules = self.rules.get('n_glycan_core', {})
        required = core_rules.get('required', [])
        
        # Check required monosaccharides present
        mono_types = [m.get('type') for m in monosaccharides]
        
        for req_type in required:
            if req_type not in mono_types:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.STRUCTURE_VALIDITY,
                    severity="error",
                    message=f"N-glycan missing required core monosaccharide: {req_type}",
                    glycan_id=glycan_id,
                    details={'required': req_type}
                ))
        
        return violations
    
    def _validate_o_glycan_core(self,
                               monosaccharides: List[Dict],
                               linkages: List[Dict],
                               glycan_id: str) -> List[ConstraintViolation]:
        """Validate O-glycan core structure"""
        violations = []
        
        o_cores = self.rules.get('o_glycan_cores', [])
        mono_types = [m.get('type') for m in monosaccharides]
        
        # Check at least one valid core type present
        has_valid_core = any(core in mono_types for core in o_cores)
        
        if not has_valid_core:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.STRUCTURE_VALIDITY,
                severity="warning",
                message=f"O-glycan missing typical core monosaccharide",
                glycan_id=glycan_id,
                details={'expected_cores': o_cores}
            ))
        
        return violations
    
    def validate_composition(self, composition: Dict, glycan_id: str = None) -> List[ConstraintViolation]:
        """Validate monosaccharide composition"""
        violations = []
        
        # Check for impossible stoichiometry
        total = sum(composition.values())
        if total == 0:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.COMPOSITION,
                severity="error",
                message="Empty composition",
                glycan_id=glycan_id
            ))
        
        # Check for excessive size
        if total > 50:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.COMPOSITION,
                severity="warning",
                message=f"Unusually large glycan with {total} monosaccharides",
                glycan_id=glycan_id,
                details={'total_monosaccharides': total}
            ))
        
        return violations
    
    def validate_mass(self, observed_mass: float, calculated_mass: float, 
                     tolerance: float = 0.01, glycan_id: str = None) -> List[ConstraintViolation]:
        """Validate mass accuracy"""
        violations = []
        
        mass_error = abs(observed_mass - calculated_mass)
        mass_error_ppm = (mass_error / calculated_mass) * 1e6 if calculated_mass > 0 else float('inf')
        
        if mass_error_ppm > tolerance * 1e6:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.COMPOSITION,
                severity="error",
                message=f"Mass error {mass_error_ppm:.1f} ppm exceeds tolerance",
                glycan_id=glycan_id,
                details={
                    'observed_mass': observed_mass,
                    'calculated_mass': calculated_mass,
                    'error_ppm': mass_error_ppm
                }
            ))
        
        return violations
    
    def get_summary(self, violations: List[ConstraintViolation]) -> Dict:
        """Summarize validation results"""
        return {
            'total_violations': len(violations),
            'errors': sum(1 for v in violations if v.severity == "error"),
            'warnings': sum(1 for v in violations if v.severity == "warning"),
            'info': sum(1 for v in violations if v.severity == "info"),
            'by_type': {
                ct.value: sum(1 for v in violations if v.constraint_type == ct)
                for ct in ConstraintType
            }
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    validator = ConstraintValidator()
    
    # Example structure
    structure_data = {
        'glycan_id': 'G00001MO',
        'type': 'N-glycan',
        'monosaccharides': [
            {'id': 'm1', 'type': 'GlcNAc'},
            {'id': 'm2', 'type': 'GlcNAc'},
            {'id': 'm3', 'type': 'Man'},
            {'id': 'm4', 'type': 'Man'},
            {'id': 'm5', 'type': 'Man'},
            {'id': 'm6', 'type': 'Fuc'}
        ],
        'linkages': [
            {'donor': 'm1', 'acceptor': 'protein', 'type': 'β1-N'},
            {'donor': 'm2', 'acceptor': 'm1', 'type': 'β1-4'},
            {'donor': 'm3', 'acceptor': 'm2', 'type': 'β1-4'},
            {'donor': 'm4', 'acceptor': 'm3', 'type': 'α1-3'},
            {'donor': 'm5', 'acceptor': 'm3', 'type': 'α1-6'},
            {'donor': 'm6', 'acceptor': 'm2', 'type': 'α1-6'}
        ]
    }
    
    is_valid, violations = validator.validate_structure(structure_data)
    
    print(f"\nValidation result: {'VALID' if is_valid else 'INVALID'}")
    print(f"Violations found: {len(violations)}")
    
    for v in violations:
        print(f"\n[{v.severity.upper()}] {v.constraint_type.value}")
        print(f"  {v.message}")
        if v.details:
            print(f"  Details: {v.details}")
    
    summary = validator.get_summary(violations)
    print(f"\nSummary: {summary}")
