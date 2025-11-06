"""
Enhanced Output Contract System for GlycoLLM API

This module provides sophisticated contract validation, serialization utilities,
and enhanced data models for the GlycoLLM platform. Ensures consistent,
reliable, and well-validated API responses across all endpoints.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import json
import time
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)

# ========================================================================================
# ENHANCED VALIDATION UTILITIES
# ========================================================================================

class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"

class OutputQuality(str, Enum):
    """Output quality assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNRELIABLE = "unreliable"

def validate_wurcs_format(wurcs_string: str) -> bool:
    """Validate WURCS format structure."""
    if not wurcs_string.startswith("WURCS="):
        return False
    
    try:
        parts = wurcs_string.split("/")
        if len(parts) < 4:
            return False
        
        # Basic format validation
        version = parts[0].split("=")[1]
        counts = parts[1].split(",")
        
        return len(counts) == 3 and all(c.isdigit() for c in counts)
    except:
        return False

def validate_spectrum_peaks(peaks: List[Dict[str, Any]]) -> bool:
    """Validate spectrum peak format."""
    required_fields = ["mz", "intensity"]
    
    for peak in peaks:
        if not all(field in peak for field in required_fields):
            return False
        
        try:
            mz = float(peak["mz"])
            intensity = float(peak["intensity"])
            
            if mz <= 0 or intensity < 0:
                return False
        except (ValueError, TypeError):
            return False
    
    return True

def calculate_confidence_category(confidence: float) -> str:
    """Categorize confidence score."""
    if confidence >= 0.9:
        return "very_high"
    elif confidence >= 0.7:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    elif confidence >= 0.3:
        return "low"
    else:
        return "very_low"

# ========================================================================================
# ENHANCED BASE MODELS
# ========================================================================================

class EnhancedBaseModel(BaseModel):
    """Enhanced base model with additional validation and serialization."""
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            set: lambda v: list(v)
        }
    
    def model_dump_enhanced(self, 
                          include_metadata: bool = True,
                          validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Any]:
        """Enhanced model dump with metadata and validation."""
        data = self.model_dump()
        
        if include_metadata:
            data["_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "model_type": self.__class__.__name__,
                "validation_level": validation_level.value,
                "data_hash": self._calculate_hash(data)
            }
        
        return data
    
    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash of data for integrity checking."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

# ========================================================================================
# ENHANCED UNCERTAINTY MODELS
# ========================================================================================

class AdvancedUncertaintyMetrics(EnhancedBaseModel):
    """Advanced uncertainty quantification with multiple methods."""
    
    # Core confidence metrics
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence [0-1]")
    confidence_category: str = Field(..., description="Categorical confidence assessment")
    
    # Prediction intervals
    prediction_interval: Dict[str, float] = Field(
        default_factory=lambda: {"lower": 0.0, "upper": 1.0},
        description="Prediction interval bounds"
    )
    coverage_probability: float = Field(0.95, ge=0.5, le=0.99, description="Interval coverage probability")
    
    # Uncertainty decomposition
    epistemic_uncertainty: float = Field(0.0, ge=0.0, description="Model/knowledge uncertainty")
    aleatoric_uncertainty: float = Field(0.0, ge=0.0, description="Data/noise uncertainty")
    total_uncertainty: float = Field(0.0, ge=0.0, description="Combined uncertainty")
    
    # Calibration metrics
    calibration_quality: str = Field("unknown", description="Calibration assessment")
    calibration_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="ECE or Brier score")
    
    # Selective prediction
    should_abstain: bool = Field(False, description="Whether to abstain from prediction")
    abstention_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Abstention threshold used")
    
    # Quality assessment
    output_quality: OutputQuality = Field(OutputQuality.ACCEPTABLE, description="Overall output quality")
    quality_factors: Dict[str, float] = Field(default_factory=dict, description="Quality contributing factors")
    
    @validator('confidence_category', pre=True, always=True)
    def set_confidence_category(cls, v, values):
        if 'confidence_score' in values:
            return calculate_confidence_category(values['confidence_score'])
        return v
    
    @validator('total_uncertainty', pre=True, always=True)
    def calculate_total_uncertainty(cls, v, values):
        epistemic = values.get('epistemic_uncertainty', 0.0)
        aleatoric = values.get('aleatoric_uncertainty', 0.0)
        return (epistemic**2 + aleatoric**2)**0.5
    
    @validator('should_abstain', pre=True, always=True)
    def determine_abstention(cls, v, values):
        confidence = values.get('confidence_score', 1.0)
        threshold = values.get('abstention_threshold', 0.5)
        return confidence < threshold

class UncertaintyBreakdown(EnhancedBaseModel):
    """Detailed breakdown of uncertainty sources."""
    model_uncertainty: float = Field(0.0, ge=0.0, description="Model architecture uncertainty")
    parameter_uncertainty: float = Field(0.0, ge=0.0, description="Parameter estimation uncertainty")
    data_uncertainty: float = Field(0.0, ge=0.0, description="Input data uncertainty")
    task_complexity: float = Field(0.0, ge=0.0, description="Task-specific complexity factor")
    domain_shift: float = Field(0.0, ge=0.0, description="Out-of-domain detection score")

# ========================================================================================
# ENHANCED GROUNDING MODELS
# ========================================================================================

class DetailedGroundingEvidence(EnhancedBaseModel):
    """Detailed knowledge graph grounding evidence."""
    
    # Core identification
    entity_id: str = Field(..., description="Unique KG entity identifier")
    entity_type: str = Field(..., description="Entity type (motif, enzyme, pathway, etc.)")
    entity_name: Optional[str] = Field(None, description="Human-readable entity name")
    
    # Confidence and validation
    confidence: float = Field(..., ge=0.0, le=1.0, description="Grounding confidence")
    validation_method: str = Field("similarity", description="Method used for validation")
    evidence_strength: str = Field("moderate", description="Strength of supporting evidence")
    
    # Supporting information
    evidence_source: str = Field(..., description="Primary evidence source")
    supporting_papers: List[str] = Field(default_factory=list, description="Supporting literature PMIDs")
    cross_references: Dict[str, str] = Field(default_factory=dict, description="Cross-database references")
    
    # Biological context
    biosynthetic_pathway: Optional[str] = Field(None, description="Associated pathway")
    organism_specificity: List[str] = Field(default_factory=list, description="Organism-specific annotations")
    tissue_specificity: List[str] = Field(default_factory=list, description="Tissue-specific annotations")
    
    # Metadata
    last_validated: Optional[datetime] = Field(None, description="Last validation timestamp")
    validation_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall validation score")

class ComprehensiveGroundingResult(EnhancedBaseModel):
    """Comprehensive knowledge graph grounding result."""
    
    # Entity collections
    structural_motifs: List[DetailedGroundingEvidence] = Field(default_factory=list, description="Structural motifs")
    enzymes: List[DetailedGroundingEvidence] = Field(default_factory=list, description="Associated enzymes")
    pathways: List[DetailedGroundingEvidence] = Field(default_factory=list, description="Biosynthetic pathways")
    organisms: List[DetailedGroundingEvidence] = Field(default_factory=list, description="Source organisms")
    diseases: List[DetailedGroundingEvidence] = Field(default_factory=list, description="Disease associations")
    
    # Overall assessment
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Overall grounding confidence")
    coverage_score: float = Field(0.0, ge=0.0, le=1.0, description="Knowledge coverage score")
    consistency_score: float = Field(0.0, ge=0.0, le=1.0, description="Cross-reference consistency")
    
    # Validation metadata
    grounding_method: str = Field("embedding_similarity", description="Grounding methodology")
    knowledge_sources: List[str] = Field(default_factory=list, description="Knowledge sources used")
    grounding_timestamp: datetime = Field(default_factory=datetime.now, description="Grounding timestamp")
    
    @validator('overall_confidence', pre=True, always=True)
    def calculate_overall_confidence(cls, v, values):
        all_evidence = []
        for field in ['structural_motifs', 'enzymes', 'pathways', 'organisms', 'diseases']:
            if field in values:
                all_evidence.extend(values[field])
        
        if all_evidence:
            confidences = [e.confidence for e in all_evidence]
            return sum(confidences) / len(confidences)
        return 0.0

# ========================================================================================
# ENHANCED CANDIDATE MODELS
# ========================================================================================

class ValidatedStructureCandidate(EnhancedBaseModel):
    """Enhanced structure candidate with comprehensive validation."""
    
    # Core structure information
    structure: str = Field(..., description="Glycan structure string")
    format: str = Field("WURCS", description="Structure format")
    canonical_form: Optional[str] = Field(None, description="Canonical structure representation")
    
    # Confidence and validation
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    structure_validity: bool = Field(True, description="Structure format validity")
    chemical_validity: bool = Field(True, description="Chemical plausibility")
    
    # Scoring metrics
    mass_accuracy: Optional[float] = Field(None, description="Mass accuracy (ppm)")
    fragmentation_score: Optional[float] = Field(None, description="Fragmentation pattern match")
    biosynthetic_plausibility: Optional[float] = Field(None, ge=0.0, le=1.0, description="Biosynthetic feasibility")
    
    # Additional properties
    molecular_weight: Optional[float] = Field(None, description="Calculated molecular weight")
    composition: Optional[Dict[str, int]] = Field(None, description="Monosaccharide composition")
    structural_features: List[str] = Field(default_factory=list, description="Key structural features")
    
    # Grounding and context
    grounding: Optional[ComprehensiveGroundingResult] = Field(None, description="KG grounding")
    alternative_representations: Dict[str, str] = Field(default_factory=dict, description="Alternative formats")
    
    @validator('structure_validity', pre=True, always=True)
    def validate_structure_format(cls, v, values):
        structure = values.get('structure', '')
        format_type = values.get('format', 'WURCS')
        
        if format_type == 'WURCS':
            return validate_wurcs_format(structure)
        return True  # Placeholder for other formats
    
    @validator('canonical_form', pre=True, always=True)
    def generate_canonical_form(cls, v, values):
        if v is None and values.get('structure'):
            # Placeholder - would implement actual canonicalization
            return values['structure']
        return v

class EnhancedSpectraCandidate(EnhancedBaseModel):
    """Enhanced spectrum candidate with detailed annotations."""
    
    # Core spectrum data
    predicted_peaks: List[Dict[str, Union[float, str]]] = Field(..., description="Predicted peaks")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    
    # Experimental parameters
    fragmentation_type: str = Field("MS/MS", description="Fragmentation method")
    collision_energy: Optional[float] = Field(None, description="Collision energy (eV)")
    charge_state: Optional[int] = Field(None, description="Ion charge state")
    
    # Mass information
    theoretical_mass: Optional[float] = Field(None, description="Theoretical molecular mass")
    observed_mass: Optional[float] = Field(None, description="Observed precursor mass")
    mass_error: Optional[float] = Field(None, description="Mass error (ppm)")
    
    # Spectrum quality metrics
    peak_quality: Dict[str, float] = Field(default_factory=dict, description="Peak quality metrics")
    coverage_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Fragment coverage %")
    intensity_distribution: Dict[str, float] = Field(default_factory=dict, description="Intensity statistics")
    
    # Annotations and assignments
    fragment_annotations: List[Dict[str, Any]] = Field(default_factory=list, description="Fragment assignments")
    unassigned_peaks: List[Dict[str, Any]] = Field(default_factory=list, description="Unassigned peaks")
    
    @validator('predicted_peaks')
    def validate_peaks(cls, v):
        if not validate_spectrum_peaks(v):
            raise ValueError("Invalid spectrum peak format")
        return v
    
    @validator('mass_error', pre=True, always=True)
    def calculate_mass_error(cls, v, values):
        theoretical = values.get('theoretical_mass')
        observed = values.get('observed_mass')
        
        if theoretical and observed and v is None:
            return ((observed - theoretical) / theoretical) * 1e6  # ppm
        return v

# ========================================================================================
# ENHANCED RATIONALE MODELS
# ========================================================================================

class DetailedRationale(EnhancedBaseModel):
    """Comprehensive reasoning rationale with methodology details."""
    
    # Core reasoning
    reasoning_steps: List[str] = Field(..., description="Step-by-step reasoning process")
    methodology: str = Field("multimodal_transformer", description="Inference methodology used")
    
    # Confidence factors
    confidence_factors: Dict[str, float] = Field(default_factory=dict, description="Confidence contributors")
    uncertainty_factors: Dict[str, float] = Field(default_factory=dict, description="Uncertainty contributors")
    
    # Evidence and support
    supporting_evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    literature_support: List[str] = Field(default_factory=list, description="Literature references")
    cross_validation: Dict[str, Any] = Field(default_factory=dict, description="Cross-validation results")
    
    # Limitations and caveats
    known_limitations: List[str] = Field(default_factory=list, description="Known limitations")
    potential_biases: List[str] = Field(default_factory=list, description="Potential biases")
    alternative_interpretations: List[str] = Field(default_factory=list, description="Alternative explanations")
    
    # Methodology details
    model_components_used: List[str] = Field(default_factory=list, description="Model components utilized")
    data_sources_consulted: List[str] = Field(default_factory=list, description="Data sources used")
    validation_methods: List[str] = Field(default_factory=list, description="Validation approaches")
    
    # Quality assessment
    rationale_quality: OutputQuality = Field(OutputQuality.ACCEPTABLE, description="Rationale quality")
    explanation_completeness: float = Field(0.0, ge=0.0, le=1.0, description="Completeness score")

class ActionableNextSteps(EnhancedBaseModel):
    """Actionable recommendations and next steps."""
    
    # Experimental recommendations
    experimental_validation: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Experimental validation protocols"
    )
    
    # Analytical recommendations  
    additional_analyses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recommended additional analyses"
    )
    
    # Database and literature queries
    database_queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Suggested database searches"
    )
    
    # Confidence improvement strategies
    confidence_improvement: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Methods to improve confidence"
    )
    
    # Priority and effort estimation
    priority_ranking: Dict[str, int] = Field(
        default_factory=dict,
        description="Priority ranking of recommendations"
    )
    effort_estimates: Dict[str, str] = Field(
        default_factory=dict,
        description="Effort level estimates (low/medium/high)"
    )
    
    # Success probability
    success_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Estimated success probability for each recommendation"
    )

# ========================================================================================
# CONTRACT VALIDATION SYSTEM
# ========================================================================================

class ContractValidator:
    """Validates and enforces output contracts."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load validation rules based on level."""
        rules = {
            "confidence_thresholds": {
                ValidationLevel.STRICT: {"min": 0.8, "abstain_below": 0.7},
                ValidationLevel.STANDARD: {"min": 0.6, "abstain_below": 0.5},
                ValidationLevel.PERMISSIVE: {"min": 0.3, "abstain_below": 0.2}
            },
            "quality_requirements": {
                ValidationLevel.STRICT: [OutputQuality.EXCELLENT, OutputQuality.GOOD],
                ValidationLevel.STANDARD: [OutputQuality.EXCELLENT, OutputQuality.GOOD, OutputQuality.ACCEPTABLE],
                ValidationLevel.PERMISSIVE: [q for q in OutputQuality]
            }
        }
        return rules
    
    def validate_response(self, response: EnhancedBaseModel) -> Dict[str, Any]:
        """Validate response against contract requirements."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "quality_score": 1.0
        }
        
        # Check confidence thresholds
        if hasattr(response, 'uncertainty'):
            confidence = response.uncertainty.confidence_score
            min_confidence = self.validation_rules["confidence_thresholds"][self.validation_level]["min"]
            
            if confidence < min_confidence:
                validation_results["warnings"].append(
                    f"Confidence {confidence:.2f} below recommended threshold {min_confidence}"
                )
                validation_results["quality_score"] *= 0.8
        
        # Check output quality
        if hasattr(response, 'uncertainty') and hasattr(response.uncertainty, 'output_quality'):
            quality = response.uncertainty.output_quality
            allowed_qualities = self.validation_rules["quality_requirements"][self.validation_level]
            
            if quality not in allowed_qualities:
                validation_results["errors"].append(
                    f"Output quality {quality} not acceptable for {self.validation_level} validation"
                )
                validation_results["valid"] = False
        
        return validation_results

# ========================================================================================
# SERIALIZATION UTILITIES
# ========================================================================================

class EnhancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for complex types."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return super().default(obj)

def serialize_response(response: EnhancedBaseModel, 
                      include_metadata: bool = True,
                      compact: bool = False) -> str:
    """Serialize response with enhanced options."""
    data = response.model_dump_enhanced(include_metadata=include_metadata)
    
    if compact:
        return json.dumps(data, cls=EnhancedJSONEncoder, separators=(',', ':'))
    else:
        return json.dumps(data, cls=EnhancedJSONEncoder, indent=2, sort_keys=True)

def validate_and_serialize(response: EnhancedBaseModel,
                          validator: ContractValidator,
                          include_validation_info: bool = True) -> Dict[str, Any]:
    """Validate and serialize response with contract checking."""
    validation_results = validator.validate_response(response)
    
    result = {
        "response": response.model_dump_enhanced(),
        "contract_validation": validation_results
    }
    
    if not validation_results["valid"] and validator.validation_level == ValidationLevel.STRICT:
        result["error"] = "Response failed contract validation"
        result["details"] = validation_results["errors"]
    
    return result