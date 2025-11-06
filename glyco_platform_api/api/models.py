"""
Advanced Pydantic Models for Sophisticated GlycoLLM API

This module defines the sophisticated request/response models for the 4 core
GlycoLLM endpoints: spec2struct, structure2spec, explain, and retrieval.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import re

# ========================================================================================
# ENUMS AND CONSTANTS
# ========================================================================================

class TaskType(str, Enum):
    """Core GlycoLLM task types."""
    SPEC2STRUCT = "spec2struct"
    STRUCTURE2SPEC = "structure2spec"
    EXPLAIN = "explain"
    RETRIEVAL = "retrieval"

class SpectraType(str, Enum):
    """Mass spectrometry types."""
    MSMS = "MS/MS"
    CID = "CID"
    HCD = "HCD"
    ETD = "ETD"
    ECD = "ECD"

class StructureFormat(str, Enum):
    """Glycan structure formats."""
    WURCS = "WURCS"
    GLYCOCT = "GlycoCT"
    IUPAC = "IUPAC"
    LINEAR_CODE = "LinearCode"

class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

# ========================================================================================
# UNCERTAINTY AND GROUNDING MODELS
# ========================================================================================

class UncertaintyMetrics(BaseModel):
    """Comprehensive uncertainty quantification."""
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence [0-1]")
    prediction_interval: Dict[str, float] = Field(
        default_factory=lambda: {"lower": 0.0, "upper": 1.0},
        description="Prediction interval bounds"
    )
    calibration_quality: str = Field("unknown", description="Calibration assessment")
    epistemic_uncertainty: float = Field(0.0, ge=0.0, description="Model uncertainty")
    aleatoric_uncertainty: float = Field(0.0, ge=0.0, description="Data uncertainty")
    confidence_level: ConfidenceLevel = Field(ConfidenceLevel.UNCERTAIN, description="Categorical confidence")
    
    @field_validator('prediction_interval')
    @classmethod
    def validate_interval(cls, v):
        if 'lower' in v and 'upper' in v:
            if v['lower'] > v['upper']:
                raise ValueError("Lower bound must be <= upper bound")
        return v

class GroundingEvidence(BaseModel):
    """Knowledge graph grounding evidence."""
    entity_id: str = Field(..., description="KG entity identifier")
    entity_type: str = Field(..., description="Entity type (motif, enzyme, pathway, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Grounding confidence")
    evidence_source: str = Field(..., description="Source of evidence")
    supporting_papers: List[str] = Field(default_factory=list, description="Supporting literature")
    biosynthetic_pathway: Optional[str] = Field(None, description="Associated biosynthetic pathway")

class GroundingResult(BaseModel):
    """Complete grounding information."""
    motifs: List[GroundingEvidence] = Field(default_factory=list, description="Structural motifs")
    enzymes: List[GroundingEvidence] = Field(default_factory=list, description="Associated enzymes")
    pathways: List[GroundingEvidence] = Field(default_factory=list, description="Biosynthetic pathways")
    organisms: List[GroundingEvidence] = Field(default_factory=list, description="Source organisms")
    validation_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall validation score")

# ========================================================================================
# CANDIDATE MODELS
# ========================================================================================

class StructureCandidate(BaseModel):
    """Candidate glycan structure with confidence."""
    structure: str = Field(..., description="Glycan structure string")
    format: StructureFormat = Field(StructureFormat.WURCS, description="Structure format")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    mass_accuracy: Optional[float] = Field(None, description="Mass accuracy (ppm)")
    fragmentation_score: Optional[float] = Field(None, description="Fragmentation pattern match")
    grounding: Optional[GroundingResult] = Field(None, description="Knowledge graph grounding")

class SpectraCandidate(BaseModel):
    """Candidate spectrum prediction with confidence."""
    predicted_peaks: List[Dict[str, Union[float, str]]] = Field(
        ..., description="Predicted peaks with m/z, intensity, assignment"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    fragmentation_type: SpectraType = Field(SpectraType.MSMS, description="Fragmentation type")
    theoretical_mass: Optional[float] = Field(None, description="Theoretical molecular mass")
    charge_state: Optional[int] = Field(None, description="Charge state")

# ========================================================================================
# REQUEST MODELS
# ========================================================================================

class Spec2StructRequest(BaseModel):
    """Request for spectrum-to-structure prediction."""
    spectra: Dict[str, Any] = Field(..., description="Mass spectrum data")
    spectra_type: SpectraType = Field(SpectraType.MSMS, description="MS fragmentation type")
    precursor_mass: Optional[float] = Field(None, description="Precursor ion mass")
    charge_state: Optional[int] = Field(None, description="Ion charge state")
    organism_context: Optional[str] = Field(None, description="Biological organism context")
    max_candidates: int = Field(5, ge=1, le=20, description="Maximum structure candidates")
    confidence_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum confidence")
    include_grounding: bool = Field(True, description="Include KG grounding")
    
    @field_validator('spectra')
    @classmethod
    def validate_spectra(cls, v):
        required_fields = ['peaks', 'precursor_mz']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v

class Structure2SpecRequest(BaseModel):
    """Request for structure-to-spectrum prediction."""
    structure: str = Field(..., description="Glycan structure")
    structure_format: StructureFormat = Field(StructureFormat.WURCS, description="Input format")
    fragmentation_type: SpectraType = Field(SpectraType.MSMS, description="Desired fragmentation")
    collision_energy: Optional[float] = Field(None, description="Collision energy (eV)")
    charge_states: List[int] = Field(default=[1], description="Charge states to predict")
    include_annotations: bool = Field(True, description="Include peak annotations")
    
    @field_validator('structure')
    @classmethod
    def validate_structure(cls, v):
        # Basic validation for common formats
        if not v or len(v.strip()) == 0:
            raise ValueError("Structure cannot be empty")
        return v.strip()

class ExplainRequest(BaseModel):
    """Request for detailed glycan analysis explanation."""
    structure: Optional[str] = Field(None, description="Glycan structure to explain")
    spectra: Optional[Dict[str, Any]] = Field(None, description="Mass spectrum to explain")
    question: Optional[str] = Field(None, description="Specific question about the glycan")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    explanation_depth: str = Field("comprehensive", description="Detail level: brief, standard, comprehensive")
    include_literature: bool = Field(True, description="Include literature references")
    
    @model_validator(mode='after')
    def validate_input(self) -> 'ExplainRequest':
        if not any([self.structure, self.spectra, self.question]):
            raise ValueError("Must provide at least one of: structure, spectra, or question")
        return self

class RetrievalRequest(BaseModel):
    """Request for glycan database retrieval."""
    query: str = Field(..., description="Search query (structure, text, or mixed)")
    query_type: str = Field("mixed", description="Query type: structure, text, similarity, mixed")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results to return")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    databases: List[str] = Field(default=["GlyTouCan", "GlyGen"], description="Databases to search")
    include_annotations: bool = Field(True, description="Include biological annotations")
    rank_by: str = Field("relevance", description="Ranking method: relevance, similarity, confidence")

# ========================================================================================
# RESPONSE MODELS
# ========================================================================================

class Rationale(BaseModel):
    """Detailed reasoning and rationale."""
    reasoning_steps: List[str] = Field(..., description="Step-by-step reasoning")
    confidence_factors: Dict[str, float] = Field(default_factory=dict, description="Factors affecting confidence")
    limitations: List[str] = Field(default_factory=list, description="Known limitations")
    literature_support: List[str] = Field(default_factory=list, description="Supporting literature")

class NextSteps(BaseModel):
    """Actionable next steps and recommendations."""
    experimental_validation: List[str] = Field(default_factory=list, description="Suggested experiments")
    additional_analyses: List[str] = Field(default_factory=list, description="Recommended analyses")
    database_queries: List[str] = Field(default_factory=list, description="Useful database searches")
    confidence_improvement: List[str] = Field(default_factory=list, description="Ways to improve confidence")

class Spec2StructResponse(BaseModel):
    """Response for spectrum-to-structure prediction."""
    task: TaskType = Field(TaskType.SPEC2STRUCT, description="Task type")
    candidates: List[StructureCandidate] = Field(..., description="Structure candidates")
    uncertainty: UncertaintyMetrics = Field(..., description="Uncertainty quantification")
    rationale: Rationale = Field(..., description="Reasoning and rationale")
    next_steps: NextSteps = Field(..., description="Recommended next steps")
    processing_time: float = Field(..., description="Processing time (seconds)")
    model_version: str = Field("glycollm-v0.1.0", description="Model version used")

class Structure2SpecResponse(BaseModel):
    """Response for structure-to-spectrum prediction."""
    task: TaskType = Field(TaskType.STRUCTURE2SPEC, description="Task type")
    candidates: List[SpectraCandidate] = Field(..., description="Spectrum predictions")
    uncertainty: UncertaintyMetrics = Field(..., description="Uncertainty quantification")
    rationale: Rationale = Field(..., description="Reasoning and rationale")
    next_steps: NextSteps = Field(..., description="Recommended next steps")
    processing_time: float = Field(..., description="Processing time (seconds)")
    model_version: str = Field("glycollm-v0.1.0", description="Model version used")

class ExplainResponse(BaseModel):
    """Response for glycan explanation."""
    task: TaskType = Field(TaskType.EXPLAIN, description="Task type")
    explanation: str = Field(..., description="Comprehensive explanation")
    key_insights: List[str] = Field(..., description="Key biological insights")
    uncertainty: UncertaintyMetrics = Field(..., description="Explanation confidence")
    grounding: GroundingResult = Field(..., description="Knowledge graph grounding")
    rationale: Rationale = Field(..., description="Reasoning methodology")
    next_steps: NextSteps = Field(..., description="Recommended next steps")
    processing_time: float = Field(..., description="Processing time (seconds)")
    model_version: str = Field("glycollm-v0.1.0", description="Model version used")

class RetrievalResult(BaseModel):
    """Single retrieval result."""
    id: str = Field(..., description="Database identifier")
    structure: Optional[str] = Field(None, description="Glycan structure")
    name: Optional[str] = Field(None, description="Glycan name")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity to query")
    source_database: str = Field(..., description="Source database")
    annotations: Dict[str, Any] = Field(default_factory=dict, description="Biological annotations")
    grounding: Optional[GroundingResult] = Field(None, description="KG grounding info")

class RetrievalResponse(BaseModel):
    """Response for glycan retrieval."""
    task: TaskType = Field(TaskType.RETRIEVAL, description="Task type")
    query: str = Field(..., description="Original query")
    results: List[RetrievalResult] = Field(..., description="Retrieved glycans")
    total_found: int = Field(..., description="Total results found")
    uncertainty: UncertaintyMetrics = Field(..., description="Retrieval confidence")
    rationale: Rationale = Field(..., description="Search methodology")
    next_steps: NextSteps = Field(..., description="Recommended follow-ups")
    processing_time: float = Field(..., description="Processing time (seconds)")
    model_version: str = Field("glycollm-v0.1.0", description="Model version used")

# ========================================================================================
# ERROR MODELS
# ========================================================================================

class APIError(BaseModel):
    """Standardized API error response."""
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    suggestion: Optional[str] = Field(None, description="Suggested resolution")
    timestamp: float = Field(..., description="Error timestamp")