"""
Applications module for GlycoGoT reasoning system.

This module provides high-level applications and use cases for
glycan reasoning, including clinical analysis, drug discovery support,
and educational tools.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

from .integration import ReasoningOrchestrator, ReasoningRequest, ReasoningResponse
from .reasoning import ReasoningType, ReasoningChain
from .operations import BatchReasoningJob, ReasoningWorkflow

logger = logging.getLogger(__name__)


class ApplicationDomain(Enum):
    """Application domains for glycan reasoning."""
    CLINICAL_DIAGNOSTICS = "clinical_diagnostics"
    DRUG_DISCOVERY = "drug_discovery"
    BIOMARKER_DISCOVERY = "biomarker_discovery"
    VACCINE_DEVELOPMENT = "vaccine_development"
    FOOD_SCIENCE = "food_science"
    EDUCATIONAL = "educational"
    RESEARCH = "research"


@dataclass
class ClinicalSample:
    """Clinical sample with glycan data."""
    
    sample_id: str
    patient_id: Optional[str] = None
    sample_type: str = "serum"  # serum, plasma, tissue, etc.
    collection_date: Optional[str] = None
    clinical_context: Dict[str, Any] = None
    glycan_data: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.clinical_context is None:
            self.clinical_context = {}
        if self.glycan_data is None:
            self.glycan_data = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DrugTarget:
    """Drug target with glycan interaction data."""
    
    target_id: str
    name: str
    protein_class: str
    glycan_binding_sites: List[Dict[str, Any]] = None
    known_ligands: List[Dict[str, Any]] = None
    therapeutic_area: str = ""
    validation_status: str = "preclinical"
    
    def __post_init__(self):
        if self.glycan_binding_sites is None:
            self.glycan_binding_sites = []
        if self.known_ligands is None:
            self.known_ligands = []


class ClinicalAnalysisApplication:
    """
    Clinical glycomics analysis application.
    
    Provides specialized reasoning for clinical samples,
    disease biomarker identification, and diagnostic support.
    """
    
    def __init__(self, orchestrator: ReasoningOrchestrator):
        self.orchestrator = orchestrator
        
        # Clinical reference data
        self.disease_glycan_associations = {
            'cancer': {
                'ovarian_cancer': [
                    'increased_branching', 'core_fucosylation', 'sialylation_changes'
                ],
                'liver_cancer': [
                    'afp_glycoforms', 'core_fucosylation', 'bisecting_glcnac'
                ],
                'breast_cancer': [
                    'n_glycan_branching', 'sialyl_lewis_x', 'truncated_o_glycans'
                ]
            },
            'inflammatory': {
                'rheumatoid_arthritis': [
                    'agalactosylated_igg', 'decreased_sialylation'
                ],
                'inflammatory_bowel_disease': [
                    'altered_mucin_glycans', 'reduced_fucosylation'
                ]
            },
            'metabolic': {
                'diabetes_type_2': [
                    'glycated_proteins', 'altered_transferrin_glycans'
                ],
                'metabolic_syndrome': [
                    'adiponectin_glycoforms', 'complement_glycans'
                ]
            }
        }
        
    async def analyze_clinical_sample(self, 
                                    sample: ClinicalSample,
                                    reference_comparison: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of clinical glycomics sample.
        
        Args:
            sample: Clinical sample data
            reference_comparison: Whether to compare against reference data
            
        Returns:
            Clinical analysis results
        """
        
        results = {
            'sample_id': sample.sample_id,
            'glycan_analyses': [],
            'clinical_interpretation': {},
            'biomarker_assessment': {},
            'recommendations': []
        }
        
        # Analyze each glycan in the sample
        for i, glycan_data in enumerate(sample.glycan_data):
            glycan_analysis = await self._analyze_single_glycan_clinical(
                glycan_data, sample.clinical_context
            )
            results['glycan_analyses'].append(glycan_analysis)
            
        # Generate clinical interpretation
        results['clinical_interpretation'] = await self._generate_clinical_interpretation(
            sample, results['glycan_analyses']
        )
        
        # Assess biomarker potential
        if reference_comparison:
            results['biomarker_assessment'] = await self._assess_biomarker_potential(
                sample, results['glycan_analyses']
            )
            
        # Generate clinical recommendations
        results['recommendations'] = self._generate_clinical_recommendations(
            sample, results
        )
        
        return results
        
    async def _analyze_single_glycan_clinical(self,
                                            glycan_data: Dict[str, Any],
                                            clinical_context: Dict[str, Any]) -> ReasoningResponse:
        """Analyze single glycan with clinical context."""
        
        # Create enhanced reasoning request with clinical context
        request = ReasoningRequest(
            request_id=f"clinical_{hash(str(glycan_data))}",
            structure=glycan_data.get('structure'),
            spectra=glycan_data.get('spectra'),
            text_description=glycan_data.get('description'),
            reasoning_tasks=[
                ReasoningType.STRUCTURE_ANALYSIS,
                ReasoningType.FUNCTION_PREDICTION,
                ReasoningType.PATHWAY_INFERENCE
            ],
            context={
                'application_domain': ApplicationDomain.CLINICAL_DIAGNOSTICS.value,
                'clinical_context': clinical_context,
                'focus_areas': ['disease_association', 'biomarker_potential']
            }
        )
        
        return await self.orchestrator.integrator.process_reasoning_request(request)
        
    async def _generate_clinical_interpretation(self,
                                             sample: ClinicalSample,
                                             analyses: List[ReasoningResponse]) -> Dict[str, Any]:
        """Generate clinical interpretation of glycan analyses."""
        
        interpretation = {
            'overall_assessment': '',
            'disease_associations': [],
            'pathway_alterations': [],
            'functional_implications': [],
            'confidence_level': 'moderate'
        }
        
        # Extract key findings from analyses
        structural_features = []
        functional_predictions = []
        
        for analysis in analyses:
            if analysis.status == 'success':
                for chain in analysis.reasoning_chains:
                    if chain.reasoning_type == ReasoningType.STRUCTURE_ANALYSIS:
                        # Extract structural features
                        for step in chain.steps:
                            if 'motifs' in step.output_data:
                                motifs = step.output_data['motifs']
                                structural_features.extend([m.get('name') for m in motifs if isinstance(m, dict)])
                                
                    elif chain.reasoning_type == ReasoningType.FUNCTION_PREDICTION:
                        # Extract functional predictions
                        for step in chain.steps:
                            if 'functions' in step.output_data:
                                functions = step.output_data['functions']
                                if isinstance(functions, list):
                                    functional_predictions.extend(functions)
                                    
        # Map to clinical conditions
        clinical_condition = sample.clinical_context.get('condition', 'unknown')
        
        if clinical_condition in ['cancer', 'oncology']:
            interpretation['disease_associations'] = self._assess_cancer_associations(structural_features)
        elif clinical_condition in ['inflammation', 'autoimmune']:
            interpretation['disease_associations'] = self._assess_inflammatory_associations(structural_features)
        elif clinical_condition in ['metabolic', 'diabetes']:
            interpretation['disease_associations'] = self._assess_metabolic_associations(structural_features)
            
        # Generate overall assessment
        if interpretation['disease_associations']:
            interpretation['overall_assessment'] = f"Glycan profile shows features consistent with {clinical_condition} conditions"
        else:
            interpretation['overall_assessment'] = "Glycan profile requires further investigation"
            
        return interpretation
        
    def _assess_cancer_associations(self, features: List[str]) -> List[Dict[str, Any]]:
        """Assess cancer-related glycan associations."""
        
        associations = []
        
        if 'core_fucose' in features or 'core_fucosylation' in features:
            associations.append({
                'cancer_type': 'hepatocellular_carcinoma',
                'feature': 'core_fucosylation',
                'clinical_significance': 'Increased core fucosylation associated with liver cancer progression',
                'confidence': 0.8
            })
            
        if 'sialyl_lewis_x' in features:
            associations.append({
                'cancer_type': 'multiple',
                'feature': 'sialyl_lewis_x',
                'clinical_significance': 'Selectin ligand associated with metastasis potential',
                'confidence': 0.7
            })
            
        if 'complex_biantennary' in features:
            associations.append({
                'cancer_type': 'ovarian_cancer',
                'feature': 'increased_branching',
                'clinical_significance': 'Complex branching patterns in ovarian cancer',
                'confidence': 0.6
            })
            
        return associations
        
    def _assess_inflammatory_associations(self, features: List[str]) -> List[Dict[str, Any]]:
        """Assess inflammation-related glycan associations."""
        
        associations = []
        
        # Look for features associated with inflammatory conditions
        if any(feature in features for feature in ['decreased_sialylation', 'agalactosylated']):
            associations.append({
                'condition': 'rheumatoid_arthritis',
                'feature': 'igg_glycosylation_changes',
                'clinical_significance': 'Altered IgG glycosylation in autoimmune inflammation',
                'confidence': 0.7
            })
            
        return associations
        
    def _assess_metabolic_associations(self, features: List[str]) -> List[Dict[str, Any]]:
        """Assess metabolic disorder glycan associations."""
        
        associations = []
        
        # Look for metabolic disorder markers
        if 'glycated' in ' '.join(features).lower():
            associations.append({
                'condition': 'diabetes',
                'feature': 'protein_glycation',
                'clinical_significance': 'Non-enzymatic glycation indicates hyperglycemia',
                'confidence': 0.9
            })
            
        return associations
        
    async def _assess_biomarker_potential(self,
                                        sample: ClinicalSample,
                                        analyses: List[ReasoningResponse]) -> Dict[str, Any]:
        """Assess biomarker potential of identified glycan features."""
        
        assessment = {
            'potential_biomarkers': [],
            'validation_requirements': [],
            'clinical_utility': 'unknown'
        }
        
        # Analyze consistency and specificity of findings
        feature_confidence = {}
        
        for analysis in analyses:
            if analysis.confidence_score > 0.7:
                # Extract high-confidence features
                for chain in analysis.reasoning_chains:
                    if chain.overall_confidence > 0.7:
                        feature_key = f"{chain.reasoning_type.value}_findings"
                        feature_confidence[feature_key] = feature_confidence.get(feature_key, 0) + 1
                        
        # Identify potential biomarkers
        for feature, count in feature_confidence.items():
            if count >= 2:  # Found in multiple analyses
                assessment['potential_biomarkers'].append({
                    'feature': feature,
                    'frequency': count,
                    'biomarker_type': 'diagnostic' if 'structure' in feature else 'functional',
                    'validation_status': 'requires_validation'
                })
                
        # Set validation requirements
        if assessment['potential_biomarkers']:
            assessment['validation_requirements'] = [
                'larger_cohort_validation',
                'analytical_method_validation',
                'clinical_correlation_studies'
            ]
            assessment['clinical_utility'] = 'promising'
        else:
            assessment['clinical_utility'] = 'limited'
            
        return assessment
        
    def _generate_clinical_recommendations(self,
                                         sample: ClinicalSample,
                                         results: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on analysis."""
        
        recommendations = []
        
        # Based on biomarker assessment
        biomarker_assessment = results.get('biomarker_assessment', {})
        
        if biomarker_assessment.get('clinical_utility') == 'promising':
            recommendations.append("Consider inclusion in biomarker validation study")
            recommendations.append("Recommend follow-up sampling for longitudinal analysis")
            
        # Based on disease associations
        disease_associations = results.get('clinical_interpretation', {}).get('disease_associations', [])
        
        if disease_associations:
            recommendations.append("Correlate findings with clinical outcomes")
            recommendations.append("Consider glycan-targeted therapeutic approaches")
            
        # General recommendations
        recommendations.append("Archive sample for future comparative studies")
        
        if not recommendations:
            recommendations.append("Standard monitoring and follow-up recommended")
            
        return recommendations


class DrugDiscoveryApplication:
    """
    Drug discovery support application.
    
    Provides glycan-based drug target analysis, ligand design support,
    and therapeutic mechanism investigation.
    """
    
    def __init__(self, orchestrator: ReasoningOrchestrator):
        self.orchestrator = orchestrator
        
        # Drug target databases (simplified)
        self.glycan_drug_targets = {
            'selectins': {
                'P-selectin': {
                    'ligands': ['sialyl_lewis_x', 'psgl1_mimetics'],
                    'therapeutic_areas': ['inflammation', 'cancer_metastasis']
                },
                'E-selectin': {
                    'ligands': ['sialyl_lewis_x', 'glycomimetics'],
                    'therapeutic_areas': ['inflammation', 'atherosclerosis']
                }
            },
            'lectins': {
                'galectin-3': {
                    'ligands': ['galactoside_derivatives', 'lactose_analogs'],
                    'therapeutic_areas': ['cancer', 'fibrosis', 'heart_failure']
                },
                'dc-sign': {
                    'ligands': ['mannose_derivatives', 'fucose_conjugates'],
                    'therapeutic_areas': ['infectious_disease', 'vaccine_adjuvants']
                }
            }
        }
        
    async def analyze_drug_target_interaction(self,
                                            glycan_structure: str,
                                            target: DrugTarget) -> Dict[str, Any]:
        """
        Analyze glycan-drug target interaction potential.
        
        Args:
            glycan_structure: WURCS or glycan structure
            target: Drug target information
            
        Returns:
            Drug target interaction analysis
        """
        
        # Perform comprehensive glycan analysis with drug discovery focus
        request = ReasoningRequest(
            request_id=f"drug_target_{target.target_id}_{hash(glycan_structure)}",
            structure=glycan_structure,
            reasoning_tasks=[
                ReasoningType.STRUCTURE_ANALYSIS,
                ReasoningType.FUNCTION_PREDICTION,
                ReasoningType.SIMILARITY_ANALYSIS
            ],
            context={
                'application_domain': ApplicationDomain.DRUG_DISCOVERY.value,
                'target_information': {
                    'target_id': target.target_id,
                    'name': target.name,
                    'protein_class': target.protein_class,
                    'binding_sites': target.glycan_binding_sites
                },
                'focus_areas': ['binding_affinity', 'selectivity', 'drug_properties']
            }
        )
        
        analysis = await self.orchestrator.integrator.process_reasoning_request(request)
        
        # Generate drug discovery insights
        insights = await self._generate_drug_discovery_insights(
            glycan_structure, target, analysis
        )
        
        return {
            'glycan_structure': glycan_structure,
            'target': target,
            'structural_analysis': analysis,
            'drug_discovery_insights': insights,
            'recommendations': self._generate_drug_discovery_recommendations(insights)
        }
        
    async def _generate_drug_discovery_insights(self,
                                              glycan_structure: str,
                                              target: DrugTarget,
                                              analysis: ReasoningResponse) -> Dict[str, Any]:
        """Generate drug discovery-specific insights."""
        
        insights = {
            'binding_potential': 'unknown',
            'selectivity_factors': [],
            'druggability_assessment': {},
            'optimization_suggestions': []
        }
        
        # Analyze structural features for binding
        if analysis.status == 'success':
            for chain in analysis.reasoning_chains:
                if chain.reasoning_type == ReasoningType.STRUCTURE_ANALYSIS:
                    # Extract binding-relevant features
                    binding_features = self._extract_binding_features(chain)
                    insights['selectivity_factors'] = binding_features
                    
                    # Assess binding potential
                    insights['binding_potential'] = self._assess_binding_potential(
                        binding_features, target
                    )
                    
        # Druggability assessment
        insights['druggability_assessment'] = self._assess_druggability(
            glycan_structure, target
        )
        
        # Optimization suggestions
        insights['optimization_suggestions'] = self._generate_optimization_suggestions(
            insights['binding_potential'], 
            insights['druggability_assessment']
        )
        
        return insights
        
    def _extract_binding_features(self, chain: ReasoningChain) -> List[str]:
        """Extract features relevant to protein binding."""
        
        binding_features = []
        
        for step in chain.steps:
            if 'motifs' in step.output_data:
                motifs = step.output_data['motifs']
                for motif in motifs:
                    if isinstance(motif, dict):
                        motif_name = motif.get('name', '')
                        
                        # Key binding motifs
                        if any(term in motif_name.lower() for term in 
                              ['lewis', 'selectin', 'galactose', 'mannose', 'sialic']):
                            binding_features.append(motif_name)
                            
        return binding_features
        
    def _assess_binding_potential(self, features: List[str], target: DrugTarget) -> str:
        """Assess binding potential based on features and target."""
        
        # Match features to known target preferences
        target_class = target.protein_class.lower()
        
        if 'selectin' in target_class:
            if any('lewis' in feature.lower() for feature in features):
                return 'high'
            elif any('sialic' in feature.lower() for feature in features):
                return 'moderate'
                
        elif 'lectin' in target_class:
            if any(term in ' '.join(features).lower() for term in ['mannose', 'galactose']):
                return 'moderate'
                
        # Default assessment
        return 'low'
        
    def _assess_druggability(self, glycan_structure: str, target: DrugTarget) -> Dict[str, Any]:
        """Assess druggability of glycan-target interaction."""
        
        assessment = {
            'molecular_weight': 'unknown',
            'synthetic_accessibility': 'moderate',
            'stability': 'unknown',
            'selectivity_potential': 'moderate',
            'overall_score': 0.5
        }
        
        # Assess based on glycan complexity
        if 'WURCS' in glycan_structure:
            # Count residues (simplified)
            residue_count = glycan_structure.count('[')
            
            if residue_count <= 3:
                assessment['synthetic_accessibility'] = 'high'
                assessment['overall_score'] += 0.2
            elif residue_count <= 6:
                assessment['synthetic_accessibility'] = 'moderate'
            else:
                assessment['synthetic_accessibility'] = 'low'
                assessment['overall_score'] -= 0.2
                
        # Target-specific factors
        if target.validation_status == 'clinical':
            assessment['overall_score'] += 0.3
        elif target.validation_status == 'preclinical':
            assessment['overall_score'] += 0.1
            
        return assessment
        
    def _generate_optimization_suggestions(self,
                                         binding_potential: str,
                                         druggability: Dict[str, Any]) -> List[str]:
        """Generate suggestions for glycan optimization."""
        
        suggestions = []
        
        if binding_potential == 'high':
            suggestions.append("Maintain core binding epitope in derivatives")
            suggestions.append("Explore minimal binding motif for synthetic accessibility")
            
        elif binding_potential == 'moderate':
            suggestions.append("Enhance binding affinity through additional interactions")
            suggestions.append("Consider multivalent presentation")
            
        else:
            suggestions.append("Significant structural modifications needed")
            suggestions.append("Consider alternative glycan scaffolds")
            
        # Druggability-based suggestions
        if druggability.get('synthetic_accessibility') == 'low':
            suggestions.append("Simplify structure for improved synthesis")
            suggestions.append("Consider glycomimetic approaches")
            
        if druggability.get('overall_score', 0) < 0.4:
            suggestions.append("Evaluate alternative therapeutic modalities")
            
        return suggestions
        
    def _generate_drug_discovery_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate drug discovery recommendations."""
        
        recommendations = []
        
        binding_potential = insights.get('binding_potential', 'unknown')
        druggability = insights.get('druggability_assessment', {})
        
        if binding_potential == 'high' and druggability.get('overall_score', 0) > 0.6:
            recommendations.append("Proceed with lead optimization")
            recommendations.append("Conduct binding affinity studies")
            recommendations.append("Evaluate ADMET properties")
            
        elif binding_potential == 'moderate':
            recommendations.append("Optimize binding interactions")
            recommendations.append("Consider structure-activity relationship studies")
            
        else:
            recommendations.append("Explore alternative approaches")
            recommendations.append("Consider immunoconjugate strategies")
            
        # General recommendations
        recommendations.append("Validate target engagement in relevant models")
        recommendations.append("Assess selectivity profile against related targets")
        
        return recommendations


class EducationalApplication:
    """
    Educational application for glycobiology and glycoinformatics.
    
    Provides interactive learning tools, case studies, and 
    step-by-step explanations of glycan analysis.
    """
    
    def __init__(self, orchestrator: ReasoningOrchestrator):
        self.orchestrator = orchestrator
        
        # Educational content templates
        self.learning_modules = {
            'basic_glycobiology': {
                'title': 'Introduction to Glycobiology',
                'topics': ['monosaccharides', 'glycosidic_bonds', 'glycan_types'],
                'difficulty': 'beginner'
            },
            'structural_analysis': {
                'title': 'Glycan Structural Analysis',
                'topics': ['wurcs_notation', 'motif_recognition', 'branching_patterns'],
                'difficulty': 'intermediate'
            },
            'mass_spectrometry': {
                'title': 'Glycan MS Analysis',
                'topics': ['fragmentation', 'peak_assignment', 'quantification'],
                'difficulty': 'intermediate'
            },
            'clinical_glycomics': {
                'title': 'Clinical Applications',
                'topics': ['biomarkers', 'disease_associations', 'therapeutic_targets'],
                'difficulty': 'advanced'
            }
        }
        
    async def create_interactive_tutorial(self,
                                        glycan_structure: str,
                                        learning_objectives: List[str]) -> Dict[str, Any]:
        """
        Create interactive tutorial for specific glycan.
        
        Args:
            glycan_structure: Glycan structure for tutorial
            learning_objectives: What students should learn
            
        Returns:
            Interactive tutorial content
        """
        
        # Generate comprehensive analysis with educational context
        request = ReasoningRequest(
            request_id=f"tutorial_{hash(glycan_structure)}",
            structure=glycan_structure,
            reasoning_tasks=[
                ReasoningType.STRUCTURE_ANALYSIS,
                ReasoningType.FRAGMENTATION_PREDICTION,
                ReasoningType.FUNCTION_PREDICTION
            ],
            context={
                'application_domain': ApplicationDomain.EDUCATIONAL.value,
                'learning_objectives': learning_objectives,
                'explanation_level': 'detailed',
                'include_examples': True
            }
        )
        
        analysis = await self.orchestrator.integrator.process_reasoning_request(request)
        
        # Create tutorial content
        tutorial = {
            'glycan_structure': glycan_structure,
            'learning_objectives': learning_objectives,
            'tutorial_steps': await self._create_tutorial_steps(analysis),
            'interactive_elements': self._create_interactive_elements(analysis),
            'assessment_questions': self._generate_assessment_questions(analysis),
            'additional_resources': self._suggest_additional_resources(learning_objectives)
        }
        
        return tutorial
        
    async def _create_tutorial_steps(self, analysis: ReasoningResponse) -> List[Dict[str, Any]]:
        """Create step-by-step tutorial from analysis."""
        
        tutorial_steps = []
        
        if analysis.status == 'success':
            for i, chain in enumerate(analysis.reasoning_chains):
                
                step = {
                    'step_number': i + 1,
                    'title': self._get_tutorial_title(chain.reasoning_type),
                    'explanation': self._create_educational_explanation(chain),
                    'key_concepts': self._extract_key_concepts(chain),
                    'visual_aids': self._suggest_visual_aids(chain.reasoning_type)
                }
                
                tutorial_steps.append(step)
                
        return tutorial_steps
        
    def _get_tutorial_title(self, reasoning_type: ReasoningType) -> str:
        """Get educational title for reasoning type."""
        
        titles = {
            ReasoningType.STRUCTURE_ANALYSIS: "Understanding Glycan Structure",
            ReasoningType.FRAGMENTATION_PREDICTION: "Mass Spectrometry Fragmentation",
            ReasoningType.FUNCTION_PREDICTION: "Biological Functions of Glycans",
            ReasoningType.PATHWAY_INFERENCE: "Glycan Biosynthesis Pathways"
        }
        
        return titles.get(reasoning_type, f"Analysis: {reasoning_type.value}")
        
    def _create_educational_explanation(self, chain: ReasoningChain) -> str:
        """Create detailed educational explanation."""
        
        explanation_parts = []
        
        # Introduction
        explanation_parts.append(
            f"In this step, we will analyze the {chain.reasoning_type.value.replace('_', ' ')} "
            f"of the glycan structure."
        )
        
        # Step-by-step breakdown
        for i, step in enumerate(chain.steps, 1):
            explanation_parts.append(
                f"\n**Sub-step {i}: {step.step_type.replace('_', ' ').title()}**\n"
                f"{step.description}\n"
            )
            
            # Add educational context
            if step.evidence:
                explanation_parts.append("**Key Evidence:**")
                for evidence in step.evidence:
                    explanation_parts.append(f"- {evidence}")
                    
        # Conclusion
        explanation_parts.append(f"\n**Conclusion:** {chain.final_conclusion}")
        
        return "\n".join(explanation_parts)
        
    def _extract_key_concepts(self, chain: ReasoningChain) -> List[str]:
        """Extract key learning concepts."""
        
        concepts = []
        
        if chain.reasoning_type == ReasoningType.STRUCTURE_ANALYSIS:
            concepts.extend([
                "Monosaccharide composition",
                "Glycosidic linkages", 
                "Structural motifs",
                "Branching patterns"
            ])
        elif chain.reasoning_type == ReasoningType.FRAGMENTATION_PREDICTION:
            concepts.extend([
                "Glycosidic bond cleavage",
                "Cross-ring fragmentation",
                "Ion types (Y, B, A, C)",
                "Mass spectrometry principles"
            ])
        elif chain.reasoning_type == ReasoningType.FUNCTION_PREDICTION:
            concepts.extend([
                "Structure-function relationships",
                "Protein-carbohydrate interactions",
                "Biological recognition",
                "Cellular functions"
            ])
            
        return concepts
        
    def _suggest_visual_aids(self, reasoning_type: ReasoningType) -> List[str]:
        """Suggest visual aids for learning."""
        
        visual_aids = []
        
        if reasoning_type == ReasoningType.STRUCTURE_ANALYSIS:
            visual_aids.extend([
                "3D molecular structure",
                "Symbolic notation diagram",
                "Monosaccharide composition chart"
            ])
        elif reasoning_type == ReasoningType.FRAGMENTATION_PREDICTION:
            visual_aids.extend([
                "Mass spectrum with peak assignments",
                "Fragmentation pathway diagram",
                "Ion structure illustrations"
            ])
            
        return visual_aids
        
    def _create_interactive_elements(self, analysis: ReasoningResponse) -> List[Dict[str, Any]]:
        """Create interactive learning elements."""
        
        elements = [
            {
                'type': 'structure_explorer',
                'description': 'Interactive 3D structure viewer',
                'features': ['rotation', 'zoom', 'highlight_motifs']
            },
            {
                'type': 'quiz',
                'description': 'Knowledge check questions',
                'question_count': 5
            },
            {
                'type': 'drag_drop',
                'description': 'Match structures to functions',
                'difficulty': 'intermediate'
            }
        ]
        
        return elements
        
    def _generate_assessment_questions(self, analysis: ReasoningResponse) -> List[Dict[str, Any]]:
        """Generate assessment questions."""
        
        questions = [
            {
                'type': 'multiple_choice',
                'question': 'What is the main biological function of this glycan?',
                'options': ['A) Energy storage', 'B) Cell recognition', 'C) Structural support', 'D) Enzymatic activity'],
                'correct_answer': 'B',
                'explanation': 'Based on the structural motifs identified, this glycan is involved in cell recognition processes.'
            },
            {
                'type': 'short_answer',
                'question': 'Describe the key structural features that determine this glycan\'s biological activity.',
                'sample_answer': 'The key features include specific motifs, branching patterns, and terminal modifications.',
                'points': 5
            }
        ]
        
        return questions
        
    def _suggest_additional_resources(self, learning_objectives: List[str]) -> List[Dict[str, Any]]:
        """Suggest additional learning resources."""
        
        resources = [
            {
                'type': 'database',
                'name': 'GlyTouCan',
                'url': 'https://glytoucan.org',
                'description': 'International glycan structure repository'
            },
            {
                'type': 'tool',
                'name': 'GlycanBuilder',
                'description': 'Interactive glycan structure drawing tool'
            },
            {
                'type': 'literature',
                'title': 'Essentials of Glycobiology',
                'description': 'Comprehensive textbook on glycobiology'
            }
        ]
        
        return resources


def create_application_suite(orchestrator: ReasoningOrchestrator) -> Dict[str, Any]:
    """
    Create complete application suite for GlycoGoT.
    
    Args:
        orchestrator: Reasoning orchestrator
        
    Returns:
        Dictionary with all applications
    """
    
    return {
        'clinical_analysis': ClinicalAnalysisApplication(orchestrator),
        'drug_discovery': DrugDiscoveryApplication(orchestrator),
        'educational': EducationalApplication(orchestrator)
    }