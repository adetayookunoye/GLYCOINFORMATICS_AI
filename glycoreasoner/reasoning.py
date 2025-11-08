"""
GlycoGOT (Glycoinformatics Graph of Thought) Reasoning Engine

This module implements advanced reasoning capabilities for glycoinformatics,
including multi-step logical inference, hypothesis generation, causal reasoning,
and graph-based analysis over knowledge graphs and AI model predictions.
"""

import logging
import json
import time
import re
import random
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import itertools

# Optional imports for enhanced functionality
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

try:
    from rdflib import Graph, URIRef, Literal, Namespace
    from rdflib.namespace import RDF, RDFS, OWL
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning supported by GlycoGOT"""
    DEDUCTIVE = "deductive"           # Logical deduction from premises
    INDUCTIVE = "inductive"           # Pattern-based generalization
    ABDUCTIVE = "abductive"           # Best explanation hypothesis
    ANALOGICAL = "analogical"         # Reasoning by analogy
    CAUSAL = "causal"                # Cause-effect relationships
    TEMPORAL = "temporal"             # Time-based reasoning
    SPATIAL = "spatial"              # Structure-based reasoning
    COMPOSITIONAL = "compositional"   # Part-whole relationships


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning conclusions"""
    CERTAIN = 1.0
    VERY_HIGH = 0.9
    HIGH = 0.8
    MODERATE = 0.6
    LOW = 0.4
    VERY_LOW = 0.2
    UNCERTAIN = 0.1


@dataclass
class ReasoningPremise:
    """A premise used in reasoning"""
    statement: str                    # The premise statement
    evidence: Optional[str] = None    # Supporting evidence
    source: Optional[str] = None      # Source of the premise
    confidence: float = 1.0           # Confidence in the premise
    premise_type: str = "fact"        # Type: fact, rule, observation, hypothesis


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain"""
    step_id: int
    reasoning_type: ReasoningType
    premises: List[ReasoningPremise]
    conclusion: str
    confidence: float
    explanation: str
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


@dataclass
class ReasoningPlan:
    """A complete reasoning plan for solving a problem"""
    plan_id: str
    goal: str                         # The reasoning goal
    context: Dict[str, Any]           # Problem context
    steps: List[Dict[str, Any]]       # Planned reasoning steps
    constraints: List[str] = field(default_factory=list)
    expected_outcome: Optional[str] = None
    max_steps: int = 10
    timeout: float = 300.0            # 5 minutes default


@dataclass
class ReasoningResult:
    """Result of a reasoning process"""
    goal: str
    conclusion: str
    confidence: float
    reasoning_chain: List[ReasoningStep]
    supporting_evidence: List[str]
    alternative_hypotheses: List[Tuple[str, float]] = field(default_factory=list)
    execution_time: float = 0.0
    total_steps: int = 0
    success: bool = True
    error_message: Optional[str] = None


class ReasoningRule:
    """A logical rule for reasoning"""
    
    def __init__(self, 
                 rule_id: str,
                 name: str,
                 premises_pattern: List[str],
                 conclusion_pattern: str,
                 confidence_modifier: float = 1.0,
                 rule_type: str = "general"):
        self.rule_id = rule_id
        self.name = name
        self.premises_pattern = premises_pattern
        self.conclusion_pattern = conclusion_pattern
        self.confidence_modifier = confidence_modifier
        self.rule_type = rule_type
        self.usage_count = 0
        
    def can_apply(self, premises: List[ReasoningPremise]) -> bool:
        """Check if this rule can be applied to given premises"""
        # Simple pattern matching - in practice would be more sophisticated
        premise_texts = [p.statement.lower() for p in premises]
        
        for pattern in self.premises_pattern:
            pattern_lower = pattern.lower()
            if not any(pattern_lower in text for text in premise_texts):
                return False
        return True
        
    def apply(self, premises: List[ReasoningPremise]) -> Tuple[str, float]:
        """Apply the rule to premises and generate conclusion"""
        self.usage_count += 1
        
        # Calculate confidence based on premises and rule confidence
        premise_confidence = min(p.confidence for p in premises) if premises else 0.5
        final_confidence = premise_confidence * self.confidence_modifier
        
        # Generate conclusion (simplified pattern substitution)
        conclusion = self.conclusion_pattern
        
        return conclusion, final_confidence


class GlycoKnowledgeBase:
    """Knowledge base for glycoinformatics reasoning"""
    
    def __init__(self):
        self.facts = []
        self.rules = []
        self.ontology_graph = None
        self.glycan_structures = {}
        self.protein_interactions = {}
        self.experimental_data = {}
        
        # Initialize with glycoinformatics-specific rules
        self._initialize_glyco_rules()
        
    def _initialize_glyco_rules(self):
        """Initialize with domain-specific reasoning rules"""
        
        # Structural reasoning rules
        self.add_rule(ReasoningRule(
            rule_id="glycan_mass_calc",
            name="Glycan Mass Calculation",
            premises_pattern=["monosaccharide composition", "linkage pattern"],
            conclusion_pattern="theoretical mass can be calculated",
            confidence_modifier=0.95,
            rule_type="structural"
        ))
        
        self.add_rule(ReasoningRule(
            rule_id="spectra_glycan_match",
            name="Spectra-Glycan Matching",
            premises_pattern=["mass spectrum", "fragmentation pattern"],
            conclusion_pattern="likely glycan structure identified",
            confidence_modifier=0.8,
            rule_type="analytical"
        ))
        
        # Functional reasoning rules
        self.add_rule(ReasoningRule(
            rule_id="glycan_protein_function",
            name="Glycan-Protein Function",
            premises_pattern=["glycan attachment site", "protein function"],
            conclusion_pattern="glycosylation affects protein function",
            confidence_modifier=0.85,
            rule_type="functional"
        ))
        
        # Biosynthetic pathway rules
        self.add_rule(ReasoningRule(
            rule_id="enzyme_substrate",
            name="Enzyme Substrate Specificity",
            premises_pattern=["enzyme present", "substrate available"],
            conclusion_pattern="glycosylation reaction possible",
            confidence_modifier=0.9,
            rule_type="biosynthetic"
        ))
        
        # Disease association rules
        self.add_rule(ReasoningRule(
            rule_id="glycan_disease_association",
            name="Glycan Disease Association",
            premises_pattern=["altered glycan pattern", "disease phenotype"],
            conclusion_pattern="glycan may be biomarker for disease",
            confidence_modifier=0.7,
            rule_type="biomedical"
        ))
        
    def add_fact(self, statement: str, evidence: Optional[str] = None, confidence: float = 1.0):
        """Add a fact to the knowledge base"""
        fact = ReasoningPremise(
            statement=statement,
            evidence=evidence,
            confidence=confidence,
            premise_type="fact"
        )
        self.facts.append(fact)
        
    def add_rule(self, rule: ReasoningRule):
        """Add a reasoning rule"""
        self.rules.append(rule)
        
    def query_facts(self, pattern: str) -> List[ReasoningPremise]:
        """Query facts matching a pattern"""
        pattern_lower = pattern.lower()
        matching_facts = []
        
        for fact in self.facts:
            if pattern_lower in fact.statement.lower():
                matching_facts.append(fact)
                
        return matching_facts
        
    def get_applicable_rules(self, premises: List[ReasoningPremise]) -> List[ReasoningRule]:
        """Get rules that can be applied to given premises"""
        applicable_rules = []
        
        for rule in self.rules:
            if rule.can_apply(premises):
                applicable_rules.append(rule)
                
        return applicable_rules


class HypothesisGenerator:
    """Generates hypotheses for glycoinformatics problems"""
    
    def __init__(self, knowledge_base: GlycoKnowledgeBase):
        self.kb = knowledge_base
        self.hypothesis_templates = self._initialize_templates()
        
    def _initialize_templates(self) -> List[Dict[str, Any]]:
        """Initialize hypothesis templates"""
        return [
            {
                "template": "The glycan {glycan} binds to protein {protein} with {affinity} affinity",
                "variables": ["glycan", "protein", "affinity"],
                "context": "protein_binding"
            },
            {
                "template": "Mass spectrum peak at {mass} m/z corresponds to {fragment} fragment",
                "variables": ["mass", "fragment"],
                "context": "mass_spectrometry"
            },
            {
                "template": "Glycan {glycan} is biosynthesized via {pathway} pathway",
                "variables": ["glycan", "pathway"],
                "context": "biosynthesis"
            },
            {
                "template": "Altered {glycan} levels indicate {condition} condition",
                "variables": ["glycan", "condition"],
                "context": "biomarker"
            }
        ]
        
    def generate_hypotheses(self, 
                          context: Dict[str, Any], 
                          max_hypotheses: int = 5) -> List[Tuple[str, float]]:
        """Generate hypotheses based on context"""
        hypotheses = []
        
        # Extract relevant information from context
        glycans = context.get("glycans", [])
        proteins = context.get("proteins", [])
        masses = context.get("masses", [])
        conditions = context.get("conditions", [])
        
        # Generate hypotheses from templates
        for template_info in self.hypothesis_templates:
            template = template_info["template"]
            variables = template_info["variables"]
            
            # Fill template with available data
            if template_info["context"] == "protein_binding" and glycans and proteins:
                for glycan in glycans[:2]:  # Limit combinations
                    for protein in proteins[:2]:
                        for affinity in ["high", "moderate", "low"]:
                            hypothesis = template.format(
                                glycan=glycan, 
                                protein=protein, 
                                affinity=affinity
                            )
                            confidence = self._estimate_hypothesis_confidence(hypothesis, context)
                            hypotheses.append((hypothesis, confidence))
                            
            elif template_info["context"] == "mass_spectrometry" and masses:
                for mass in masses:
                    for fragment_type in ["Y-ion", "B-ion", "cross-ring", "internal"]:
                        hypothesis = template.format(mass=mass, fragment=fragment_type)
                        confidence = self._estimate_hypothesis_confidence(hypothesis, context)
                        hypotheses.append((hypothesis, confidence))
                        
        # Sort by confidence and return top hypotheses
        hypotheses.sort(key=lambda x: x[1], reverse=True)
        return hypotheses[:max_hypotheses]
        
    def _estimate_hypothesis_confidence(self, hypothesis: str, context: Dict[str, Any]) -> float:
        """Estimate confidence in a hypothesis based on available evidence"""
        # Simple confidence estimation - in practice would be more sophisticated
        base_confidence = 0.5
        
        # Boost confidence based on supporting evidence
        if "experimental_evidence" in context:
            base_confidence += 0.2
            
        if "literature_support" in context:
            base_confidence += 0.1
            
        if "structural_similarity" in context:
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)


class CausalReasoner:
    """Performs causal reasoning for glycoinformatics"""
    
    def __init__(self):
        self.causal_graph = nx.DiGraph() if HAS_NETWORKX else None
        self.causal_relationships = []
        
    def add_causal_relationship(self, 
                              cause: str, 
                              effect: str, 
                              strength: float = 0.8,
                              evidence: Optional[str] = None):
        """Add a causal relationship"""
        relationship = {
            "cause": cause,
            "effect": effect,
            "strength": strength,
            "evidence": evidence
        }
        self.causal_relationships.append(relationship)
        
        if self.causal_graph is not None:
            self.causal_graph.add_edge(cause, effect, weight=strength)
            
    def find_causal_path(self, cause: str, effect: str) -> Optional[List[str]]:
        """Find causal path between cause and effect"""
        if self.causal_graph is None or not HAS_NETWORKX:
            return None
            
        try:
            return nx.shortest_path(self.causal_graph, cause, effect)
        except nx.NetworkXNoPath:
            return None
            
    def analyze_causality(self, 
                         observations: List[str],
                         target_effect: str) -> Dict[str, float]:
        """Analyze potential causes for an observed effect"""
        potential_causes = {}
        
        for obs in observations:
            causal_path = self.find_causal_path(obs, target_effect)
            if causal_path:
                # Calculate causal strength along path
                strength = self._calculate_path_strength(causal_path)
                potential_causes[obs] = strength
                
        return potential_causes
        
    def _calculate_path_strength(self, path: List[str]) -> float:
        """Calculate strength of causal path"""
        if self.causal_graph is None or len(path) < 2:
            return 0.0
            
        total_strength = 1.0
        for i in range(len(path) - 1):
            edge_data = self.causal_graph.get_edge_data(path[i], path[i+1])
            if edge_data:
                total_strength *= edge_data.get('weight', 0.5)
            else:
                total_strength *= 0.1  # Weak connection if no explicit weight
                
        return total_strength


class GlycoGOTReasoner:
    """Main reasoning engine for Glycoinformatics Graph of Thought"""
    
    def __init__(self, knowledge_base: Optional[GlycoKnowledgeBase] = None):
        self.kb = knowledge_base or GlycoKnowledgeBase()
        self.hypothesis_generator = HypothesisGenerator(self.kb)
        self.causal_reasoner = CausalReasoner()
        self.reasoning_history = []
        
        # Initialize glycoinformatics-specific causal relationships
        self._initialize_causal_relationships()
        
    def _initialize_causal_relationships(self):
        """Initialize domain-specific causal relationships"""
        
        # Enzyme-substrate relationships
        self.causal_reasoner.add_causal_relationship(
            "GalT enzyme expression",
            "galactose addition to glycan",
            strength=0.9,
            evidence="Enzymatic specificity studies"
        )
        
        # Disease-glycan relationships  
        self.causal_reasoner.add_causal_relationship(
            "genetic mutation in glycosyltransferase",
            "altered glycan structure",
            strength=0.85,
            evidence="Genetic studies"
        )
        
        self.causal_reasoner.add_causal_relationship(
            "altered glycan structure",
            "changed protein function",
            strength=0.7,
            evidence="Functional studies"
        )
        
        # Environmental factors
        self.causal_reasoner.add_causal_relationship(
            "inflammatory conditions",
            "altered glycosylation patterns",
            strength=0.75,
            evidence="Clinical observations"
        )
        
    def reason(self, 
               goal: str,
               context: Dict[str, Any],
               reasoning_plan: Optional[ReasoningPlan] = None) -> ReasoningResult:
        """
        Perform multi-step reasoning to achieve a goal.
        
        Args:
            goal: The reasoning goal
            context: Problem context and available data
            reasoning_plan: Optional pre-defined reasoning plan
            
        Returns:
            ReasoningResult with conclusions and reasoning chain
        """
        start_time = time.time()
        
        logger.info(f"Starting GlycoGOT reasoning for goal: {goal}")
        
        try:
            # Generate reasoning plan if not provided
            if reasoning_plan is None:
                reasoning_plan = self._generate_reasoning_plan(goal, context)
                
            # Execute reasoning steps
            reasoning_chain = []
            current_premises = self._extract_initial_premises(context)
            
            for step_config in reasoning_plan.steps:
                step_result = self._execute_reasoning_step(
                    step_config, 
                    current_premises, 
                    len(reasoning_chain) + 1
                )
                reasoning_chain.append(step_result)
                
                # Add step conclusion as new premise for next step
                new_premise = ReasoningPremise(
                    statement=step_result.conclusion,
                    confidence=step_result.confidence,
                    premise_type="derived"
                )
                current_premises.append(new_premise)
                
                # Check if goal is achieved
                if self._goal_achieved(goal, step_result.conclusion):
                    break
                    
            # Generate final conclusion
            final_conclusion = self._synthesize_conclusion(goal, reasoning_chain)
            
            # Generate alternative hypotheses
            alternatives = self.hypothesis_generator.generate_hypotheses(context)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(reasoning_chain)
            
            # Collect supporting evidence
            supporting_evidence = []
            for step in reasoning_chain:
                for premise in step.premises:
                    if premise.evidence:
                        supporting_evidence.append(premise.evidence)
                        
            execution_time = time.time() - start_time
            
            result = ReasoningResult(
                goal=goal,
                conclusion=final_conclusion,
                confidence=overall_confidence,
                reasoning_chain=reasoning_chain,
                supporting_evidence=supporting_evidence,
                alternative_hypotheses=alternatives,
                execution_time=execution_time,
                total_steps=len(reasoning_chain),
                success=True
            )
            
            self.reasoning_history.append(result)
            logger.info(f"Reasoning completed successfully in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = ReasoningResult(
                goal=goal,
                conclusion="Reasoning failed due to error",
                confidence=0.0,
                reasoning_chain=[],
                supporting_evidence=[],
                execution_time=execution_time,
                total_steps=0,
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Reasoning failed: {e}")
            return error_result
            
    def _generate_reasoning_plan(self, goal: str, context: Dict[str, Any]) -> ReasoningPlan:
        """Generate a reasoning plan for the given goal"""
        
        plan_steps = []
        
        # Analyze goal type and generate appropriate steps
        goal_lower = goal.lower()
        
        if "structure" in goal_lower and ("predict" in goal_lower or "identify" in goal_lower):
            # Structure prediction workflow
            plan_steps = [
                {
                    "type": ReasoningType.INDUCTIVE.value,
                    "description": "Analyze mass spectral patterns",
                    "premises_required": ["mass_spectrum", "fragmentation_data"]
                },
                {
                    "type": ReasoningType.ANALOGICAL.value,
                    "description": "Compare with known structures",
                    "premises_required": ["structure_database", "similarity_metrics"]
                },
                {
                    "type": ReasoningType.DEDUCTIVE.value,
                    "description": "Apply structural constraints",
                    "premises_required": ["chemical_rules", "biosynthetic_pathways"]
                }
            ]
            
        elif "function" in goal_lower or "interaction" in goal_lower:
            # Functional analysis workflow
            plan_steps = [
                {
                    "type": ReasoningType.CAUSAL.value,
                    "description": "Analyze structure-function relationships",
                    "premises_required": ["glycan_structure", "protein_target"]
                },
                {
                    "type": ReasoningType.ABDUCTIVE.value,
                    "description": "Generate functional hypotheses",
                    "premises_required": ["binding_data", "functional_assays"]
                },
                {
                    "type": ReasoningType.DEDUCTIVE.value,
                    "description": "Validate against known mechanisms",
                    "premises_required": ["mechanism_database", "literature_evidence"]
                }
            ]
            
        elif "disease" in goal_lower or "biomarker" in goal_lower:
            # Disease association workflow
            plan_steps = [
                {
                    "type": ReasoningType.INDUCTIVE.value,
                    "description": "Identify altered glycan patterns",
                    "premises_required": ["patient_samples", "control_samples"]
                },
                {
                    "type": ReasoningType.CAUSAL.value,
                    "description": "Analyze causal relationships",
                    "premises_required": ["disease_mechanisms", "glycan_functions"]
                },
                {
                    "type": ReasoningType.ABDUCTIVE.value,
                    "description": "Propose biomarker candidates",
                    "premises_required": ["statistical_analysis", "clinical_relevance"]
                }
            ]
            
        else:
            # General reasoning workflow
            plan_steps = [
                {
                    "type": ReasoningType.INDUCTIVE.value,
                    "description": "Analyze available data patterns",
                    "premises_required": ["observational_data"]
                },
                {
                    "type": ReasoningType.ABDUCTIVE.value,
                    "description": "Generate explanatory hypotheses",
                    "premises_required": ["pattern_analysis"]
                },
                {
                    "type": ReasoningType.DEDUCTIVE.value,
                    "description": "Validate hypotheses against knowledge",
                    "premises_required": ["domain_knowledge"]
                }
            ]
            
        return ReasoningPlan(
            plan_id=f"plan_{int(time.time())}",
            goal=goal,
            context=context,
            steps=plan_steps,
            max_steps=len(plan_steps),
            timeout=300.0
        )
        
    def _extract_initial_premises(self, context: Dict[str, Any]) -> List[ReasoningPremise]:
        """Extract initial premises from context"""
        premises = []
        
        # Extract explicit facts from context
        if "facts" in context:
            for fact in context["facts"]:
                if isinstance(fact, str):
                    premises.append(ReasoningPremise(statement=fact, premise_type="fact"))
                elif isinstance(fact, dict):
                    premises.append(ReasoningPremise(
                        statement=fact.get("statement", ""),
                        evidence=fact.get("evidence"),
                        confidence=fact.get("confidence", 1.0),
                        premise_type="fact"
                    ))
                    
        # Extract observations
        if "observations" in context:
            for obs in context["observations"]:
                premises.append(ReasoningPremise(
                    statement=obs,
                    premise_type="observation",
                    confidence=0.8
                ))
                
        # Extract experimental data as premises
        if "experimental_data" in context:
            data = context["experimental_data"]
            if isinstance(data, dict):
                for key, value in data.items():
                    statement = f"{key}: {value}"
                    premises.append(ReasoningPremise(
                        statement=statement,
                        premise_type="experimental",
                        confidence=0.9
                    ))
                    
        return premises
        
    def _execute_reasoning_step(self, 
                               step_config: Dict[str, Any],
                               premises: List[ReasoningPremise],
                               step_id: int) -> ReasoningStep:
        """Execute a single reasoning step"""
        
        step_start_time = time.time()
        reasoning_type = ReasoningType(step_config["type"])
        
        # Filter relevant premises for this step
        relevant_premises = self._filter_relevant_premises(
            premises, 
            step_config.get("premises_required", [])
        )
        
        # Apply reasoning based on type
        if reasoning_type == ReasoningType.DEDUCTIVE:
            conclusion, confidence = self._apply_deductive_reasoning(relevant_premises)
        elif reasoning_type == ReasoningType.INDUCTIVE:
            conclusion, confidence = self._apply_inductive_reasoning(relevant_premises)
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            conclusion, confidence = self._apply_abductive_reasoning(relevant_premises)
        elif reasoning_type == ReasoningType.ANALOGICAL:
            conclusion, confidence = self._apply_analogical_reasoning(relevant_premises)
        elif reasoning_type == ReasoningType.CAUSAL:
            conclusion, confidence = self._apply_causal_reasoning(relevant_premises)
        else:
            conclusion = "Unable to apply reasoning type"
            confidence = 0.1
            
        explanation = step_config.get("description", f"Applied {reasoning_type.value} reasoning")
        execution_time = time.time() - step_start_time
        
        return ReasoningStep(
            step_id=step_id,
            reasoning_type=reasoning_type,
            premises=relevant_premises,
            conclusion=conclusion,
            confidence=confidence,
            explanation=explanation,
            execution_time=execution_time
        )
        
    def _filter_relevant_premises(self, 
                                premises: List[ReasoningPremise],
                                required_types: List[str]) -> List[ReasoningPremise]:
        """Filter premises relevant to the reasoning step"""
        if not required_types:
            return premises
            
        relevant = []
        for premise in premises:
            for req_type in required_types:
                if req_type.lower() in premise.statement.lower():
                    relevant.append(premise)
                    break
                    
        return relevant if relevant else premises  # Fallback to all premises
        
    def _apply_deductive_reasoning(self, premises: List[ReasoningPremise]) -> Tuple[str, float]:
        """Apply deductive reasoning using logical rules"""
        
        # Find applicable rules
        applicable_rules = self.kb.get_applicable_rules(premises)
        
        if applicable_rules:
            # Use the most confident applicable rule
            best_rule = max(applicable_rules, key=lambda r: r.confidence_modifier)
            conclusion, confidence = best_rule.apply(premises)
            return conclusion, confidence
        else:
            # Fallback to simple logical combination
            if premises:
                statements = [p.statement for p in premises]
                conclusion = f"Based on {', '.join(statements[:2])}, logical deduction suggests a related outcome"
                confidence = min(p.confidence for p in premises) * 0.8
                return conclusion, confidence
            else:
                return "No premises available for deduction", 0.1
                
    def _apply_inductive_reasoning(self, premises: List[ReasoningPremise]) -> Tuple[str, float]:
        """Apply inductive reasoning to find patterns"""
        
        if len(premises) < 2:
            return "Insufficient data for pattern recognition", 0.2
            
        # Simple pattern analysis - look for common terms
        all_text = " ".join(p.statement.lower() for p in premises)
        
        # Count common glycoinformatics terms
        glycan_terms = ["glycan", "sugar", "carbohydrate", "monosaccharide"]
        protein_terms = ["protein", "enzyme", "receptor", "antibody"]
        functional_terms = ["binding", "activity", "function", "interaction"]
        
        pattern_counts = {
            "structural": sum(all_text.count(term) for term in glycan_terms),
            "protein": sum(all_text.count(term) for term in protein_terms),
            "functional": sum(all_text.count(term) for term in functional_terms)
        }
        
        # Find dominant pattern
        dominant_pattern = max(pattern_counts, key=pattern_counts.get)
        pattern_strength = pattern_counts[dominant_pattern] / len(premises)
        
        conclusion = f"Pattern analysis indicates a primarily {dominant_pattern} relationship"
        confidence = min(0.9, pattern_strength / 3.0)  # Scale confidence
        
        return conclusion, confidence
        
    def _apply_abductive_reasoning(self, premises: List[ReasoningPremise]) -> Tuple[str, float]:
        """Apply abductive reasoning to find best explanations"""
        
        if not premises:
            return "No observations available for explanation", 0.1
            
        # Extract key observations
        observations = [p.statement for p in premises if p.premise_type == "observation"]
        
        if not observations:
            observations = [p.statement for p in premises[:2]]  # Use first few premises
            
        # Generate explanatory hypotheses
        context = {"observations": observations}
        hypotheses = self.hypothesis_generator.generate_hypotheses(context, max_hypotheses=3)
        
        if hypotheses:
            best_hypothesis, confidence = hypotheses[0]
            conclusion = f"Best explanation: {best_hypothesis}"
            return conclusion, confidence
        else:
            conclusion = "Multiple explanations possible for observed data"
            return conclusion, 0.5
            
    def _apply_analogical_reasoning(self, premises: List[ReasoningPremise]) -> Tuple[str, float]:
        """Apply analogical reasoning by finding similar cases"""
        
        # Simple analogical reasoning - look for structural or functional similarities
        premise_text = " ".join(p.statement for p in premises).lower()
        
        # Check for known analogous relationships
        analogies = {
            "antibody binding": "similar to lectin-glycan interactions",
            "enzyme specificity": "similar to other glycosyltransferases",
            "disease association": "similar to other glycan biomarkers",
            "mass spectrum": "similar to known glycan fragmentation patterns"
        }
        
        for key, analogy in analogies.items():
            if key in premise_text:
                conclusion = f"By analogy: {analogy}"
                confidence = 0.6  # Moderate confidence for analogical reasoning
                return conclusion, confidence
                
        conclusion = "Analogical patterns suggest similar behavior to known cases"
        return conclusion, 0.5
        
    def _apply_causal_reasoning(self, premises: List[ReasoningPremise]) -> Tuple[str, float]:
        """Apply causal reasoning to identify cause-effect relationships"""
        
        # Extract potential causes and effects from premises
        statements = [p.statement.lower() for p in premises]
        
        # Look for causal indicators
        causal_indicators = ["causes", "leads to", "results in", "affects", "influences"]
        causal_relationships = []
        
        for statement in statements:
            for indicator in causal_indicators:
                if indicator in statement:
                    causal_relationships.append(statement)
                    
        if causal_relationships:
            conclusion = f"Causal analysis identifies: {causal_relationships[0]}"
            confidence = 0.75
        else:
            # Use causal reasoner to find potential relationships
            observations = [p.statement for p in premises]
            if observations:
                # Simple causal analysis
                conclusion = "Causal relationships likely exist between observed factors"
                confidence = 0.6
            else:
                conclusion = "Insufficient information for causal analysis"
                confidence = 0.2
                
        return conclusion, confidence
        
    def _goal_achieved(self, goal: str, conclusion: str) -> bool:
        """Check if the reasoning goal has been achieved"""
        
        goal_lower = goal.lower()
        conclusion_lower = conclusion.lower()
        
        # Simple keyword matching - in practice would be more sophisticated
        goal_keywords = goal_lower.split()
        
        # Check if key goal terms appear in conclusion
        matches = sum(1 for keyword in goal_keywords 
                     if len(keyword) > 3 and keyword in conclusion_lower)
        
        return matches >= len(goal_keywords) * 0.6  # 60% keyword overlap threshold
        
    def _synthesize_conclusion(self, goal: str, reasoning_chain: List[ReasoningStep]) -> str:
        """Synthesize final conclusion from reasoning chain"""
        
        if not reasoning_chain:
            return "No reasoning steps completed"
            
        # Get the most confident conclusion
        best_step = max(reasoning_chain, key=lambda s: s.confidence)
        
        # Combine insights from multiple steps
        key_insights = []
        for step in reasoning_chain[-3:]:  # Last 3 steps
            if step.confidence > 0.5:
                key_insights.append(step.conclusion)
                
        if key_insights:
            synthesis = f"Analysis concludes: {best_step.conclusion}. "
            if len(key_insights) > 1:
                synthesis += f"Supporting evidence includes: {'; '.join(key_insights[1:])}"
            return synthesis
        else:
            return best_step.conclusion
            
    def _calculate_overall_confidence(self, reasoning_chain: List[ReasoningStep]) -> float:
        """Calculate overall confidence in the reasoning result"""
        
        if not reasoning_chain:
            return 0.0
            
        # Weighted average of step confidences, with more weight on later steps
        total_weight = 0
        weighted_sum = 0
        
        for i, step in enumerate(reasoning_chain):
            weight = i + 1  # Later steps have higher weight
            weighted_sum += step.confidence * weight
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0
        
    def get_reasoning_explanation(self, result: ReasoningResult) -> str:
        """Generate a human-readable explanation of the reasoning process"""
        
        explanation = f"Reasoning Goal: {result.goal}\n\n"
        explanation += f"Final Conclusion: {result.conclusion}\n"
        explanation += f"Overall Confidence: {result.confidence:.2f}\n\n"
        
        explanation += "Reasoning Steps:\n"
        for i, step in enumerate(result.reasoning_chain, 1):
            explanation += f"{i}. {step.explanation} ({step.reasoning_type.value})\n"
            explanation += f"   Premises: {len(step.premises)} items\n"
            explanation += f"   Conclusion: {step.conclusion}\n"
            explanation += f"   Confidence: {step.confidence:.2f}\n\n"
            
        if result.alternative_hypotheses:
            explanation += "Alternative Hypotheses:\n"
            for i, (hypothesis, confidence) in enumerate(result.alternative_hypotheses, 1):
                explanation += f"{i}. {hypothesis} (confidence: {confidence:.2f})\n"
                
        return explanation
    
    def rank_candidates(self, 
                       candidates: List[Dict[str, Any]], 
                       ranking_method: str = "confidence",
                       context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Advanced candidate ranking with multiple criteria.
        
        Args:
            candidates: List of prediction candidates
            ranking_method: Ranking strategy (confidence, consensus, evidence, hybrid)
            context: Additional context for ranking decisions
            
        Returns:
            Ranked list of candidates
        """
        if not candidates:
            return []
        
        logger.info(f"Ranking {len(candidates)} candidates using {ranking_method} method")
        
        if ranking_method == "confidence":
            return self._rank_by_confidence(candidates)
        elif ranking_method == "consensus":
            return self._rank_by_consensus(candidates, context)
        elif ranking_method == "evidence":
            return self._rank_by_evidence(candidates)
        elif ranking_method == "hybrid":
            return self._rank_by_hybrid_score(candidates, context)
        elif ranking_method == "uncertainty_aware":
            return self._rank_by_uncertainty_awareness(candidates)
        else:
            logger.warning(f"Unknown ranking method: {ranking_method}, using confidence")
            return self._rank_by_confidence(candidates)
    
    def _rank_by_confidence(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank candidates by confidence score."""
        return sorted(candidates, 
                     key=lambda c: c.get('confidence', 0.0), 
                     reverse=True)
    
    def _rank_by_consensus(self, 
                          candidates: List[Dict[str, Any]], 
                          context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Rank candidates by cross-modal consensus."""
        for candidate in candidates:
            consensus_score = self._calculate_consensus_score(candidate, context)
            candidate['consensus_score'] = consensus_score
            candidate['ranking_score'] = (
                0.6 * candidate.get('confidence', 0.0) + 
                0.4 * consensus_score
            )
        
        return sorted(candidates, 
                     key=lambda c: c['ranking_score'], 
                     reverse=True)
    
    def _rank_by_evidence(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank candidates by supporting evidence strength."""
        for candidate in candidates:
            evidence_score = self._calculate_evidence_strength(candidate)
            candidate['evidence_score'] = evidence_score
            candidate['ranking_score'] = (
                0.5 * candidate.get('confidence', 0.0) + 
                0.5 * evidence_score
            )
        
        return sorted(candidates, 
                     key=lambda c: c['ranking_score'], 
                     reverse=True)
    
    def _rank_by_hybrid_score(self, 
                             candidates: List[Dict[str, Any]], 
                             context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Rank using hybrid scoring combining multiple factors."""
        for candidate in candidates:
            # Calculate individual scores
            confidence = candidate.get('confidence', 0.0)
            evidence_score = self._calculate_evidence_strength(candidate)
            consensus_score = self._calculate_consensus_score(candidate, context)
            novelty_score = self._calculate_novelty_score(candidate)
            
            # Weighted hybrid score
            hybrid_score = (
                0.4 * confidence +
                0.25 * evidence_score + 
                0.2 * consensus_score +
                0.15 * novelty_score
            )
            
            candidate['hybrid_score'] = hybrid_score
            candidate['ranking_score'] = hybrid_score
        
        return sorted(candidates, 
                     key=lambda c: c['ranking_score'], 
                     reverse=True)
    
    def _rank_by_uncertainty_awareness(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank candidates considering uncertainty and reliability."""
        for candidate in candidates:
            confidence = candidate.get('confidence', 0.0)
            uncertainty = candidate.get('uncertainty', {})
            
            # Penalize high uncertainty
            epistemic_penalty = uncertainty.get('epistemic_uncertainty', 0.0)
            aleatoric_penalty = uncertainty.get('aleatoric_uncertainty', 0.0)
            
            # Reward prediction interval tightness
            interval = uncertainty.get('prediction_interval', {})
            interval_width = interval.get('upper', 1.0) - interval.get('lower', 0.0)
            sharpness_reward = 1.0 - interval_width
            
            # Calculate uncertainty-aware score
            uncertainty_score = (
                confidence - 
                0.3 * epistemic_penalty - 
                0.2 * aleatoric_penalty + 
                0.1 * sharpness_reward
            )
            
            candidate['uncertainty_score'] = max(0.0, uncertainty_score)
            candidate['ranking_score'] = candidate['uncertainty_score']
        
        return sorted(candidates, 
                     key=lambda c: c['ranking_score'], 
                     reverse=True)
    
    def _calculate_consensus_score(self, 
                                  candidate: Dict[str, Any], 
                                  context: Dict[str, Any] = None) -> float:
        """Calculate cross-modal consensus score."""
        if not context:
            return 0.5
        
        # Check agreement across modalities
        modal_predictions = context.get('modal_predictions', {})
        if len(modal_predictions) < 2:
            return 0.5
        
        # Simple agreement calculation
        agreements = []
        candidate_structure = candidate.get('structure', '')
        
        for modality, prediction in modal_predictions.items():
            if isinstance(prediction, str):
                # Simple string similarity
                similarity = len(set(candidate_structure.split()) & 
                               set(prediction.split())) / max(1, len(set(prediction.split())))
                agreements.append(similarity)
        
        return sum(agreements) / len(agreements) if agreements else 0.5
    
    def _calculate_evidence_strength(self, candidate: Dict[str, Any]) -> float:
        """Calculate supporting evidence strength."""
        grounding = candidate.get('grounding', {})
        
        # Count high-confidence grounding evidence
        total_evidence = 0
        strong_evidence = 0
        
        for evidence_type in ['structural_motifs', 'enzymes', 'pathways', 'organisms']:
            evidence_list = grounding.get(evidence_type, [])
            for evidence in evidence_list:
                total_evidence += 1
                if evidence.get('confidence', 0.0) > 0.7:
                    strong_evidence += 1
        
        return strong_evidence / max(1, total_evidence)
    
    def _calculate_novelty_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate novelty/diversity score."""
        # Simple novelty estimation based on structure complexity
        structure = candidate.get('structure', '')
        
        # Count unique components
        unique_chars = len(set(structure))
        length = len(structure)
        
        # Normalize novelty score
        novelty = min(1.0, unique_chars / max(1, length * 0.3))
        return novelty
    
    def generate_actionable_recommendations(self, 
                                         candidates: List[Dict[str, Any]],
                                         uncertainty_metrics: Dict[str, Any],
                                         task_type: str = "general") -> List[Dict[str, Any]]:
        """
        Generate actionable next-step recommendations based on results.
        
        Args:
            candidates: Ranked prediction candidates
            uncertainty_metrics: Uncertainty quantification results
            task_type: Type of task (spec2struct, structure2spec, etc.)
            
        Returns:
            List of actionable recommendations with priorities
        """
        recommendations = []
        
        if not candidates:
            recommendations.append({
                "action": "data_collection",
                "description": "Collect additional experimental data for analysis",
                "priority": "high",
                "effort": "medium",
                "expected_impact": "high"
            })
            return recommendations
        
        # Get top candidate and confidence
        top_candidate = candidates[0]
        top_confidence = top_candidate.get('confidence', 0.0)
        
        # Confidence-based recommendations
        if top_confidence < 0.5:
            recommendations.extend(self._generate_low_confidence_recommendations(task_type))
        elif top_confidence < 0.8:
            recommendations.extend(self._generate_medium_confidence_recommendations(task_type))
        else:
            recommendations.extend(self._generate_high_confidence_recommendations(task_type))
        
        # Uncertainty-based recommendations
        uncertainty = uncertainty_metrics.get('epistemic_uncertainty', 0.0)
        if uncertainty > 0.3:
            recommendations.extend(self._generate_high_uncertainty_recommendations())
        
        # Task-specific recommendations
        if task_type == "spec2struct":
            recommendations.extend(self._generate_spec2struct_recommendations(candidates))
        elif task_type == "structure2spec":
            recommendations.extend(self._generate_structure2spec_recommendations(candidates))
        elif task_type == "explain":
            recommendations.extend(self._generate_explanation_recommendations(candidates))
        elif task_type == "retrieval":
            recommendations.extend(self._generate_retrieval_recommendations(candidates))
        
        # Remove duplicates and rank by priority
        unique_recommendations = self._deduplicate_recommendations(recommendations)
        return self._prioritize_recommendations(unique_recommendations)
    
    def _generate_low_confidence_recommendations(self, task_type: str) -> List[Dict[str, Any]]:
        """Generate recommendations for low confidence results."""
        return [
            {
                "action": "additional_data_acquisition",
                "description": "Acquire higher quality or complementary experimental data",
                "priority": "high",
                "effort": "high", 
                "expected_impact": "high",
                "timeframe": "short_term"
            },
            {
                "action": "method_validation",
                "description": "Validate analysis method with known standards",
                "priority": "high",
                "effort": "medium",
                "expected_impact": "medium",
                "timeframe": "short_term"
            },
            {
                "action": "expert_consultation",
                "description": "Consult domain experts for interpretation guidance",
                "priority": "medium",
                "effort": "low",
                "expected_impact": "medium",
                "timeframe": "immediate"
            }
        ]
    
    def _generate_medium_confidence_recommendations(self, task_type: str) -> List[Dict[str, Any]]:
        """Generate recommendations for medium confidence results."""
        return [
            {
                "action": "orthogonal_validation",
                "description": "Use complementary analytical methods for validation",
                "priority": "medium",
                "effort": "medium",
                "expected_impact": "high",
                "timeframe": "short_term"
            },
            {
                "action": "literature_validation",
                "description": "Cross-reference results with published literature",
                "priority": "medium",
                "effort": "low",
                "expected_impact": "medium",
                "timeframe": "immediate"
            },
            {
                "action": "replicate_analysis", 
                "description": "Repeat analysis with independent samples",
                "priority": "low",
                "effort": "medium",
                "expected_impact": "medium",
                "timeframe": "medium_term"
            }
        ]
    
    def _generate_high_confidence_recommendations(self, task_type: str) -> List[Dict[str, Any]]:
        """Generate recommendations for high confidence results."""
        return [
            {
                "action": "publish_results",
                "description": "Consider publishing findings in appropriate journal",
                "priority": "medium",
                "effort": "high",
                "expected_impact": "high", 
                "timeframe": "long_term"
            },
            {
                "action": "functional_validation",
                "description": "Conduct functional studies to validate biological relevance",
                "priority": "high",
                "effort": "high",
                "expected_impact": "high",
                "timeframe": "medium_term"
            },
            {
                "action": "expand_study",
                "description": "Extend analysis to related structures or conditions",
                "priority": "low",
                "effort": "high",
                "expected_impact": "medium",
                "timeframe": "long_term"
            }
        ]
    
    def _generate_high_uncertainty_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for high uncertainty results."""
        return [
            {
                "action": "model_improvement",
                "description": "Improve model training with additional data",
                "priority": "high",
                "effort": "high",
                "expected_impact": "high",
                "timeframe": "long_term"
            },
            {
                "action": "ensemble_analysis",
                "description": "Use ensemble of multiple prediction methods",
                "priority": "medium",
                "effort": "medium",
                "expected_impact": "medium",
                "timeframe": "short_term"
            }
        ]
    
    def _generate_spec2struct_recommendations(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate spec2struct-specific recommendations."""
        return [
            {
                "action": "synthetic_validation",
                "description": "Synthesize top candidate structures for MS comparison",
                "priority": "high",
                "effort": "high",
                "expected_impact": "very_high",
                "timeframe": "medium_term"
            },
            {
                "action": "fragmentation_analysis",
                "description": "Perform detailed fragmentation pathway analysis",
                "priority": "medium",
                "effort": "medium", 
                "expected_impact": "medium",
                "timeframe": "short_term"
            }
        ]
    
    def _generate_structure2spec_recommendations(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate structure2spec-specific recommendations."""
        return [
            {
                "action": "experimental_validation",
                "description": "Acquire experimental spectra for predicted structures",
                "priority": "high",
                "effort": "medium",
                "expected_impact": "high",
                "timeframe": "short_term"
            },
            {
                "action": "instrument_optimization",
                "description": "Optimize MS parameters for better fragmentation",
                "priority": "medium",
                "effort": "low",
                "expected_impact": "medium",
                "timeframe": "immediate"
            }
        ]
    
    def _generate_explanation_recommendations(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate explanation-specific recommendations."""
        return [
            {
                "action": "literature_expansion",
                "description": "Conduct comprehensive literature review",
                "priority": "medium",
                "effort": "medium",
                "expected_impact": "medium",
                "timeframe": "short_term"
            },
            {
                "action": "mechanism_studies",
                "description": "Design experiments to test proposed mechanisms",
                "priority": "high",
                "effort": "high", 
                "expected_impact": "high",
                "timeframe": "long_term"
            }
        ]
    
    def _generate_retrieval_recommendations(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate retrieval-specific recommendations."""
        return [
            {
                "action": "database_expansion",
                "description": "Search additional glycan databases",
                "priority": "medium",
                "effort": "low",
                "expected_impact": "medium",
                "timeframe": "immediate"
            },
            {
                "action": "similarity_analysis",
                "description": "Perform detailed similarity analysis of top hits",
                "priority": "medium",
                "effort": "medium",
                "expected_impact": "medium",
                "timeframe": "short_term"
            }
        ]
    
    def _deduplicate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate recommendations."""
        seen_actions = set()
        unique_recommendations = []
        
        for rec in recommendations:
            action = rec.get('action', '')
            if action not in seen_actions:
                seen_actions.add(action)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize recommendations by impact and effort."""
        priority_scores = {
            "high": 3,
            "medium": 2, 
            "low": 1
        }
        
        effort_scores = {
            "low": 3,
            "medium": 2,
            "high": 1
        }
        
        impact_scores = {
            "very_high": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        
        for rec in recommendations:
            priority_score = priority_scores.get(rec.get('priority', 'medium'), 2)
            effort_score = effort_scores.get(rec.get('effort', 'medium'), 2)  
            impact_score = impact_scores.get(rec.get('expected_impact', 'medium'), 2)
            
            # Combined score favors high impact, high priority, low effort
            combined_score = (0.4 * impact_score + 0.4 * priority_score + 0.2 * effort_score)
            rec['combined_score'] = combined_score
        
        return sorted(recommendations, key=lambda r: r['combined_score'], reverse=True)
        
    def save_reasoning_session(self, filepath: str):
        """Save reasoning history to file"""
        session_data = {
            "timestamp": time.time(),
            "reasoning_history": [asdict(result) for result in self.reasoning_history],
            "knowledge_base_stats": {
                "facts_count": len(self.kb.facts),
                "rules_count": len(self.kb.rules)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
            
        logger.info(f"Reasoning session saved to {filepath}")


# Example usage and testing functions
def create_sample_reasoning_context() -> Dict[str, Any]:
    """Create a sample context for testing reasoning"""
    return {
        "facts": [
            "Mass spectrum shows peak at 365 m/z",
            "Glycan contains N-acetylglucosamine residues",
            "Protein target is CD22 receptor"
        ],
        "observations": [
            "Strong binding affinity observed",
            "Sialic acid residues present",
            "Terminal galactose detected"
        ],
        "experimental_data": {
            "binding_kd": "10 μM",
            "mass_accuracy": "2 ppm",
            "purity": "95%"
        },
        "glycans": ["Neu5Ac-Gal-GlcNAc", "GalNAc-Gal"],
        "proteins": ["CD22", "Siglec-2"],
        "masses": [365.1, 542.2, 203.1]
    }


def demo_glycogot_reasoning():
    """Demonstrate GlycoGOT reasoning capabilities"""
    
    print("=== GlycoGOT Reasoning Engine Demo ===\n")
    
    # Initialize reasoner
    reasoner = GlycoGOTReasoner()
    
    # Sample reasoning task
    goal = "Identify the most likely glycan structure from mass spectral data"
    context = create_sample_reasoning_context()
    
    print(f"Reasoning Goal: {goal}\n")
    print("Context:")
    for key, value in context.items():
        print(f"  {key}: {value}")
    print()
    
    # Perform reasoning
    result = reasoner.reason(goal, context)
    
    # Display results
    print("=== Reasoning Results ===\n")
    print(reasoner.get_reasoning_explanation(result))
    
    return result


if __name__ == "__main__":
    # Run demonstration
    demo_result = demo_glycogot_reasoning()