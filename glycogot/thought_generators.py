"""
Thought Generation Functions for GoT Implementation
==================================================

This module provides various thought generators that can create
new thoughts based on existing ones, implementing different
reasoning patterns and strategies.
"""

import re
import logging
import random
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

from .graph_reasoning import ThoughtNode, ThoughtGraph, ThoughtType, ThoughtStatus

logger = logging.getLogger(__name__)


class ThoughtGenerator(ABC):
    """Abstract base class for thought generators"""
    
    @abstractmethod
    def generate(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> List[ThoughtNode]:
        """
        Generate new thoughts based on a source thought.
        
        Args:
            source_thought: The thought to generate from
            graph: Current thought graph for context
            
        Returns:
            List of new ThoughtNode objects
        """
        pass
    
    @abstractmethod
    def get_generator_type(self) -> str:
        """Return the type/name of this generator"""
        pass


class DeductiveReasoningGenerator(ThoughtGenerator):
    """
    Generates deductive conclusions from premises.
    
    This generator looks for logical implications and applies
    deductive reasoning rules to generate conclusions.
    """
    
    def __init__(self):
        # Common deductive reasoning patterns for glycoinformatics
        self.deductive_patterns = [
            {
                'trigger': 'fragment.*m/z.*204',
                'conclusion_template': 'Presence of HexNAc residue confirmed by diagnostic fragment at m/z 204',
                'confidence_boost': 0.8,
                'evidence_type': 'mass_spectrometry'
            },
            {
                'trigger': 'fragment.*m/z.*366',
                'conclusion_template': 'Presence of Hex residue confirmed by diagnostic fragment at m/z 366',
                'confidence_boost': 0.7,
                'evidence_type': 'mass_spectrometry'
            },
            {
                'trigger': 'precursor.*loss.*162',
                'conclusion_template': 'Hex loss detected, indicating terminal hexose residue',
                'confidence_boost': 0.75,
                'evidence_type': 'neutral_loss'
            },
            {
                'trigger': 'linkage.*beta.*1.*4',
                'conclusion_template': 'β(1→4) linkage suggests lactosamine or similar structure',
                'confidence_boost': 0.6,
                'evidence_type': 'structural_analysis'
            }
        ]
    
    def generate(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Generate deductive conclusions"""
        new_thoughts = []
        
        # Apply pattern matching for glycoinformatics deductions
        for pattern in self.deductive_patterns:
            if re.search(pattern['trigger'], source_thought.content, re.IGNORECASE):
                conclusion_thought = ThoughtNode(
                    content=pattern['conclusion_template'],
                    thought_type=ThoughtType.DEDUCTION,
                    confidence=min(source_thought.confidence + pattern['confidence_boost'], 1.0),
                    relevance=source_thought.relevance,
                    evidence_strength=pattern['confidence_boost'],
                    creator=f"deductive_generator",
                    evidence=[source_thought.content],
                    context={
                        'source_thought': source_thought.id,
                        'evidence_type': pattern['evidence_type'],
                        'reasoning_type': 'deductive'
                    }
                )
                new_thoughts.append(conclusion_thought)
        
        # Logical implication detection
        if any(keyword in source_thought.content.lower() for keyword in 
               ['if', 'when', 'given that', 'since', 'because']):
            # Extract implications
            implication_thought = self._extract_logical_implication(source_thought)
            if implication_thought:
                new_thoughts.append(implication_thought)
        
        return new_thoughts
    
    def _extract_logical_implication(self, source_thought: ThoughtNode) -> Optional[ThoughtNode]:
        """Extract logical implications from conditional statements"""
        content = source_thought.content.lower()
        
        # Simple pattern matching for logical implications
        if 'if' in content and 'then' in content:
            parts = content.split('then', 1)
            if len(parts) == 2:
                conclusion = parts[1].strip()
                return ThoughtNode(
                    content=f"Deduction: {conclusion}",
                    thought_type=ThoughtType.DEDUCTION,
                    confidence=source_thought.confidence * 0.8,
                    relevance=source_thought.relevance,
                    evidence=[source_thought.content],
                    creator="deductive_generator"
                )
        
        return None
    
    def get_generator_type(self) -> str:
        return "deductive_reasoning"


class InductiveReasoningGenerator(ThoughtGenerator):
    """
    Generates inductive generalizations from specific observations.
    
    This generator identifies patterns and creates generalizations
    based on multiple observations.
    """
    
    def generate(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Generate inductive generalizations"""
        new_thoughts = []
        
        # Look for patterns across similar observations
        similar_thoughts = self._find_similar_thoughts(source_thought, graph)
        
        if len(similar_thoughts) >= 2:  # Need multiple observations for induction
            pattern_thought = self._create_pattern_generalization(similar_thoughts)
            if pattern_thought:
                new_thoughts.append(pattern_thought)
        
        # Frequency-based induction for glycan analysis
        if any(keyword in source_thought.content.lower() for keyword in 
               ['occurs', 'found', 'detected', 'present']):
            frequency_thought = self._create_frequency_generalization(source_thought, graph)
            if frequency_thought:
                new_thoughts.append(frequency_thought)
        
        return new_thoughts
    
    def _find_similar_thoughts(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Find thoughts with similar content or context"""
        similar = []
        source_words = set(source_thought.content.lower().split())
        
        for thought in graph.nodes.values():
            if thought.id == source_thought.id:
                continue
                
            thought_words = set(thought.content.lower().split())
            overlap = len(source_words.intersection(thought_words))
            
            # Consider thoughts similar if they share significant vocabulary
            if overlap >= 3 or overlap / max(len(source_words), len(thought_words)) > 0.4:
                similar.append(thought)
        
        return similar
    
    def _create_pattern_generalization(self, similar_thoughts: List[ThoughtNode]) -> Optional[ThoughtNode]:
        """Create a pattern generalization from similar thoughts"""
        if len(similar_thoughts) < 2:
            return None
            
        # Extract common elements
        common_elements = []
        for thought in similar_thoughts:
            # Simple keyword extraction
            words = thought.content.lower().split()
            for word in words:
                if len(word) > 3 and word not in ['this', 'that', 'with', 'from', 'have']:
                    common_elements.append(word)
        
        # Find most frequent elements
        element_counts = {}
        for element in common_elements:
            element_counts[element] = element_counts.get(element, 0) + 1
        
        frequent_elements = [elem for elem, count in element_counts.items() 
                           if count >= len(similar_thoughts) // 2]
        
        if frequent_elements:
            pattern_content = f"Pattern observed: {', '.join(frequent_elements[:3])} commonly occur together"
            
            avg_confidence = sum(t.confidence for t in similar_thoughts) / len(similar_thoughts)
            
            return ThoughtNode(
                content=pattern_content,
                thought_type=ThoughtType.INDUCTION,
                confidence=min(avg_confidence * 0.7, 0.9),  # Lower confidence for induction
                relevance=max(t.relevance for t in similar_thoughts),
                evidence=[t.content for t in similar_thoughts[:3]],  # Limit evidence
                creator="inductive_generator",
                context={
                    'pattern_size': len(similar_thoughts),
                    'reasoning_type': 'inductive'
                }
            )
        
        return None
    
    def _create_frequency_generalization(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> Optional[ThoughtNode]:
        """Create frequency-based generalizations"""
        # Simple frequency analysis
        content = source_thought.content.lower()
        
        if 'frequently' in content or 'often' in content or 'commonly' in content:
            return ThoughtNode(
                content=f"General pattern: {source_thought.content.replace('is', 'tends to be')}",
                thought_type=ThoughtType.INDUCTION,
                confidence=source_thought.confidence * 0.6,
                relevance=source_thought.relevance,
                evidence=[source_thought.content],
                creator="inductive_generator"
            )
        
        return None
    
    def get_generator_type(self) -> str:
        return "inductive_reasoning"


class AnalogicalReasoningGenerator(ThoughtGenerator):
    """
    Generates analogical reasoning thoughts by finding structural similarities.
    
    This generator creates thoughts based on analogies between different
    glycan structures or analysis scenarios.
    """
    
    def __init__(self):
        # Common analogies in glycoinformatics
        self.structural_analogies = [
            {
                'source_pattern': r'n.*acetyl.*glucosamine',
                'target_domain': 'protein modifications',
                'analogy_template': 'Similar to protein N-linked glycosylation patterns'
            },
            {
                'source_pattern': r'sialic.*acid',
                'target_domain': 'cell_surface',
                'analogy_template': 'Analogous to cell surface recognition motifs'
            },
            {
                'source_pattern': r'beta.*1.*4.*linkage',
                'target_domain': 'polymer_structure',
                'analogy_template': 'Similar to linear polymer backbone structures'
            }
        ]
    
    def generate(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Generate analogical reasoning thoughts"""
        new_thoughts = []
        
        # Apply structural analogies
        for analogy in self.structural_analogies:
            if re.search(analogy['source_pattern'], source_thought.content, re.IGNORECASE):
                analogy_thought = ThoughtNode(
                    content=f"Analogy: {analogy['analogy_template']}",
                    thought_type=ThoughtType.ANALOGY,
                    confidence=source_thought.confidence * 0.5,  # Analogies are less certain
                    relevance=source_thought.relevance * 0.8,
                    novelty=0.7,  # Analogies can be novel
                    evidence=[source_thought.content],
                    creator="analogical_generator",
                    context={
                        'source_thought': source_thought.id,
                        'analogy_domain': analogy['target_domain'],
                        'reasoning_type': 'analogical'
                    }
                )
                new_thoughts.append(analogy_thought)
        
        # Cross-domain analogies
        cross_domain_thought = self._create_cross_domain_analogy(source_thought, graph)
        if cross_domain_thought:
            new_thoughts.append(cross_domain_thought)
        
        return new_thoughts
    
    def _create_cross_domain_analogy(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> Optional[ThoughtNode]:
        """Create cross-domain analogical thoughts"""
        content = source_thought.content.lower()
        
        # Biology to chemistry analogies
        if any(bio_term in content for bio_term in ['cell', 'membrane', 'receptor', 'enzyme']):
            return ThoughtNode(
                content=f"Chemical analogy: Similar to molecular recognition in {source_thought.content}",
                thought_type=ThoughtType.ANALOGY,
                confidence=source_thought.confidence * 0.4,
                relevance=source_thought.relevance * 0.6,
                novelty=0.8,
                evidence=[source_thought.content],
                creator="analogical_generator"
            )
        
        # Structure to function analogies
        if any(struct_term in content for struct_term in ['structure', 'shape', 'conformation']):
            return ThoughtNode(
                content=f"Structure-function analogy: Structure determines function as in {source_thought.content}",
                thought_type=ThoughtType.ANALOGY,
                confidence=source_thought.confidence * 0.5,
                relevance=source_thought.relevance,
                novelty=0.6,
                evidence=[source_thought.content],
                creator="analogical_generator"
            )
        
        return None
    
    def get_generator_type(self) -> str:
        return "analogical_reasoning"


class CausalReasoningGenerator(ThoughtGenerator):
    """
    Generates causal reasoning thoughts identifying cause-effect relationships.
    
    This generator creates thoughts about what causes certain glycan
    structures or analytical results.
    """
    
    def __init__(self):
        # Causal patterns in glycoinformatics
        self.causal_patterns = [
            {
                'trigger': r'fragment.*at.*m/z',
                'effect': 'specific glycan structural element',
                'confidence_modifier': 0.8
            },
            {
                'trigger': r'linkage.*position',
                'effect': 'specific fragmentation pattern',
                'confidence_modifier': 0.7
            },
            {
                'trigger': r'enzyme.*activity',
                'effect': 'glycan synthesis pathway',
                'confidence_modifier': 0.6
            }
        ]
    
    def generate(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Generate causal reasoning thoughts"""
        new_thoughts = []
        
        # Apply causal pattern matching
        for pattern in self.causal_patterns:
            if re.search(pattern['trigger'], source_thought.content, re.IGNORECASE):
                causal_thought = ThoughtNode(
                    content=f"Causal relationship: {source_thought.content} indicates {pattern['effect']}",
                    thought_type=ThoughtType.CAUSAL,
                    confidence=source_thought.confidence * pattern['confidence_modifier'],
                    relevance=source_thought.relevance,
                    evidence=[source_thought.content],
                    creator="causal_generator",
                    context={
                        'causal_type': 'direct_causation',
                        'reasoning_type': 'causal'
                    }
                )
                new_thoughts.append(causal_thought)
        
        # Mechanism-based causation
        mechanism_thought = self._identify_mechanism(source_thought)
        if mechanism_thought:
            new_thoughts.append(mechanism_thought)
        
        return new_thoughts
    
    def _identify_mechanism(self, source_thought: ThoughtNode) -> Optional[ThoughtNode]:
        """Identify causal mechanisms"""
        content = source_thought.content.lower()
        
        # Enzymatic mechanisms
        if any(enzyme_term in content for enzyme_term in ['transferase', 'synthase', 'hydrolase']):
            return ThoughtNode(
                content=f"Enzymatic mechanism: {source_thought.content} suggests specific enzymatic pathway",
                thought_type=ThoughtType.CAUSAL,
                confidence=source_thought.confidence * 0.7,
                relevance=source_thought.relevance,
                evidence=[source_thought.content],
                creator="causal_generator",
                context={'mechanism_type': 'enzymatic'}
            )
        
        # Structural mechanisms
        if any(struct_term in content for struct_term in ['conformation', 'binding', 'interaction']):
            return ThoughtNode(
                content=f"Structural mechanism: {source_thought.content} implies conformational effects",
                thought_type=ThoughtType.CAUSAL,
                confidence=source_thought.confidence * 0.6,
                relevance=source_thought.relevance,
                evidence=[source_thought.content],
                creator="causal_generator",
                context={'mechanism_type': 'structural'}
            )
        
        return None
    
    def get_generator_type(self) -> str:
        return "causal_reasoning"


class SynthesisGenerator(ThoughtGenerator):
    """
    Generates synthesis thoughts that combine multiple existing thoughts.
    
    This generator creates new insights by combining information
    from multiple sources.
    """
    
    def generate(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Generate synthesis thoughts"""
        new_thoughts = []
        
        # Find related thoughts to synthesize with
        related_thoughts = self._find_related_thoughts(source_thought, graph)
        
        if len(related_thoughts) >= 1:
            # Combine with each related thought
            for related in related_thoughts[:3]:  # Limit combinations
                synthesis_thought = self._synthesize_thoughts(source_thought, related)
                if synthesis_thought:
                    new_thoughts.append(synthesis_thought)
        
        # Multi-thought synthesis
        if len(related_thoughts) >= 2:
            multi_synthesis = self._multi_thought_synthesis([source_thought] + related_thoughts[:2])
            if multi_synthesis:
                new_thoughts.append(multi_synthesis)
        
        return new_thoughts
    
    def _find_related_thoughts(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Find thoughts related to the source thought"""
        related = []
        
        # Find thoughts with overlapping concepts
        source_concepts = self._extract_concepts(source_thought.content)
        
        for thought in graph.nodes.values():
            if thought.id == source_thought.id:
                continue
                
            thought_concepts = self._extract_concepts(thought.content)
            overlap = len(source_concepts.intersection(thought_concepts))
            
            if overlap > 0:
                related.append((thought, overlap))
        
        # Sort by concept overlap
        related.sort(key=lambda x: x[1], reverse=True)
        
        return [thought for thought, overlap in related[:5]]
    
    def _extract_concepts(self, content: str) -> set:
        """Extract key concepts from thought content"""
        # Simple concept extraction - could be enhanced with NLP
        words = content.lower().split()
        
        # Filter to meaningful concepts
        concepts = set()
        for word in words:
            if (len(word) > 4 and 
                word not in ['this', 'that', 'with', 'from', 'have', 'been', 'will', 'would'] and
                not word.isdigit()):
                concepts.add(word)
        
        return concepts
    
    def _synthesize_thoughts(self, thought1: ThoughtNode, thought2: ThoughtNode) -> Optional[ThoughtNode]:
        """Combine two thoughts into a synthesis"""
        # Create synthesis content
        synthesis_content = f"Synthesis: Combining '{thought1.content[:50]}...' with '{thought2.content[:50]}...'"
        
        # Calculate combined confidence (conservative approach)
        combined_confidence = min(thought1.confidence, thought2.confidence) * 0.8
        
        # Calculate combined relevance
        combined_relevance = max(thought1.relevance, thought2.relevance)
        
        return ThoughtNode(
            content=synthesis_content,
            thought_type=ThoughtType.SYNTHESIS,
            confidence=combined_confidence,
            relevance=combined_relevance,
            novelty=0.7,  # Synthesis can be novel
            evidence=[thought1.content, thought2.content],
            creator="synthesis_generator",
            context={
                'synthesized_thoughts': [thought1.id, thought2.id],
                'reasoning_type': 'synthesis'
            }
        )
    
    def _multi_thought_synthesis(self, thoughts: List[ThoughtNode]) -> Optional[ThoughtNode]:
        """Synthesize multiple thoughts"""
        if len(thoughts) < 2:
            return None
            
        synthesis_content = f"Multi-synthesis: Combining {len(thoughts)} related insights"
        
        # Conservative confidence calculation
        avg_confidence = sum(t.confidence for t in thoughts) / len(thoughts)
        synthesis_confidence = avg_confidence * 0.7
        
        # Maximum relevance
        max_relevance = max(t.relevance for t in thoughts)
        
        return ThoughtNode(
            content=synthesis_content,
            thought_type=ThoughtType.SYNTHESIS,
            confidence=synthesis_confidence,
            relevance=max_relevance,
            novelty=0.8,  # High novelty for multi-synthesis
            evidence=[t.content for t in thoughts[:3]],  # Limit evidence
            creator="synthesis_generator",
            context={
                'synthesized_thoughts': [t.id for t in thoughts],
                'synthesis_size': len(thoughts),
                'reasoning_type': 'multi_synthesis'
            }
        )
    
    def get_generator_type(self) -> str:
        return "synthesis"


class EvaluationGenerator(ThoughtGenerator):
    """
    Generates evaluation thoughts that assess other thoughts.
    
    This generator creates metacognitive thoughts that evaluate
    the quality and validity of other reasoning.
    """
    
    def generate(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Generate evaluation thoughts"""
        new_thoughts = []
        
        # Confidence evaluation
        confidence_eval = self._evaluate_confidence(source_thought)
        if confidence_eval:
            new_thoughts.append(confidence_eval)
        
        # Evidence evaluation
        evidence_eval = self._evaluate_evidence(source_thought, graph)
        if evidence_eval:
            new_thoughts.append(evidence_eval)
        
        # Consistency evaluation
        consistency_eval = self._evaluate_consistency(source_thought, graph)
        if consistency_eval:
            new_thoughts.append(consistency_eval)
        
        return new_thoughts
    
    def _evaluate_confidence(self, source_thought: ThoughtNode) -> Optional[ThoughtNode]:
        """Evaluate confidence level of a thought"""
        if source_thought.confidence < 0.3:
            eval_content = f"Low confidence concern: '{source_thought.content[:50]}...' has low confidence ({source_thought.confidence:.2f})"
            confidence_modifier = 0.8
        elif source_thought.confidence > 0.9:
            eval_content = f"High confidence note: '{source_thought.content[:50]}...' has high confidence ({source_thought.confidence:.2f})"
            confidence_modifier = 0.9
        else:
            return None  # Moderate confidence doesn't need evaluation
        
        return ThoughtNode(
            content=eval_content,
            thought_type=ThoughtType.EVALUATION,
            confidence=confidence_modifier,
            relevance=source_thought.relevance * 0.8,
            evidence=[f"Source confidence: {source_thought.confidence}"],
            creator="evaluation_generator",
            context={
                'evaluated_thought': source_thought.id,
                'evaluation_type': 'confidence',
                'reasoning_type': 'evaluation'
            }
        )
    
    def _evaluate_evidence(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> Optional[ThoughtNode]:
        """Evaluate the evidence supporting a thought"""
        evidence_count = len(source_thought.evidence)
        
        if evidence_count == 0:
            eval_content = f"Evidence gap: '{source_thought.content[:50]}...' lacks supporting evidence"
            eval_confidence = 0.7
        elif evidence_count >= 3:
            eval_content = f"Well-supported: '{source_thought.content[:50]}...' has strong evidence base"
            eval_confidence = 0.8
        else:
            eval_content = f"Moderate evidence: '{source_thought.content[:50]}...' has some supporting evidence"
            eval_confidence = 0.6
        
        return ThoughtNode(
            content=eval_content,
            thought_type=ThoughtType.EVALUATION,
            confidence=eval_confidence,
            relevance=source_thought.relevance * 0.7,
            evidence=[f"Evidence count: {evidence_count}"],
            creator="evaluation_generator",
            context={
                'evaluated_thought': source_thought.id,
                'evaluation_type': 'evidence',
                'evidence_count': evidence_count
            }
        )
    
    def _evaluate_consistency(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> Optional[ThoughtNode]:
        """Evaluate consistency with other thoughts"""
        # Find potentially conflicting thoughts
        conflicts = 0
        supports = 0
        
        source_concepts = self._extract_key_concepts(source_thought.content)
        
        for thought in graph.nodes.values():
            if thought.id == source_thought.id:
                continue
                
            thought_concepts = self._extract_key_concepts(thought.content)
            overlap = len(source_concepts.intersection(thought_concepts))
            
            if overlap > 0:
                # Simple heuristic for agreement/disagreement
                if any(neg_word in thought.content.lower() for neg_word in ['not', 'no', 'never', 'cannot']):
                    conflicts += 1
                else:
                    supports += 1
        
        if conflicts > supports:
            eval_content = f"Consistency concern: '{source_thought.content[:50]}...' conflicts with other thoughts"
            eval_confidence = 0.7
        elif supports > conflicts:
            eval_content = f"Well-supported: '{source_thought.content[:50]}...' is consistent with other evidence"
            eval_confidence = 0.8
        else:
            return None  # Neutral consistency
        
        return ThoughtNode(
            content=eval_content,
            thought_type=ThoughtType.EVALUATION,
            confidence=eval_confidence,
            relevance=source_thought.relevance * 0.8,
            evidence=[f"Supports: {supports}, Conflicts: {conflicts}"],
            creator="evaluation_generator",
            context={
                'evaluated_thought': source_thought.id,
                'evaluation_type': 'consistency',
                'supports': supports,
                'conflicts': conflicts
            }
        )
    
    def _extract_key_concepts(self, content: str) -> set:
        """Extract key concepts for consistency checking"""
        # Enhanced concept extraction for evaluation
        words = content.lower().split()
        concepts = set()
        
        for word in words:
            # Include technical terms and meaningful words
            if (len(word) > 3 and 
                not word.isdigit() and
                word not in ['this', 'that', 'with', 'from', 'have', 'been', 'will']):
                concepts.add(word)
        
        return concepts
    
    def get_generator_type(self) -> str:
        return "evaluation"


class HypothesisGenerator(ThoughtGenerator):
    """
    Generates hypothesis thoughts based on observations and patterns.
    
    This generator creates testable hypotheses from existing observations.
    """
    
    def __init__(self):
        # Hypothesis templates for glycoinformatics
        self.hypothesis_templates = [
            "Hypothesis: {observation} suggests {mechanism}",
            "Hypothesis: If {condition}, then {prediction}",
            "Hypothesis: {pattern} indicates {underlying_cause}",
            "Hypothesis: {structure} implies {function}"
        ]
    
    def generate(self, source_thought: ThoughtNode, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Generate hypothesis thoughts"""
        new_thoughts = []
        
        # Only generate hypotheses from observations or deductions
        if source_thought.thought_type in [ThoughtType.OBSERVATION, ThoughtType.DEDUCTION]:
            # Pattern-based hypothesis
            pattern_hypothesis = self._create_pattern_hypothesis(source_thought)
            if pattern_hypothesis:
                new_thoughts.append(pattern_hypothesis)
                
            # Mechanistic hypothesis
            mechanism_hypothesis = self._create_mechanism_hypothesis(source_thought)
            if mechanism_hypothesis:
                new_thoughts.append(mechanism_hypothesis)
                
            # Structural hypothesis
            if any(struct_word in source_thought.content.lower() for struct_word in 
                   ['structure', 'linkage', 'conformation', 'shape']):
                struct_hypothesis = self._create_structural_hypothesis(source_thought)
                if struct_hypothesis:
                    new_thoughts.append(struct_hypothesis)
        
        return new_thoughts
    
    def _create_pattern_hypothesis(self, source_thought: ThoughtNode) -> Optional[ThoughtNode]:
        """Create pattern-based hypothesis"""
        content = source_thought.content
        
        # Extract potential patterns
        if 'fragment' in content.lower():
            hypothesis_content = f"Hypothesis: The fragmentation pattern in '{content[:50]}...' suggests specific structural motifs"
        elif 'linkage' in content.lower():
            hypothesis_content = f"Hypothesis: The linkage described in '{content[:50]}...' determines specific biological function"
        elif 'mass' in content.lower():
            hypothesis_content = f"Hypothesis: The mass data in '{content[:50]}...' corresponds to known glycan compositions"
        else:
            return None
        
        return ThoughtNode(
            content=hypothesis_content,
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=source_thought.confidence * 0.6,  # Hypotheses are speculative
            relevance=source_thought.relevance,
            novelty=0.8,  # Hypotheses can be novel
            evidence=[source_thought.content],
            creator="hypothesis_generator",
            context={
                'source_thought': source_thought.id,
                'hypothesis_type': 'pattern_based',
                'reasoning_type': 'hypothesis'
            }
        )
    
    def _create_mechanism_hypothesis(self, source_thought: ThoughtNode) -> Optional[ThoughtNode]:
        """Create mechanism-based hypothesis"""
        content = source_thought.content.lower()
        
        if any(mech_word in content for mech_word in ['enzyme', 'catalysis', 'synthesis', 'pathway']):
            hypothesis_content = f"Hypothesis: The mechanism underlying '{source_thought.content[:50]}...' involves specific enzymatic processes"
        elif any(bio_word in content for bio_word in ['cell', 'membrane', 'receptor', 'binding']):
            hypothesis_content = f"Hypothesis: The biological process in '{source_thought.content[:50]}...' depends on molecular recognition"
        else:
            return None
        
        return ThoughtNode(
            content=hypothesis_content,
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=source_thought.confidence * 0.5,
            relevance=source_thought.relevance,
            novelty=0.7,
            evidence=[source_thought.content],
            creator="hypothesis_generator",
            context={
                'hypothesis_type': 'mechanism_based',
                'reasoning_type': 'hypothesis'
            }
        )
    
    def _create_structural_hypothesis(self, source_thought: ThoughtNode) -> Optional[ThoughtNode]:
        """Create structure-based hypothesis"""
        hypothesis_content = f"Hypothesis: The structural features in '{source_thought.content[:50]}...' determine functional properties"
        
        return ThoughtNode(
            content=hypothesis_content,
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=source_thought.confidence * 0.6,
            relevance=source_thought.relevance,
            novelty=0.6,
            evidence=[source_thought.content],
            creator="hypothesis_generator",
            context={
                'hypothesis_type': 'structural',
                'reasoning_type': 'hypothesis'
            }
        )
    
    def get_generator_type(self) -> str:
        return "hypothesis"


# Factory function to create all standard generators
def create_standard_generators() -> List[ThoughtGenerator]:
    """Create a list of all standard thought generators"""
    return [
        DeductiveReasoningGenerator(),
        InductiveReasoningGenerator(),
        AnalogicalReasoningGenerator(),
        CausalReasoningGenerator(),
        SynthesisGenerator(),
        EvaluationGenerator(),
        HypothesisGenerator()
    ]


# Convenience function to convert generators to callable functions
def generators_to_functions(generators: List[ThoughtGenerator]) -> List[Callable]:
    """Convert ThoughtGenerator objects to callable functions"""
    return [gen.generate for gen in generators]


# Example usage
if __name__ == "__main__":
    from .graph_reasoning import ThoughtNode, ThoughtGraph, ThoughtType
    
    # Create test setup
    graph = ThoughtGraph()
    
    # Create a test observation
    obs_thought = ThoughtNode(
        content="Fragment ion observed at m/z 204.09 indicating HexNAc residue",
        thought_type=ThoughtType.OBSERVATION,
        confidence=0.8,
        relevance=0.9
    )
    
    graph.add_thought(obs_thought)
    
    # Test generators
    generators = create_standard_generators()
    
    for generator in generators:
        new_thoughts = generator.generate(obs_thought, graph)
        print(f"\n{generator.get_generator_type()} generated {len(new_thoughts)} thoughts:")
        
        for thought in new_thoughts:
            print(f"  - {thought.content}")
            graph.add_thought(thought)
    
    print(f"\nTotal thoughts in graph: {len(graph.nodes)}")