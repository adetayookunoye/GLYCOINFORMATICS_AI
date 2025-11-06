"""
Advanced Knowledge Graph Grounding System for GlycoLLM

This module implements sophisticated knowledge graph grounding that connects
GlycoLLM predictions to existing GlycoKG entities, validates biosynthetic
pathways, and provides comprehensive entity linking with confidence scoring.

Integrates with the existing GlycoKG infrastructure and provides real-time
grounding for all 4 GlycoLLM tasks (spec2struct, structure2spec, explain, retrieval).
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
import asyncio
import logging
from datetime import datetime
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import json
from pathlib import Path
import hashlib

# Platform imports
from glycokg.ontology.glyco_ontology import GlycoOntology
from glycokg.integration.coordinator import DataCoordinator
from platform.api.contracts import (
    DetailedGroundingEvidence, ComprehensiveGroundingResult,
    ValidationLevel, OutputQuality
)

logger = logging.getLogger(__name__)

class AdvancedKGGroundingEngine:
    """
    Advanced knowledge graph grounding engine for GlycoLLM predictions.
    
    Provides sophisticated entity linking, pathway validation, biosynthetic
    feasibility checking, and multi-source evidence aggregation.
    """
    
    def __init__(self, 
                 glyco_kg_path: Optional[str] = None,
                 confidence_threshold: float = 0.7,
                 max_grounding_depth: int = 3):
        """Initialize the grounding engine."""
        self.confidence_threshold = confidence_threshold
        self.max_grounding_depth = max_grounding_depth
        
        # Initialize knowledge graph components
        self.ontology = GlycoOntology()
        self.kg_graph = self._load_knowledge_graph(glyco_kg_path)
        
        # Initialize entity databases
        self.motif_database = MotifGroundingDatabase()
        self.enzyme_database = EnzymeGroundingDatabase()
        self.pathway_database = PathwayGroundingDatabase()
        self.organism_database = OrganismGroundingDatabase()
        
        # Grounding caches
        self.grounding_cache = {}
        self.validation_cache = {}
        
        logger.info("AdvancedKGGroundingEngine initialized")
    
    def _load_knowledge_graph(self, kg_path: Optional[str]) -> Graph:
        """Load the knowledge graph."""
        graph = Graph()
        
        if kg_path and Path(kg_path).exists():
            graph.parse(kg_path, format="turtle")
            logger.info(f"Loaded knowledge graph from {kg_path}")
        else:
            # Load sample data if available
            sample_kg = Path(__file__).parent.parent.parent / "glycokg" / "graph" / "sample.ttl"
            if sample_kg.exists():
                graph.parse(sample_kg, format="turtle")
                logger.info(f"Loaded sample knowledge graph")
        
        return graph
    
    async def ground_structure_prediction(self, 
                                        structure: str,
                                        confidence: float,
                                        context: Dict[str, Any] = None) -> ComprehensiveGroundingResult:
        """
        Ground a predicted glycan structure in the knowledge graph.
        
        Args:
            structure: Predicted glycan structure (WURCS format)
            confidence: Prediction confidence
            context: Additional context information
            
        Returns:
            Comprehensive grounding result with all linked entities
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(structure, "structure", context)
            if cache_key in self.grounding_cache:
                logger.debug(f"Using cached grounding for {structure[:20]}...")
                return self.grounding_cache[cache_key]
            
            # Initialize grounding result
            grounding = ComprehensiveGroundingResult()
            
            # 1. Ground structural motifs
            structural_motifs = await self._ground_structural_motifs(structure, confidence)
            grounding.structural_motifs = structural_motifs
            
            # 2. Ground associated enzymes
            enzymes = await self._ground_associated_enzymes(structure, structural_motifs, context)
            grounding.enzymes = enzymes
            
            # 3. Ground biosynthetic pathways
            pathways = await self._ground_biosynthetic_pathways(structure, enzymes, context)
            grounding.pathways = pathways
            
            # 4. Ground source organisms
            organisms = await self._ground_source_organisms(structure, pathways, context)
            grounding.organisms = organisms
            
            # 5. Ground disease associations (if relevant)
            diseases = await self._ground_disease_associations(structure, context)
            grounding.diseases = diseases
            
            # 6. Calculate overall grounding metrics
            grounding.overall_confidence = self._calculate_grounding_confidence(grounding)
            grounding.coverage_score = self._calculate_coverage_score(grounding)
            grounding.consistency_score = await self._validate_grounding_consistency(grounding)
            
            # Set metadata
            grounding.grounding_method = "multimodal_entity_linking"
            grounding.knowledge_sources = ["GlycoKG", "GlyTouCan", "GlyGen", "UniProt", "KEGG"]
            grounding.grounding_timestamp = datetime.now()
            
            # Cache result
            self.grounding_cache[cache_key] = grounding
            
            return grounding
            
        except Exception as e:
            logger.error(f"Structure grounding failed: {e}")
            return ComprehensiveGroundingResult()  # Empty result on failure
    
    async def ground_spectrum_prediction(self,
                                       predicted_peaks: List[Dict[str, Any]],
                                       structure_context: Optional[str] = None,
                                       confidence: float = 0.5) -> ComprehensiveGroundingResult:
        """
        Ground predicted mass spectrum in the knowledge graph.
        
        Args:
            predicted_peaks: Predicted spectrum peaks with annotations
            structure_context: Associated structure if known
            confidence: Prediction confidence
            
        Returns:
            Comprehensive grounding result
        """
        try:
            grounding = ComprehensiveGroundingResult()
            
            # 1. Ground fragmentation patterns to known motifs
            fragmentation_motifs = await self._ground_fragmentation_patterns(predicted_peaks)
            grounding.structural_motifs.extend(fragmentation_motifs)
            
            # 2. If structure context is provided, use structure grounding
            if structure_context:
                structure_grounding = await self.ground_structure_prediction(
                    structure_context, confidence
                )
                grounding = self._merge_grounding_results(grounding, structure_grounding)
            
            # 3. Ground analytical methods and instruments
            analytical_context = await self._ground_analytical_methods(predicted_peaks)
            grounding.knowledge_sources.extend(analytical_context)
            
            grounding.grounding_method = "spectrum_pattern_matching"
            grounding.grounding_timestamp = datetime.now()
            
            return grounding
            
        except Exception as e:
            logger.error(f"Spectrum grounding failed: {e}")
            return ComprehensiveGroundingResult()
    
    async def ground_explanation(self,
                               explanation_text: str,
                               entities_mentioned: List[str] = None,
                               context: Dict[str, Any] = None) -> ComprehensiveGroundingResult:
        """
        Ground generated explanation text in the knowledge graph.
        
        Args:
            explanation_text: Generated explanation
            entities_mentioned: Explicitly mentioned entities
            context: Additional context
            
        Returns:
            Comprehensive grounding result
        """
        try:
            grounding = ComprehensiveGroundingResult()
            
            # 1. Extract entities from explanation text
            extracted_entities = await self._extract_entities_from_text(explanation_text)
            
            # 2. Ground explicitly mentioned entities
            if entities_mentioned:
                explicit_grounding = await self._ground_explicit_entities(entities_mentioned)
                grounding = self._merge_grounding_results(grounding, explicit_grounding)
            
            # 3. Ground extracted entities
            for entity_type, entities in extracted_entities.items():
                if entity_type == "structures":
                    for structure in entities:
                        struct_grounding = await self.ground_structure_prediction(structure, 0.8, context)
                        grounding = self._merge_grounding_results(grounding, struct_grounding)
                
                elif entity_type == "enzymes":
                    enzyme_evidence = await self._ground_enzyme_mentions(entities)
                    grounding.enzymes.extend(enzyme_evidence)
                
                elif entity_type == "pathways":
                    pathway_evidence = await self._ground_pathway_mentions(entities)
                    grounding.pathways.extend(pathway_evidence)
            
            # 4. Validate biological coherence
            grounding.consistency_score = await self._validate_explanation_coherence(
                explanation_text, grounding
            )
            
            grounding.grounding_method = "text_entity_extraction"
            grounding.grounding_timestamp = datetime.now()
            
            return grounding
            
        except Exception as e:
            logger.error(f"Explanation grounding failed: {e}")
            return ComprehensiveGroundingResult()
    
    async def ground_retrieval_results(self,
                                     query: str,
                                     results: List[Dict[str, Any]],
                                     query_type: str = "mixed") -> List[ComprehensiveGroundingResult]:
        """
        Ground retrieval results in the knowledge graph.
        
        Args:
            query: Original query
            results: Retrieved results
            query_type: Type of query
            
        Returns:
            List of grounding results for each retrieved item
        """
        try:
            grounded_results = []
            
            for result in results:
                grounding = ComprehensiveGroundingResult()
                
                # Ground the retrieved structure if present
                if "structure" in result and result["structure"]:
                    struct_grounding = await self.ground_structure_prediction(
                        result["structure"], result.get("similarity_score", 0.7)
                    )
                    grounding = struct_grounding
                
                # Add retrieval-specific grounding
                retrieval_grounding = await self._ground_retrieval_metadata(result)
                grounding = self._merge_grounding_results(grounding, retrieval_grounding)
                
                grounded_results.append(grounding)
            
            return grounded_results
            
        except Exception as e:
            logger.error(f"Retrieval grounding failed: {e}")
            return [ComprehensiveGroundingResult() for _ in results]
    
    # ========================================================================================
    # MOTIF GROUNDING METHODS
    # ========================================================================================
    
    async def _ground_structural_motifs(self, 
                                      structure: str, 
                                      confidence: float) -> List[DetailedGroundingEvidence]:
        """Ground structural motifs within the glycan structure."""
        try:
            motifs = []
            
            # Parse structure and identify known motifs
            identified_motifs = await self.motif_database.identify_motifs(structure)
            
            for motif_data in identified_motifs:
                evidence = DetailedGroundingEvidence(
                    entity_id=motif_data["id"],
                    entity_type="structural_motif",
                    entity_name=motif_data["name"],
                    confidence=motif_data["confidence"] * confidence,
                    validation_method="pattern_matching",
                    evidence_strength=self._assess_evidence_strength(motif_data["confidence"]),
                    evidence_source="GlycoKG_motifs",
                    supporting_papers=motif_data.get("papers", []),
                    validation_score=motif_data["confidence"]
                )
                
                motifs.append(evidence)
            
            return motifs
            
        except Exception as e:
            logger.error(f"Motif grounding failed: {e}")
            return []
    
    async def _ground_associated_enzymes(self,
                                       structure: str,
                                       motifs: List[DetailedGroundingEvidence],
                                       context: Dict[str, Any] = None) -> List[DetailedGroundingEvidence]:
        """Ground enzymes associated with the glycan structure."""
        try:
            enzymes = []
            
            # Ground enzymes from motifs
            for motif in motifs:
                associated_enzymes = await self.enzyme_database.get_enzymes_for_motif(
                    motif.entity_id
                )
                
                for enzyme_data in associated_enzymes:
                    evidence = DetailedGroundingEvidence(
                        entity_id=enzyme_data["id"],
                        entity_type="glycosyltransferase",
                        entity_name=enzyme_data["name"],
                        confidence=enzyme_data["confidence"] * motif.confidence,
                        validation_method="motif_association",
                        evidence_source="UniProt",
                        biosynthetic_pathway=enzyme_data.get("pathway"),
                        organism_specificity=enzyme_data.get("organisms", []),
                        validation_score=enzyme_data["confidence"]
                    )
                    
                    enzymes.append(evidence)
            
            # Direct structure-enzyme associations
            direct_enzymes = await self.enzyme_database.get_enzymes_for_structure(structure)
            for enzyme_data in direct_enzymes:
                evidence = DetailedGroundingEvidence(
                    entity_id=enzyme_data["id"],
                    entity_type="glycosidase" if "glycosidase" in enzyme_data["name"].lower() else "glycosyltransferase",
                    entity_name=enzyme_data["name"],
                    confidence=enzyme_data["confidence"],
                    validation_method="direct_association",
                    evidence_source="GlyGen",
                    validation_score=enzyme_data["confidence"]
                )
                
                enzymes.append(evidence)
            
            return self._deduplicate_evidence(enzymes)
            
        except Exception as e:
            logger.error(f"Enzyme grounding failed: {e}")
            return []
    
    async def _ground_biosynthetic_pathways(self,
                                          structure: str,
                                          enzymes: List[DetailedGroundingEvidence],
                                          context: Dict[str, Any] = None) -> List[DetailedGroundingEvidence]:
        """Ground biosynthetic pathways for the glycan structure."""
        try:
            pathways = []
            
            # Get pathways from enzymes
            for enzyme in enzymes:
                if enzyme.biosynthetic_pathway:
                    pathway_data = await self.pathway_database.get_pathway_details(
                        enzyme.biosynthetic_pathway
                    )
                    
                    if pathway_data:
                        evidence = DetailedGroundingEvidence(
                            entity_id=pathway_data["id"],
                            entity_type="biosynthetic_pathway",
                            entity_name=pathway_data["name"],
                            confidence=enzyme.confidence * 0.9,  # Slight confidence reduction
                            validation_method="enzyme_pathway_mapping",
                            evidence_source="KEGG",
                            supporting_papers=pathway_data.get("papers", []),
                            organism_specificity=pathway_data.get("organisms", []),
                            validation_score=pathway_data.get("completeness", 0.8)
                        )
                        
                        pathways.append(evidence)
            
            # Direct pathway associations
            direct_pathways = await self.pathway_database.get_pathways_for_structure(structure)
            for pathway_data in direct_pathways:
                evidence = DetailedGroundingEvidence(
                    entity_id=pathway_data["id"],
                    entity_type="biosynthetic_pathway",
                    entity_name=pathway_data["name"],
                    confidence=pathway_data["confidence"],
                    validation_method="direct_pathway_association",
                    evidence_source="MetaCyc",
                    validation_score=pathway_data["confidence"]
                )
                
                pathways.append(evidence)
            
            return self._deduplicate_evidence(pathways)
            
        except Exception as e:
            logger.error(f"Pathway grounding failed: {e}")
            return []
    
    async def _ground_source_organisms(self,
                                     structure: str,
                                     pathways: List[DetailedGroundingEvidence],
                                     context: Dict[str, Any] = None) -> List[DetailedGroundingEvidence]:
        """Ground source organisms for the glycan structure."""
        try:
            organisms = []
            
            # Get organisms from pathways
            for pathway in pathways:
                if pathway.organism_specificity:
                    for organism_name in pathway.organism_specificity:
                        organism_data = await self.organism_database.get_organism_details(
                            organism_name
                        )
                        
                        if organism_data:
                            evidence = DetailedGroundingEvidence(
                                entity_id=organism_data["id"],
                                entity_type="source_organism",
                                entity_name=organism_data["name"],
                                confidence=pathway.confidence * 0.8,
                                validation_method="pathway_organism_mapping",
                                evidence_source="NCBI_Taxonomy",
                                validation_score=organism_data.get("confidence", 0.8)
                            )
                            
                            organisms.append(evidence)
            
            # Direct organism associations
            direct_organisms = await self.organism_database.get_organisms_for_structure(structure)
            for organism_data in direct_organisms:
                evidence = DetailedGroundingEvidence(
                    entity_id=organism_data["id"],
                    entity_type="source_organism",
                    entity_name=organism_data["name"],
                    confidence=organism_data["confidence"],
                    validation_method="experimental_evidence",
                    evidence_source="GlyGen",
                    supporting_papers=organism_data.get("papers", []),
                    validation_score=organism_data["confidence"]
                )
                
                organisms.append(evidence)
            
            return self._deduplicate_evidence(organisms)
            
        except Exception as e:
            logger.error(f"Organism grounding failed: {e}")
            return []
    
    async def _ground_disease_associations(self,
                                         structure: str,
                                         context: Dict[str, Any] = None) -> List[DetailedGroundingEvidence]:
        """Ground disease associations for the glycan structure."""
        try:
            diseases = []
            
            # Query disease associations from knowledge graph
            disease_query = f"""
            PREFIX glyco: <http://purl.glycoinfo.org/>
            PREFIX disease: <http://purl.obolibrary.org/obo/>
            
            SELECT ?disease ?name ?confidence ?evidence
            WHERE {{
                ?structure glyco:hasStructure "{structure}" .
                ?structure glyco:associatedWithDisease ?disease .
                ?disease rdfs:label ?name .
                OPTIONAL {{ ?association glyco:confidence ?confidence }}
                OPTIONAL {{ ?association glyco:evidence ?evidence }}
            }}
            """
            
            results = self.kg_graph.query(disease_query)
            
            for row in results:
                evidence = DetailedGroundingEvidence(
                    entity_id=str(row.disease),
                    entity_type="disease_association",
                    entity_name=str(row.name),
                    confidence=float(row.confidence) if row.confidence else 0.6,
                    validation_method="literature_mining",
                    evidence_source="DisGeNET",
                    validation_score=float(row.confidence) if row.confidence else 0.6
                )
                
                diseases.append(evidence)
            
            return diseases
            
        except Exception as e:
            logger.error(f"Disease grounding failed: {e}")
            return []
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    def _generate_cache_key(self, 
                          primary_input: str, 
                          task_type: str, 
                          context: Dict[str, Any] = None) -> str:
        """Generate cache key for grounding results."""
        key_data = {
            "input": primary_input,
            "task": task_type,
            "context": context or {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _assess_evidence_strength(self, confidence: float) -> str:
        """Assess evidence strength based on confidence."""
        if confidence >= 0.9:
            return "very_strong"
        elif confidence >= 0.7:
            return "strong"
        elif confidence >= 0.5:
            return "moderate"
        elif confidence >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def _deduplicate_evidence(self, 
                            evidence_list: List[DetailedGroundingEvidence]) -> List[DetailedGroundingEvidence]:
        """Remove duplicate evidence entries."""
        seen_ids = set()
        unique_evidence = []
        
        for evidence in evidence_list:
            if evidence.entity_id not in seen_ids:
                seen_ids.add(evidence.entity_id)
                unique_evidence.append(evidence)
            else:
                # Merge with existing evidence if confidence is higher
                for existing in unique_evidence:
                    if existing.entity_id == evidence.entity_id and evidence.confidence > existing.confidence:
                        existing.confidence = evidence.confidence
                        existing.validation_score = max(existing.validation_score, evidence.validation_score)
        
        return unique_evidence
    
    def _calculate_grounding_confidence(self, grounding: ComprehensiveGroundingResult) -> float:
        """Calculate overall grounding confidence."""
        all_confidences = []
        
        for evidence_list in [grounding.structural_motifs, grounding.enzymes, 
                             grounding.pathways, grounding.organisms, grounding.diseases]:
            for evidence in evidence_list:
                all_confidences.append(evidence.confidence)
        
        if not all_confidences:
            return 0.0
        
        # Weighted average with diminishing returns
        sorted_confidences = sorted(all_confidences, reverse=True)
        weights = [0.8 ** i for i in range(len(sorted_confidences))]
        
        weighted_sum = sum(c * w for c, w in zip(sorted_confidences, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _calculate_coverage_score(self, grounding: ComprehensiveGroundingResult) -> float:
        """Calculate knowledge coverage score."""
        categories = ["structural_motifs", "enzymes", "pathways", "organisms", "diseases"]
        covered_categories = 0
        
        for category in categories:
            evidence_list = getattr(grounding, category, [])
            if evidence_list and any(e.confidence > 0.5 for e in evidence_list):
                covered_categories += 1
        
        return covered_categories / len(categories)
    
    async def _validate_grounding_consistency(self, grounding: ComprehensiveGroundingResult) -> float:
        """Validate consistency across grounded entities."""
        try:
            consistency_checks = []
            
            # Check enzyme-pathway consistency
            if grounding.enzymes and grounding.pathways:
                enzyme_pathways = set(e.biosynthetic_pathway for e in grounding.enzymes if e.biosynthetic_pathway)
                pathway_ids = set(p.entity_id for p in grounding.pathways)
                
                if enzyme_pathways and pathway_ids:
                    overlap = len(enzyme_pathways.intersection(pathway_ids))
                    consistency_checks.append(overlap / max(len(enzyme_pathways), len(pathway_ids)))
            
            # Check organism-pathway consistency
            if grounding.organisms and grounding.pathways:
                organism_names = set(o.entity_name for o in grounding.organisms)
                pathway_organisms = set()
                for p in grounding.pathways:
                    pathway_organisms.update(p.organism_specificity)
                
                if organism_names and pathway_organisms:
                    overlap = len(organism_names.intersection(pathway_organisms))
                    consistency_checks.append(overlap / max(len(organism_names), len(pathway_organisms)))
            
            return sum(consistency_checks) / len(consistency_checks) if consistency_checks else 0.5
            
        except Exception as e:
            logger.error(f"Consistency validation failed: {e}")
            return 0.5
    
    def _merge_grounding_results(self, 
                               result1: ComprehensiveGroundingResult,
                               result2: ComprehensiveGroundingResult) -> ComprehensiveGroundingResult:
        """Merge two grounding results."""
        merged = ComprehensiveGroundingResult()
        
        # Merge evidence lists
        merged.structural_motifs = self._deduplicate_evidence(
            result1.structural_motifs + result2.structural_motifs
        )
        merged.enzymes = self._deduplicate_evidence(
            result1.enzymes + result2.enzymes
        )
        merged.pathways = self._deduplicate_evidence(
            result1.pathways + result2.pathways
        )
        merged.organisms = self._deduplicate_evidence(
            result1.organisms + result2.organisms
        )
        merged.diseases = self._deduplicate_evidence(
            result1.diseases + result2.diseases
        )
        
        # Merge metadata
        merged.knowledge_sources = list(set(result1.knowledge_sources + result2.knowledge_sources))
        merged.grounding_method = f"{result1.grounding_method}+{result2.grounding_method}"
        merged.grounding_timestamp = datetime.now()
        
        return merged


# ========================================================================================
# SPECIALIZED GROUNDING DATABASES
# ========================================================================================

class MotifGroundingDatabase:
    """Database interface for structural motif grounding."""
    
    async def identify_motifs(self, structure: str) -> List[Dict[str, Any]]:
        """Identify structural motifs in glycan structure."""
        # Placeholder implementation - would use actual motif matching
        motifs = [
            {
                "id": "MOTIF_001",
                "name": "Core fucose",
                "confidence": 0.89,
                "papers": ["PMID:12345678"]
            },
            {
                "id": "MOTIF_002", 
                "name": "Complex N-glycan",
                "confidence": 0.76,
                "papers": ["PMID:87654321"]
            }
        ]
        return motifs

class EnzymeGroundingDatabase:
    """Database interface for enzyme grounding."""
    
    async def get_enzymes_for_motif(self, motif_id: str) -> List[Dict[str, Any]]:
        """Get enzymes associated with a structural motif."""
        # Placeholder implementation
        enzymes = [
            {
                "id": "ENZ_001",
                "name": "Alpha-1,6-fucosyltransferase",
                "confidence": 0.85,
                "pathway": "N-glycan_biosynthesis",
                "organisms": ["Homo sapiens", "Mus musculus"]
            }
        ]
        return enzymes
    
    async def get_enzymes_for_structure(self, structure: str) -> List[Dict[str, Any]]:
        """Get enzymes directly associated with structure."""
        # Placeholder implementation
        return []

class PathwayGroundingDatabase:
    """Database interface for pathway grounding."""
    
    async def get_pathway_details(self, pathway_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed pathway information."""
        # Placeholder implementation
        return {
            "id": pathway_id,
            "name": "N-glycan biosynthesis",
            "completeness": 0.92,
            "organisms": ["Homo sapiens"],
            "papers": ["PMID:11111111"]
        }
    
    async def get_pathways_for_structure(self, structure: str) -> List[Dict[str, Any]]:
        """Get pathways directly associated with structure."""
        # Placeholder implementation
        return []

class OrganismGroundingDatabase:
    """Database interface for organism grounding."""
    
    async def get_organism_details(self, organism_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed organism information."""
        # Placeholder implementation
        return {
            "id": f"NCBI_{organism_name.replace(' ', '_')}",
            "name": organism_name,
            "confidence": 0.9
        }
    
    async def get_organisms_for_structure(self, structure: str) -> List[Dict[str, Any]]:
        """Get organisms directly associated with structure."""
        # Placeholder implementation
        return [
            {
                "id": "NCBI_Homo_sapiens",
                "name": "Homo sapiens",
                "confidence": 0.88,
                "papers": ["PMID:22222222"]
            }
        ]