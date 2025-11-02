"""
Core Glycoinformatics Ontology for GlycoKG

This module defines the core ontological framework for representing
glycan structures, protein associations, experimental evidence, and
biological relationships in RDF.
"""

import logging
from typing import Dict, List, Optional, Set, Any, Union
from datetime import datetime
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, DCTERMS
import json

logger = logging.getLogger(__name__)


class GlycoOntology:
    """
    Core glycoinformatics ontology manager.
    
    Defines classes, properties, and relationships for comprehensive
    representation of glycan knowledge in RDF format.
    """
    
    def __init__(self, base_uri: str = "http://purl.glycoinfo.org/"):
        """
        Initialize the glyco ontology.
        
        Args:
            base_uri: Base URI for the ontology namespace
        """
        self.base_uri = base_uri
        
        # Define namespaces
        self.GLYCO = Namespace(f"{base_uri}ontology/")
        self.GLYCAN = Namespace(f"{base_uri}glycan/")
        self.PROTEIN = Namespace("http://purl.uniprot.org/uniprot/")
        self.PUBMED = Namespace("http://rdf.ncbi.nlm.nih.gov/pubmed/")
        self.TAXONOMY = Namespace("http://purl.uniprot.org/taxonomy/")
        self.TISSUE = Namespace("http://purl.obolibrary.org/obo/UBERON_")
        self.DISEASE = Namespace("http://purl.obolibrary.org/obo/DOID_")
        self.CHEBI = Namespace("http://purl.obolibrary.org/obo/CHEBI_")
        
        # Initialize graph
        self.graph = Graph()
        
        # Bind namespaces
        self.graph.bind("glyco", self.GLYCO)
        self.graph.bind("glycan", self.GLYCAN)
        self.graph.bind("uniprot", self.PROTEIN)
        self.graph.bind("pubmed", self.PUBMED)
        self.graph.bind("taxonomy", self.TAXONOMY)
        self.graph.bind("tissue", self.TISSUE)
        self.graph.bind("disease", self.DISEASE)
        self.graph.bind("chebi", self.CHEBI)
        self.graph.bind("owl", OWL)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("dcterms", DCTERMS)
        
        # Initialize ontology
        self._define_core_classes()
        self._define_properties()
        self._define_constraints()
        
    def _define_core_classes(self):
        """Define core ontology classes"""
        
        # Glycan structure classes
        classes = [
            # Core glycan classes
            (self.GLYCO.Glycan, "A carbohydrate structure composed of monosaccharides"),
            (self.GLYCO.Monosaccharide, "A simple sugar unit"),
            (self.GLYCO.Linkage, "A glycosidic bond between monosaccharides"),
            (self.GLYCO.GlycanMotif, "A recurring structural pattern in glycans"),
            (self.GLYCO.Composition, "Monosaccharide composition of a glycan"),
            
            # Protein-related classes
            (self.GLYCO.Protein, "A protein that can be glycosylated"),
            (self.GLYCO.GlycosylationSite, "A specific amino acid position where glycosylation occurs"),
            (self.GLYCO.ProteinGlycanAssociation, "An association between a protein and glycan"),
            (self.GLYCO.GlycoproteinComplex, "A complex of protein and attached glycans"),
            
            # Biological context classes
            (self.GLYCO.Organism, "A biological organism that produces glycans"),
            (self.GLYCO.Tissue, "A tissue type where glycans are found"),
            (self.GLYCO.CellType, "A specific cell type"),
            (self.GLYCO.Disease, "A disease associated with glycan alterations"),
            (self.GLYCO.BiosyntheticPathway, "A pathway for glycan biosynthesis"),
            (self.GLYCO.Enzyme, "An enzyme involved in glycan metabolism"),
            
            # Experimental classes
            (self.GLYCO.MSExperiment, "A mass spectrometry experiment"),
            (self.GLYCO.MSSpectrum, "A mass spectrum"),
            (self.GLYCO.ExperimentalEvidence, "Evidence linking structures to experimental data"),
            (self.GLYCO.Publication, "A scientific publication"),
            
            # Structural representation classes
            (self.GLYCO.StructuralRepresentation, "A way to represent glycan structure"),
            (self.GLYCO.WURCSRepresentation, "WURCS notation representation"),
            (self.GLYCO.GlycoCTRepresentation, "GlycoCT representation"),
            (self.GLYCO.IUPACRepresentation, "IUPAC nomenclature representation"),
            
            # Functional classes
            (self.GLYCO.BiologicalFunction, "A biological function of glycans"),
            (self.GLYCO.MolecularInteraction, "An interaction involving glycans"),
            (self.GLYCO.Biomarker, "A glycan biomarker for disease or condition")
        ]
        
        for class_uri, description in classes:
            self.graph.add((class_uri, RDF.type, OWL.Class))
            self.graph.add((class_uri, RDFS.comment, Literal(description)))
            
        # Define class hierarchies
        hierarchies = [
            (self.GLYCO.WURCSRepresentation, self.GLYCO.StructuralRepresentation),
            (self.GLYCO.GlycoCTRepresentation, self.GLYCO.StructuralRepresentation),
            (self.GLYCO.IUPACRepresentation, self.GLYCO.StructuralRepresentation),
            (self.GLYCO.GlycoproteinComplex, self.GLYCO.Protein),
            (self.GLYCO.MSSpectrum, self.GLYCO.MSExperiment)
        ]
        
        for subclass, superclass in hierarchies:
            self.graph.add((subclass, RDFS.subClassOf, superclass))
            
    def _define_properties(self):
        """Define object and data properties"""
        
        # Object properties (relationships between entities)
        object_properties = [
            # Glycan relationships
            (self.GLYCO.hasMonosaccharide, "Links a glycan to its constituent monosaccharides"),
            (self.GLYCO.hasLinkage, "Links monosaccharides via glycosidic bonds"),
            (self.GLYCO.hasMotif, "Links a glycan to structural motifs it contains"),
            (self.GLYCO.isIsomerOf, "Relates glycan isomers"),
            (self.GLYCO.hasComposition, "Links a glycan to its composition"),
            
            # Protein-glycan relationships  
            (self.GLYCO.associatedWith, "General association between entities"),
            (self.GLYCO.attachedTo, "Links glycan to protein attachment site"),
            (self.GLYCO.hasGlycosylationSite, "Links protein to glycosylation sites"),
            (self.GLYCO.modifies, "Links glycan to the protein it modifies"),
            
            # Biological context relationships
            (self.GLYCO.foundIn, "Links glycan to organism/tissue where found"),
            (self.GLYCO.expressedIn, "Links to expression context"),
            (self.GLYCO.involvedInPathway, "Links to biosynthetic pathway"),
            (self.GLYCO.synthesizedBy, "Links to synthesizing enzyme"),
            (self.GLYCO.associatedWithDisease, "Links to disease association"),
            
            # Experimental relationships
            (self.GLYCO.hasExperimentalEvidence, "Links to experimental evidence"),
            (self.GLYCO.identifiedIn, "Links structure to identifying experiment"),
            (self.GLYCO.hasSpectrum, "Links to mass spectrum"),
            (self.GLYCO.publishedIn, "Links to publication"),
            
            # Functional relationships
            (self.GLYCO.hasFunction, "Links to biological function"),
            (self.GLYCO.participatesIn, "Links to molecular interaction"),
            (self.GLYCO.regulatedBy, "Links to regulatory mechanism"),
            (self.GLYCO.servesAsBiomarker, "Links to biomarker role")
        ]
        
        for prop_uri, description in object_properties:
            self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
            self.graph.add((prop_uri, RDFS.comment, Literal(description)))
            
        # Data properties (literal values)
        data_properties = [
            # Structural properties
            (self.GLYCO.hasWURCSSequence, XSD.string, "WURCS notation string"),
            (self.GLYCO.hasGlycoCTSequence, XSD.string, "GlycoCT sequence"),
            (self.GLYCO.hasIUPACName, XSD.string, "IUPAC nomenclature"),
            (self.GLYCO.hasMonoisotopicMass, XSD.decimal, "Monoisotopic mass in Daltons"),
            (self.GLYCO.hasAverageMass, XSD.decimal, "Average mass in Daltons"),
            (self.GLYCO.hasMolecularFormula, XSD.string, "Molecular formula"),
            
            # Identifiers
            (self.GLYCO.hasGlyTouCanID, XSD.string, "GlyTouCan accession ID"),
            (self.GLYCO.hasUniProtID, XSD.string, "UniProt accession ID"),
            (self.GLYCO.hasPubMedID, XSD.string, "PubMed identifier"),
            (self.GLYCO.hasTaxonomyID, XSD.integer, "NCBI Taxonomy ID"),
            
            # Quantitative properties
            (self.GLYCO.hasConfidenceScore, XSD.decimal, "Confidence score (0-1)"),
            (self.GLYCO.hasAbundance, XSD.decimal, "Relative abundance"),
            (self.GLYCO.hasPrecursorMZ, XSD.decimal, "Precursor m/z value"),
            (self.GLYCO.hasChargeState, XSD.integer, "Ion charge state"),
            (self.GLYCO.hasCollisionEnergy, XSD.decimal, "Collision energy (eV)"),
            
            # Positional properties
            (self.GLYCO.hasGlycosylationPosition, XSD.integer, "Amino acid position"),
            (self.GLYCO.hasLinkagePosition, XSD.string, "Linkage position (e.g., '1-4')"),
            (self.GLYCO.hasAnomericConfiguration, XSD.string, "Anomeric configuration (alpha/beta)"),
            
            # Metadata properties
            (self.GLYCO.hasCreationDate, XSD.dateTime, "Creation timestamp"),
            (self.GLYCO.hasUpdateDate, XSD.dateTime, "Last update timestamp"),
            (self.GLYCO.hasSource, XSD.string, "Data source"),
            (self.GLYCO.hasEvidenceType, XSD.string, "Type of experimental evidence"),
            (self.GLYCO.hasExperimentalMethod, XSD.string, "Experimental method used")
        ]
        
        for prop_uri, datatype, description in data_properties:
            self.graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            self.graph.add((prop_uri, RDFS.range, datatype))
            self.graph.add((prop_uri, RDFS.comment, Literal(description)))
            
    def _define_constraints(self):
        """Define ontological constraints and restrictions"""
        
        # Cardinality constraints
        constraints = [
            # Each glycan must have at least one structural representation
            (self.GLYCO.Glycan, self.GLYCO.hasWURCSSequence, "some"),
            
            # Each glycosylation site belongs to exactly one protein
            (self.GLYCO.GlycosylationSite, self.GLYCO.belongsTo, "exactly", 1),
            
            # Each MS spectrum has exactly one precursor m/z
            (self.GLYCO.MSSpectrum, self.GLYCO.hasPrecursorMZ, "exactly", 1),
            
            # Each experimental evidence links to at least one publication
            (self.GLYCO.ExperimentalEvidence, self.GLYCO.publishedIn, "min", 1)
        ]
        
        for class_uri, property_uri, restriction_type, *args in constraints:
            restriction = BNode()
            self.graph.add((restriction, RDF.type, OWL.Restriction))
            self.graph.add((restriction, OWL.onProperty, property_uri))
            
            if restriction_type == "some":
                self.graph.add((restriction, OWL.someValuesFrom, OWL.Thing))
            elif restriction_type == "exactly":
                self.graph.add((restriction, OWL.cardinality, Literal(args[0])))
            elif restriction_type == "min":
                self.graph.add((restriction, OWL.minCardinality, Literal(args[0])))
            elif restriction_type == "max":
                self.graph.add((restriction, OWL.maxCardinality, Literal(args[0])))
                
            self.graph.add((class_uri, RDFS.subClassOf, restriction))
            
    def add_glycan(self, 
                  glytoucan_id: str,
                  wurcs_sequence: Optional[str] = None,
                  glycoct_sequence: Optional[str] = None,
                  iupac_name: Optional[str] = None,
                  mass_mono: Optional[float] = None,
                  mass_avg: Optional[float] = None,
                  composition: Optional[Dict[str, int]] = None,
                  **kwargs) -> URIRef:
        """
        Add a glycan structure to the ontology.
        
        Args:
            glytoucan_id: GlyTouCan accession ID
            wurcs_sequence: WURCS notation
            glycoct_sequence: GlycoCT sequence
            iupac_name: IUPAC nomenclature
            mass_mono: Monoisotopic mass
            mass_avg: Average mass
            composition: Monosaccharide composition
            **kwargs: Additional properties
            
        Returns:
            URIRef for the created glycan
        """
        glycan_uri = self.GLYCAN[glytoucan_id]
        
        # Core type and identifier
        self.graph.add((glycan_uri, RDF.type, self.GLYCO.Glycan))
        self.graph.add((glycan_uri, self.GLYCO.hasGlyTouCanID, Literal(glytoucan_id)))
        
        # Structural representations
        if wurcs_sequence:
            self.graph.add((glycan_uri, self.GLYCO.hasWURCSSequence, Literal(wurcs_sequence)))
        if glycoct_sequence:
            self.graph.add((glycan_uri, self.GLYCO.hasGlycoCTSequence, Literal(glycoct_sequence)))
        if iupac_name:
            self.graph.add((glycan_uri, self.GLYCO.hasIUPACName, Literal(iupac_name)))
            
        # Mass properties
        if mass_mono:
            self.graph.add((glycan_uri, self.GLYCO.hasMonoisotopicMass, Literal(mass_mono)))
        if mass_avg:
            self.graph.add((glycan_uri, self.GLYCO.hasAverageMass, Literal(mass_avg)))
            
        # Composition
        if composition:
            comp_node = BNode()
            self.graph.add((glycan_uri, self.GLYCO.hasComposition, comp_node))
            self.graph.add((comp_node, RDF.type, self.GLYCO.Composition))
            
            for mono, count in composition.items():
                self.graph.add((comp_node, self.GLYCO[f"has{mono}Count"], Literal(count)))
                
        # Metadata
        self.graph.add((glycan_uri, self.GLYCO.hasCreationDate, 
                       Literal(datetime.now(), datatype=XSD.dateTime)))
        self.graph.add((glycan_uri, self.GLYCO.hasSource, Literal("GlyTouCan")))
        
        # Additional properties
        for key, value in kwargs.items():
            if hasattr(self.GLYCO, key):
                prop_uri = getattr(self.GLYCO, key)
                self.graph.add((glycan_uri, prop_uri, Literal(value)))
                
        logger.debug(f"Added glycan {glytoucan_id} to ontology")
        return glycan_uri
        
    def add_glycoprotein_association(self,
                                   glytoucan_id: str,
                                   uniprot_id: str,
                                   glycosylation_site: Optional[int] = None,
                                   evidence_type: Optional[str] = None,
                                   **kwargs) -> URIRef:
        """
        Add a glycoprotein association (alias for protein_glycan_association).
        
        Args:
            glytoucan_id: GlyTouCan accession ID
            uniprot_id: UniProt accession ID  
            glycosylation_site: Amino acid position
            evidence_type: Type of experimental evidence
            **kwargs: Additional properties
            
        Returns:
            URIRef for the created association
        """
        return self.add_protein_glycan_association(
            uniprot_id=uniprot_id,
            glytoucan_id=glytoucan_id,
            glycosylation_site=glycosylation_site,
            evidence_type=evidence_type,
            **kwargs
        )
    
    def add_protein_glycan_association(self,
                                     uniprot_id: str,
                                     glytoucan_id: str,
                                     glycosylation_site: Optional[int] = None,
                                     site_type: Optional[str] = None,
                                     organism_taxid: Optional[int] = None,
                                     tissue: Optional[str] = None,
                                     confidence_score: Optional[float] = None,
                                     evidence_type: Optional[str] = None,
                                     **kwargs) -> URIRef:
        """
        Add a protein-glycan association to the ontology.
        
        Args:
            uniprot_id: UniProt accession ID
            glytoucan_id: GlyTouCan accession ID  
            glycosylation_site: Amino acid position
            site_type: Type of glycosylation (N-linked, O-linked)
            organism_taxid: NCBI taxonomy ID
            tissue: Tissue type
            confidence_score: Association confidence (0-1)
            **kwargs: Additional properties
            
        Returns:
            URIRef for the created association
        """
        # Create URIs
        protein_uri = self.PROTEIN[uniprot_id]
        glycan_uri = self.GLYCAN[glytoucan_id]
        
        # Create association as a reified relationship
        assoc_id = f"{uniprot_id}_{glytoucan_id}_{glycosylation_site or 'unknown'}"
        assoc_uri = self.GLYCO[f"association_{assoc_id}"]
        
        # Core association
        self.graph.add((assoc_uri, RDF.type, self.GLYCO.ProteinGlycanAssociation))
        self.graph.add((assoc_uri, self.GLYCO.hasProtein, protein_uri))
        self.graph.add((assoc_uri, self.GLYCO.hasGlycan, glycan_uri))
        
        # Direct association for simple queries
        self.graph.add((protein_uri, self.GLYCO.associatedWith, glycan_uri))
        self.graph.add((glycan_uri, self.GLYCO.attachedTo, protein_uri))
        
        # Protein information
        self.graph.add((protein_uri, RDF.type, self.GLYCO.Protein))
        self.graph.add((protein_uri, self.GLYCO.hasUniProtID, Literal(uniprot_id)))
        
        # Site information
        if glycosylation_site:
            site_uri = self.GLYCO[f"site_{uniprot_id}_{glycosylation_site}"]
            self.graph.add((site_uri, RDF.type, self.GLYCO.GlycosylationSite))
            self.graph.add((site_uri, self.GLYCO.hasGlycosylationPosition, 
                           Literal(glycosylation_site)))
            self.graph.add((site_uri, self.GLYCO.belongsTo, protein_uri))
            self.graph.add((assoc_uri, self.GLYCO.atSite, site_uri))
            
            if site_type:
                self.graph.add((site_uri, self.GLYCO.hasSiteType, Literal(site_type)))
                
        # Biological context
        if organism_taxid:
            organism_uri = self.TAXONOMY[str(organism_taxid)]
            self.graph.add((organism_uri, RDF.type, self.GLYCO.Organism))
            self.graph.add((organism_uri, self.GLYCO.hasTaxonomyID, Literal(organism_taxid)))
            self.graph.add((assoc_uri, self.GLYCO.foundIn, organism_uri))
            
        if tissue:
            tissue_uri = self.TISSUE[tissue.replace(" ", "_")]
            self.graph.add((tissue_uri, RDF.type, self.GLYCO.Tissue))
            self.graph.add((assoc_uri, self.GLYCO.expressedIn, tissue_uri))
            
        # Confidence
        if confidence_score:
            self.graph.add((assoc_uri, self.GLYCO.hasConfidenceScore, 
                           Literal(confidence_score)))
            
        # Metadata
        self.graph.add((assoc_uri, self.GLYCO.hasCreationDate,
                       Literal(datetime.now(), datatype=XSD.dateTime)))
        self.graph.add((assoc_uri, self.GLYCO.hasSource, Literal("GlyGen")))
        
        # Evidence type
        if evidence_type:
            self.graph.add((assoc_uri, self.GLYCO.hasEvidenceType, Literal(evidence_type)))
        
        # Additional properties
        for key, value in kwargs.items():
            if hasattr(self.GLYCO, key):
                prop_uri = getattr(self.GLYCO, key)
                self.graph.add((assoc_uri, prop_uri, Literal(value)))
                
        logger.debug(f"Added association {uniprot_id}-{glytoucan_id} to ontology")
        return assoc_uri
        
    def add_ms_spectrum(self,
                       spectrum_id: str,
                       glytoucan_id: Optional[str] = None,
                       precursor_mz: Optional[float] = None,
                       charge_state: Optional[int] = None,
                       collision_energy: Optional[float] = None,
                       peaks: Optional[List[tuple[float, float]]] = None,
                       **kwargs) -> URIRef:
        """
        Add an MS spectrum to the ontology.
        
        Args:
            spectrum_id: Spectrum identifier
            glytoucan_id: Associated glycan ID
            precursor_mz: Precursor m/z value
            charge_state: Ion charge state
            collision_energy: Collision energy
            peaks: Peak list [(mz, intensity), ...]
            **kwargs: Additional properties
            
        Returns:
            URIRef for the created spectrum
        """
        spectrum_uri = self.GLYCO[f"spectrum_{spectrum_id}"]
        
        # Core spectrum
        self.graph.add((spectrum_uri, RDF.type, self.GLYCO.MSSpectrum))
        self.graph.add((spectrum_uri, self.GLYCO.hasSpectrumID, Literal(spectrum_id)))
        
        # Link to glycan if known
        if glytoucan_id:
            glycan_uri = self.GLYCAN[glytoucan_id]
            self.graph.add((spectrum_uri, self.GLYCO.identifiesGlycan, glycan_uri))
            self.graph.add((glycan_uri, self.GLYCO.hasSpectrum, spectrum_uri))
            
        # MS parameters
        if precursor_mz:
            self.graph.add((spectrum_uri, self.GLYCO.hasPrecursorMZ, Literal(precursor_mz)))
        if charge_state:
            self.graph.add((spectrum_uri, self.GLYCO.hasChargeState, Literal(charge_state)))
        if collision_energy:
            self.graph.add((spectrum_uri, self.GLYCO.hasCollisionEnergy, 
                           Literal(collision_energy)))
            
        # Peak data (simplified representation)
        if peaks:
            peaks_json = json.dumps(peaks)
            self.graph.add((spectrum_uri, self.GLYCO.hasPeakData, Literal(peaks_json)))
            
        # Metadata
        self.graph.add((spectrum_uri, self.GLYCO.hasCreationDate,
                       Literal(datetime.now(), datatype=XSD.dateTime)))
        self.graph.add((spectrum_uri, self.GLYCO.hasSource, Literal("GlycoPOST")))
        
        # Additional properties
        for key, value in kwargs.items():
            if hasattr(self.GLYCO, key):
                prop_uri = getattr(self.GLYCO, key)
                self.graph.add((spectrum_uri, prop_uri, Literal(value)))
                
        logger.debug(f"Added spectrum {spectrum_id} to ontology")
        return spectrum_uri
        
    def serialize(self, format: str = "turtle") -> str:
        """
        Serialize the ontology graph.
        
        Args:
            format: RDF serialization format (turtle, xml, n3, etc.)
            
        Returns:
            Serialized RDF string
        """
        return self.graph.serialize(format=format)
        
    def save_ontology(self, filepath: str, format: str = "turtle"):
        """
        Save ontology to file (alias for save_to_file).
        
        Args:
            filepath: Output file path
            format: RDF serialization format
        """
        self.save_to_file(filepath, format)
        
    def save_to_file(self, filepath: str, format: str = "turtle"):
        """
        Save ontology to file.
        
        Args:
            filepath: Output file path
            format: RDF serialization format
        """
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.serialize(format=format))
            
        logger.info(f"Ontology saved to {filepath} in {format} format")
        
    def query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query on the ontology.
        
        Args:
            sparql_query: SPARQL query string
            
        Returns:
            List of query results
        """
        results = self.graph.query(sparql_query)
        return [dict(row.asdict()) for row in results]
        
    def get_statistics(self) -> Dict[str, int]:
        """
        Get ontology statistics.
        
        Returns:
            Dictionary with counts of different entity types
        """
        stats = {}
        
        # Count triples by class
        classes = [
            (self.GLYCO.Glycan, "glycans"),
            (self.GLYCO.Protein, "proteins"), 
            (self.GLYCO.ProteinGlycanAssociation, "associations"),
            (self.GLYCO.MSSpectrum, "spectra"),
            (self.GLYCO.GlycosylationSite, "sites"),
            (self.GLYCO.Organism, "organisms")
        ]
        
        for class_uri, name in classes:
            count_query = f"""
            SELECT (COUNT(?s) as ?count)
            WHERE {{
                ?s a <{class_uri}> .
            }}
            """
            result = self.graph.query(count_query)
            count = int(list(result)[0][0])
            stats[name] = count
            
        stats["total_triples"] = len(self.graph)
        stats["last_updated"] = datetime.now().isoformat()
        
        return stats