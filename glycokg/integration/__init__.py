"""
GlycoKG Data Integration Module

This module provides clients and coordinators for integrating data from
multiple glycoinformatics databases including GlyTouCan, GlyGen, and GlycoPOST.
"""

from .glytoucan_client import GlyTouCanClient, GlycanStructure
from .glygen_client import GlyGenClient, ProteinGlycanAssociation, ProteinInfo  
from .glycopost_client import GlycoPOSTClient, MSSpectrum, ExperimentalEvidence
from .coordinator import DataIntegrationCoordinator

__all__ = [
    'GlyTouCanClient',
    'GlycanStructure',
    'GlyGenClient', 
    'ProteinGlycanAssociation',
    'ProteinInfo',
    'GlycoPOSTClient',
    'MSSpectrum',
    'ExperimentalEvidence',
    'DataIntegrationCoordinator'
]