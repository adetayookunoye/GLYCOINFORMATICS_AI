"""
Advanced contrastive learning implementation for GlycoLLM.

This module implements sophisticated contrastive learning strategies for
multimodal glycoinformatics representation learning with hard negative mining,
temperature scaling, and specialized loss functions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Optional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Mock classes for development without PyTorch
    class nn:
        class Module:
            pass
    torch = None
    F = None

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveBatch:
    """Container for contrastive learning batch data."""
    
    # Anchor samples
    anchor_structures: Optional[torch.Tensor] = None
    anchor_spectra: Optional[torch.Tensor] = None
    anchor_text: Optional[torch.Tensor] = None
    
    # Positive samples (same glycan, different modality)
    positive_structures: Optional[torch.Tensor] = None
    positive_spectra: Optional[torch.Tensor] = None
    positive_text: Optional[torch.Tensor] = None
    
    # Negative samples (different glycans)
    negative_structures: Optional[torch.Tensor] = None
    negative_spectra: Optional[torch.Tensor] = None
    negative_text: Optional[torch.Tensor] = None
    
    # Metadata
    glycan_ids: Optional[List[str]] = None
    batch_size: int = 0


class NegativeSampler(ABC):
    """Abstract base class for negative sampling strategies."""
    
    @abstractmethod
    def sample_negatives(self, 
                        anchor_idx: int, 
                        batch_data: Dict[str, Any],
                        num_negatives: int) -> List[int]:
        """
        Sample negative examples for contrastive learning.
        
        Args:
            anchor_idx: Index of anchor sample
            batch_data: Batch data dictionary
            num_negatives: Number of negatives to sample
            
        Returns:
            List of negative sample indices
        """
        pass


class RandomNegativeSampler(NegativeSampler):
    """Random negative sampling strategy."""
    
    def __init__(self, exclude_positives: bool = True):
        self.exclude_positives = exclude_positives
        
    def sample_negatives(self, 
                        anchor_idx: int, 
                        batch_data: Dict[str, Any],
                        num_negatives: int) -> List[int]:
        """Sample random negatives excluding anchor and positives."""
        
        batch_size = len(batch_data.get('glycan_ids', []))
        if batch_size <= 1:
            return []
            
        # Get all possible indices
        all_indices = set(range(batch_size))
        
        # Remove anchor
        all_indices.discard(anchor_idx)
        
        # Remove positives if specified
        if self.exclude_positives and 'positive_indices' in batch_data:
            positive_indices = batch_data['positive_indices'].get(anchor_idx, [])
            all_indices.difference_update(positive_indices)
            
        # Sample negatives
        available_indices = list(all_indices)
        if not available_indices:
            return []
            
        num_to_sample = min(num_negatives, len(available_indices))
        return random.sample(available_indices, num_to_sample)


class HardNegativeSampler(NegativeSampler):
    """Hard negative sampling based on embedding similarity."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 fallback_sampler: Optional[NegativeSampler] = None):
        self.similarity_threshold = similarity_threshold
        self.fallback_sampler = fallback_sampler or RandomNegativeSampler()
        
    def sample_negatives(self, 
                        anchor_idx: int, 
                        batch_data: Dict[str, Any],
                        num_negatives: int) -> List[int]:
        """Sample hard negatives based on similarity."""
        
        if not HAS_TORCH or 'embeddings' not in batch_data:
            # Fall back to random sampling
            return self.fallback_sampler.sample_negatives(
                anchor_idx, batch_data, num_negatives
            )
            
        embeddings = batch_data['embeddings']
        anchor_embedding = embeddings[anchor_idx]
        
        # Compute similarities
        similarities = torch.cosine_similarity(
            anchor_embedding.unsqueeze(0), embeddings, dim=1
        )
        
        # Find hard negatives (high similarity but different glycan)
        batch_size = embeddings.size(0)
        glycan_ids = batch_data.get('glycan_ids', [])
        anchor_glycan_id = glycan_ids[anchor_idx] if glycan_ids else None
        
        hard_negatives = []
        
        for idx in range(batch_size):
            if idx == anchor_idx:
                continue
                
            # Skip if same glycan (positive)
            if glycan_ids and glycan_ids[idx] == anchor_glycan_id:
                continue
                
            # Check if similarity is above threshold (hard negative)
            if similarities[idx] > self.similarity_threshold:
                hard_negatives.append((idx, similarities[idx].item()))
                
        # Sort by similarity (descending) and take top negatives
        hard_negatives.sort(key=lambda x: x[1], reverse=True)
        
        negative_indices = [idx for idx, _ in hard_negatives[:num_negatives]]
        
        # Fill remaining with random negatives if needed
        if len(negative_indices) < num_negatives:
            remaining = num_negatives - len(negative_indices)
            random_negatives = self.fallback_sampler.sample_negatives(
                anchor_idx, batch_data, remaining
            )
            # Avoid duplicates
            for neg_idx in random_negatives:
                if neg_idx not in negative_indices:
                    negative_indices.append(neg_idx)
                    
        return negative_indices[:num_negatives]


class StructuralNegativeSampler(NegativeSampler):
    """Structural similarity-based negative sampling."""
    
    def __init__(self, 
                 structural_similarity_fn: Optional[callable] = None,
                 similarity_threshold: float = 0.6):
        self.structural_similarity_fn = structural_similarity_fn
        self.similarity_threshold = similarity_threshold
        
    def sample_negatives(self, 
                        anchor_idx: int, 
                        batch_data: Dict[str, Any],
                        num_negatives: int) -> List[int]:
        """Sample negatives based on structural similarity."""
        
        if not self.structural_similarity_fn or 'structures' not in batch_data:
            # Fall back to random sampling
            return RandomNegativeSampler().sample_negatives(
                anchor_idx, batch_data, num_negatives
            )
            
        structures = batch_data['structures']
        anchor_structure = structures[anchor_idx]
        
        # Compute structural similarities
        similarities = []
        for idx, structure in enumerate(structures):
            if idx != anchor_idx:
                sim = self.structural_similarity_fn(anchor_structure, structure)
                similarities.append((idx, sim))
                
        # Select hard negatives (structurally similar but different)
        hard_negatives = [
            idx for idx, sim in similarities 
            if sim > self.similarity_threshold
        ]
        
        # Sort by similarity and sample
        hard_negatives.sort(key=lambda idx: similarities[idx][1], reverse=True)
        
        selected = hard_negatives[:num_negatives]
        
        # Fill with random if needed
        if len(selected) < num_negatives:
            remaining = num_negatives - len(selected)
            all_indices = [idx for idx, _ in similarities if idx not in selected]
            if all_indices:
                random_selection = random.sample(
                    all_indices, 
                    min(remaining, len(all_indices))
                )
                selected.extend(random_selection)
                
        return selected[:num_negatives]


if HAS_TORCH:
    class MultiModalContrastiveLoss(nn.Module):
        """
        Advanced multi-modal contrastive loss for GlycoLLM.
        
        Supports:
        - Temperature scaling
        - Hard negative mining
        - Symmetric loss computation
        - Curriculum-aware weighting
        """
        
        def __init__(self, 
                     temperature: float = 0.07,
                     negative_sampler: Optional[NegativeSampler] = None,
                     loss_type: str = "info_nce",
                     symmetric: bool = True):
            super().__init__()
            
            self.temperature = temperature
            self.negative_sampler = negative_sampler or HardNegativeSampler()
            self.loss_type = loss_type
            self.symmetric = symmetric
            
            # Learnable temperature parameter
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
            
        def forward(self, 
                   anchor_embeddings: torch.Tensor,
                   positive_embeddings: torch.Tensor,
                   negative_embeddings: Optional[torch.Tensor] = None,
                   **kwargs) -> Dict[str, torch.Tensor]:
            """
            Compute contrastive loss.
            
            Args:
                anchor_embeddings: Anchor embeddings [batch_size, dim]
                positive_embeddings: Positive embeddings [batch_size, dim]
                negative_embeddings: Negative embeddings [batch_size, num_negatives, dim]
                
            Returns:
                Loss dictionary
            """
            
            batch_size = anchor_embeddings.size(0)
            device = anchor_embeddings.device
            
            # Normalize embeddings
            anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
            positive_norm = F.normalize(positive_embeddings, p=2, dim=1)
            
            # Compute temperature
            temperature = torch.exp(self.log_temperature)
            
            if self.loss_type == "info_nce":
                loss = self._compute_info_nce_loss(
                    anchor_norm, positive_norm, negative_embeddings, temperature
                )
            elif self.loss_type == "triplet":
                loss = self._compute_triplet_loss(
                    anchor_norm, positive_norm, negative_embeddings
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
                
            # Add symmetric loss if enabled
            if self.symmetric and self.loss_type == "info_nce":
                symmetric_loss = self._compute_info_nce_loss(
                    positive_norm, anchor_norm, negative_embeddings, temperature
                )
                loss = (loss + symmetric_loss) / 2
                
            return {
                'contrastive_loss': loss,
                'temperature': temperature.detach(),
                'positive_similarity': torch.cosine_similarity(anchor_norm, positive_norm).mean()
            }
            
        def _compute_info_nce_loss(self, 
                                  anchor: torch.Tensor,
                                  positive: torch.Tensor,
                                  negatives: Optional[torch.Tensor],
                                  temperature: torch.Tensor) -> torch.Tensor:
            """Compute InfoNCE loss."""
            
            batch_size = anchor.size(0)
            
            # Positive similarities
            pos_sim = torch.sum(anchor * positive, dim=1) / temperature  # [batch_size]
            
            if negatives is not None:
                # Negative similarities
                neg_sim = torch.bmm(
                    anchor.unsqueeze(1),  # [batch_size, 1, dim]
                    negatives.transpose(1, 2)  # [batch_size, dim, num_negatives]
                ).squeeze(1) / temperature  # [batch_size, num_negatives]
                
                # Combine positive and negative scores
                logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
                labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
                
            else:
                # Use in-batch negatives
                similarity_matrix = torch.matmul(anchor, positive.T) / temperature
                labels = torch.arange(batch_size, device=anchor.device)
                logits = similarity_matrix
                
            loss = F.cross_entropy(logits, labels)
            return loss
            
        def _compute_triplet_loss(self, 
                                 anchor: torch.Tensor,
                                 positive: torch.Tensor,
                                 negatives: Optional[torch.Tensor],
                                 margin: float = 0.2) -> torch.Tensor:
            """Compute triplet loss."""
            
            if negatives is None:
                # Use hardest in-batch negative
                batch_size = anchor.size(0)
                similarity_matrix = torch.matmul(anchor, positive.T)
                
                # Get hardest negatives (excluding positives)
                mask = torch.eye(batch_size, device=anchor.device).bool()
                similarity_matrix.masked_fill_(mask, -float('inf'))
                hardest_neg_idx = similarity_matrix.argmax(dim=1)
                
                hardest_negatives = positive[hardest_neg_idx]
            else:
                # Use provided negatives (take first one for simplicity)
                hardest_negatives = negatives[:, 0, :]
                
            # Compute distances
            pos_dist = 1 - torch.cosine_similarity(anchor, positive)
            neg_dist = 1 - torch.cosine_similarity(anchor, hardest_negatives)
            
            # Triplet loss
            loss = F.relu(pos_dist - neg_dist + margin).mean()
            return loss


    class AdaptiveContrastiveLoss(nn.Module):
        """
        Adaptive contrastive loss that adjusts based on curriculum stage.
        
        Features:
        - Dynamic temperature scaling
        - Curriculum-aware negative sampling
        - Progressive difficulty increase
        """
        
        def __init__(self, 
                     base_temperature: float = 0.07,
                     temperature_schedule: Optional[Dict[str, float]] = None,
                     curriculum_weights: Optional[Dict[str, float]] = None):
            super().__init__()
            
            self.base_temperature = base_temperature
            self.temperature_schedule = temperature_schedule or {}
            self.curriculum_weights = curriculum_weights or {}
            
            # Multiple contrastive loss modules
            self.structure_text_loss = MultiModalContrastiveLoss(
                temperature=base_temperature,
                negative_sampler=HardNegativeSampler(similarity_threshold=0.8)
            )
            
            self.structure_spectra_loss = MultiModalContrastiveLoss(
                temperature=base_temperature,
                negative_sampler=HardNegativeSampler(similarity_threshold=0.7)
            )
            
            self.spectra_text_loss = MultiModalContrastiveLoss(
                temperature=base_temperature,
                negative_sampler=RandomNegativeSampler()
            )
            
        def forward(self, 
                   embeddings: Dict[str, torch.Tensor],
                   curriculum_stage: Optional[str] = None,
                   **kwargs) -> Dict[str, torch.Tensor]:
            """
            Compute adaptive contrastive loss.
            
            Args:
                embeddings: Dictionary of modality embeddings
                curriculum_stage: Current curriculum stage
                
            Returns:
                Loss dictionary
            """
            
            losses = {}
            total_loss = 0.0
            
            # Get curriculum weights
            stage_weights = self.curriculum_weights.get(curriculum_stage, {})
            
            # Structure-Text contrastive loss
            if 'structure' in embeddings and 'text' in embeddings:
                struct_text_loss = self.structure_text_loss(
                    embeddings['structure'], embeddings['text']
                )
                weight = stage_weights.get('structure_text', 1.0)
                losses['structure_text_loss'] = struct_text_loss['contrastive_loss']
                total_loss += weight * struct_text_loss['contrastive_loss']
                
            # Structure-Spectra contrastive loss
            if 'structure' in embeddings and 'spectra' in embeddings:
                struct_spectra_loss = self.structure_spectra_loss(
                    embeddings['structure'], embeddings['spectra']
                )
                weight = stage_weights.get('structure_spectra', 1.0)
                losses['structure_spectra_loss'] = struct_spectra_loss['contrastive_loss']
                total_loss += weight * struct_spectra_loss['contrastive_loss']
                
            # Spectra-Text contrastive loss
            if 'spectra' in embeddings and 'text' in embeddings:
                spectra_text_loss = self.spectra_text_loss(
                    embeddings['spectra'], embeddings['text']
                )
                weight = stage_weights.get('spectra_text', 0.5)  # Lower weight by default
                losses['spectra_text_loss'] = spectra_text_loss['contrastive_loss']
                total_loss += weight * spectra_text_loss['contrastive_loss']
                
            losses['total_contrastive_loss'] = total_loss
            
            return losses


    class ContrastiveLearningManager:
        """
        Manages contrastive learning setup and batch preparation.
        
        Handles:
        - Batch construction for contrastive learning
        - Negative sampling coordination
        - Loss computation orchestration
        """
        
        def __init__(self, 
                     negative_sampler: Optional[NegativeSampler] = None,
                     num_negatives: int = 4,
                     in_batch_negatives: bool = True):
            
            self.negative_sampler = negative_sampler or HardNegativeSampler()
            self.num_negatives = num_negatives
            self.in_batch_negatives = in_batch_negatives
            
        def prepare_contrastive_batch(self, 
                                    batch_data: Dict[str, Any]) -> ContrastiveBatch:
            """
            Prepare batch for contrastive learning.
            
            Args:
                batch_data: Original batch data
                
            Returns:
                Contrastive batch with anchors, positives, and negatives
            """
            
            if not HAS_TORCH:
                return ContrastiveBatch()
                
            batch_size = len(batch_data.get('glycan_ids', []))
            contrastive_batch = ContrastiveBatch(batch_size=batch_size)
            
            # Extract anchors (use all samples as anchors)
            if 'structure_tokens' in batch_data:
                contrastive_batch.anchor_structures = batch_data['structure_tokens']
            if 'spectra_tokens' in batch_data:
                contrastive_batch.anchor_spectra = batch_data['spectra_tokens']
            if 'text_tokens' in batch_data:
                contrastive_batch.anchor_text = batch_data['text_tokens']
                
            # For multimodal contrastive learning, each modality serves as
            # positive for the others from the same glycan
            contrastive_batch.positive_structures = contrastive_batch.anchor_structures
            contrastive_batch.positive_spectra = contrastive_batch.anchor_spectra
            contrastive_batch.positive_text = contrastive_batch.anchor_text
            
            # Sample negatives if not using in-batch negatives
            if not self.in_batch_negatives:
                negative_indices = []
                for i in range(batch_size):
                    neg_idx = self.negative_sampler.sample_negatives(
                        i, batch_data, self.num_negatives
                    )
                    negative_indices.append(neg_idx)
                    
                # Construct negative tensors
                if contrastive_batch.anchor_structures is not None:
                    neg_structures = []
                    for neg_idx_list in negative_indices:
                        neg_batch = torch.stack([
                            contrastive_batch.anchor_structures[idx] 
                            for idx in neg_idx_list
                        ])
                        neg_structures.append(neg_batch)
                    contrastive_batch.negative_structures = torch.stack(neg_structures)
                    
                # Similar for other modalities...
                
            contrastive_batch.glycan_ids = batch_data.get('glycan_ids', [])
            return contrastive_batch
            
        def compute_contrastive_embeddings(self, 
                                         model: nn.Module,
                                         contrastive_batch: ContrastiveBatch) -> Dict[str, torch.Tensor]:
            """
            Compute embeddings for contrastive learning.
            
            Args:
                model: GlycoLLM model
                contrastive_batch: Prepared contrastive batch
                
            Returns:
                Dictionary of embeddings by modality
            """
            
            embeddings = {}
            
            # Get embeddings for each modality
            if contrastive_batch.anchor_structures is not None:
                struct_emb = model.encode_structure(contrastive_batch.anchor_structures)
                embeddings['structure'] = struct_emb
                
            if contrastive_batch.anchor_spectra is not None:
                spectra_emb = model.encode_spectra(contrastive_batch.anchor_spectra)
                embeddings['spectra'] = spectra_emb
                
            if contrastive_batch.anchor_text is not None:
                text_emb = model.encode_text(contrastive_batch.anchor_text)
                embeddings['text'] = text_emb
                
            return embeddings

else:
    # Mock classes when PyTorch is not available
    class MultiModalContrastiveLoss:
        def __init__(self, **kwargs):
            pass
            
    class AdaptiveContrastiveLoss:
        def __init__(self, **kwargs):
            pass
            
    class ContrastiveLearningManager:
        def __init__(self, **kwargs):
            pass


def create_contrastive_manager(config: Optional[Dict[str, Any]] = None) -> ContrastiveLearningManager:
    """
    Create a contrastive learning manager with specified configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured contrastive learning manager
    """
    
    if not config:
        config = {}
        
    # Select negative sampler
    sampler_type = config.get('negative_sampler', 'hard')
    
    if sampler_type == 'hard':
        negative_sampler = HardNegativeSampler(
            similarity_threshold=config.get('similarity_threshold', 0.7)
        )
    elif sampler_type == 'structural':
        negative_sampler = StructuralNegativeSampler(
            similarity_threshold=config.get('similarity_threshold', 0.6)
        )
    else:
        negative_sampler = RandomNegativeSampler()
        
    return ContrastiveLearningManager(
        negative_sampler=negative_sampler,
        num_negatives=config.get('num_negatives', 4),
        in_batch_negatives=config.get('in_batch_negatives', True)
    )


def create_contrastive_loss(config: Optional[Dict[str, Any]] = None) -> Union[MultiModalContrastiveLoss, AdaptiveContrastiveLoss]:
    """
    Create a contrastive loss function with specified configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured contrastive loss function
    """
    
    if not HAS_TORCH:
        logger.warning("PyTorch not available - returning mock contrastive loss")
        return MultiModalContrastiveLoss()
        
    if not config:
        config = {}
        
    loss_type = config.get('type', 'adaptive')
    
    if loss_type == 'adaptive':
        return AdaptiveContrastiveLoss(
            base_temperature=config.get('temperature', 0.07),
            temperature_schedule=config.get('temperature_schedule'),
            curriculum_weights=config.get('curriculum_weights')
        )
    else:
        return MultiModalContrastiveLoss(
            temperature=config.get('temperature', 0.07),
            loss_type=config.get('loss_function', 'info_nce'),
            symmetric=config.get('symmetric', True)
        )