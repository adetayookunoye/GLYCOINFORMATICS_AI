"""
Loss functions for GlycoLLM training.

This module implements specialized loss functions for multimodal
glycoinformatics learning including contrastive learning, 
structure prediction, and cross-modal alignment.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Optional imports - available when packages installed
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module:
            pass

logger = logging.getLogger(__name__)


class LossType(Enum):
    """Types of losses for GlycoLLM training"""
    CROSS_ENTROPY = "cross_entropy"
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    MSE = "mse"
    COSINE_SIMILARITY = "cosine_similarity"


@dataclass
class LossConfig:
    """Configuration for loss computation"""
    
    # Loss weights for different tasks
    structure_weight: float = 1.0
    spectra_weight: float = 1.0  
    text_weight: float = 1.0
    contrastive_weight: float = 0.5
    
    # Contrastive learning settings
    temperature: float = 0.07
    margin: float = 0.2
    
    # Label smoothing
    label_smoothing: float = 0.0
    
    # Ignore index for padding tokens
    ignore_index: int = -100


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for cross-modal representation learning.
    
    Encourages representations from the same glycan to be similar
    across modalities while pushing different glycans apart.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.temperature = temperature
        
    def forward(self, 
                text_embeddings: torch.Tensor,
                structure_embeddings: torch.Tensor,
                spectra_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss between modalities.
        
        Args:
            text_embeddings: Text representations [batch_size, d_model]
            structure_embeddings: Structure representations [batch_size, d_model]
            spectra_embeddings: Spectra representations [batch_size, d_model]
            
        Returns:
            Contrastive loss value
        """
        batch_size = text_embeddings.size(0)
        device = text_embeddings.device
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        structure_embeddings = F.normalize(structure_embeddings, p=2, dim=1)
        
        # Compute similarity matrices
        text_structure_sim = torch.matmul(text_embeddings, structure_embeddings.T) / self.temperature
        
        # Create positive labels (diagonal elements)
        labels = torch.arange(batch_size, device=device)
        
        # Compute loss in both directions
        loss_text_to_structure = F.cross_entropy(text_structure_sim, labels)
        loss_structure_to_text = F.cross_entropy(text_structure_sim.T, labels)
        
        total_loss = (loss_text_to_structure + loss_structure_to_text) / 2
        
        # Add spectra modality if provided
        if spectra_embeddings is not None:
            spectra_embeddings = F.normalize(spectra_embeddings, p=2, dim=1)
            
            # Text-Spectra contrastive loss
            text_spectra_sim = torch.matmul(text_embeddings, spectra_embeddings.T) / self.temperature
            loss_text_to_spectra = F.cross_entropy(text_spectra_sim, labels)
            loss_spectra_to_text = F.cross_entropy(text_spectra_sim.T, labels)
            
            # Structure-Spectra contrastive loss
            structure_spectra_sim = torch.matmul(structure_embeddings, spectra_embeddings.T) / self.temperature
            loss_structure_to_spectra = F.cross_entropy(structure_spectra_sim, labels)
            loss_spectra_to_structure = F.cross_entropy(structure_spectra_sim.T, labels)
            
            # Average all contrastive losses
            spectra_loss = (loss_text_to_spectra + loss_spectra_to_text + 
                          loss_structure_to_spectra + loss_spectra_to_structure) / 4
            
            total_loss = (total_loss + spectra_loss) / 2
            
        return total_loss


class StructurePredictionLoss(nn.Module):
    """
    Loss function for WURCS structure prediction.
    
    Uses cross-entropy with label smoothing and special handling
    for structural tokens.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 label_smoothing: float = 0.1,
                 ignore_index: int = -100,
                 structure_token_weight: float = 2.0):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.structure_token_weight = structure_token_weight
        
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor,
                structure_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute structure prediction loss.
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            structure_token_mask: Mask for important structural tokens
            
        Returns:
            Structure prediction loss
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for cross-entropy computation
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        
        # Create mask for valid targets
        valid_mask = (targets != self.ignore_index)
        
        if self.label_smoothing > 0:
            # Label smoothing
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Create smoothed targets
            smoothed_targets = torch.full_like(log_probs, self.label_smoothing / (vocab_size - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Compute KL divergence loss
            loss = F.kl_div(log_probs, smoothed_targets, reduction='none').sum(dim=1)
            
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
            
        # Apply valid mask
        loss = loss * valid_mask.float()
        
        # Weight important structural tokens more heavily
        if structure_token_mask is not None:
            structure_mask = structure_token_mask.view(-1) * valid_mask
            loss = loss * (1.0 + (self.structure_token_weight - 1.0) * structure_mask.float())
            
        # Return mean loss
        return loss.sum() / valid_mask.float().sum().clamp(min=1)


class SpectraPredictionLoss(nn.Module):
    """
    Loss function for mass spectra prediction.
    
    Combines cross-entropy for peak tokens with MSE for intensity values.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 intensity_weight: float = 0.5,
                 peak_weight: float = 1.0):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.vocab_size = vocab_size
        self.intensity_weight = intensity_weight
        self.peak_weight = peak_weight
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                intensity_targets: Optional[torch.Tensor] = None,
                peak_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute spectra prediction loss.
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]  
            targets: Target token IDs [batch_size, seq_len]
            intensity_targets: Target intensity values [batch_size, seq_len]
            peak_mask: Mask for peak positions
            
        Returns:
            Spectra prediction loss
        """
        # Cross-entropy loss for token prediction
        token_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=-100
        )
        
        total_loss = self.peak_weight * token_loss
        
        # MSE loss for intensity prediction if provided
        if intensity_targets is not None:
            # Extract intensity predictions (assume last dimension represents intensity)
            intensity_logits = logits[:, :, -100:]  # Last 100 tokens for intensities
            intensity_probs = F.softmax(intensity_logits, dim=-1)
            
            # Convert to continuous intensity values (0-100)
            intensity_bins = torch.arange(100, dtype=torch.float, device=logits.device)
            predicted_intensities = torch.sum(intensity_probs * intensity_bins, dim=-1)
            
            # MSE loss with peak mask
            intensity_loss = F.mse_loss(predicted_intensities, intensity_targets, reduction='none')
            
            if peak_mask is not None:
                intensity_loss = intensity_loss * peak_mask
                intensity_loss = intensity_loss.sum() / peak_mask.sum().clamp(min=1)
            else:
                intensity_loss = intensity_loss.mean()
                
            total_loss = total_loss + self.intensity_weight * intensity_loss
            
        return total_loss


class MultiModalLoss(nn.Module):
    """
    Combined loss function for multimodal GlycoLLM training.
    
    Integrates structure prediction, spectra prediction, text generation,
    and contrastive learning losses.
    """
    
    def __init__(self, config: LossConfig):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.config = config
        
        # Initialize component losses
        self.contrastive_loss = ContrastiveLoss(temperature=config.temperature)
        
        self.structure_loss = StructurePredictionLoss(
            vocab_size=50000,  # Will be updated from model config
            label_smoothing=config.label_smoothing,
            ignore_index=config.ignore_index
        )
        
        self.spectra_loss = SpectraPredictionLoss(vocab_size=50000)
        
    def forward(self,
                model_outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                embeddings: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined multimodal loss.
        
        Args:
            model_outputs: Dictionary of model predictions
            targets: Dictionary of target values  
            embeddings: Dictionary of embeddings for contrastive learning
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        total_loss = 0.0
        
        # Structure prediction loss
        if 'structure_logits' in model_outputs and 'structure_targets' in targets:
            structure_loss = self.structure_loss(
                model_outputs['structure_logits'],
                targets['structure_targets'],
                targets.get('structure_token_mask')
            )
            losses['structure_loss'] = structure_loss
            total_loss += self.config.structure_weight * structure_loss
            
        # Spectra prediction loss
        if 'spectra_logits' in model_outputs and 'spectra_targets' in targets:
            spectra_loss = self.spectra_loss(
                model_outputs['spectra_logits'],
                targets['spectra_targets'],
                targets.get('intensity_targets'),
                targets.get('peak_mask')
            )
            losses['spectra_loss'] = spectra_loss
            total_loss += self.config.spectra_weight * spectra_loss
            
        # Text generation loss
        if 'text_logits' in model_outputs and 'text_targets' in targets:
            text_loss = F.cross_entropy(
                model_outputs['text_logits'].view(-1, model_outputs['text_logits'].size(-1)),
                targets['text_targets'].view(-1),
                ignore_index=self.config.ignore_index,
                label_smoothing=self.config.label_smoothing
            )
            losses['text_loss'] = text_loss
            total_loss += self.config.text_weight * text_loss
            
        # Contrastive loss for cross-modal alignment
        if embeddings is not None and self.config.contrastive_weight > 0:
            if 'text_embeddings' in embeddings and 'structure_embeddings' in embeddings:
                contrastive_loss = self.contrastive_loss(
                    embeddings['text_embeddings'],
                    embeddings['structure_embeddings'],
                    embeddings.get('spectra_embeddings')
                )
                losses['contrastive_loss'] = contrastive_loss
                total_loss += self.config.contrastive_weight * contrastive_loss
                
        losses['total_loss'] = total_loss
        return losses


class TripletLoss(nn.Module):
    """
    Triplet loss for learning embeddings where similar structures
    are closer than dissimilar ones.
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.margin = margin
        
    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, d_model]
            positive: Positive embeddings [batch_size, d_model]
            negative: Negative embeddings [batch_size, d_model]
            
        Returns:
            Triplet loss value
        """
        # Compute distances
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss with margin
        loss = F.relu(pos_distance - neg_distance + self.margin)
        
        return loss.mean()


class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity loss for alignment tasks.
    """
    
    def __init__(self, target_similarity: float = 1.0):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.target_similarity = target_similarity
        
    def forward(self, 
                embeddings1: torch.Tensor,
                embeddings2: torch.Tensor,
                similarity_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute cosine similarity loss.
        
        Args:
            embeddings1: First set of embeddings [batch_size, d_model]
            embeddings2: Second set of embeddings [batch_size, d_model]
            similarity_targets: Target similarity values [batch_size]
            
        Returns:
            Cosine similarity loss
        """
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(embeddings1, embeddings2, dim=1)
        
        # Use provided targets or default target
        if similarity_targets is None:
            targets = torch.full_like(cos_sim, self.target_similarity)
        else:
            targets = similarity_targets
            
        # MSE loss between predicted and target similarities
        loss = F.mse_loss(cos_sim, targets)
        
        return loss


def create_loss_function(loss_config: Optional[LossConfig] = None) -> MultiModalLoss:
    """
    Create multimodal loss function with default or custom configuration.
    
    Args:
        loss_config: Loss configuration (uses defaults if None)
        
    Returns:
        Configured multimodal loss function
    """
    if loss_config is None:
        loss_config = LossConfig()
        
    return MultiModalLoss(loss_config)


# Loss computation utilities
def compute_accuracy(logits: torch.Tensor, 
                    targets: torch.Tensor,
                    ignore_index: int = -100) -> float:
    """
    Compute prediction accuracy.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target token IDs [batch_size, seq_len]  
        ignore_index: Index to ignore in accuracy computation
        
    Returns:
        Accuracy percentage
    """
    if not HAS_TORCH:
        return 0.0
        
    predictions = torch.argmax(logits, dim=-1)
    
    # Create mask for valid targets
    valid_mask = (targets != ignore_index)
    
    # Compute correct predictions
    correct = (predictions == targets) * valid_mask
    
    # Calculate accuracy
    accuracy = correct.sum().float() / valid_mask.sum().float()
    
    return accuracy.item() * 100.0


def compute_perplexity(logits: torch.Tensor,
                      targets: torch.Tensor, 
                      ignore_index: int = -100) -> float:
    """
    Compute perplexity from logits and targets.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target token IDs [batch_size, seq_len]
        ignore_index: Index to ignore in computation
        
    Returns:
        Perplexity value
    """
    if not HAS_TORCH:
        return 0.0
        
    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    # Convert to perplexity
    perplexity = torch.exp(loss)
    
    return perplexity.item()