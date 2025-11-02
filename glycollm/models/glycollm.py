"""
GlycoLLM: Multimodal Large Language Model for Glycoinformatics

This module implements the core architecture for GlycoLLM, a specialized
multimodal transformer designed for glycan structure prediction, mass spectra
analysis, and scientific text understanding.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Optional imports - will be available when packages are installed
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create mock classes for development
    class nn:
        class Module:
            pass
        class ModuleList:
            pass
        class Linear:
            pass
        class Embedding:
            pass
        class LayerNorm:
            pass
        class Dropout:
            pass
        class MultiheadAttention:
            pass
        class TransformerEncoderLayer:
            pass
        class TransformerEncoder:
            pass

logger = logging.getLogger(__name__)


@dataclass
class GlycoLLMConfig:
    """Configuration for GlycoLLM model architecture"""
    
    # Model dimensions
    d_model: int = 768                    # Model dimension
    d_ff: int = 3072                      # Feed-forward dimension
    n_heads: int = 12                     # Number of attention heads
    n_layers: int = 12                    # Number of transformer layers
    
    # Vocabulary and sequence settings
    vocab_size: int = 50000               # Total vocabulary size
    max_seq_length: int = 2048            # Maximum sequence length
    
    # Modality-specific settings
    text_max_length: int = 512            # Max text sequence length
    structure_max_length: int = 256       # Max WURCS sequence length
    spectra_max_length: int = 1024        # Max spectrum sequence length
    
    # Embedding dimensions (can be different per modality)
    text_d_model: int = 768
    structure_d_model: int = 512
    spectra_d_model: int = 512
    
    # Cross-modal fusion
    fusion_layers: int = 4                # Number of cross-modal fusion layers
    fusion_heads: int = 8                 # Attention heads for cross-modal fusion
    
    # Regularization
    dropout: float = 0.1                  # Dropout probability
    attention_dropout: float = 0.1        # Attention dropout
    
    # Activation functions
    activation: str = "gelu"              # Activation function
    
    # Positional encoding
    max_position_embeddings: int = 2048   # Maximum position embeddings
    
    # Task-specific heads
    enable_structure_prediction: bool = True    # Structure prediction head
    enable_spectra_prediction: bool = True      # Spectra prediction head
    enable_text_generation: bool = True         # Text generation head
    enable_cross_modal_retrieval: bool = True  # Cross-modal retrieval
    
    # Model initialization
    initializer_range: float = 0.02       # Parameter initialization range
    
    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing
    use_cache: bool = True               # Use key-value caching


class MultiModalEmbedding(nn.Module):
    """
    Multimodal embedding layer that handles different input modalities
    and projects them to a common embedding space.
    """
    
    def __init__(self, config: GlycoLLMConfig):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.config = config
        
        # Modality-specific embeddings
        self.text_embedding = nn.Embedding(config.vocab_size, config.text_d_model)
        self.structure_embedding = nn.Embedding(config.vocab_size, config.structure_d_model)
        self.spectra_embedding = nn.Embedding(config.vocab_size, config.spectra_d_model)
        
        # Modality projection layers to common dimension
        self.text_projection = nn.Linear(config.text_d_model, config.d_model)
        self.structure_projection = nn.Linear(config.structure_d_model, config.d_model)
        self.spectra_projection = nn.Linear(config.spectra_d_model, config.d_model)
        
        # Positional encodings for each modality
        self.text_pos_encoding = PositionalEncoding(config.d_model, config.text_max_length)
        self.structure_pos_encoding = PositionalEncoding(config.d_model, config.structure_max_length)
        self.spectra_pos_encoding = PositionalEncoding(config.d_model, config.spectra_max_length)
        
        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(3, config.d_model)  # 3 modalities
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Modality indices
        self.TEXT_MODALITY = 0
        self.STRUCTURE_MODALITY = 1
        self.SPECTRA_MODALITY = 2
        
    def forward(self, 
                text_input_ids: Optional[torch.Tensor] = None,
                structure_input_ids: Optional[torch.Tensor] = None,
                spectra_input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multimodal embedding.
        
        Args:
            text_input_ids: Text token IDs [batch_size, text_seq_len]
            structure_input_ids: Structure token IDs [batch_size, struct_seq_len]
            spectra_input_ids: Spectra token IDs [batch_size, spectra_seq_len]
            attention_mask: Attention mask for padded tokens
            
        Returns:
            Tuple of (embeddings, combined_attention_mask)
        """
        embeddings = []
        masks = []
        
        batch_size = None
        
        # Process text modality
        if text_input_ids is not None:
            batch_size = text_input_ids.size(0)
            seq_len = text_input_ids.size(1)
            
            # Token embeddings
            text_emb = self.text_embedding(text_input_ids)
            text_emb = self.text_projection(text_emb)
            
            # Add positional encoding
            text_emb = self.text_pos_encoding(text_emb)
            
            # Add modality type embedding
            modality_emb = self.modality_embeddings(
                torch.full((batch_size, seq_len), self.TEXT_MODALITY, 
                          device=text_input_ids.device)
            )
            text_emb = text_emb + modality_emb
            
            embeddings.append(text_emb)
            
            # Create attention mask
            if attention_mask is not None:
                text_mask = attention_mask[:, :seq_len]
            else:
                text_mask = torch.ones(batch_size, seq_len, device=text_input_ids.device)
            masks.append(text_mask)
        
        # Process structure modality  
        if structure_input_ids is not None:
            if batch_size is None:
                batch_size = structure_input_ids.size(0)
            seq_len = structure_input_ids.size(1)
            
            # Token embeddings
            struct_emb = self.structure_embedding(structure_input_ids)
            struct_emb = self.structure_projection(struct_emb)
            
            # Add positional encoding
            struct_emb = self.structure_pos_encoding(struct_emb)
            
            # Add modality type embedding
            modality_emb = self.modality_embeddings(
                torch.full((batch_size, seq_len), self.STRUCTURE_MODALITY,
                          device=structure_input_ids.device)
            )
            struct_emb = struct_emb + modality_emb
            
            embeddings.append(struct_emb)
            
            # Create attention mask
            struct_mask = torch.ones(batch_size, seq_len, device=structure_input_ids.device)
            masks.append(struct_mask)
            
        # Process spectra modality
        if spectra_input_ids is not None:
            if batch_size is None:
                batch_size = spectra_input_ids.size(0)
            seq_len = spectra_input_ids.size(1)
            
            # Token embeddings
            spectra_emb = self.spectra_embedding(spectra_input_ids)
            spectra_emb = self.spectra_projection(spectra_emb)
            
            # Add positional encoding
            spectra_emb = self.spectra_pos_encoding(spectra_emb)
            
            # Add modality type embedding  
            modality_emb = self.modality_embeddings(
                torch.full((batch_size, seq_len), self.SPECTRA_MODALITY,
                          device=spectra_input_ids.device)
            )
            spectra_emb = spectra_emb + modality_emb
            
            embeddings.append(spectra_emb)
            
            # Create attention mask
            spectra_mask = torch.ones(batch_size, seq_len, device=spectra_input_ids.device)
            masks.append(spectra_mask)
            
        # Concatenate all modalities
        if embeddings:
            combined_embeddings = torch.cat(embeddings, dim=1)  # [batch_size, total_seq_len, d_model]
            combined_mask = torch.cat(masks, dim=1)  # [batch_size, total_seq_len]
            
            # Apply layer norm and dropout
            combined_embeddings = self.layer_norm(combined_embeddings)
            combined_embeddings = self.dropout(combined_embeddings)
            
            return combined_embeddings, combined_mask
        else:
            raise ValueError("At least one modality must be provided")


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions.
    """
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        if not HAS_TORCH:
            return
        
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusion between different modalities.
    """
    
    def __init__(self, config: GlycoLLMConfig):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.fusion_heads
        self.head_dim = config.d_model // config.fusion_heads
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Multi-head attention components
        self.query_projection = nn.Linear(config.d_model, config.d_model)
        self.key_projection = nn.Linear(config.d_model, config.d_model)
        self.value_projection = nn.Linear(config.d_model, config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Cross-modal attention forward pass.
        
        Args:
            query: Query tensor [batch_size, query_len, d_model]
            key: Key tensor [batch_size, key_len, d_model]
            value: Value tensor [batch_size, value_len, d_model]
            attention_mask: Mask for padded positions
            
        Returns:
            Attended output [batch_size, query_len, d_model]
        """
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)
        
        # Store residual connection
        residual = query
        
        # Layer normalization
        query = self.layer_norm1(query)
        key = self.layer_norm1(key)
        value = self.layer_norm1(value)
        
        # Project to query, key, value
        Q = self.query_projection(query)  # [batch_size, query_len, d_model]
        K = self.key_projection(key)      # [batch_size, key_len, d_model]
        V = self.value_projection(value)  # [batch_size, value_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, key_len]
            mask = mask.expand(-1, self.n_heads, query_len, -1)
            scores.masked_fill_(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch_size, n_heads, query_len, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        
        # Output projection
        output = self.output_projection(attended)
        
        # First residual connection
        output = output + residual
        
        # Feed-forward network with second residual connection
        ffn_output = self.ffn(self.layer_norm2(output))
        output = output + ffn_output
        
        return output


class GlycoTransformerLayer(nn.Module):
    """
    Custom transformer layer for glycoinformatics with cross-modal capabilities.
    """
    
    def __init__(self, config: GlycoLLMConfig):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.config = config
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(config)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.layer_norm3 = nn.LayerNorm(config.d_model)
        
    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                cross_modal_context: Optional[torch.Tensor] = None,
                cross_modal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through glyco transformer layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Self-attention mask
            cross_modal_context: Context from other modalities
            cross_modal_mask: Mask for cross-modal attention
            
        Returns:
            Transformed tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        residual = x
        x = self.layer_norm1(x)
        
        attn_output, _ = self.self_attention(
            query=x, key=x, value=x, 
            key_padding_mask=attention_mask == 0 if attention_mask is not None else None
        )
        x = residual + attn_output
        
        # Cross-modal attention if context provided
        if cross_modal_context is not None:
            residual = x
            cross_output = self.cross_modal_attention(
                query=x,
                key=cross_modal_context,
                value=cross_modal_context,
                attention_mask=cross_modal_mask
            )
            x = residual + cross_output
            
        # Feed-forward network with residual connection
        residual = x
        x = self.layer_norm3(x)
        ffn_output = self.ffn(x)
        x = residual + ffn_output
        
        return x


class GlycoLLMEncoder(nn.Module):
    """
    Main encoder for GlycoLLM with multimodal capabilities.
    """
    
    def __init__(self, config: GlycoLLMConfig):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.config = config
        
        # Embedding layer
        self.embeddings = MultiModalEmbedding(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GlycoTransformerLayer(config) for _ in range(config.n_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
    def forward(self,
                text_input_ids: Optional[torch.Tensor] = None,
                structure_input_ids: Optional[torch.Tensor] = None, 
                spectra_input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                output_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GlycoLLM encoder.
        
        Args:
            text_input_ids: Text token IDs
            structure_input_ids: Structure token IDs
            spectra_input_ids: Spectra token IDs
            attention_mask: Attention mask
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Dictionary with encoded representations
        """
        # Get embeddings
        hidden_states, combined_mask = self.embeddings(
            text_input_ids=text_input_ids,
            structure_input_ids=structure_input_ids,
            spectra_input_ids=spectra_input_ids,
            attention_mask=attention_mask
        )
        
        # Store all hidden states if requested
        all_hidden_states = [] if output_hidden_states else None
        
        # Pass through transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
                
            hidden_states = layer(
                x=hidden_states,
                attention_mask=combined_mask
            )
        
        # Final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attention_mask': combined_mask
        }


class GlycoLLMTaskHeads(nn.Module):
    """
    Task-specific prediction heads for GlycoLLM.
    """
    
    def __init__(self, config: GlycoLLMConfig):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.config = config
        
        # Structure prediction head (WURCS generation)
        if config.enable_structure_prediction:
            self.structure_prediction_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.vocab_size)
            )
            
        # Spectra prediction head (peak prediction)
        if config.enable_spectra_prediction:
            self.spectra_prediction_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.vocab_size)
            )
            
        # Text generation head
        if config.enable_text_generation:
            self.text_generation_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.vocab_size)
            )
            
        # Cross-modal retrieval head
        if config.enable_cross_modal_retrieval:
            self.retrieval_projection = nn.Linear(config.d_model, config.d_model)
            
    def forward(self,
                hidden_states: torch.Tensor,
                task: str = "all") -> Dict[str, torch.Tensor]:
        """
        Forward pass through task heads.
        
        Args:
            hidden_states: Encoded representations [batch_size, seq_len, d_model]
            task: Specific task to compute ("structure", "spectra", "text", "retrieval", "all")
            
        Returns:
            Dictionary with task-specific predictions
        """
        outputs = {}
        
        if task in ["structure", "all"] and hasattr(self, 'structure_prediction_head'):
            outputs['structure_logits'] = self.structure_prediction_head(hidden_states)
            
        if task in ["spectra", "all"] and hasattr(self, 'spectra_prediction_head'):
            outputs['spectra_logits'] = self.spectra_prediction_head(hidden_states)
            
        if task in ["text", "all"] and hasattr(self, 'text_generation_head'):
            outputs['text_logits'] = self.text_generation_head(hidden_states)
            
        if task in ["retrieval", "all"] and hasattr(self, 'retrieval_projection'):
            # Pool sequence to single vector for retrieval
            pooled = torch.mean(hidden_states, dim=1)  # Simple mean pooling
            outputs['retrieval_embeddings'] = self.retrieval_projection(pooled)
            
        return outputs


class GlycoLLM(nn.Module):
    """
    Complete GlycoLLM model for multimodal glycoinformatics.
    """
    
    def __init__(self, config: GlycoLLMConfig):
        super().__init__()
        
        if not HAS_TORCH:
            return
            
        self.config = config
        
        # Main encoder
        self.encoder = GlycoLLMEncoder(config)
        
        # Task-specific heads
        self.task_heads = GlycoLLMTaskHeads(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self,
                text_input_ids: Optional[torch.Tensor] = None,
                structure_input_ids: Optional[torch.Tensor] = None,
                spectra_input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                task: str = "all",
                output_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete GlycoLLM model.
        
        Args:
            text_input_ids: Text token IDs
            structure_input_ids: Structure token IDs
            spectra_input_ids: Spectra token IDs
            attention_mask: Attention mask
            labels: Target labels for training
            task: Specific task to compute
            output_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary with model outputs
        """
        # Encode inputs
        encoder_outputs = self.encoder(
            text_input_ids=text_input_ids,
            structure_input_ids=structure_input_ids,
            spectra_input_ids=spectra_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )
        
        # Get task-specific predictions
        task_outputs = self.task_heads(
            hidden_states=encoder_outputs['last_hidden_state'],
            task=task
        )
        
        # Combine outputs
        outputs = {
            'last_hidden_state': encoder_outputs['last_hidden_state'],
            'attention_mask': encoder_outputs['attention_mask'],
            **task_outputs
        }
        
        if output_hidden_states:
            outputs['hidden_states'] = encoder_outputs['hidden_states']
            
        # Compute loss if labels provided
        if labels is not None:
            outputs['loss'] = self._compute_loss(task_outputs, labels, task)
            
        return outputs
        
    def _compute_loss(self, 
                     predictions: Dict[str, torch.Tensor],
                     labels: torch.Tensor,
                     task: str) -> torch.Tensor:
        """Compute task-specific loss."""
        losses = []
        
        if task in ["structure", "all"] and 'structure_logits' in predictions:
            structure_loss = F.cross_entropy(
                predictions['structure_logits'].view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            losses.append(structure_loss)
            
        if task in ["spectra", "all"] and 'spectra_logits' in predictions:
            spectra_loss = F.cross_entropy(
                predictions['spectra_logits'].view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            losses.append(spectra_loss)
            
        if task in ["text", "all"] and 'text_logits' in predictions:
            text_loss = F.cross_entropy(
                predictions['text_logits'].view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            losses.append(text_loss)
            
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)
        
    def generate(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                max_length: int = 100,
                num_beams: int = 1,
                temperature: float = 1.0,
                top_p: float = 1.0) -> torch.Tensor:
        """
        Generate sequences using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            Generated token IDs
        """
        # Simple greedy generation for now
        self.eval()
        
        with torch.no_grad():
            batch_size = input_ids.size(0)
            current_length = input_ids.size(1)
            
            # Generate tokens one by one
            for _ in range(max_length - current_length):
                # Get model outputs
                outputs = self.forward(
                    text_input_ids=input_ids,
                    attention_mask=attention_mask,
                    task="text"
                )
                
                # Get next token logits
                next_token_logits = outputs['text_logits'][:, -1, :] / temperature
                
                # Apply top-p sampling if specified
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if temperature == 0.0:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                else:
                    next_tokens = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                    next_tokens = next_tokens.squeeze(1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
                
                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask, 
                        torch.ones(batch_size, 1, device=attention_mask.device)
                    ], dim=1)
                
        return input_ids
        
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'GlycoLLM':
        """Load model from saved checkpoint."""
        # Implementation for loading pretrained model
        # This would load config and weights from saved files
        pass
        
    def save_pretrained(self, save_path: str):
        """Save model to directory."""
        # Implementation for saving model
        # This would save config and weights to files
        pass


def create_glycollm_model(vocab_size: int = 50000,
                         d_model: int = 768,
                         n_layers: int = 12,
                         n_heads: int = 12) -> GlycoLLM:
    """
    Convenience function to create GlycoLLM model with common settings.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of layers
        n_heads: Number of attention heads
        
    Returns:
        Configured GlycoLLM model
    """
    config = GlycoLLMConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads
    )
    
    return GlycoLLM(config)