"""
LLM Fine-tuning module for GlycoLLM.

This module integrates pre-trained language models (GPT, LLaMA, T5, etc.) 
with glycomics and glycoproteomics domain fine-tuning capabilities.
Supports LoRA, QLoRA, and full fine-tuning approaches.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

# Optional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        AutoModelForSeq2SeqLM, TrainingArguments, Trainer,
        BitsAndBytesConfig, PreTrainedModel
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    # Mock classes for development
    class PreTrainedModel:
        pass
    class Trainer:
        pass
    class TrainingArguments:
        pass

logger = logging.getLogger(__name__)


class LLMType(Enum):
    """Supported pre-trained LLM types."""
    GPT2 = "gpt2"
    GPT_NEO = "EleutherAI/gpt-neo-1.3B"
    LLAMA2_7B = "meta-llama/Llama-2-7b-hf"
    LLAMA2_13B = "meta-llama/Llama-2-13b-hf"
    T5_BASE = "t5-base"
    T5_LARGE = "t5-large"
    FLAN_T5_BASE = "google/flan-t5-base"
    FLAN_T5_LARGE = "google/flan-t5-large"
    MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
    BIOMISTRAL = "BioMistral/BioMistral-7B"
    SCIGPT = "AI4Science/SciGPT"
    TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class FineTuningMethod(Enum):
    """Fine-tuning methods."""
    FULL = "full"  # Full parameter fine-tuning
    LORA = "lora"  # LoRA (Low-Rank Adaptation)
    QLORA = "qlora"  # QLoRA (Quantized LoRA)
    ADAPTER = "adapter"  # Adapter layers
    PREFIX_TUNING = "prefix_tuning"  # Prefix tuning


@dataclass
class LLMFineTuningConfig:
    """Configuration for LLM fine-tuning."""
    
    # Model selection
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_type: LLMType = LLMType.TINYLLAMA
    
    # Fine-tuning method
    method: FineTuningMethod = FineTuningMethod.LORA
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    
    # Quantization (for QLoRA)
    use_quantization: bool = False
    quantization_bits: int = 4
    
    # Training parameters
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_steps: int = 1000
    logging_steps: int = 10
    save_steps: int = 100
    evaluation_steps: int = 100
    
    # Data parameters
    max_sequence_length: int = 512
    truncation: bool = True
    padding: bool = True
    
    # Task-specific configurations
    task_type: str = "general"  # spec2struct, structure2spec, explain, retrieval
    enable_uncertainty: bool = True
    enable_grounding: bool = True
    
    # Specialized model components
    use_task_specific_heads: bool = True
    cross_modal_attention: bool = True
    uncertainty_estimation: bool = True
    
    # Output directory
    output_dir: str = "./llm_checkpoints"
    logging_dir: str = "./llm_logs"
    
    # Glycomics-specific parameters
    include_structure_tokens: bool = True
    include_spectra_tokens: bool = True
    structure_embedding_dim: int = 256
    spectra_embedding_dim: int = 128


class GlycanDatasetForLLM:
    """
    Dataset class for LLM fine-tuning on glycomics data.
    
    Formats glycan structures, spectra, and text for language model training.
    """
    
    def __init__(self, 
                 data: List[Dict[str, Any]],
                 tokenizer: Any,
                 config: LLMFineTuningConfig):
        
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        
        # Glycomics-specific token mapping
        self.special_tokens = {
            '<WURCS>': '[WURCS]',
            '</WURCS>': '[/WURCS]',
            '<SPECTRUM>': '[SPECTRUM]',
            '</SPECTRUM>': '[/SPECTRUM]',
            '<GLYCAN>': '[GLYCAN]', 
            '</GLYCAN>': '[/GLYCAN]',
            '<FRAGMENT>': '[FRAGMENT]',
            '</FRAGMENT>': '[/FRAGMENT]'
        }
        
        # Add special tokens to tokenizer
        if hasattr(tokenizer, 'add_special_tokens'):
            new_tokens = list(self.special_tokens.values())
            tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get formatted sample for LLM training."""
        
        sample = self.data[idx]
        
        # Format input text with glycomics structure
        input_text = self._format_input_text(sample)
        
        # Format target text
        target_text = self._format_target_text(sample)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.max_sequence_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.config.max_sequence_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'original_sample': sample
        }
        
    def _format_input_text(self, sample: Dict[str, Any]) -> str:
        """Format input text with glycan structure information."""
        
        text_parts = []
        
        # Add task instruction
        if 'task' in sample:
            task = sample['task']
            if task == 'structure_prediction':
                text_parts.append("Predict the glycan structure from the given mass spectrum:")
            elif task == 'fragmentation':
                text_parts.append("Analyze the fragmentation pattern of this glycan structure:")
            elif task == 'identification':
                text_parts.append("Identify and describe this glycan structure:")
            else:
                text_parts.append("Analyze the following glycomics data:")
        
        # Add structure information
        if 'structure' in sample and self.config.include_structure_tokens:
            structure = sample['structure']
            text_parts.append(f"{self.special_tokens['<WURCS>']}{structure}{self.special_tokens['</WURCS>']}")
            
        # Add spectra information
        if 'spectra' in sample and self.config.include_spectra_tokens:
            spectra = sample['spectra']
            if isinstance(spectra, list):
                # Format peak list
                peaks_str = "; ".join([f"{mz:.2f}:{intensity:.1f}" for mz, intensity in spectra[:20]])  # Limit peaks
                text_parts.append(f"{self.special_tokens['<SPECTRUM>']}{peaks_str}{self.special_tokens['</SPECTRUM>']}")
                
        # Add context text
        if 'text' in sample:
            text_parts.append(sample['text'])
            
        # Add question/prompt
        if 'question' in sample:
            text_parts.append(f"Question: {sample['question']}")
            
        return "\n\n".join(text_parts)
        
    def _format_target_text(self, sample: Dict[str, Any]) -> str:
        """Format target text for training."""
        
        target_parts = []
        
        # Add structured answer
        if 'answer' in sample:
            target_parts.append(sample['answer'])
        elif 'target_structure' in sample:
            structure = sample['target_structure']
            target_parts.append(f"The glycan structure is: {self.special_tokens['<WURCS>']}{structure}{self.special_tokens['</WURCS>']}")
        elif 'explanation' in sample:
            target_parts.append(sample['explanation'])
            
        # Add reasoning steps if available
        if 'reasoning_steps' in sample:
            steps = sample['reasoning_steps']
            if isinstance(steps, list):
                target_parts.append("Analysis steps:")
                for i, step in enumerate(steps, 1):
                    target_parts.append(f"{i}. {step}")
                    
        # Add fragments if available
        if 'fragments' in sample:
            fragments = sample['fragments']
            if isinstance(fragments, list):
                target_parts.append("Key fragments:")
                for fragment in fragments:
                    if isinstance(fragment, dict):
                        mz = fragment.get('mz', 'unknown')
                        structure = fragment.get('structure', 'unknown')
                        target_parts.append(f"- m/z {mz}: {self.special_tokens['<FRAGMENT>']}{structure}{self.special_tokens['</FRAGMENT>']}")
                        
        return "\n\n".join(target_parts) if target_parts else "No specific target provided."


if HAS_TRANSFORMERS:
    class GlycoLLMWithFineTuning(nn.Module):
        """
        GlycoLLM with integrated LLM fine-tuning capabilities.
        
        Combines our custom multimodal architecture with fine-tuned 
        pre-trained language models for enhanced reasoning.
        """
        
        def __init__(self, 
                     custom_glycollm: nn.Module,
                     llm_config: LLMFineTuningConfig):
            super().__init__()
            
            self.custom_glycollm = custom_glycollm
            self.llm_config = llm_config
            
            # Load pre-trained LLM
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_config.model_name)
            
            # Add padding token if not present
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
            # Load model based on type
            if "t5" in llm_config.model_name.lower():
                self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                    llm_config.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_config.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
            # Apply fine-tuning method
            self._setup_fine_tuning()
            
            # Cross-modal projection layers
            self.structure_to_llm = nn.Linear(
                custom_glycollm.config.hidden_size, 
                self.llm_model.config.hidden_size
            )
            self.spectra_to_llm = nn.Linear(
                custom_glycollm.config.hidden_size,
                self.llm_model.config.hidden_size  
            )
            
        def _setup_fine_tuning(self):
            """Set up fine-tuning method (LoRA, QLoRA, etc.)."""
            
            if self.llm_config.method == FineTuningMethod.LORA:
                # LoRA configuration
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM if "t5" not in self.llm_config.model_name.lower() else TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    r=self.llm_config.lora_rank,
                    lora_alpha=self.llm_config.lora_alpha,
                    lora_dropout=self.llm_config.lora_dropout,
                    target_modules=self.llm_config.lora_target_modules
                )
                
                self.llm_model = get_peft_model(self.llm_model, lora_config)
                logger.info(f"Applied LoRA fine-tuning with rank {self.llm_config.lora_rank}")
                
            elif self.llm_config.method == FineTuningMethod.QLORA:
                # QLoRA with quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                # Reload model with quantization
                if "t5" in self.llm_config.model_name.lower():
                    self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.llm_config.model_name,
                        quantization_config=bnb_config
                    )
                else:
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        self.llm_config.model_name,
                        quantization_config=bnb_config
                    )
                    
                # Apply LoRA on top of quantization
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM if "t5" not in self.llm_config.model_name.lower() else TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    r=self.llm_config.lora_rank,
                    lora_alpha=self.llm_config.lora_alpha,
                    lora_dropout=self.llm_config.lora_dropout
                )
                
                self.llm_model = get_peft_model(self.llm_model, lora_config)
                logger.info("Applied QLoRA fine-tuning with 4-bit quantization")
                
            elif self.llm_config.method == FineTuningMethod.FULL:
                logger.info("Using full fine-tuning (all parameters)")
                
        def forward(self, 
                   structure_input: Optional[torch.Tensor] = None,
                   spectra_input: Optional[torch.Tensor] = None,
                   text_input: Optional[torch.Tensor] = None,
                   llm_input_ids: Optional[torch.Tensor] = None,
                   llm_attention_mask: Optional[torch.Tensor] = None,
                   labels: Optional[torch.Tensor] = None,
                   use_llm: bool = True,
                   **kwargs) -> Dict[str, torch.Tensor]:
            """
            Forward pass through combined architecture.
            
            Args:
                structure_input: WURCS structure tokens
                spectra_input: Spectra tokens 
                text_input: Text tokens
                llm_input_ids: LLM input token IDs
                llm_attention_mask: LLM attention mask
                labels: Target labels for training
                use_llm: Whether to use LLM component
                
            Returns:
                Dictionary of outputs and losses
            """
            
            outputs = {}
            
            # Custom GlycoLLM forward pass
            if structure_input is not None or spectra_input is not None or text_input is not None:
                custom_outputs = self.custom_glycollm(
                    structure_input=structure_input,
                    spectra_input=spectra_input, 
                    text_input=text_input,
                    **kwargs
                )
                outputs.update(custom_outputs)
                
            # LLM forward pass
            if use_llm and llm_input_ids is not None:
                
                # Prepare LLM inputs
                llm_inputs = {
                    'input_ids': llm_input_ids,
                    'attention_mask': llm_attention_mask
                }
                
                if labels is not None:
                    llm_inputs['labels'] = labels
                    
                # Add cross-modal embeddings if available
                if 'structure_embeddings' in outputs:
                    structure_proj = self.structure_to_llm(outputs['structure_embeddings'])
                    # Could inject into LLM embeddings here
                    
                if 'spectra_embeddings' in outputs:
                    spectra_proj = self.spectra_to_llm(outputs['spectra_embeddings'])
                    # Could inject into LLM embeddings here
                    
                # LLM forward pass
                llm_outputs = self.llm_model(**llm_inputs)
                
                outputs['llm_logits'] = llm_outputs.logits
                if hasattr(llm_outputs, 'loss') and llm_outputs.loss is not None:
                    outputs['llm_loss'] = llm_outputs.loss
                    
            return outputs
            
        def generate_text(self,
                         prompt: str,
                         structure: Optional[str] = None,
                         spectra: Optional[List[Tuple[float, float]]] = None,
                         max_length: int = 256,
                         temperature: float = 0.7,
                         do_sample: bool = True) -> str:
            """
            Generate text response using the fine-tuned LLM.
            
            Args:
                prompt: Input text prompt
                structure: Optional WURCS structure
                spectra: Optional mass spectra peaks
                max_length: Maximum generation length
                temperature: Sampling temperature
                do_sample: Whether to use sampling
                
            Returns:
                Generated text response
            """
            
            # Format input with glycomics data
            formatted_prompt = prompt
            
            if structure:
                formatted_prompt += f"\n\n[WURCS]{structure}[/WURCS]"
                
            if spectra:
                peaks_str = "; ".join([f"{mz:.2f}:{intensity:.1f}" for mz, intensity in spectra[:15]])
                formatted_prompt += f"\n\n[SPECTRUM]{peaks_str}[/SPECTRUM]"
                
            # Tokenize
            inputs = self.llm_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.llm_config.max_sequence_length
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id
                )
                
            # Decode
            generated_text = self.llm_tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Remove input prompt from output
            if generated_text.startswith(formatted_prompt):
                generated_text = generated_text[len(formatted_prompt):].strip()
                
            return generated_text


    class GlycoLLMFineTuner:
        """
        Trainer class for fine-tuning GlycoLLM with pre-trained language models.
        
        Handles data preparation, training loop, and model management.
        """
        
        def __init__(self, 
                     config: LLMFineTuningConfig,
                     custom_glycollm: Optional[nn.Module] = None):
            
            self.config = config
            self.custom_glycollm = custom_glycollm
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Initialize model
            self.model = None
            
        def prepare_model(self) -> GlycoLLMWithFineTuning:
            """Prepare the combined model for training."""
            
            if self.custom_glycollm is None:
                logger.warning("No custom GlycoLLM provided - using LLM only")
                # Could create a minimal custom model here
                
            self.model = GlycoLLMWithFineTuning(
                custom_glycollm=self.custom_glycollm,
                llm_config=self.config
            )
            
            return self.model
            
        def prepare_dataset(self, 
                           glycan_data: List[Dict[str, Any]],
                           validation_split: float = 0.1) -> Tuple[GlycanDatasetForLLM, GlycanDatasetForLLM]:
            """
            Prepare training and validation datasets.
            
            Args:
                glycan_data: List of glycan samples
                validation_split: Fraction for validation
                
            Returns:
                Tuple of (train_dataset, val_dataset)
            """
            
            # Split data
            split_idx = int(len(glycan_data) * (1 - validation_split))
            train_data = glycan_data[:split_idx]
            val_data = glycan_data[split_idx:]
            
            # Create datasets
            train_dataset = GlycanDatasetForLLM(train_data, self.tokenizer, self.config)
            val_dataset = GlycanDatasetForLLM(val_data, self.tokenizer, self.config)
            
            logger.info(f"Prepared datasets: {len(train_dataset)} train, {len(val_dataset)} validation")
            
            return train_dataset, val_dataset
            
        def train(self, 
                 train_dataset: GlycanDatasetForLLM,
                 val_dataset: GlycanDatasetForLLM) -> str:
            """
            Fine-tune the model.
            
            Args:
                train_dataset: Training dataset
                val_dataset: Validation dataset
                
            Returns:
                Path to saved model
            """
            
            if self.model is None:
                self.prepare_model()
                
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.evaluation_steps,
                logging_dir=self.config.logging_dir,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="tensorboard",
                fp16=torch.cuda.is_available(),
                dataloader_pin_memory=False,
                remove_unused_columns=False
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer
            )
            
            # Start training
            logger.info("Starting LLM fine-tuning...")
            trainer.train()
            
            # Save final model
            final_model_path = f"{self.config.output_dir}/final_model"
            trainer.save_model(final_model_path)
            
            logger.info(f"Fine-tuning completed. Model saved to: {final_model_path}")
            
            return final_model_path

else:
    # Mock classes when transformers is not available
    class GlycoLLMWithFineTuning:
        def __init__(self, **kwargs):
            logger.warning("Transformers not available - LLM fine-tuning disabled")
            
    class GlycoLLMFineTuner:
        def __init__(self, **kwargs):
            logger.warning("Transformers not available - LLM fine-tuning disabled")


def create_glycomics_training_data() -> List[Dict[str, Any]]:
    """
    Create sample training data for glycomics LLM fine-tuning.
    
    Returns:
        List of formatted training samples
    """
    
    samples = [
        {
            'task': 'structure_prediction',
            'structure': 'WURCS=2.0/3,3,2/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5]/1-2-3/a4-b1_b4-c1',
            'spectra': [(204.087, 100.0), (366.140, 80.0), (528.193, 60.0), (690.246, 40.0)],
            'question': 'What is the structure of this N-glycan based on the mass spectrum?',
            'answer': 'This is a high-mannose type N-glycan with core GlcNAc residues and three mannose residues. The fragmentation pattern shows sequential loss of mannose units.',
            'reasoning_steps': [
                'Molecular ion at m/z 690.246 suggests a trisaccharide core structure',
                'Fragment at m/z 528.193 corresponds to loss of one hexose (162 Da)',
                'Fragment at m/z 366.140 shows loss of two hexose units',
                'Base peak at m/z 204.087 represents the core GlcNAc unit'
            ]
        },
        {
            'task': 'fragmentation',
            'structure': 'WURCS=2.0/4,4,3/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5][a112h-1b_1-5]/1-2-3-4/a4-b1_b4-c1_c3-d1',
            'text': 'Complex N-glycan with core fucosylation and high mannose content',
            'question': 'Explain the fragmentation pattern of this fucosylated N-glycan.',
            'answer': 'The fragmentation shows characteristic losses: fucose (146 Da), mannose (162 Da), and GlcNAc (203 Da). Core fucosylation is evident from the specific fragmentation pathway.',
            'fragments': [
                {'mz': 836.299, 'structure': 'Complete glycan', 'type': 'molecular_ion'},
                {'mz': 690.246, 'structure': 'Loss of fucose', 'type': 'Y_fragment'},
                {'mz': 528.193, 'structure': 'Loss of fucose + mannose', 'type': 'Y_fragment'}
            ]
        },
        {
            'task': 'identification',
            'structure': 'WURCS=2.0/5,7,6/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1221m-1a_1-5][a112h-1b_1-5][a2122A-2a_2-6_5*NCCO/3=O]/1-2-3-2-4-3-5/a4-b1_b4-c1_c3-d1_d2-e1_e4-f1_f6-g1',
            'text': 'Sialylated complex N-glycan found in human serum glycoproteins',
            'question': 'Describe the biological significance of this sialylated N-glycan structure.',
            'answer': 'This is a biantennary complex N-glycan with core fucosylation and terminal sialic acid residues. It is commonly found on serum glycoproteins and plays important roles in cell recognition, immune modulation, and protein stability. The sialic acid residues contribute to negative charge and influence protein half-life in circulation.',
            'reasoning_steps': [
                'Identify core GlcNAc2Man3 pentasaccharide structure',
                'Recognize biantennary branching pattern with two antennae',
                'Note presence of core fucose linked α1-6 to reducing GlcNAc', 
                'Identify terminal Neu5Ac (sialic acid) residues on both antennae',
                'Assess biological context and functional implications'
            ]
        }
    ]
    
    return samples


def load_llm_fine_tuning_config(config_path: Optional[str] = None) -> LLMFineTuningConfig:
    """
    Load LLM fine-tuning configuration from file or create default.
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        LLM fine-tuning configuration
    """
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return LLMFineTuningConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            
    # Return default configuration optimized for glycomics
    return LLMFineTuningConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Small, fast model for glycomics
        model_type=LLMType.TINYLLAMA,
        method=FineTuningMethod.LORA,  # Memory efficient
        lora_rank=16,
        lora_alpha=32,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=1000,
        include_structure_tokens=True,
        include_spectra_tokens=True
    )