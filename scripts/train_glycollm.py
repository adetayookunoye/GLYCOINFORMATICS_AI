"""
Training Script for GlycoLLM

Fine-tune GlycoLLM on spec→struct task using existing trainer infrastructure.

Supports:
- Multimodal training (spectra + structure + text)
- Distributed training
- Mixed precision
- Contrastive learning
- Curriculum learning
- LLM fine-tuning with TinyLlama (default) or other models

Author: Adetayo Research Team
Date: November 2025
"""

import logging
import argparse
from pathlib import Path
import json
import torch
from typing import Optional

from glycollm.models.glycollm import GlycoLLM, GlycoLLMConfig
from glycollm.training.trainer import GlycoLLMTrainer, TrainingConfig
from glycollm.training.evaluation import StructureEvaluator, SpectraEvaluator
from glycollm.data.multimodal_dataset import MultimodalGlycanDataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GlycoLLM")
    
    # Data
    parser.add_argument("--train-data", type=str, required=True, help="Training data path")
    parser.add_argument("--val-data", type=str, required=True, help="Validation data path")
    parser.add_argument("--test-data", type=str, help="Test data path")
    
    # Model
    parser.add_argument("--model-config", type=str, help="Model config JSON")
    parser.add_argument("--pretrained-model", type=str, help="Path to pretrained model")
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    
    # Training
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    
    # Optimization
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    
    # Advanced
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--curriculum-learning", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    
    # Task
    parser.add_argument("--task", type=str, default="spec_to_struct",
                       choices=["spec_to_struct", "struct_to_text", "multimodal"])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("GlycoLLM Training")
    logger.info("=" * 80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading datasets...")
    train_dataset = MultimodalGlycanDataset(
        data_path=args.train_data,
        max_seq_length=512
    )
    val_dataset = MultimodalGlycanDataset(
        data_path=args.val_data,
        max_seq_length=512
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    if args.test_data:
        test_dataset = MultimodalGlycanDataset(
            data_path=args.test_data,
            max_seq_length=512
        )
        logger.info(f"Test samples: {len(test_dataset)}")
    else:
        test_dataset = None
    
    # Initialize or load model
    if args.pretrained_model:
        logger.info(f"Loading pretrained model from {args.pretrained_model}")
        model = GlycoLLM.from_pretrained(args.pretrained_model)
    elif args.model_config:
        logger.info(f"Loading model config from {args.model_config}")
        with open(args.model_config) as f:
            config_dict = json.load(f)
        config = GlycoLLMConfig(**config_dict)
        model = GlycoLLM(config)
    else:
        logger.info("Creating default model")
        config = GlycoLLMConfig(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            vocab_size=1000,  # Should match tokenizer
            max_seq_length=512
        )
        model = GlycoLLM(config)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training configuration
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
        distributed=args.distributed,
        contrastive_loss_weight=args.contrastive_loss_weight,
        use_curriculum_learning=args.curriculum_learning,
        early_stopping_patience=args.early_stopping_patience,
        save_steps=1000,
        eval_steps=500,
        logging_steps=100
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = GlycoLLMTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on test set if provided
    if test_dataset:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {json.dumps(test_results, indent=2)}")
        
        # Save test results
        test_results_path = Path(args.output_dir) / "test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
    
    # Save final model
    final_model_path = Path(args.output_dir) / "final_model"
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
