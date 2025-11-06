#!/usr/bin/env python3
"""
GlycoWorks Data Processing Only
===============================

Lightweight script for processing GlycoWorks experimental data without
heavy ML dependencies. Perfect for data preparation and validation.

This script processes CSV files and creates training samples without
importing torch, tensorflow, or other heavy ML libraries.

Author: Glycoinformatics AI Team
Date: November 5, 2025
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
import pandas as pd
import pickle
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Lightweight imports only - no torch/tensorflow
from glycollm.data.glycoworks_processor import GlycoWorksProcessor
from glycollm.data.candycrunch_validator import CandycrunchValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('glycoworks_data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LightweightDataProcessor:
    """
    Lightweight data processing without heavy ML dependencies.

    Processes GlycoWorks CSV files and candycrunch validation data
    for training preparation.
    """

    def __init__(self,
                 glycoworks_dir: str = "data/raw/dataset/glycoworks_glycan_data",
                 candycrunch_dir: str = "data/raw/dataset/candycrunch_glycan_data",
                 output_dir: str = "data/processed/glycoworks_data_only"):
        """
        Initialize the lightweight data processor.

        Args:
            glycoworks_dir: Directory containing GlycoWorks CSV files
            candycrunch_dir: Directory containing candycrunch validation data
            output_dir: Output directory for processed data
        """
        self.glycoworks_dir = Path(glycoworks_dir)
        self.candycrunch_dir = Path(candycrunch_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.processor = None
        self.validator = None

        # Data storage
        self.processed_data = {}
        self.validation_data = {}

        logger.info("🧬 Lightweight Data Processor initialized")
        logger.info(f"   📂 GlycoWorks data: {glycoworks_dir}")
        logger.info(f"   📂 Validation data: {candycrunch_dir}")
        logger.info(f"   📂 Output directory: {output_dir}")

    def initialize_processors(self):
        """Initialize data processing components."""
        logger.info("🔧 Initializing data processors...")

        # Initialize GlycoWorks processor
        self.processor = GlycoWorksProcessor(
            data_dir=str(self.glycoworks_dir),
            output_dir=str(self.output_dir / "processed"),
            tokenizer_path=None  # No tokenizer needed for data processing
        )

        # Initialize candycrunch validator
        self.validator = CandycrunchValidator(str(self.candycrunch_dir))

        logger.info("✅ Data processors initialized successfully")

    def process_glycoworks_data(self,
                              task_types: List[str] = None,
                              save_format: str = 'pickle') -> Dict[str, Any]:
        """
        Process GlycoWorks experimental data into training samples.

        Args:
            task_types: Types of tasks to create samples for
            save_format: Format to save processed data ('pickle', 'json', or 'both')

        Returns:
            Processed data summary
        """
        if not self.processor:
            raise ValueError("Processor not initialized. Call initialize_processors() first.")

        logger.info("🔬 Processing GlycoWorks experimental data...")

        # Process CSV files
        glycoworks_dataset = self.processor.process_all_csv_files()

        # Create multimodal samples
        if task_types is None:
            task_types = [
                'abundance_prediction',
                'tissue_classification',
                'biomarker_discovery',
                'quantitative_structure_elucidation'
            ]

        multimodal_samples = self.processor.create_multimodal_samples(task_types)

        # Store processed data
        self.processed_data = {
            'dataset': glycoworks_dataset,
            'samples': multimodal_samples,
            'metadata': {
                'source': 'glycoworks_experimental',
                'processing_date': datetime.now().isoformat(),
                'task_types': task_types,
                'statistics': glycoworks_dataset.statistics,
                'num_samples': len(multimodal_samples)
            }
        }

        # Save processed data
        self._save_processed_data(save_format)

        logger.info("✅ GlycoWorks data processing complete")
        logger.info(f"   📊 Created {len(multimodal_samples)} training samples")
        logger.info(f"   🎯 Task types: {task_types}")
        logger.info(f"   💾 Data saved in {save_format} format")

        return self.processed_data['metadata']

    def load_validation_knowledge_base(self) -> Dict[str, Any]:
        """
        Load and setup the candycrunch validation knowledge base.

        Returns:
            Validation knowledge base summary
        """
        if not self.validator:
            raise ValueError("Validator not initialized. Call initialize_processors() first.")

        logger.info("📚 Loading candycrunch validation knowledge base...")

        # Load knowledge base
        knowledge_base = self.validator.load_knowledge_base()

        # Create validation samples for testing
        validation_samples = self.validator.create_validation_samples(
            num_samples=500,  # Reasonable validation set size
            task_types=['structure_validation', 'glytoucan_mapping']
        )

        # Store validation data
        self.validation_data = {
            'knowledge_base': knowledge_base,
            'validation_samples': validation_samples,
            'metadata': {
                'source': 'candycrunch_validation',
                'processing_date': datetime.now().isoformat(),
                'knowledge_base_stats': knowledge_base.get_validation_statistics(),
                'num_validation_samples': len(validation_samples)
            }
        }

        # Save validation data
        self._save_validation_data()

        logger.info("✅ Validation knowledge base loaded")
        logger.info(f"   📊 {len(knowledge_base.entries)} reference structures")
        logger.info(f"   🎯 {knowledge_base.glytoucan_coverage:.1%} GlyTouCan coverage")
        logger.info(f"   📋 {len(validation_samples)} validation samples created")

        return self.validation_data['metadata']

    def _save_processed_data(self, save_format: str):
        """Save processed GlycoWorks data."""

        if save_format in ['pickle', 'both']:
            # Save as pickle for Python use
            pickle_file = self.output_dir / "glycoworks_processed.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.processed_data, f)
            logger.info(f"💾 Pickle data saved to {pickle_file}")

        if save_format in ['json', 'both']:
            # Save metadata as JSON for inspection
            json_file = self.output_dir / "glycoworks_metadata.json"
            with open(json_file, 'w') as f:
                json.dump(self.processed_data['metadata'], f, indent=2, default=str)
            logger.info(f"💾 JSON metadata saved to {json_file}")

        # Save samples as CSV for external tools
        self._save_samples_as_csv()

    def _save_samples_as_csv(self):
        """Save processed samples as CSV for external analysis."""

        if not self.processed_data.get('samples'):
            return

        samples_data = []
        for sample in self.processed_data['samples']:
            # Extract glycan information from labels or text
            glycan = 'unknown'
            abundance = 0
            tissue = sample.tissue or 'unknown'

            # Try to extract from labels
            if sample.labels and 'glycan_sequence' in sample.labels:
                glycan = sample.labels['glycan_sequence']
            elif sample.labels and 'target_abundance' in sample.labels:
                abundance = sample.labels.get('target_abundance', 0)

            # Try to extract from text (simple parsing)
            if sample.text and 'glycan' in sample.text.lower():
                # Simple extraction - look for patterns like "glycan XXX"
                import re
                glycan_match = re.search(r'glycan\s+([^\s]+)', sample.text, re.IGNORECASE)
                if glycan_match:
                    glycan = glycan_match.group(1)

            sample_dict = {
                'glycan': glycan,
                'abundance': abundance,
                'tissue': tissue,
                'task_type': sample.labels.get('task_type', 'unknown') if sample.labels else 'unknown',
                'glytoucan_id': sample.glytoucan_id or '',
                'structure_hash': hashlib.md5(glycan.encode()).hexdigest()[:8]
            }
            samples_data.append(sample_dict)

        df = pd.DataFrame(samples_data)
        csv_file = self.output_dir / "glycoworks_samples.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"💾 CSV samples saved to {csv_file} ({len(samples_data)} rows)")

    def _save_validation_data(self):
        """Save validation knowledge base data."""

        # Save metadata as JSON
        json_file = self.output_dir / "candycrunch_metadata.json"
        with open(json_file, 'w') as f:
            json.dump(self.validation_data['metadata'], f, indent=2, default=str)
        logger.info(f"💾 Validation metadata saved to {json_file}")

        # Save validation samples as CSV
        if self.validation_data.get('validation_samples'):
            validation_data = []
            for sample in self.validation_data['validation_samples']:
                sample_dict = {
                    'glycan': sample.text or '',  # Use text field instead of experimental_context
                    'glytoucan_id': sample.glytoucan_id or '',
                    'validation_type': sample.labels.get('task_type', '') if sample.labels else '',
                    'structure_hash': hashlib.md5(
                        (sample.text or '').encode()
                    ).hexdigest()[:8]
                }
                validation_data.append(sample_dict)

            df = pd.DataFrame(validation_data)
            csv_file = self.output_dir / "candycrunch_validation.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"💾 Validation CSV saved to {csv_file} ({len(validation_data)} rows)")

    def generate_processing_report(self) -> str:
        """Generate a comprehensive processing report."""

        report_file = self.output_dir / "data_processing_report.md"

        report = f"""# GlycoWorks Data Processing Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Processing Summary

### GlycoWorks Training Data
"""

        if self.processed_data.get('metadata'):
            meta = self.processed_data['metadata']
            report += f"""- **Source:** {meta['source']}
- **Processing Date:** {meta['processing_date']}
- **Total Samples:** {meta['num_samples']}
- **Task Types:** {', '.join(meta['task_types'])}
"""

            stats = meta.get('statistics', {})
            report += f"""- **CSV Files Processed:** {stats.get('total_csv_files', 0)}
- **Total Glycan Entries:** {stats.get('total_glycan_entries', 0)}
- **Unique Glycans:** {stats.get('unique_glycans', 0)}
- **Tissues Covered:** {stats.get('tissues_covered', 0)}
"""

        report += """
### Candycrunch Validation Data
"""

        if self.validation_data.get('metadata'):
            meta = self.validation_data['metadata']
            report += f"""- **Source:** {meta['source']}
- **Processing Date:** {meta['processing_date']}
- **Reference Structures:** {len(self.validator.knowledge_base.entries) if self.validator else 0}
- **Validation Samples:** {meta['num_validation_samples']}
"""

            kb_stats = meta.get('knowledge_base_stats', {})
            glytoucan_coverage = self.validator.knowledge_base.glytoucan_coverage if self.validator else 0
            report += f"""- **GlyTouCan Coverage:** {glytoucan_coverage:.1%}
- **Structure Validation Rate:** {kb_stats.get('validation_coverage', 0):.1%}
"""

        report += f"""
## Output Files
- **Processed Data:** `{self.output_dir}/glycoworks_processed.pkl`
- **Metadata:** `{self.output_dir}/glycoworks_metadata.json`
- **Training Samples:** `{self.output_dir}/glycoworks_samples.csv`
- **Validation Metadata:** `{self.output_dir}/candycrunch_metadata.json`
- **Validation Samples:** `{self.output_dir}/candycrunch_validation.csv`

## Next Steps
1. **Training:** Use the processed pickle files with your ML training pipeline
2. **Validation:** Use validation samples to evaluate model performance
3. **Analysis:** Review CSV files for data exploration and visualization
4. **Integration:** Import processed data into your preferred ML framework

## Data Quality Notes
"""

        if self.processed_data.get('metadata'):
            stats = self.processed_data['metadata'].get('statistics', {})
            if stats.get('data_quality_score', 0) > 0.8:
                report += "- ✅ High data quality - ready for training\n"
            elif stats.get('data_quality_score', 0) > 0.6:
                report += "- ⚠️ Moderate data quality - review outliers\n"
            else:
                report += "- ❌ Low data quality - additional preprocessing needed\n"

        report += "\n---\n*Generated by Lightweight GlycoWorks Data Processor*"

        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"📋 Processing report saved to {report_file}")
        return str(report_file)


def main():
    """Main execution function for lightweight data processing."""

    parser = argparse.ArgumentParser(description='Process GlycoWorks data without heavy ML dependencies')
    parser.add_argument('--glycoworks-dir', default='data/raw/dataset/glycoworks_glycan_data',
                       help='Directory containing GlycoWorks CSV files')
    parser.add_argument('--candycrunch-dir', default='data/raw/dataset/candycrunch_glycan_data',
                       help='Directory containing candycrunch validation data')
    parser.add_argument('--output-dir', default='data/processed/glycoworks_data_only',
                       help='Output directory for results')
    parser.add_argument('--task-types', nargs='+',
                       default=['abundance_prediction', 'tissue_classification',
                               'biomarker_discovery', 'quantitative_structure_elucidation'],
                       help='Task types for training samples')
    parser.add_argument('--save-format', choices=['pickle', 'json', 'both'], default='both',
                       help='Format to save processed data')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip loading candycrunch validation data')

    args = parser.parse_args()

    # Initialize processor
    processor = LightweightDataProcessor(
        glycoworks_dir=args.glycoworks_dir,
        candycrunch_dir=args.candycrunch_dir,
        output_dir=args.output_dir
    )

    try:
        # Initialize components
        processor.initialize_processors()

        # Process GlycoWorks data
        processing_results = processor.process_glycoworks_data(
            task_types=args.task_types,
            save_format=args.save_format
        )

        # Load validation data (unless skipped)
        if not args.skip_validation:
            validation_results = processor.load_validation_knowledge_base()
        else:
            validation_results = None
            logger.info("⏭️ Skipping validation data loading")

        # Generate report
        report_file = processor.generate_processing_report()

        logger.info("🎉 Data processing completed successfully!")
        logger.info(f"   📊 Training samples: {processing_results['num_samples']}")
        if validation_results:
            logger.info(f"   ✅ Validation structures: {validation_results['num_validation_samples']}")
        logger.info(f"   📋 Report: {report_file}")

    except Exception as e:
        logger.error(f"❌ Data processing failed: {e}")
        raise


if __name__ == '__main__':
    main()