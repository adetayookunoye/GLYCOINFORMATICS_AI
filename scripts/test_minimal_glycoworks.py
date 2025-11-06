#!/usr/bin/env python3
"""
Minimal Test for GlycoWorks CSV Processing
==========================================

Tests just the CSV processing part without full imports.

Author: Glycoinformatics AI Team
Date: November 5, 2025
"""

import pandas as pd
import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def test_csv_reading():
    """Test basic CSV reading functionality."""
    logger.info("🧪 Testing CSV reading...")

    try:
        # Test reading one CSV file
        csv_path = "dataset/glycoworks_glycan_data/glycomics_human_brain_N_PMID38343116.csv"

        if not os.path.exists(csv_path):
            logger.error(f"❌ Test CSV not found: {csv_path}")
            return False

        # Read CSV
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        logger.info(f"✅ Successfully read CSV: {df.shape[0]} rows, {df.shape[1]} columns")

        # Check required columns
        if 'glycan' not in df.columns:
            logger.error("❌ Missing 'glycan' column")
            return False

        logger.info(f"✅ Found glycan column with {len(df['glycan'].unique())} unique glycans")

        # Check sample columns
        sample_cols = [col for col in df.columns if col != 'glycan']
        logger.info(f"✅ Found {len(sample_cols)} sample columns")

        # Test data extraction
        sample_data = {}
        for idx, row in df.head(5).iterrows():  # Test first 5 rows
            glycan = str(row['glycan']).strip()
            for sample_col in sample_cols[:3]:  # Test first 3 samples
                abundance = row[sample_col]
                if pd.notna(abundance) and abundance > 0:
                    if glycan not in sample_data:
                        sample_data[glycan] = []
                    sample_data[glycan].append(abundance)

        logger.info(f"✅ Extracted abundance data for {len(sample_data)} glycans")

        return True

    except Exception as e:
        logger.error(f"❌ CSV reading test failed: {e}")
        return False

def test_metadata_parsing():
    """Test sample metadata parsing."""
    logger.info("🧪 Testing metadata parsing...")

    try:
        # Test parsing sample names
        test_samples = [
            "NG_Control_1",
            "NG_SSSMuG_2",
            "SC_160320_EC_2_NG",
            "HV_9F_NG"
        ]

        for sample in test_samples:
            logger.info(f"   Parsing: {sample}")
            # Basic parsing logic (simplified)
            parts = sample.split('_')
            if len(parts) >= 2:
                tissue_condition = parts[0]
                replicate = parts[-1] if parts[-1].isdigit() else 'unknown'
                logger.info(f"     → Tissue/Condition: {tissue_condition}, Replicate: {replicate}")

        logger.info("✅ Metadata parsing test completed")
        return True

    except Exception as e:
        logger.error(f"❌ Metadata parsing test failed: {e}")
        return False

def main():
    """Main test function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("🚀 Starting minimal GlycoWorks CSV processing test...")

    # Run tests
    csv_test = test_csv_reading()
    metadata_test = test_metadata_parsing()

    if csv_test and metadata_test:
        logger.info("🎉 All minimal tests passed!")
        logger.info("   ✅ CSV processing foundation is working")
        logger.info("   ✅ Ready to build full GlycoWorks processor")
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1)

if __name__ == '__main__':
    main()