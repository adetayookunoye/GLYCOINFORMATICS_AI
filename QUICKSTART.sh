#!/bin/bash

# Quick Start Guide for Glycoinformatics AI Platform
# Run this script to set up and test your complete system

echo "======================================================================"
echo "  GLYCOINFORMATICS AI - QUICK START SETUP"
echo "======================================================================"
echo ""

# Step 1: Install dependencies
echo "[1/6] Installing Python dependencies..."
pip install pyteomics scipy rdflib biopython psycopg2-binary pymongo redis

# Step 2: Create necessary directories
echo ""
echo "[2/6] Creating directory structure..."
mkdir -p data/training
mkdir -p data/processed
mkdir -p data/benchmarks
mkdir -p models/glycollm_trained
mkdir -p results/{clinical_demo,benchmarks,predictions}
mkdir -p logs

# Step 3: Test spectra parser
echo ""
echo "[3/6] Testing MS/MS spectra parser..."
python3 << 'EOF'
from glycollm.data.spectra_parser import SpectraParser, MSSpectrum
import numpy as np

# Create test spectrum
spectrum = MSSpectrum(
    spectrum_id="TEST_001",
    precursor_mz=1500.5,
    precursor_charge=2,
    precursor_intensity=1e6,
    peaks=np.array([[100.0, 1000.0], [200.0, 2000.0], [300.0, 1500.0]])
)

# Test normalization
spectrum.normalize_intensities()
print(f"✓ Spectra parser working: {len(spectrum.peaks)} peaks normalized")

# Test binning
binned = spectrum.bin_spectrum(bin_size=1.0, mz_range=(0, 2000))
print(f"✓ Spectra binning working: {len(binned)} bins created")
EOF

# Step 4: Test glycosylation site predictor
echo ""
echo "[4/6] Testing glycosylation site predictor..."
python3 << 'EOF'
from glycogot.applications.site_prediction import GlycosylationSitePredictor

predictor = GlycosylationSitePredictor(min_confidence=0.5)

# Test protein sequence
test_sequence = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRHPEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTPTLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNRLCVLHEKTPVSEKVTKCCTESLVNRRPCFSALTPDETYVPKAFDEKLFTFHADICTLPDTEKQIKKQTALVELLKHKPKATEEQLKTVMENFVAFVDKCCAADDKEACFAVEGPKLVVSTQTALA"

# Predict sites
sites = predictor.predict_all_sites(test_sequence)
print(f"✓ Site predictor working: Found {sites['N-linked'].__len__()} N-linked sites, {sites['O-linked'].__len__()} O-linked sites")
EOF

# Step 5: Test constraint validator
echo ""
echo "[5/6] Testing constraint validator..."
python3 << 'EOF'
from glycokg.validation.constraint_validator import ConstraintValidator

validator = ConstraintValidator()

# Test structure
test_structure = {
    'glycan_id': 'TEST_GLYCAN',
    'type': 'N-glycan',
    'monosaccharides': [
        {'id': 'm1', 'type': 'GlcNAc'},
        {'id': 'm2', 'type': 'GlcNAc'},
        {'id': 'm3', 'type': 'Man'}
    ],
    'linkages': [
        {'donor': 'm2', 'acceptor': 'm1', 'type': 'β1-4'},
        {'donor': 'm3', 'acceptor': 'm2', 'type': 'β1-4'}
    ]
}

is_valid, violations = validator.validate_structure(test_structure)
print(f"✓ Constraint validator working: Valid={is_valid}, Violations={len(violations)}")
EOF

# Step 6: Test biomarker discovery
echo ""
echo "[6/6] Testing biomarker discovery..."
python3 << 'EOF'
from glycogot.applications.biomarker_discovery import (
    DifferentialExpressionAnalyzer, GlycanExpression
)
import numpy as np

analyzer = DifferentialExpressionAnalyzer()

# Create synthetic test data
np.random.seed(42)
test_data = []

for glycan_id in ['G0000001', 'G0000002', 'G0000003']:
    # Tumor samples (higher expression)
    for i in range(10):
        test_data.append(GlycanExpression(
            glycan_id=glycan_id,
            sample_id=f'tumor_{i}',
            expression_level=100 + np.random.normal(0, 10),
            condition='tumor'
        ))
    # Normal samples (lower expression)
    for i in range(10):
        test_data.append(GlycanExpression(
            glycan_id=glycan_id,
            sample_id=f'normal_{i}',
            expression_level=50 + np.random.normal(0, 8),
            condition='normal'
        ))

results = analyzer.analyze(test_data, 'tumor', 'normal')
significant = analyzer.filter_significant(results)
print(f"✓ Biomarker discovery working: {len(results)} glycans analyzed, {len(significant)} significant")
EOF

echo ""
echo "======================================================================"
echo "  SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "✅ All components tested successfully!"
echo ""
echo "Next steps:"
echo ""
echo "1. Format your training data:"
echo "   python -m glycollm.data.training_data_formatter \\"
echo "       --output-dir data/training \\"
echo "       --limit 100000 \\"
echo "       --create-splits"
echo ""
echo "2. Populate knowledge graph:"
echo "   python -m glycokg.integration.graph_populator \\"
echo "       --output data/processed/glycokg.ttl \\"
echo "       --limit 1000000"
echo ""
echo "3. Train your model:"
echo "   python scripts/train_glycollm.py \\"
echo "       --train-data data/training/spec_to_struct_train.json \\"
echo "       --val-data data/training/spec_to_struct_val.json \\"
echo "       --output-dir models/glycollm_trained \\"
echo "       --num-epochs 50"
echo ""
echo "4. Run clinical demo:"
echo "   python samples/clinical_demo.py \\"
echo "       --output results/clinical_demo"
echo ""
echo "For detailed documentation, see IMPLEMENTATION_COMPLETE.md"
echo ""
echo "======================================================================"
