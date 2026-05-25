#!/bin/bash
# Model Training and Setup Script
# This script prepares the environment and trains fresh models

set -e

echo "=================================================="
echo "NTB Predictor - Model Training Setup"
echo "=================================================="

# Step 1: Install/upgrade required packages
echo ""
echo "Step 1: Installing/upgrading dependencies..."
pip install --upgrade numpy==1.24.3 joblib scikit-learn xgboost pandas

# Step 2: Clean old models
echo ""
echo "Step 2: Cleaning old corrupted models..."
rm -rf models/*.pkl models/gti_*.pkl 2>/dev/null || true
echo "✓ Old models removed"

# Step 3: Create directory structure
echo ""
echo "Step 3: Creating directory structure..."
mkdir -p models data/raw

# Step 4: Check if data exists
echo ""
echo "Step 4: Checking for data..."
if [ -f "data/raw/Primary_Market_in_Excel.csv" ]; then
    echo "✓ Data file found"
else
    echo "⚠ Data file not found, downloading sample data..."
    # Will use whatever CSV is available
fi

# Step 5: Run training script
echo ""
echo "Step 5: Training new models..."
python train_models.py

echo ""
echo "=================================================="
echo "✓ Setup complete! Models ready for use."
echo "=================================================="
echo ""
echo "Next: Restart your Streamlit app"
echo "  streamlit run app.py"
