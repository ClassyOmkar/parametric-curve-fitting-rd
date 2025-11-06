#!/bin/bash
# Validation script to compute L1 score and verify outputs

# Default threshold (can be overridden by environment variable)
L1_THRESHOLD=${L1_THRESHOLD:-1.0}

# Check if params.json exists
if [ ! -f "results/params.json" ]; then
    echo "Error: results/params.json not found. Run the pipeline first."
    exit 1
fi

# Extract L1 score from params.json
L1_SCORE=$(python3 -c "import json; print(json.load(open('results/params.json'))['l1'])")

echo "L1 Score: $L1_SCORE"
echo "Threshold: $L1_THRESHOLD"

# Compare L1 score with threshold
python3 -c "
import sys
l1 = float('$L1_SCORE')
threshold = float('$L1_THRESHOLD')
if l1 <= threshold:
    print('PASS: L1 score is within threshold')
    sys.exit(0)
else:
    print('FAIL: L1 score exceeds threshold')
    sys.exit(1)
"
