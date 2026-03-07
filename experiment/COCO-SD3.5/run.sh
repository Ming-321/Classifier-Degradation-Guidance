#!/bin/bash
# Compare CFG, CADS, ICG, CDG methods on SD3.5 model. Evaluate with FID, CLIPScore, AestheticScore, VQAScore.

set -e

# Experiment settings
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BASE_DIR/../.." && pwd)"
PROFILES_DIR="$BASE_DIR/profiles"
OUTPUT_BASE_DIR="$BASE_DIR/outputs"

# Experimental parameters
METHODS=("cfg/sd35:cfg.json" "cads/sd35:cads.json" "icg/sd35:icg.json" "cdg/sd35:cdg.json" "pag/sd35:pag.json" "seg/sd35:seg.json")
MAX_PROMPTS=5000
GPUS="0"
SEED=42
PROMPT_SEED=2025

echo "=== SD3.5 Methods Comparison Experiment ==="
echo "Methods: CFG, CADS, ICG, CDG, PAG, SEG"
echo "Prompts: $MAX_PROMPTS, Seed: $SEED, Prompt Seed: $PROMPT_SEED"
mkdir -p "$OUTPUT_BASE_DIR"

# Generation phase
echo "=== Step 1/3: Distributed Generation ==="
cd "$PROJECT_ROOT"
GENERATED_PATHS=()

for method in "${METHODS[@]}"; do
    # Parse method and config file
    model_type=$(echo $method | cut -d':' -f1)
    config_file=$(echo $method | cut -d':' -f2)
    method_name=$(echo $model_type | tr '/' '_')
    
    echo "Generating with $model_type..."
    
    # Config file path
    config_path="$PROFILES_DIR/$config_file"
    output_dir="$OUTPUT_BASE_DIR/${method_name}"
    
    # Check config file exists
    if [ ! -f "$config_path" ]; then
        echo "Config file not found: $config_path, skipping"
        continue
    fi
    
    # Run distributed generation
    python -m evaluation.generate_distributed \
        --model "$model_type" \
        --config "$config_path" \
        --gpus "$GPUS" \
        --max_prompts "$MAX_PROMPTS" \
        --output_dir "$output_dir" \
        --prompt_seed "$PROMPT_SEED" \
        --seed "$SEED"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed $model_type"
        GENERATED_PATHS+=("$output_dir")
    else
        echo "✗ Failed $model_type"
    fi
    echo ""
done

# Evaluation phase
echo "=== Step 2/3: Distributed Evaluation ==="
if [ ${#GENERATED_PATHS[@]} -eq 0 ]; then
    echo "Error: No generated results found"
    exit 1
fi

echo "Found ${#GENERATED_PATHS[@]} result directories"
for path in "${GENERATED_PATHS[@]}"; do
    echo "  - $(basename "$path"): $path"
done

# Run distributed evaluation
python -m evaluation.eval_distributed \
    --generated_paths ${GENERATED_PATHS[@]} \
    --metrics "FID,CLIPScore,AestheticScore,VQAScore" \
    --device_ids "$GPUS"

if [ $? -eq 0 ]; then
    echo "✓ Evaluation completed"
else
    echo "✗ Evaluation failed"
    exit 1
fi

# Result analysis
echo "=== Step 3/3: Result Analysis ==="
python "$BASE_DIR/analysis.py" \
    --output_dir "$OUTPUT_BASE_DIR" \
    --save_dir "$OUTPUT_BASE_DIR"

echo "=== Experiment Completed ==="
echo "Results saved in: $OUTPUT_BASE_DIR"
echo "Comparison table: $OUTPUT_BASE_DIR/sd35_methods_comparison.csv"