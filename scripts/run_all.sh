#!/bin/bash
# ============================================================
# Full experiment pipeline for rl-memory-curriculum
#
# Trains 3 curriculum configs (AA + MM), evaluates on 2 benchmarks,
# runs analysis and generates paper tables.
#
# Usage:
#   bash scripts/run_all.sh                          # default (LoRA, ≥48GB GPU)
#   bash scripts/run_all.sh --config-dir configs/full_ft  # full FT (≥80GB GPU)
#   bash scripts/run_all.sh --dry-run                # quick test (1 step per config)
#
# Hardware:
#   Default configs: 1× GPU ≥48GB (L40S, A100 40GB+)
#   Full FT configs: 1× GPU ≥80GB (H100, A100 80GB)
#
# Time estimate:
#   LoRA:    ~10h training + eval
#   Full FT: ~14h training + eval
# ============================================================
set -e

if ! command -v uv >/dev/null 2>&1; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the cargo/uv env so the current shell can find it
    source "$HOME/.local/bin/env" 2>/dev/null || source "$HOME/.cargo/env" 2>/dev/null || true
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv installation failed. Please install manually:"
        echo "  https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    echo "uv installed successfully: $(uv --version)"
fi

CONFIG_DIR="configs"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config-dir=*) CONFIG_DIR="${1#*=}"; shift ;;
        --config-dir) CONFIG_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) shift ;;
    esac
done

CONFIG_A="${CONFIG_DIR}/train_locomo_only.yaml"
CONFIG_B="${CONFIG_DIR}/train_mixed.yaml"
CONFIG_C="${CONFIG_DIR}/train_longmemeval_only.yaml"
EVAL_CONFIG="configs/eval.yaml"
RESULTS_DIR="results"

echo "============================================"
echo "  rl-memory-curriculum experiment"
echo "============================================"
echo "Config dir: ${CONFIG_DIR}"
echo "Dry run:    ${DRY_RUN}"
echo "Start:      $(date)"
echo ""

# Verify configs exist
for cfg in "$CONFIG_A" "$CONFIG_B" "$CONFIG_C"; do
    if [ ! -f "$cfg" ]; then
        echo "ERROR: Config not found: $cfg"
        exit 1
    fi
done

# 0. Verify GPU
echo "=== Step 0: GPU Check ==="
uv run python3 -c "
import torch
assert torch.cuda.is_available(), 'No GPU found!'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {name} ({vram:.1f} GB)')
"
echo ""

# Build dry-run flags
EVAL_EXTRA=""
if [ "$DRY_RUN" = true ]; then
    echo ">>> DRY RUN MODE: limited eval (5 examples per benchmark) <<<"
    echo ">>> Training is NOT shortened — dry-run only affects eval + analysis <<<"
    EVAL_EXTRA="--max-examples 5"
    echo ""
fi

# 1. Data prep
echo "=== Step 1: Data Preparation ==="
uv run python3 data/prepare_locomo.py
uv run python3 data/prepare_longmemeval.py
uv run python3 data/prepare_mixed.py
echo ""

START_TIME=$(date +%s)

# 2-4. Train Answer Agents (3 configs)
for label_config in "A:$CONFIG_A:config_a_locomo_only" "B:$CONFIG_B:config_b_mixed" "C:$CONFIG_C:config_c_longmemeval_only"; do
    IFS=':' read -r label config prefix <<< "$label_config"

    if [ ! -f "checkpoints/${prefix}/answer_agent/training_meta.json" ]; then
        echo "=== Training Config ${label} — Answer Agent ==="
        echo "Config: $config"
        echo "Start: $(date)"
        uv run python3 src/train_grpo.py --config "$config" --agent answer_agent
        echo "Config ${label} AA done: $(date)"
        echo ""
    else
        echo "=== Config ${label} AA — SKIPPED (training_meta.json exists) ==="
    fi
done

# 5-7. Train Memory Managers (3 configs)
for label_config in "A:$CONFIG_A:config_a_locomo_only" "B:$CONFIG_B:config_b_mixed" "C:$CONFIG_C:config_c_longmemeval_only"; do
    IFS=':' read -r label config prefix <<< "$label_config"

    if [ ! -f "checkpoints/${prefix}/memory_manager/training_meta.json" ]; then
        echo "=== Training Config ${label} — Memory Manager ==="
        echo "Config: $config"
        echo "Start: $(date)"
        uv run python3 src/train_grpo.py --config "$config" --agent memory_manager
        echo "Config ${label} MM done: $(date)"
        echo ""
    else
        echo "=== Config ${label} MM — SKIPPED (training_meta.json exists) ==="
    fi
done

# 8. Evaluation
echo "=== Step 8: Evaluation ==="
echo "Start: $(date)"
uv run python3 eval/run_eval.py --config "$EVAL_CONFIG" --skip-judge ${EVAL_EXTRA}
echo "Eval done: $(date)"
echo ""

# 9. Analysis + paper tables
echo "=== Step 9: Analysis ==="
uv run python3 eval/analyze_results.py --results "${RESULTS_DIR}/all_results.json" --output paper/tables/
echo ""

# Timing
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
HOURS=$(( ELAPSED / 60 ))
MINS=$(( ELAPSED % 60 ))

echo "============================================"
echo "  PIPELINE COMPLETE"
echo "============================================"
echo "Total time: ${HOURS}h ${MINS}m"
echo "End:        $(date)"
echo "Results:    ${RESULTS_DIR}/"
echo "Tables:     paper/tables/"
echo "============================================"
