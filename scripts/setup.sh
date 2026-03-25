#!/bin/bash
# Generic setup: create uv-managed venv, sync deps, download tokenizer.
# Works on any machine with Python 3.10+ and a GPU.
set -e

echo "=== rl-memory-curriculum setup ==="

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed."
    echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Install dependencies into uv-managed .venv
echo "Syncing dependencies with uv..."
uv sync

# Pre-download tokenizer (model weights download on first training run)
echo "Pre-downloading tokenizer..."
uv run python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print(f'Tokenizer ready: vocab_size={tok.vocab_size}')
"

# Verify GPU
echo ""
echo "Checking GPU..."
uv run python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'GPU: {name} ({vram:.1f} GB)')
    if vram < 48:
        print('WARNING: <48GB VRAM. Default configs need ≥48GB. See configs/full_ft/ for ≥80GB.')
else:
    print('WARNING: No GPU detected. Training requires a CUDA GPU.')
"

echo ""
echo "Checking data..."
LOCOMO_RAW_DIR="data/raw/locomo"
LONGMEMEVAL_RAW_DIR="data/raw/longmemeval"

if [ -f "data/processed/locomo_train.jsonl" ]; then
    echo "Processed data found. Ready to train."
else
    if [ -f "$LOCOMO_RAW_DIR/data/locomo10.json" ]; then
        echo "Raw data found. run_all.sh will process it automatically."
    else
        echo "Raw data not found. Cloning required repositories..."
        mkdir -p data/raw

        if [ ! -d "$LOCOMO_RAW_DIR/.git" ]; then
            git clone https://github.com/snap-research/locomo "$LOCOMO_RAW_DIR"
        else
            echo "locomo repo already present: $LOCOMO_RAW_DIR"
        fi

        if [ ! -d "$LONGMEMEVAL_RAW_DIR/.git" ]; then
            git clone https://github.com/xiaowu0162/LongMemEval "$LONGMEMEVAL_RAW_DIR"
        else
            echo "LongMemEval repo already present: $LONGMEMEVAL_RAW_DIR"
        fi
    fi
fi

echo ""
echo "Setup complete. Run: bash scripts/run_all.sh"
