#!/usr/bin/env bash
# 08_export_gguf.sh — Export merged model to GGUF with all quantization variants
# Usage: ./08_export_gguf.sh <merged_model_dir> <output_dir> <model_short_name>
#
# Requires: Python environment with unsloth installed
# Example:  ./08_export_gguf.sh outputs/llama-3.2-1b-finance/merged_float16 gguf/llama-3.2-1b-finance llama-3.2-1b

set -euo pipefail

MERGED_DIR="${1:?Usage: $0 <merged_model_dir> <output_dir> <model_short_name>}"
OUTPUT_DIR="${2:?}"
MODEL_SHORT="${3:?}"

echo "Exporting GGUF from: $MERGED_DIR"
echo "Output directory:    $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

python3 - <<PYEOF
from unsloth import FastLanguageModel
from pathlib import Path
import hashlib, json, time

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="$MERGED_DIR",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

for quant in ["q4_k_m", "q5_k_m", "q3_k_m"]:
    print(f"Quantizing {quant.upper()}...")
    model.save_pretrained_gguf("$OUTPUT_DIR", tokenizer, quantization_method=quant)
    print(f"  Done: {quant.upper()}")

print("GGUF export complete.")
for f in Path("$OUTPUT_DIR").glob("*.gguf"):
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    print(f"  {f.name}: {f.stat().st_size/1024/1024:.1f} MB | sha256: {sha[:16]}...")
PYEOF

echo "✅ GGUF export done: $OUTPUT_DIR"
