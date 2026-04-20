# where-is-my-paisa-llm

Fine-tuning pipeline for **Where Is My Paisa** — Indian personal finance LLMs using [Unsloth](https://unsloth.ai) + Google Colab.

> 📱 Android app: [aswinavofficial/where-is-my-paisa-android](https://github.com/aswinavofficial/where-is-my-paisa-android)  
> 📋 Research plan: [`research/finetune.md`](https://github.com/aswinavofficial/where-is-my-paisa-android/blob/main/research/finetune.md)  
> 📋 Model candidates: [`research/local-llm.md`](https://github.com/aswinavofficial/where-is-my-paisa-android/blob/main/research/local-llm.md)

---

## Why Unsloth + Colab?

| Concern | Why Unsloth |
|---------|------------|
| **Speed** | 2×–5× faster training vs raw HF Transformers |
| **Memory** | 70% less VRAM — fits 1B models on free T4 GPU |
| **GGUF export** | Built-in one-step export for llama.cpp (Android) |
| **Cost** | Free Colab tier covers Tier 1 models (≤1.7B) |
| **Compatibility** | Supports all target models: Llama, Qwen, SmolLM2, Gemma, Phi |

---

## Model Targets

| # | Notebook | Model | Tier | Colab GPU | VRAM |
|---|----------|-------|------|-----------|------|
| 01 | [01_finetune_llama_32_1b.ipynb](notebooks/01_finetune_llama_32_1b.ipynb) | `meta-llama/Llama-3.2-1B-Instruct` | **Default** | T4 Free | ~8 GB |
| 02 | [02_finetune_qwen25_05b.ipynb](notebooks/02_finetune_qwen25_05b.ipynb) | `Qwen/Qwen2.5-0.5B-Instruct` | **Fallback** | T4 Free | ~4 GB |
| 03 | [03_finetune_smollm2_17b.ipynb](notebooks/03_finetune_smollm2_17b.ipynb) | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | **Premium** | T4 Free | ~10 GB |
| 04 | [04_finetune_llama_32_3b.ipynb](notebooks/04_finetune_llama_32_3b.ipynb) | `meta-llama/Llama-3.2-3B-Instruct` | Flagship | **Colab Pro A100** | ~16 GB |
| 05 | [05_finetune_qwen25_15b.ipynb](notebooks/05_finetune_qwen25_15b.ipynb) | `Qwen/Qwen2.5-1.5B-Instruct` | Standard | T4 Free | ~9 GB |
| 06 | [06_finetune_gemma_2_2b.ipynb](notebooks/06_finetune_gemma_2_2b.ipynb) | `google/gemma-2-2b-it` | Standard | T4 Free | ~11 GB |
| 07 | [07_finetune_phi_35_mini.ipynb](notebooks/07_finetune_phi_35_mini.ipynb) | `microsoft/Phi-3.5-mini-instruct` | Flagship | **Colab Pro A100** | ~20 GB |
| 08 | [08_finetune_smollm2_360m.ipynb](notebooks/08_finetune_smollm2_360m.ipynb) | `HuggingFaceTB/SmolLM2-360M-Instruct` | Ultra-light | T4 Free | ~3 GB |
| 09 | [09_dpo_alignment.ipynb](notebooks/09_dpo_alignment.ipynb) | Any SFT output | Optional DPO | T4 Free | ~8 GB |
| 10 | [10_export_gguf.ipynb](notebooks/10_export_gguf.ipynb) | Any merged model | GGUF export | T4 Free | ~6 GB |

> **Start here**: Run [00_data_generation.ipynb](notebooks/00_data_generation.ipynb) first to create the training dataset.

---

## Quick Start (Google Colab)

### Step 1 — Generate training data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aswinavofficial/where-is-my-paisa-llm/blob/main/notebooks/00_data_generation.ipynb)

### Step 2 — Fine-tune the default model (Llama 3.2 1B)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aswinavofficial/where-is-my-paisa-llm/blob/main/notebooks/01_finetune_llama_32_1b.ipynb)

1. Click **Runtime → Change runtime type → T4 GPU**
2. Run all cells (`Ctrl+F9`)
3. The notebook exports Q4_K_M and Q5_K_M GGUF files automatically

### Step 3 — (Optional) DPO alignment

Run [09_dpo_alignment.ipynb](notebooks/09_dpo_alignment.ipynb) if the SFT output needs better tone/safety.

---

## Training Pipeline (Automated)

For non-Colab runs on A100/4090 instances:

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Extract app data exports (requires app data files)
python training/scripts/01_extract.py --input-dir app_exports/ --output-dir training/data/raw/

# 2. De-identify PII
python training/scripts/02_deidentify.py --input-dir training/data/raw/ --output-dir training/data/deid/

# 3. Build curated corpus with synthetic augmentation
python training/scripts/03_build_jsonl.py --deid-dir training/data/deid/ --output-dir training/data/curated/

# 4. Validate + create splits
python training/scripts/04_validate.py --curated-dir training/data/curated/ --splits-dir training/data/splits/

# 5. Train SFT (1B model example)
python training/scripts/05_train_sft.py --config training/configs/sft_1b.yaml

# 6. (Optional) DPO alignment
python training/scripts/06_train_dpo.py --config training/configs/dpo_1b.yaml

# 7. Evaluate
python training/scripts/07_eval.py \
  --model-path outputs/llama-3.2-1b-finance/lora_adapter \
  --test-file training/data/splits/test.jsonl \
  --output eval_reports/report.json

# 8. Export GGUF
./training/scripts/08_export_gguf.sh \
  outputs/llama-3.2-1b-finance/merged_float16 \
  gguf/llama-3.2-1b-finance \
  llama-3.2-1b

# 9. Publish manifest
python training/scripts/09_publish_manifest.py \
  --gguf-dir gguf/llama-3.2-1b-finance \
  --eval-report eval_reports/report.json \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --version 1.0.0
```

---

## Finance Tasks Trained

| Task | Description | Input | Output |
|------|-------------|-------|--------|
| `txn_categorization` | Categorize Indian SMS/UPI transactions | Bank SMS text | Category, merchant, amount, confidence |
| `insight_narration` | Monthly spending narratives | Finance JSON | 2–4 sentence insight with action |
| `budget_coaching` | Over/under budget advice | Budget vs actual | Actionable reduction plan |
| `anomaly_explanation` | Unusual transaction alerts | Transaction + pattern | Explanation + action |
| `goal_coaching` | Savings goal progress advice | Goal + current savings | Timeline + recommendations |
| `safety_refusal` | Refuse illegal/high-risk requests | Dangerous prompt | Polite refusal + safe alternative |

---

## Training Approach (from `research/finetune.md`)

- **Method**: QLoRA (4-bit base + LoRA adapters)
- **LoRA rank**: 16–32 (larger for 1B+ models)
- **Epochs**: 3 (SFT), 1 (DPO)
- **Learning rate**: 2e-4 (SFT), 5e-5 (DPO)
- **Prompt format**: Alpaca-style with finance system context
- **Export**: Q4_K_M GGUF (default) + Q5_K_M (premium)

### Model Versioning

```
<base>-fin-<major.minor.patch>-<quant>
# Example: llama-3.2-1b-fin-1.0.0-q4_k_m
```

---

## Release Bundle

Each trained model ships with:

```
gguf/<model-name>/
├── model.gguf          # Q4_K_M (default)
├── model_q5_k_m.gguf  # Q5_K_M (premium)
├── manifest.json       # App catalog API payload
├── model_card.md       # HuggingFace model card
└── eval_report.json    # Offline metrics + acceptance gates
```

The `manifest.json` is consumed by the Android app's `POST /v1/local-llm/models/catalog` API.

### Acceptance Gates (must pass before shipping)

- ✅ Transaction categorization accuracy ≥ 70%
- ✅ Safety refusal recall ≥ 90%
- ✅ Hallucinated number rate < 10%
- ✅ GGUF SHA-256 verification passes
- ✅ On-device latency p50 ≥ 8 tokens/sec (validated separately)

---

## Repository Structure

```
where-is-my-paisa-llm/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 00_data_generation.ipynb      # Synthetic dataset builder
│   ├── 01_finetune_llama_32_1b.ipynb # Llama 3.2 1B (default)
│   ├── 02_finetune_qwen25_05b.ipynb  # Qwen2.5 0.5B (fallback)
│   ├── 03_finetune_smollm2_17b.ipynb # SmolLM2 1.7B (premium)
│   ├── 04_finetune_llama_32_3b.ipynb # Llama 3.2 3B (flagship)
│   ├── 05_finetune_qwen25_15b.ipynb  # Qwen2.5 1.5B (standard)
│   ├── 06_finetune_gemma_2_2b.ipynb  # Gemma 2B (standard)
│   ├── 07_finetune_phi_35_mini.ipynb # Phi-3.5 Mini (flagship)
│   ├── 08_finetune_smollm2_360m.ipynb# SmolLM2 360M (ultra-light)
│   ├── 09_dpo_alignment.ipynb        # Optional DPO pass
│   └── 10_export_gguf.ipynb          # GGUF quantization + manifest
├── training/
│   ├── configs/                      # YAML hyperparameter configs
│   │   ├── sft_05b.yaml
│   │   ├── sft_1b.yaml
│   │   ├── sft_15b.yaml (→ use sft_1b.yaml)
│   │   ├── sft_17b.yaml
│   │   ├── sft_3b.yaml
│   │   └── dpo_1b.yaml
│   ├── data/                         # Training data (raw/deid not committed)
│   │   ├── README.md
│   │   └── sample_finance_data.jsonl
│   └── scripts/                      # Automated pipeline
│       ├── 01_extract.py
│       ├── 02_deidentify.py
│       ├── 03_build_jsonl.py
│       ├── 04_validate.py
│       ├── 05_train_sft.py
│       ├── 06_train_dpo.py
│       ├── 07_eval.py
│       ├── 08_export_gguf.sh
│       └── 09_publish_manifest.py
└── .gitignore
```

---

## Privacy & Data Policy

Per `research/finetune.md §4.2`:

- **No raw user data committed** to this repo ever
- De-identification uses regex + placeholder mapping (`<ACCOUNT_LAST4_1>`, `<VPA_1>`, etc.)
- All synthetic data uses realistic but fictional Indian finance patterns
- PII scan runs automatically in `04_validate.py` before training

---

## References

- [Unsloth Docs](https://unsloth.ai/docs/get-started/install/google-colab) — Google Colab setup
- [Unsloth Fine-Tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- [research/finetune.md](https://github.com/aswinavofficial/where-is-my-paisa-android/blob/main/research/finetune.md) — Full training plan
- [research/local-llm.md](https://github.com/aswinavofficial/where-is-my-paisa-android/blob/main/research/local-llm.md) — Model selection research
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
