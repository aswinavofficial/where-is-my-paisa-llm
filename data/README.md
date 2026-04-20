# Training Data

This directory holds training data at each pipeline stage.

## Directory Layout

```
data/
├── raw/          # Output of 01_extract.py — app logic outputs, not yet de-identified
├── deid/         # Output of 02_deidentify.py — PII replaced with stable placeholders
├── curated/      # Output of 03_build_jsonl.py — merged + formatted corpus
├── splits/       # Output of 04_validate.py — train/validation/test splits (80/10/10)
└── sample_finance_data.jsonl   # 15 hand-crafted seed examples (safe to commit)
```

## Data Schema (per `research/finetune.md`)

Each JSONL line:

```json
{
  "id": "uuid",
  "task": "txn_categorization|insight_narration|budget_coaching|anomaly_explanation|goal_coaching|safety_refusal",
  "instruction": "...",
  "input": "...",
  "output": "...",
  "text": "<formatted Alpaca prompt + response>",
  "metadata": {
    "source": "historical|synthetic",
    "language": "en",
    "safety_tags": [],
    "difficulty": "easy|medium|hard"
  }
}
```

## Privacy Rules

- `raw/` and `deid/` **must not be committed** to git (listed in `.gitignore`)
- Only `sample_finance_data.jsonl` and `splits/` are safe to share
- Run `04_validate.py` to confirm no PII leaks before training

## Volume Targets (from `research/finetune.md §6`)

| Task | Minimum |
|------|---------|
| Transaction categorization | 20k–40k |
| Insight narration | 8k–15k |
| Budget coaching | 6k–10k |
| Anomaly explanation | 4k–8k |
| Goal coaching | 4k–8k |
| Safety refusal pairs | 5k+ |
| **Total** | **50k–80k** |
