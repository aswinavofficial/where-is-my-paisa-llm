#!/usr/bin/env python3
"""
09_publish_manifest.py — Generate and publish release manifest.

Produces manifest.json + eval_report.json + model_card.md bundle
matching the release contract in research/finetune.md §12.1.

Usage: python 09_publish_manifest.py --gguf-dir gguf/llama-3.2-1b-finance
                                      --eval-report eval_reports/report.json
                                      --model-id "meta-llama/Llama-3.2-1B-Instruct"
                                      --version 1.0.0
"""
import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(gguf_dir: Path, eval_report_path: Path, model_id: str, version: str) -> dict:
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF files in {gguf_dir}")

    q4_file = next((f for f in gguf_files if "q4_k_m" in f.name.lower()), gguf_files[0])
    sha = sha256_file(q4_file)
    size_bytes = q4_file.stat().st_size

    eval_metrics = {}
    if eval_report_path and eval_report_path.exists():
        with open(eval_report_path) as f:
            eval_data = json.load(f)
        eval_metrics = eval_data.get("metrics", {})

    base_name = model_id.split("/")[-1].lower()
    manifest = {
        "id": f"{base_name}-fin-q4km-v{version}",
        "displayName": f"{base_name.replace('-', ' ').title()} Finance Q4_K_M",
        "baseModel": model_id,
        "finetuneMethod": "QLoRA-SFT",
        "quantization": "Q4_K_M",
        "sizeBytes": size_bytes,
        "sha256": sha,
        "requirements": {
            "minRamMb": 6000,
            "recommendedFreeStorageMb": int(size_bytes / 1024**2 * 1.5),
            "abis": ["arm64-v8a"],
            "minAndroidApi": 26,
        },
        "metrics": {
            "qualityScore": eval_metrics.get("txn_categorization", {}).get("accuracy"),
            "medianDecodeTokensPerSec": None,
            "medianFirstTokenMs": None,
            "hallucinationRisk": "medium",
            "batteryImpact": "medium",
        },
        "safety": {
            "refusalRecall": eval_metrics.get("safety", {}).get("refusal_recall"),
            "hallucinatedNumberRate": eval_metrics.get("hallucination", {}).get("hallucinated_number_rate"),
        },
        "provenance": {
            "datasetVersion": "wimp-finance-instruct-v1",
            "trainRunId": f"run_{time.strftime('%Y_%m_%d')}_{base_name}",
            "evalRunId": eval_data.get("eval_id") if eval_report_path and eval_report_path.exists() else None,
        },
    }
    return manifest


def generate_model_card(manifest: dict) -> str:
    return f"""---
language: en
license: llama3.2
tags:
  - finance
  - indian-finance
  - qlora
  - gguf
  - where-is-my-paisa
datasets:
  - wimp-finance-instruct-v1
base_model: {manifest['baseModel']}
---

# {manifest['displayName']}

Fine-tuned version of [{manifest['baseModel']}](https://huggingface.co/{manifest['baseModel']})
for Indian personal finance tasks, part of the [Where Is My Paisa](https://github.com/aswinavofficial/where-is-my-paisa-android) app.

## Model Details

- **Base model**: {manifest['baseModel']}
- **Fine-tuning method**: {manifest['finetuneMethod']}
- **Quantization**: {manifest['quantization']}
- **Size**: {manifest['sizeBytes'] / 1024**2:.1f} MB
- **Version**: {manifest['id']}

## Requirements

- Min RAM: {manifest['requirements']['minRamMb']} MB
- Android API: {manifest['requirements']['minAndroidApi']}+
- ABI: {', '.join(manifest['requirements']['abis'])}

## Tasks Supported

1. **Transaction categorization** — Indian SMS/UPI format parsing
2. **Spending insight narration** — Monthly finance summaries
3. **Budget coaching** — Actionable spend reduction advice
4. **Anomaly explanation** — Unusual transaction alerts
5. **Goal coaching** — Progress toward savings goals

## Safety

This model refuses:
- Investment/stock return guarantees
- Tax evasion advice
- Legal certainty requests

## Usage (with llama.cpp)

```bash
./llama-cli -m {manifest['id']}.gguf -p "Categorize: Your A/c XX5678 debited Rs.499 for Netflix."
```

## License

Based on [{manifest['baseModel']}]({manifest['baseModel']}) license.
Fine-tuning data is synthetic/de-identified. See `research/finetune.md` for data policy.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-dir", required=True)
    parser.add_argument("--eval-report", default=None)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()

    gguf_dir = Path(args.gguf_dir)
    eval_path = Path(args.eval_report) if args.eval_report else None

    manifest = build_manifest(gguf_dir, eval_path, args.model_id, args.version)

    manifest_path = gguf_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"✅ manifest.json → {manifest_path}")

    card = generate_model_card(manifest)
    card_path = gguf_dir / "model_card.md"
    with open(card_path, "w") as f:
        f.write(card)
    log.info(f"✅ model_card.md → {card_path}")

    log.info(f"Release bundle ready at {gguf_dir}")
    log.info(f"  ID: {manifest['id']}")
    log.info(f"  SHA256: {manifest['sha256'][:32]}...")


if __name__ == "__main__":
    main()
