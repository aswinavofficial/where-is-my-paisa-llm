#!/usr/bin/env python3
"""
01_extract.py — Extract raw training signals from app logic outputs.

Reads app-generated data (transaction parser outputs, category rules, anomaly flags,
budget summaries) and writes raw JSONL to training/data/raw/.

Usage: python 01_extract.py --input-dir /path/to/app/exports --output-dir training/data/raw
"""
import argparse
import json
import logging
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def extract_transactions(app_export_path: Path, out_dir: Path) -> int:
    """Extract transaction categorization examples from app export."""
    rows = []
    export_file = app_export_path / "transactions.jsonl"
    if not export_file.exists():
        log.warning(f"No transactions.jsonl at {export_file}")
        return 0

    with open(export_file) as f:
        for line in f:
            txn = json.loads(line)
            # Only include transactions where user has confirmed the category
            if txn.get("user_confirmed_category"):
                rows.append({
                    "id": str(uuid.uuid4()),
                    "task": "txn_categorization",
                    "instruction": "Categorize this Indian bank transaction SMS and extract key details.",
                    "input": txn.get("raw_sms", txn.get("description", "")),
                    "output": (
                        f"Category: {txn['category']}\n"
                        f"Merchant: {txn.get('merchant', 'Unknown')}\n"
                        f"Amount: ₹{txn['amount']:,.2f}\n"
                        f"Type: {txn.get('txn_type', 'debit')}\n"
                        f"Confidence: high\n"
                        f"Rationale: User-confirmed category."
                    ),
                    "metadata": {
                        "source": "historical",
                        "language": "en",
                        "safety_tags": [],
                        "difficulty": "medium",
                    },
                })

    out_file = out_dir / "transactions_raw.jsonl"
    with open(out_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"Extracted {len(rows)} transaction examples → {out_file}")
    return len(rows)


def extract_budget_summaries(app_export_path: Path, out_dir: Path) -> int:
    """Extract budget coaching examples from monthly summaries."""
    rows = []
    export_file = app_export_path / "budget_summaries.jsonl"
    if not export_file.exists():
        log.warning(f"No budget_summaries.jsonl at {export_file}")
        return 0

    with open(export_file) as f:
        for line in f:
            summary = json.loads(line)
            structured = {
                "income": summary.get("income", 0),
                "categories": summary.get("spend_by_category", {}),
                "budgets": summary.get("budget_by_category", {}),
                "savings": summary.get("savings", 0),
            }
            rows.append({
                "id": str(uuid.uuid4()),
                "task": "insight_narration",
                "instruction": "Generate a concise spending insight for this Indian user's monthly finances.",
                "input": json.dumps(structured),
                "output": summary.get("ai_insight", ""),
                "metadata": {
                    "source": "historical",
                    "language": "en",
                    "safety_tags": [],
                    "difficulty": "medium",
                },
            })

    out_file = out_dir / "budget_summaries_raw.jsonl"
    with open(out_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"Extracted {len(rows)} budget examples → {out_file}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="app_exports", help="Path to app data exports")
    parser.add_argument("--output-dir", default="training/data/raw", help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    total += extract_transactions(input_dir, out_dir)
    total += extract_budget_summaries(input_dir, out_dir)
    log.info(f"Total extracted: {total} raw samples")


if __name__ == "__main__":
    main()
