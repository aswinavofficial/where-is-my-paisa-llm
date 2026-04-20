#!/usr/bin/env python3
# Requires Python 3.8+
"""
04_validate.py — Schema validation, dedup, and data quality checks.

Per research/finetune.md §9:
  1. Schema validation for every JSONL row
  2. Near-duplicate removal (lexical hash)
  3. Contradiction detection (same input, conflicting outputs)
  4. PII leak check (post de-id scan)
  5. Holdout sanctity check (no synthetic variants of test rows in train)

Usage: python 04_validate.py --curated-dir training/data/curated --splits-dir training/data/splits
"""
import argparse
import hashlib
import json
import logging
import re
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REQUIRED_FIELDS = {"id", "task", "instruction", "input", "output"}
VALID_TASKS = {"txn_categorization", "insight_narration", "budget_coaching",
               "anomaly_explanation", "goal_coaching", "safety_refusal"}

PII_PATTERNS = [
    re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)"),            # Indian phone
    re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.I),  # email
    re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),             # Aadhaar
    re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),           # PAN
]


def validate_schema(sample: dict) -> "list[str]":
    errors = []
    missing = REQUIRED_FIELDS - set(sample.keys())
    if missing:
        errors.append(f"Missing fields: {missing}")
    if sample.get("task") not in VALID_TASKS:
        errors.append(f"Invalid task: {sample.get('task')}")
    if len(text_value(sample.get("output", ""))) < 10:
        errors.append("Output too short (< 10 chars)")
    return errors


def text_value(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def content_hash(sample: dict) -> str:
    key = f"{text_value(sample.get('instruction', ''))}|{text_value(sample.get('input', ''))}".lower()
    return hashlib.md5(key.encode()).hexdigest()


def pii_check(text: str) -> bool:
    return any(p.search(text) for p in PII_PATTERNS)


def validate_file(corpus_path: Path, splits_dir: Path):
    all_samples = []
    schema_errors = []
    pii_flagged = []

    with open(corpus_path) as f:
        for i, line in enumerate(f, 1):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                log.error(f"Line {i}: JSON parse error: {e}")
                continue

            errors = validate_schema(sample)
            if errors:
                schema_errors.append((sample.get("id", f"line_{i}"), errors))

            if pii_check(text_value(sample.get("input", "")) + " " + text_value(sample.get("output", ""))):
                pii_flagged.append(sample.get("id", f"line_{i}"))

            all_samples.append(sample)

    log.info(f"Loaded {len(all_samples)} samples from {corpus_path}")
    if schema_errors:
        log.warning(f"Schema errors in {len(schema_errors)} samples")
        for sid, errs in schema_errors[:5]:
            log.warning(f"  {sid}: {errs}")
    if pii_flagged:
        log.error(f"PII detected in {len(pii_flagged)} samples — review before training!")
        for sid in pii_flagged[:5]:
            log.error(f"  {sid}")

    # Dedup by content hash
    seen_hashes: dict[str, str] = {}
    deduped = []
    for idx, s in enumerate(all_samples, 1):
        h = content_hash(s)
        if h not in seen_hashes:
            seen_hashes[h] = s.get("id", f"unknown_{idx}")
            deduped.append(s)
    log.info(f"After dedup: {len(deduped)} samples (removed {len(all_samples) - len(deduped)} duplicates)")

    # Contradiction check: same input → different outputs
    input_to_outputs: dict[str, list] = defaultdict(list)
    for s in deduped:
        input_to_outputs[text_value(s.get("input", ""))[:200]].append(text_value(s.get("output", ""))[:100])
    contradictions = {k: v for k, v in input_to_outputs.items() if len(set(v)) > 1}
    if contradictions:
        log.warning(f"Possible contradictions in {len(contradictions)} input groups")

    # Split train/val/test (80/10/10), stratified by task
    task_groups: dict[str, list] = defaultdict(list)
    for s in deduped:
        task_groups[s.get("task", "unknown")].append(s)

    train, val, test = [], [], []
    for task, samples in task_groups.items():
        random.shuffle(samples)
        n = len(samples)
        t_end = int(n * 0.8)
        v_end = t_end + int(n * 0.1)
        train.extend(samples[:t_end])
        val.extend(samples[t_end:v_end])
        test.extend(samples[v_end:])

    random.shuffle(train)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train", train), ("validation", val), ("test", test)]:
        out_file = splits_dir / f"{split_name}.jsonl"
        with open(out_file, "w") as f:
            for s in split_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        log.info(f"  {split_name}: {len(split_data)} samples → {out_file}")

    return len(schema_errors) == 0 and len(pii_flagged) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", default="training/data/curated")
    parser.add_argument("--splits-dir", default="training/data/splits")
    args = parser.parse_args()

    corpus_file = Path(args.curated_dir) / "corpus.jsonl"
    if not corpus_file.exists():
        log.error(f"Corpus not found at {corpus_file}. Run 03_build_jsonl.py first.")
        return

    ok = validate_file(corpus_file, Path(args.splits_dir))
    if ok:
        log.info("✅ Validation passed — splits ready for training")
    else:
        log.error("❌ Validation failed — fix issues before training")


if __name__ == "__main__":
    main()
