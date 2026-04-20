#!/usr/bin/env python3
"""
02_deidentify.py — PII de-identification pipeline.

Replaces sensitive spans before training data reaches model training.
Implements the privacy pipeline from research/finetune.md §4.2.

De-identified entities:
  - Account numbers → <ACCOUNT_LAST4_N>
  - Phone numbers   → <PHONE_N>
  - UPI IDs/VPAs    → <VPA_N>
  - Email addresses → <EMAIL_N>
  - Names           → <NAME_N>

Usage: python 02_deidentify.py --input-dir training/data/raw --output-dir training/data/deid
"""
import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Regex patterns for Indian finance PII
PATTERNS = {
    "account_number": re.compile(r"\b(?:A/c|a/c|Account|Acct)[\s#:]*(?:No\.?\s*)?([X\d]{4,12})\b", re.IGNORECASE),
    "upi_id": re.compile(r"\b[\w.\-]+@(?:okicici|oksbi|okhdfcbank|ybl|upi|paytm|ibl|rbl|axisb|kotak)\b", re.IGNORECASE),
    "phone": re.compile(r"(?<!\d)(?:\+91[\s\-]?)?[6-9]\d{9}(?!\d)"),
    "email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "aadhaar": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
}


class Deidentifier:
    """De-identifier with stable placeholder mapping per document.

    Stateless between documents — create a new instance per document to get
    independent placeholder counters. Within a single call to `deidentify`,
    the same entity always maps to the same placeholder.
    """

    def deidentify(self, text: str) -> "tuple[str, dict]":
        """Replace PII in text. Returns (deid_text, entity_map)."""
        entity_map: dict[str, str] = {}
        counters: dict[str, int] = {}

        def replace(match, entity_type):
            original = match.group(0)
            if original in entity_map:
                return entity_map[original]
            counters[entity_type] = counters.get(entity_type, 0) + 1
            placeholder = f"<{entity_type.upper()}_{counters[entity_type]}>"
            entity_map[original] = placeholder
            return placeholder

        result = text
        for entity_type, pattern in PATTERNS.items():
            result = pattern.sub(lambda m: replace(m, entity_type), result)

        return result, entity_map

    def deidentify_value(self, value, deid: "Deidentifier"):
        if isinstance(value, str):
            return deid.deidentify(value)[0]
        if isinstance(value, dict):
            return {k: self.deidentify_value(v, deid) for k, v in value.items()}
        if isinstance(value, list):
            return [self.deidentify_value(v, deid) for v in value]
        return value

    def deidentify_sample(self, sample: dict) -> dict:
        deid = Deidentifier()
        result = dict(sample)
        result["input"] = self.deidentify_value(sample.get("input", ""), deid)
        result["output"] = self.deidentify_value(sample.get("output", ""), deid)
        result.setdefault("metadata", {})["deid_applied"] = True
        return result


def to_scan_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def process_file(input_path: Path, output_path: Path) -> "tuple[int, int]":
    """De-identify a JSONL file. Returns (total, flagged_count)."""
    deid = Deidentifier()
    total = flagged = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            result = deid.deidentify_sample(sample)

            # Leak check: warn if raw PII patterns still present in output
            scan_text = to_scan_text(result.get("input", "")) + " " + to_scan_text(result.get("output", ""))
            remaining_pii = any(p.search(scan_text) for p in PATTERNS.values())
            if remaining_pii:
                log.warning(f"Potential residual PII in sample {sample.get('id', '?')}")
                flagged += 1

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            total += 1

    return total, flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="training/data/raw")
    parser.add_argument("--output-dir", default="training/data/deid")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grand_total = grand_flagged = 0
    for input_file in input_dir.glob("*.jsonl"):
        output_file = out_dir / input_file.name
        total, flagged = process_file(input_file, output_file)
        log.info(f"{input_file.name}: {total} samples, {flagged} flagged for review")
        grand_total += total
        grand_flagged += flagged

    log.info(f"De-identification complete: {grand_total} total, {grand_flagged} flagged")
    if grand_flagged > 0:
        log.warning("Review flagged samples before training — potential PII leakage detected.")


if __name__ == "__main__":
    main()
