#!/usr/bin/env python3
"""
03_build_jsonl.py — Build and augment the curated JSONL training corpus.

Merges de-identified historical data with synthetic augmentation,
applies the Alpaca prompt format, and writes training/data/curated/.

Usage: python 03_build_jsonl.py --deid-dir training/data/deid --output-dir training/data/curated
"""
import argparse
import json
import logging
import random
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
random.seed(42)

ALPACA_PROMPT = """Below is an instruction describing a finance task, paired with input context. Write a concise, accurate response.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

SYNTHETIC_MERCHANTS = {
    "Food": ["Swiggy", "Zomato", "Dunzo", "EatFit", "Rebel Foods"],
    "Groceries": ["DMart", "BigBasket", "Blinkit", "Zepto", "JioMart"],
    "Transport": ["Ola", "Uber", "IRCTC", "RedBus", "Rapido"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Meesho", "Nykaa"],
    "Entertainment": ["Netflix", "Hotstar", "BookMyShow", "Spotify", "Gaana"],
    "Utilities": ["BESCOM", "Airtel", "Jio", "Mahanagar Gas", "BWSSB"],
    "Healthcare": ["Apollo Pharmacy", "1mg", "PharmEasy", "Practo", "Netmeds"],
}

SMS_TEMPLATES = [
    "Your A/c XX{acct} debited by Rs.{amount} for {merchant} on {date}. Avl Bal Rs.{bal}",
    "UPI/DR/{ref}/{merchant}/HDFC{branch}. Rs {amount} debited. Avl Bal Rs {bal}",
    "Alert: INR {amount} spent on {merchant} using your card ending {acct} on {date}.",
    "Dear Customer, Rs.{amount} debited from A/c XX{acct} on {date} at {merchant}. Avl Bal INR {bal}",
]


def synthetic_txn() -> dict:
    cat = random.choice(list(SYNTHETIC_MERCHANTS.keys()))
    merchant = random.choice(SYNTHETIC_MERCHANTS[cat])
    amount = round(random.uniform(50, 20000), 2)
    sms = random.choice(SMS_TEMPLATES).format(
        acct=random.randint(1000, 9999),
        amount=f"{amount:,.2f}",
        merchant=merchant,
        date=f"{random.randint(1, 28):02d}-Apr",
        bal=f"{random.uniform(500, 80000):,.2f}",
        ref=str(random.randint(100000000000, 999999999999)),
        branch=random.randint(1000, 9999),
    )
    output = (
        f"Category: {cat}\nMerchant: {merchant}\nAmount: ₹{amount:,.2f}\n"
        f"Type: UPI\nConfidence: high\nRationale: {merchant} is a well-known {cat.lower()} provider."
    )
    return {
        "id": str(uuid.uuid4()),
        "task": "txn_categorization",
        "instruction": "Categorize this Indian bank transaction SMS and extract key details.",
        "input": sms,
        "output": output,
        "metadata": {"source": "synthetic", "language": "en", "safety_tags": [], "difficulty": "easy"},
    }


def format_alpaca(sample: dict, eos_token: str = "</s>") -> dict:
    text = ALPACA_PROMPT.format(
        sample.get("instruction", ""),
        sample.get("input", ""),
        sample.get("output", ""),
    ) + eos_token
    return {**sample, "text": text}


def build_corpus(deid_dir: Path, output_dir: Path, synthetic_count: int = 5000):
    samples = []

    # Load de-identified historical samples
    for f in deid_dir.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                samples.append(json.loads(line))
    log.info(f"Loaded {len(samples)} historical samples from {deid_dir}")

    # Add synthetic augmentation
    for _ in range(synthetic_count):
        samples.append(synthetic_txn())
    log.info(f"Added {synthetic_count} synthetic samples. Total: {len(samples)}")

    random.shuffle(samples)

    # Format all samples
    formatted = [format_alpaca(s) for s in samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "corpus.jsonl"
    with open(out_file, "w") as f:
        for s in formatted:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info(f"Corpus written to {out_file} ({len(formatted)} samples)")
    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deid-dir", default="training/data/deid")
    parser.add_argument("--output-dir", default="training/data/curated")
    parser.add_argument("--synthetic-count", type=int, default=5000)
    args = parser.parse_args()

    build_corpus(Path(args.deid_dir), Path(args.output_dir), args.synthetic_count)


if __name__ == "__main__":
    main()
