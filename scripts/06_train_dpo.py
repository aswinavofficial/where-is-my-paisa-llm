#!/usr/bin/env python3
"""
06_train_dpo.py — DPO alignment training runner (non-Colab).

Run after SFT to improve tone, safety, and brevity via Direct Preference Optimization.
See research/finetune.md §7.3 for when to run DPO.

Usage: python 06_train_dpo.py --config training/configs/dpo_1b.yaml
"""
import argparse
import json
import logging
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_dpo(cfg: dict):
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset
    import json

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    dpo_cfg = cfg.get("dpo", {})
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    log.info(f"Loading SFT model: {model_cfg['sft_model_path']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["sft_model_path"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=42,
    )

    pref_data = []
    with open(data_cfg["preference_file"]) as f:
        for line in f:
            pref_data.append(json.loads(line))
    dataset = Dataset.from_list(pref_data)
    log.info(f"Preference pairs: {len(dataset)}")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            num_train_epochs=train_cfg["num_train_epochs"],
            learning_rate=train_cfg["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=train_cfg.get("logging_steps", 5),
            optim=train_cfg.get("optim", "adamw_8bit"),
            output_dir=train_cfg["output_dir"],
            report_to="none",
            max_length=dpo_cfg.get("max_length", 1024),
            max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
            beta=dpo_cfg.get("beta", 0.1),
        ),
    )

    log.info("Starting DPO training...")
    start = time.time()
    trainer.train()
    log.info(f"DPO training complete in {(time.time()-start)/60:.1f}min")

    output_dir = Path(train_cfg["output_dir"])
    model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "lora_adapter")
    log.info(f"DPO adapter saved to {output_dir / 'lora_adapter'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dpo(cfg)


if __name__ == "__main__":
    main()
