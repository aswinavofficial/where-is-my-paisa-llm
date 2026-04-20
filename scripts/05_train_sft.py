#!/usr/bin/env python3
"""
05_train_sft.py — Command-line SFT training runner (non-Colab).

Loads config YAML, runs QLoRA SFT with unsloth + TRL SFTTrainer.
For Colab, use the notebooks in notebooks/. This script is for
automated/CI pipeline runs on A100/4090 instances.

Usage: python 05_train_sft.py --config training/configs/sft_1b.yaml
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


def run_training(cfg: dict):
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    import json

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    export_cfg = cfg.get("export", {})

    log.info(f"Loading model: {model_cfg['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
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
    model.print_trainable_parameters()

    # Load dataset
    train_data = []
    with open(data_cfg["train_file"]) as f:
        for line in f:
            train_data.append(json.loads(line))
    dataset = Dataset.from_list(train_data)
    log.info(f"Training samples: {len(dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=data_cfg.get("text_field", "text"),
        max_seq_length=model_cfg["max_seq_length"],
        packing=train_cfg.get("packing", False),
        args=TrainingArguments(
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            warmup_steps=train_cfg.get("warmup_steps", 10),
            num_train_epochs=train_cfg["num_train_epochs"],
            learning_rate=train_cfg["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=train_cfg.get("logging_steps", 10),
            optim=train_cfg.get("optim", "adamw_8bit"),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            seed=train_cfg.get("seed", 42),
            output_dir=train_cfg["output_dir"],
            save_strategy=train_cfg.get("save_strategy", "epoch"),
            report_to="none",
        ),
    )

    log.info("Starting training...")
    start = time.time()
    stats = trainer.train()
    elapsed = time.time() - start
    log.info(f"Training complete in {elapsed/60:.1f}min. Loss: {stats.metrics['train_loss']:.4f}")

    # Save
    output_dir = Path(train_cfg["output_dir"])
    model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "lora_adapter")
    log.info(f"LoRA adapter saved to {output_dir / 'lora_adapter'}")

    # Export GGUF if configured
    if export_cfg.get("quantization_methods"):
        merged_dir = output_dir / "merged_float16"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        gguf_dir = Path(export_cfg.get("gguf_output_dir", str(output_dir / "gguf")))
        gguf_dir.mkdir(parents=True, exist_ok=True)
        for quant in export_cfg["quantization_methods"]:
            model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)
            log.info(f"GGUF {quant.upper()} saved to {gguf_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info(f"Config: {args.config}")
    run_training(cfg)


if __name__ == "__main__":
    main()
