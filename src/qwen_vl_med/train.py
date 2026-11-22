"""LoRA SFT on label-derived VQA targets (loss on assistant tokens only)."""
from __future__ import annotations

import yaml
from PIL import Image

from .data import read_jsonl
from .model import build_messages


class VQADataset:
    def __init__(self, path: str, include_eval_only: bool = False):
        self.rows = [r for r in read_jsonl(path) if include_eval_only or not r["eval_only"]]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """Tokenise one sample at a time; mask everything before the assistant answer."""

    def __init__(self, processor):
        self.p = processor

    def __call__(self, batch):
        import torch

        feats = []
        for r in batch:
            img = Image.open(r["image"]).convert("RGB")
            full = self.p.apply_chat_template(
                build_messages(r["question"], img, r["target"]), tokenize=True,
                return_dict=True, return_tensors="pt")
            prompt = self.p.apply_chat_template(
                build_messages(r["question"], img), tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            labels = full["input_ids"].clone()
            labels[:, : prompt["input_ids"].shape[1]] = -100
            full["labels"] = labels
            feats.append(full)
        if len(feats) != 1:
            raise ValueError("Use per_device_train_batch_size=1 and gradient accumulation.")
        return feats[0]


def main(config_path: str):
    import torch  # noqa: F401
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from .model import load_model

    cfg = yaml.safe_load(open(config_path))
    model, processor = load_model(cfg["model_id"], max_visual_tokens=cfg["max_visual_tokens"])
    # Vision encoder stays frozen: LoRA only targets the LLM projections.
    lora = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=2 * cfg["lora"]["r"], lora_dropout=cfg["lora"]["dropout"],
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=cfg["output_dir"], per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"], bf16=True, logging_steps=10, save_strategy="epoch",
        remove_unused_columns=False, gradient_checkpointing=True, report_to="none",
        seed=cfg["seed"],
    )
    Trainer(
        model=model, args=args, train_dataset=VQADataset(cfg["train_file"]),
        data_collator=Collator(processor),
    ).train()
    model.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
