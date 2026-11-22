"""Load Qwen3-VL (optionally with a LoRA adapter) and run generation."""
from __future__ import annotations

from PIL import Image

from .schema import SYSTEM_PROMPT

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def load_model(model_id: str = DEFAULT_MODEL, adapter: str | None = None, max_visual_tokens: int = 1280):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id, max_pixels=max_visual_tokens * 32 * 32  # one token per 32x32 px
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


def build_messages(question: str, image: Image.Image | None = None, answer: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": ([{"type": "image", "image": image}] if image is not None else [])
            + [{"type": "text", "text": question}],
        },
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def generate(model, processor, image: Image.Image, question: str, max_new_tokens: int = 256) -> str:
    """Greedy decoding (used for evaluation and the demo)."""
    import torch

    inputs = processor.apply_chat_template(
        build_messages(question, image), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True)[0].strip()
