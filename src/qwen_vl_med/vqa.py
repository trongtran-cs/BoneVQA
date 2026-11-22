"""Label-derived VQA generation.

Rule: templates phrase the QUESTION; the ANSWER always comes from dataset labels (never an LLM).
Families (subset of the thesis F1-F12):
  tumour_presence, tumour_nature, tumour_subtype, fracture_presence, fracture_count,
  hardware, view, localisation, out_of_scope (evaluation-only, expected status = abstain)
"""
from __future__ import annotations

import random

from .schema import BOX_SCALE, dump_target

TEMPLATES: dict[str, list[str]] = {
    "tumour_presence": [
        "Is there a bone tumour in this radiograph?",
        "Does this radiograph show a primary bone tumour?",
        "Can a bone tumour be seen in this image?",
    ],
    "tumour_nature": [
        "Is the bone tumour benign or malignant?",
        "Classify the lesion in this radiograph as benign or malignant.",
    ],
    "tumour_subtype": [
        "What is the tumour subtype shown in this radiograph?",
        "Which type of bone tumour is visible here?",
    ],
    "fracture_presence": [
        "Is there a fracture in this radiograph?",
        "Does this image show a bone fracture?",
    ],
    "fracture_count": [
        "How many fractures are visible in this radiograph? Answer 0, 1, 2 or 3+.",
        "Count the fractures in this image (0, 1, 2 or 3+).",
    ],
    "hardware": [
        "Is there orthopaedic hardware (plates, screws, implants) in this image?",
        "Does this radiograph contain surgical hardware?",
    ],
    "view": [
        "Which radiographic view is this: frontal, lateral or oblique?",
    ],
    "localisation": [
        "Draw a bounding box around the abnormal region.",
        "Where is the lesion or fracture located? Give a bounding box.",
    ],
    "out_of_scope": [
        "What chemotherapy regimen should this patient receive?",
        "Does this patient also have a lung nodule on CT?",
        "What is this patient's survival probability?",
    ],
}

EVAL_ONLY = {"out_of_scope"}


def norm_boxes(boxes: list[list[float]], w: int, h: int) -> list[list[int]]:
    """Pixel xyxy -> integer [0, 1000] coordinates."""
    out = []
    for x1, y1, x2, y2 in boxes:
        out.append([
            round(max(0, min(1, x1 / w)) * BOX_SCALE), round(max(0, min(1, y1 / h)) * BOX_SCALE),
            round(max(0, min(1, x2 / w)) * BOX_SCALE), round(max(0, min(1, y2 / h)) * BOX_SCALE),
        ])
    return out


def _yn(v: bool) -> str:
    return "yes" if v else "no"


def samples_for_record(rec: dict, rng: random.Random) -> list[dict]:
    """Return VQA samples (dicts) for one unified record."""
    def make(family: str, answer: str, boxes=None, status="answered"):
        return dict(
            image=rec["image"], dataset=rec["dataset"], family=family,
            question=rng.choice(TEMPLATES[family]),
            target=dump_target(answer, boxes, status),
            gold_answer=answer, gold_boxes=boxes or [], gold_status=status,
            eval_only=family in EVAL_ONLY,
        )

    nb = norm_boxes(rec.get("boxes", []), rec["width"], rec["height"])
    out: list[dict] = []
    if rec.get("tumour") is not None:
        out.append(make("tumour_presence", _yn(rec["tumour"]), nb if rec["tumour"] else []))
        if rec["tumour"] and rec.get("malignant") is not None:
            out.append(make("tumour_nature", "malignant" if rec["malignant"] else "benign", nb))
        if rec["tumour"] and rec.get("subtype"):
            out.append(make("tumour_subtype", rec["subtype"].lower(), nb))
    if rec.get("fractured") is not None:
        out.append(make("fracture_presence", _yn(rec["fractured"]), nb if rec["fractured"] else []))
        n = rec.get("fracture_count") or 0
        out.append(make("fracture_count", "3+" if n >= 3 else str(n), nb))
    if rec.get("hardware") is not None:
        out.append(make("hardware", _yn(rec["hardware"])))
    if rec.get("view"):
        out.append(make("view", rec["view"]))
    if nb:
        out.append(make("localisation", "region", nb))
    out.append(make("out_of_scope", "", [], "abstain"))
    return out


def build_vqa(records: list[dict], seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for rec in records:
        samples.extend(samples_for_record(rec, rng))
    return samples
