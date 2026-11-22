"""Shared output contract: every answer is a JSON object.

{"answer": str, "boxes": [[x1, y1, x2, y2], ...], "finding": str, "status": "answered" | "abstain"}

Boxes are normalised to [0, 1000] (Qwen3-VL grounding convention).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

BOX_SCALE = 1000

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about plain bone radiographs. "
    "Reply ONLY with a JSON object with keys: "
    '"answer" (string), "boxes" (list of [x1,y1,x2,y2] in 0-1000 coordinates), '
    '"finding" (short string), "status" ("answered" or "abstain"). '
    "If the question cannot be answered from the image, set status to abstain. "
    "This is a research tool, not a medical device."
)

# Closed answer sets per question family (used for option scoring / confidence).
OPTIONS: dict[str, list[str]] = {
    "tumour_presence": ["yes", "no"],
    "tumour_nature": ["benign", "malignant"],
    "fracture_presence": ["yes", "no"],
    "hardware": ["yes", "no"],
    "fracture_count": ["0", "1", "2", "3+"],
    "view": ["frontal", "lateral", "oblique"],
}


@dataclass
class Prediction:
    answer: str = ""
    boxes: list[list[float]] = field(default_factory=list)
    finding: str = ""
    status: str = "answered"
    confidence: float | None = None
    raw: str = ""
    parse_ok: bool = True


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> Prediction:
    """Fixed, forgiving parser. Never raises; sets parse_ok=False on failure."""
    pred = Prediction(raw=text)
    m = _JSON_RE.search(text or "")
    if not m:
        pred.parse_ok = False
        pred.answer = (text or "").strip()
        return pred
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        pred.parse_ok = False
        pred.answer = (text or "").strip()
        return pred
    pred.answer = str(obj.get("answer", "")).strip()
    pred.finding = str(obj.get("finding", "")).strip()
    pred.status = "abstain" if str(obj.get("status", "answered")).lower() == "abstain" else "answered"
    boxes = []
    for b in obj.get("boxes", []) or []:
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue
        boxes.append([x1, y1, x2, y2])
    pred.boxes = boxes
    return pred


def dump_target(answer: str, boxes: list[list[int]] | None = None, status: str = "answered") -> str:
    """Serialise a training target. `finding` is intentionally left empty (label-derived only)."""
    return json.dumps(
        {"answer": answer, "boxes": boxes or [], "finding": "", "status": status},
        ensure_ascii=False,
    )
