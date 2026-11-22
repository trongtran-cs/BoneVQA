"""Metrics on a predictions JSONL (each row = gold sample fields + `pred_*` fields).

Per row required: family, gold_answer, gold_boxes, gold_status,
                  pred_answer, pred_boxes, pred_status, (optional) confidence
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .data import read_jsonl


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def acc_at_iou(pred_boxes, gold_boxes, thr: float = 0.5) -> bool:
    """Every gold box must be matched by some predicted box with IoU >= thr."""
    return bool(gold_boxes) and all(any(iou(p, g) >= thr for p in pred_boxes) for g in gold_boxes)


def balanced_accuracy(gold: list[str], pred: list[str]) -> float:
    recalls = []
    for c in set(gold):
        idx = [i for i, g in enumerate(gold) if g == c]
        recalls.append(sum(pred[i] == c for i in idx) / len(idx))
    return float(np.mean(recalls)) if recalls else float("nan")


def recall_of(gold: list[str], pred: list[str], cls: str) -> float:
    idx = [i for i, g in enumerate(gold) if g == cls]
    return sum(pred[i] == cls for i in idx) / len(idx) if idx else float("nan")


def evaluate(rows: list[dict]) -> dict:
    out: dict = {}
    closed = defaultdict(list)
    for r in rows:
        closed[r["family"]].append(r)

    for fam, rs in closed.items():
        if fam == "localisation":
            out[fam] = {"acc@iou0.5": float(np.mean([acc_at_iou(r["pred_boxes"], r["gold_boxes"]) for r in rs])), "n": len(rs)}
        elif fam == "out_of_scope":
            out[fam] = {"abstain_rate": float(np.mean([r["pred_status"] == "abstain" for r in rs])), "n": len(rs)}
        else:
            g, p = [r["gold_answer"] for r in rs], [r["pred_answer"].lower() for r in rs]
            out[fam] = {
                "accuracy": float(np.mean([a == b for a, b in zip(g, p)])),
                "balanced_accuracy": balanced_accuracy(g, p), "n": len(rs),
            }
            if fam == "tumour_nature":
                out[fam]["malignant_sensitivity"] = recall_of(g, p, "malignant")

    # hallucination proxy: boxes predicted where gold has none (negative images)
    neg = [r for r in rows if r["family"] in ("tumour_presence", "fracture_presence") and r["gold_answer"] == "no"]
    if neg:
        out["false_positive_finding_rate"] = float(np.mean([r["pred_answer"].lower() == "yes" or bool(r["pred_boxes"]) for r in neg]))

    # selective prediction on closed questions
    conf_rows = [r for r in rows if r.get("confidence") is not None and r["family"] not in ("localisation", "out_of_scope")]
    if conf_rows:
        c = np.array([r["confidence"] for r in conf_rows])
        ok = np.array([r["pred_answer"].lower() == r["gold_answer"] for r in conf_rows])
        kept = [r for r in conf_rows if r.get("pred_status") != "abstain"]
        out["selective"] = {"coverage": len(kept) / len(conf_rows), "ece_10bin": ece(c, ok)}
    return out


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def main(path: str):
    import json

    print(json.dumps(evaluate(read_jsonl(path)), indent=2))


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
