"""Leakage-aware split: near-duplicate images stay in the same partition.

Simple version: group by 8x8 average hash (exact match). Replace `group_key` with an
embedding-based grouping later (thesis open decision: grouping encoder + thresholds).
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from PIL import Image


def average_hash(path: str, size: int = 8) -> str:
    with Image.open(path) as im:
        px = np.asarray(im.convert("L").resize((size, size))).ravel()
    return "".join("1" if p > px.mean() else "0" for p in px)


def group_key(rec: dict) -> str:
    try:
        return average_hash(rec["image"])
    except Exception:  # unreadable image -> unique group
        return rec["image"]


def stratum(rec: dict) -> str:
    cls = "malignant" if rec.get("malignant") else "benign" if rec.get("tumour") else (
        "fracture" if rec.get("fractured") else "normal")
    return f"{rec['dataset']}|{cls}"


def grouped_stratified_split(
    records: list[dict], fractions=(0.70, 0.15, 0.15), seed: int = 0, key=group_key
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    by_stratum: dict[str, list[list[dict]]] = defaultdict(list)
    for g in groups.values():
        by_stratum[stratum(g[0])].append(g)

    names = ("train", "val", "test")
    out: dict[str, list[dict]] = {n: [] for n in names}
    for glist in by_stratum.values():
        rng.shuffle(glist)
        total = sum(len(g) for g in glist)
        targets = [f * total for f in fractions]
        counts = [0, 0, 0]
        for g in glist:  # assign to the split furthest below its target
            i = max(range(3), key=lambda k: targets[k] - counts[k])
            out[names[i]].extend(g)
            counts[i] += len(g)
    return out
