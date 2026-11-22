"""Dataset adapters: BTXRD and FracAtlas -> one unified record format (JSONL).

Unified record keys:
  image (abs path), dataset, width, height, boxes (pixel xyxy list),
  tumour (bool|None), malignant (bool|None), subtype (str|None), site (str|None),
  fractured (bool|None), fracture_count (int|None), hardware (bool|None), view (str|None)

NOTE: column names below follow the public releases as best known. Check them against
your downloaded copy and adjust the constants at the top if they differ.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


# ---------------------------------------------------------------- FracAtlas
def load_fracatlas(root: str | Path) -> list[dict]:
    """Expects `dataset.csv` + `images/` + COCO json with fracture boxes."""
    import csv

    root = Path(root)
    coco_files = list(root.rglob("COCO_fracture_masks.json")) or list(root.rglob("*.json"))
    boxes_by_name: dict[str, list[list[float]]] = {}
    if coco_files:
        coco = json.loads(coco_files[0].read_text())
        id2name = {i["id"]: i["file_name"] for i in coco["images"]}
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            boxes_by_name.setdefault(id2name[a["image_id"]], []).append([x, y, x + w, y + h])

    index = {p.name: p for p in root.rglob("*.jpg")}
    records = []
    with open(next(root.rglob("dataset.csv")), newline="") as f:
        for row in csv.DictReader(f):
            name = row["image_id"]
            path = index.get(name)
            if path is None:
                continue
            w, h = _size(path)
            view = next((v for v in ("frontal", "lateral", "oblique") if row.get(v) == "1"), None)
            records.append(
                dict(
                    image=str(path), dataset="fracatlas", width=w, height=h,
                    boxes=boxes_by_name.get(name, []),
                    tumour=None, malignant=None, subtype=None, site=None,
                    fractured=row.get("fractured") == "1",
                    fracture_count=int(row.get("fracture_count") or 0),
                    hardware=row.get("hardware") == "1",
                    view=view,
                )
            )
    return records


# -------------------------------------------------------------------- BTXRD
def load_btxrd(root: str | Path) -> list[dict]:
    """Expects `dataset.xlsx` (needs openpyxl), `images/`, and LabelMe-style `Annotations/*.json`."""
    import openpyxl

    root = Path(root)
    wb = openpyxl.load_workbook(next(root.rglob("dataset.xlsx")), read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    header = [str(c).strip().lower() for c in rows[0]]
    col = {name: i for i, name in enumerate(header)}
    index = {p.name: p for p in root.rglob("*.jpeg")} | {p.name: p for p in root.rglob("*.jpg")}
    ann_dir = next((p for p in root.rglob("Annotations") if p.is_dir()), None)

    def flag(r, key):
        return bool(r[col[key]]) if key in col and r[col[key]] not in (None, "") else None

    records = []
    for r in rows[1:]:
        name = str(r[col["image_id"]])
        path = index.get(name)
        if path is None:
            continue
        w, h = _size(path)
        boxes = []
        if ann_dir and (ann_dir / f"{path.stem}.json").exists():
            for shp in json.loads((ann_dir / f"{path.stem}.json").read_text()).get("shapes", []):
                xs, ys = zip(*shp["points"])
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
        tumour = flag(r, "tumor")
        records.append(
            dict(
                image=str(path), dataset="btxrd", width=w, height=h, boxes=boxes,
                tumour=tumour, malignant=flag(r, "malignant"),
                subtype=(str(r[col["label"]]) if "label" in col and r[col["label"]] else None),
                site=(str(r[col["site"]]) if "site" in col and r[col["site"]] else None),
                fractured=None, fracture_count=None, hardware=None, view=None,
            )
        )
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
