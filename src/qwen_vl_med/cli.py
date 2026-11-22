"""Command line: build-data | train | predict-eval | evaluate."""
from __future__ import annotations

import argparse
import json

from .data import load_btxrd, load_fracatlas, read_jsonl, write_jsonl
from .split import grouped_stratified_split
from .vqa import build_vqa


def build_data(args):
    records = []
    if args.btxrd:
        records += load_btxrd(args.btxrd)
    if args.fracatlas:
        records += load_fracatlas(args.fracatlas)
    splits = grouped_stratified_split(records, seed=args.seed)
    for name, recs in splits.items():
        write_jsonl(build_vqa(recs, seed=args.seed), f"{args.out}/vqa_{name}.jsonl")
        print(name, len(recs), "images")


def predict_eval(args):
    from PIL import Image

    from .infer import MockPredictor, Predictor

    pred = MockPredictor() if args.mock else Predictor.from_pretrained(args.model, args.adapter, temperature=args.temperature, tau=args.tau)
    rows = []
    for r in read_jsonl(args.data):
        p = pred.predict(Image.open(r["image"]).convert("RGB"), r["question"])
        rows.append({**r, "pred_answer": p.answer, "pred_boxes": p.boxes, "pred_status": p.status, "confidence": p.confidence})
    write_jsonl(rows, args.out)


def main():
    ap = argparse.ArgumentParser(prog="qwen-vl-med")
    sub = ap.add_subparsers(required=True)

    b = sub.add_parser("build-data"); b.set_defaults(fn=build_data)
    b.add_argument("--btxrd"); b.add_argument("--fracatlas"); b.add_argument("--out", default="data/vqa"); b.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train"); t.add_argument("--config", default="configs/default.yaml")
    t.set_defaults(fn=lambda a: __import__("qwen_vl_med.train", fromlist=["main"]).main(a.config))

    p = sub.add_parser("predict"); p.set_defaults(fn=predict_eval)
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct"); p.add_argument("--adapter")
    p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--mock", action="store_true")

    e = sub.add_parser("evaluate"); e.add_argument("preds")
    e.set_defaults(fn=lambda a: print(json.dumps(__import__("qwen_vl_med.evaluate", fromlist=["evaluate"]).evaluate(read_jsonl(a.preds)), indent=2)))

    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
