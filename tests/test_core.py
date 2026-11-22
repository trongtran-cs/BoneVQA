import random

import numpy as np
from PIL import Image

from qwen_vl_med.evaluate import acc_at_iou, balanced_accuracy, evaluate, iou
from qwen_vl_med.reliability import fit_temperature, select_threshold, softmax
from qwen_vl_med.schema import parse_response
from qwen_vl_med.split import grouped_stratified_split
from qwen_vl_med.vqa import norm_boxes, samples_for_record


def test_parse_ok_and_fallback():
    p = parse_response('noise {"answer": "yes", "boxes": [[1,2,3,4]], "status": "answered"} tail')
    assert p.parse_ok and p.answer == "yes" and p.boxes == [[1, 2, 3, 4]]
    assert not parse_response("not json").parse_ok


def test_norm_boxes():
    assert norm_boxes([[0, 0, 50, 100]], 100, 200) == [[0, 0, 500, 500]]


def test_vqa_from_labels():
    rec = dict(image="x.jpg", dataset="btxrd", width=100, height=100, boxes=[[10, 10, 50, 50]],
               tumour=True, malignant=True, subtype="osteosarcoma", fractured=None)
    fams = {s["family"]: s for s in samples_for_record(rec, random.Random(0))}
    assert fams["tumour_nature"]["gold_answer"] == "malignant"
    assert fams["out_of_scope"]["gold_status"] == "abstain" and fams["out_of_scope"]["eval_only"]
    assert fams["localisation"]["gold_boxes"] == [[100, 100, 500, 500]]


def test_iou_and_metrics():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert not acc_at_iou([[0, 0, 1, 1]], [[5, 5, 10, 10]])
    assert balanced_accuracy(["a", "a", "b"], ["a", "b", "b"]) == 0.75


def test_evaluate_runs():
    rows = [dict(family="tumour_nature", gold_answer="malignant", pred_answer="benign", gold_boxes=[], pred_boxes=[],
                 gold_status="answered", pred_status="answered", confidence=0.9),
            dict(family="tumour_nature", gold_answer="benign", pred_answer="benign", gold_boxes=[], pred_boxes=[],
                 gold_status="answered", pred_status="answered", confidence=0.6)]
    r = evaluate(rows)
    assert r["tumour_nature"]["malignant_sensitivity"] == 0.0 and "selective" in r


def test_reliability():
    assert abs(softmax(np.array([[1.0, 1.0]]))[0, 0] - 0.5) < 1e-9
    t = fit_temperature(np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([0, 1]))
    assert t > 0
    conf = np.array([0.5, 0.6, 0.9, 0.95]); ok = np.array([0, 0, 1, 1])
    assert select_threshold(conf, ok, rho=0.0) == 0.9


def test_split_keeps_groups(tmp_path):
    recs = []
    for i in range(40):
        p = tmp_path / f"{i}.png"
        Image.fromarray((np.random.RandomState(i % 20).rand(16, 16) * 255).astype("uint8")).save(p)  # duplicates
        recs.append(dict(image=str(p), dataset="btxrd", tumour=True, malignant=i % 4 == 0))
    s = grouped_stratified_split(recs)
    where = {}
    for name, rs in s.items():
        for r in rs:
            where.setdefault(r["image"].split("/")[-1].split(".")[0] and (int(r["image"].split("/")[-1][:-4]) % 20), set()).add(name)
    assert all(len(v) == 1 for v in where.values())
    assert sum(len(v) for v in s.values()) == 40
