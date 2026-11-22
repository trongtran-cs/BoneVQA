# Qwen-VL-Med: Bone Radiograph VQA with Qwen3-VL

Visual question answering on **bone tumour (BTXRD)** and **fracture (FracAtlas)** radiographs.
A Qwen3-VL-8B-Instruct model is adapted with **LoRA**, returns **structured JSON answers with
bounding-box evidence**, and passes through a **confidence / abstention gate**.
Includes a **Gradio demo UI**.

> Research prototype. Not a medical device. No clinical claims.
> "Qwen-VL-Med" is the working name of this project's adaptation, not a released model.

## Features
- **Label-derived VQA**: question templates are paraphrased, but answers always come from dataset labels (never from an LLM).
- **Leakage-aware split**: near-duplicate images (average-hash groups) stay in one partition, stratified by class.
- **LoRA SFT** on LLM projections (vision encoder frozen), loss on answer tokens only.
- **Reliability gate**: option log-likelihood → temperature scaling → abstain below threshold τ; out-of-scope questions should abstain.
- **Evaluation**: accuracy, balanced accuracy, malignant sensitivity, Acc@IoU≥0.5, false-positive finding rate, ECE.
- **UI**: upload a radiograph, ask a question, see answer, evidence box, confidence.

## Layout
```
src/qwen_vl_med/  schema, data, vqa, split, model, train, reliability, infer, evaluate, cli
app/app.py        Gradio demo
configs/          training config
tests/            unit tests (no GPU needed)
```

## Quick start
```bash
pip install -r requirements.txt && pip install -e .

# 1. Download BTXRD and FracAtlas (check their licences) into data/raw/
qwen-vl-med build-data --btxrd data/raw/btxrd --fracatlas data/raw/fracatlas --out data/vqa

# 2. Train LoRA (needs a GPU with ~24-40 GB)
qwen-vl-med train --config configs/default.yaml

# 3. Predict + evaluate on the test split
qwen-vl-med predict --data data/vqa/vqa_test.jsonl --out outputs/preds.jsonl --adapter outputs/qwen-vl-med-lora
qwen-vl-med evaluate outputs/preds.jsonl

# 4. Demo UI
python app/app.py --adapter outputs/qwen-vl-med-lora     # or: python app/app.py --mock
```
Run tests: `pytest`.

## Output contract
```json
{"answer": "malignant", "boxes": [[210, 340, 560, 720]], "finding": "", "status": "answered"}
```
Boxes use Qwen3-VL's normalised `[0, 1000]` coordinates.

## Known TODOs
- Verify BTXRD / FracAtlas column names in `data.py` against your download.
- Replace exact-hash grouping with embedding-based near-duplicate grouping.
- Fit temperature and τ on the validation split (`reliability.py`), then pass `--temperature/--tau`.
- Add baselines (zero-shot Qwen3-VL, MedGemma, Lingshu) and ablations.
- Check dataset licences before redistributing derived data.
