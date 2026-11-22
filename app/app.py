"""Gradio demo: upload a radiograph, ask a question, see answer + evidence box + confidence.

  python app/app.py --mock                          # UI only, no GPU
  python app/app.py --adapter outputs/qwen-vl-med-lora
"""
from __future__ import annotations

import argparse

import gradio as gr
from PIL import Image, ImageDraw

from qwen_vl_med.infer import MockPredictor, Predictor

DISCLAIMER = "Research prototype. Not a medical device. Do not use for diagnosis or treatment decisions."
EXAMPLES = [
    "Is there a fracture in this radiograph?",
    "Is the bone tumour benign or malignant?",
    "Is there orthopaedic hardware in this image?",
    "Draw a bounding box around the abnormal region.",
]


def draw_boxes(img: Image.Image, boxes) -> Image.Image:
    img = img.convert("RGB").copy()
    d, (w, h) = ImageDraw.Draw(img), img.size
    for x1, y1, x2, y2 in boxes:
        d.rectangle([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h], outline="red", width=max(2, w // 200))
    return img


def build_ui(predictor) -> gr.Blocks:
    def run(image, question):
        if image is None or not question.strip():
            return None, "Please upload an image and enter a question.", "", ""
        p = predictor.predict(image, question)
        if p.status == "abstain":
            ans = "⚠️ The model abstains: low confidence or out of scope. Refer to a specialist."
        else:
            ans = p.answer
        conf = f"{p.confidence:.2f}" if p.confidence is not None else "n/a"
        return draw_boxes(image, p.boxes), ans, conf, p.raw

    with gr.Blocks(title="Qwen-VL-Med") as ui:
        gr.Markdown(f"# Qwen-VL-Med: Bone Radiograph VQA\n{DISCLAIMER}")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Radiograph")
                q = gr.Textbox(label="Question", value=EXAMPLES[0])
                gr.Examples(EXAMPLES, inputs=q)
                btn = gr.Button("Ask", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Evidence (predicted boxes)")
                ans = gr.Textbox(label="Answer")
                conf = gr.Textbox(label="Confidence")
                raw = gr.Textbox(label="Raw model output", lines=4)
        btn.click(run, [img, q], [out_img, ans, conf, raw])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.0)
    a = ap.parse_args()
    pred = MockPredictor() if a.mock else Predictor.from_pretrained(a.model, a.adapter, temperature=a.temperature, tau=a.tau)
    build_ui(pred).launch()
