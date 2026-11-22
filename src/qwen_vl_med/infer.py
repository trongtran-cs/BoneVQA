"""High-level predictor with the reliability gate.

Gate: parse -> (closed question) calibrated confidence >= tau -> answer, else abstain.
`MockPredictor` lets the UI and tests run without GPU or weights.
"""
from __future__ import annotations

from PIL import Image

from .reliability import option_scores, softmax
from .schema import OPTIONS, Prediction, parse_response

FAMILY_HINTS = {  # crude keyword routing for free-text questions in the UI
    "tumour_nature": ("benign", "malignant"),
    "tumour_presence": ("tumour", "tumor"),
    "fracture_count": ("how many",),
    "fracture_presence": ("fracture",),
    "hardware": ("hardware", "implant", "screw", "plate"),
    "view": ("view",),
}


def guess_family(question: str) -> str | None:
    q = question.lower()
    for fam, keys in FAMILY_HINTS.items():
        if any(k in q for k in keys):
            return fam
    return None


class Predictor:
    def __init__(self, model, processor, temperature: float = 1.0, tau: float = 0.0):
        self.model, self.processor = model, processor
        self.temperature, self.tau = temperature, tau

    @classmethod
    def from_pretrained(cls, model_id: str, adapter: str | None = None, **kw):
        from .model import load_model

        m, p = load_model(model_id, adapter)
        return cls(m, p, **kw)

    def predict(self, image: Image.Image, question: str) -> Prediction:
        from .model import generate

        pred = parse_response(generate(self.model, self.processor, image, question))
        fam = guess_family(question)
        if fam in OPTIONS:
            probs = softmax(option_scores(self.model, self.processor, image, question, OPTIONS[fam]), self.temperature)
            pred.confidence = float(probs.max())
            if pred.confidence < self.tau:
                pred.status = "abstain"
        return pred


class MockPredictor:
    """Deterministic fake output so the UI can be demoed without a model."""

    def predict(self, image: Image.Image, question: str) -> Prediction:
        w, h = image.size
        return Prediction(
            answer="[mock] connect a real model to get answers", boxes=[[300, 300, 700, 700]],
            finding="demo output", status="answered", confidence=0.5,
            raw='{"answer": "mock"}',
        )
