"""Confidence, calibration and abstention (pure numpy except option scoring).

Pipeline: option log-likelihoods -> temperature-scaled softmax -> confidence.
Abstain when confidence < tau, where tau is the lowest threshold whose validation
selective risk is <= rho.
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=float) / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(scores: np.ndarray, labels: np.ndarray, grid=np.linspace(0.05, 5, 100)) -> float:
    """Pick T minimising NLL on the validation set. scores: (N, K), labels: (N,)."""
    best_t, best = 1.0, np.inf
    for t in grid:
        p = softmax(scores, t)[np.arange(len(labels)), labels]
        nll = -np.log(np.clip(p, 1e-12, None)).mean()
        if nll < best:
            best, best_t = nll, float(t)
    return best_t


def select_threshold(conf: np.ndarray, correct: np.ndarray, rho: float = 0.10) -> float:
    """Lowest tau with selective risk (error rate among kept answers) <= rho."""
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=bool)
    for tau in np.unique(conf):  # ascending
        kept = conf >= tau
        if kept.any() and 1 - correct[kept].mean() <= rho:
            return float(tau)
    return 1.01  # nothing meets the risk target -> abstain on everything


def option_scores(model, processor, image, question: str, options: list[str]) -> np.ndarray:
    """Length-normalised log-likelihood of each candidate answer JSON."""
    import torch

    from .model import build_messages
    from .schema import dump_target

    prompt = processor.apply_chat_template(
        build_messages(question, image), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    n_prompt = prompt["input_ids"].shape[1]
    scores = []
    for opt in options:
        full = processor.apply_chat_template(
            build_messages(question, image, dump_target(opt)), tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**full).logits[0, :-1]
        ids = full["input_ids"][0, 1:]
        lp = torch.log_softmax(logits.float(), -1).gather(-1, ids[:, None])[:, 0]
        scores.append(lp[n_prompt - 1:].mean().item())
    return np.array(scores)
