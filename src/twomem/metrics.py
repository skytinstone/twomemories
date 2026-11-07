"""Teacher-forced gold NLL, task scores, and the dataset-level retained ratio.

retained = (NLL_blind - NLL_cond) / (NLL_blind - NLL_full), each NLL a dataset
mean of per-sample token-mean gold NLL. It is defined at the dataset level only
(no per-sample ratio is computed anywhere); its CI is a paired bootstrap that
resamples the same sample indices for all three arrays.
"""
from __future__ import annotations

import re
import string

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["gold_nll", "normalize", "anls", "relaxed_acc", "vqa_acc", "pope_label", "pope_f1", "score",
           "retained", "paired_bootstrap"]


@torch.no_grad()
def gold_nll(logits: torch.Tensor, gold_ids: torch.Tensor) -> tuple[float, int]:
    """Sum of NLL of ``gold_ids`` [n] under ``logits`` [n, V] (row i predicts gold i) and the token count."""
    nll = F.cross_entropy(logits.float(), gold_ids.to(logits.device), reduction="sum")
    return float(nll), int(gold_ids.numel())


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile("[" + re.escape(string.punctuation.replace(".", "").replace("%", "")) + "]")


def normalize(s: str) -> str:
    """The single answer normalizer used by every metric."""
    s = str(s).lower().strip()
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)   # drop periods except decimal points
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, answers: list[str], tau: float = 0.5) -> float:
    p = normalize(pred)
    best = 0.0
    for a in answers:
        g = normalize(a)
        if not p and not g:
            return 1.0
        nld = _levenshtein(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - nld if nld < tau else 0.0)
    return best


def _to_float(s: str) -> float | None:
    s = normalize(s).replace(",", "").replace("$", "").strip()
    pct = s.endswith("%")
    try:
        v = float(s.rstrip("%").strip())
    except ValueError:
        return None
    return v / 100 if pct else v


def relaxed_acc(pred: str, answers: list[str], tol: float = 0.05) -> float:
    """ChartQA relaxed accuracy: numbers within 5% relative error, otherwise exact match."""
    for a in answers:
        pv, gv = _to_float(pred), _to_float(a)
        if pv is not None and gv is not None:
            ok = pv == gv if gv == 0 else abs(pv - gv) / abs(gv) <= tol
        else:
            ok = normalize(pred) == normalize(a)
        if ok:
            return 1.0
    return 0.0


def vqa_acc(pred: str, answers: list[str]) -> float:
    """VQA accuracy min(#matching humans / 3, 1), averaged over the 10 leave-one-out subsets."""
    p = normalize(pred)
    gts = [normalize(a) for a in answers]
    if len(gts) < 2:
        return float(p in gts)
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(g == p for g in others) / 3))
    return float(np.mean(accs))


def pope_label(pred: str) -> str:
    p = normalize(pred)
    if p.startswith("yes"):
        return "yes"
    if p.startswith("no"):
        return "no"
    return "other"


def pope_f1(preds: list[str], golds: list[str]) -> dict:
    """Dataset-level POPE precision/recall/F1 with "yes" as the positive class."""
    lab = [pope_label(p) for p in preds]
    g = [normalize(x) for x in golds]
    tp = sum(l == "yes" and y == "yes" for l, y in zip(lab, g))
    fp = sum(l == "yes" and y == "no" for l, y in zip(lab, g))
    fn = sum(l != "yes" and y == "yes" for l, y in zip(lab, g))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "acc": float(np.mean([l == y for l, y in zip(lab, g)]))}


def score(task: str, pred: str, answers: list[str]) -> float:
    """Per-sample score. POPE: 1 if the yes/no label matches (F1 is dataset-level, see pope_f1)."""
    if task == "docvqa":
        return anls(pred, answers)
    if task == "chartqa":
        return relaxed_acc(pred, answers)
    if task == "textvqa":
        return vqa_acc(pred, answers)
    if task == "pope":
        return float(pope_label(pred) == normalize(answers[0]))
    raise KeyError(task)


def retained(nll_blind_mean: float, nll_full_mean: float, nll_cond_mean: float) -> float:
    den = nll_blind_mean - nll_full_mean
    if den == 0:
        return float("nan")
    return (nll_blind_mean - nll_cond_mean) / den


def paired_bootstrap(nll_blind, nll_full, nll_cond, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Point estimate and percentile CI of the dataset-level retained ratio.

    The three arrays hold per-sample token-mean gold NLL for the same samples in the same order.
    """
    b, f, c = (np.asarray(x, dtype=np.float64) for x in (nll_blind, nll_full, nll_cond))
    if not (len(b) == len(f) == len(c)) or len(b) == 0:
        raise ValueError("paired arrays must be non-empty and of equal length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n, len(b)))
    bm, fm, cm = b[idx].mean(1), f[idx].mean(1), c[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = (bm - cm) / (bm - fm)
    boots = boots[np.isfinite(boots)]
    point = retained(b.mean(), f.mean(), c.mean())
    lo, hi = (np.quantile(boots, [alpha / 2, 1 - alpha / 2]).tolist() if len(boots) else (float("nan"),) * 2)
    return {"retained": point, "lo": lo, "hi": hi, "n": len(b), "n_boot": int(len(boots))}
