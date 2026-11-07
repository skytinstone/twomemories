"""Visual token deletion.

"Deleting" a visual token means dropping its row from the input embeddings
before the language model sees it. The kept tokens keep their original M-RoPE
positions (taken from the full image grid), so the model sees a sparser image
at the same place. Text tokens are never touched. The vision encoder still runs
on the full image; only its output tokens are subsampled.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import Prepared

__all__ = ["keep_indices", "reduce"]


def keep_indices(n_vis: int, keep: float, method: str, seed: int = 0) -> np.ndarray:
    """Sorted indices (0..n_vis-1) of the visual tokens to keep."""
    if keep == 1.0 or method == "none":
        return np.arange(n_vis)
    k = max(1, int(round(n_vis * keep)))
    if method == "uniform":
        return np.unique(np.floor(np.arange(k) * (n_vis / k)).astype(np.int64))
    if method == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_vis, size=k, replace=False))
    raise ValueError(f"unknown token method {method!r}")


def reduce(inputs: Prepared, vis_span: tuple[int, int], keep: float, method: str, seed: int = 0):
    """Return (inputs', vis_span') with only the kept visual tokens."""
    s, e = vis_span
    n_vis = e - s
    T = inputs.embeds.shape[1]
    kept = keep_indices(n_vis, keep, method, seed)
    if len(kept) == n_vis:
        return inputs, vis_span
    idx = torch.cat([torch.arange(s), torch.as_tensor(kept) + s, torch.arange(e, T)]).to(inputs.embeds.device)
    out = Prepared(
        embeds=inputs.embeds.index_select(1, idx),
        pos=inputs.pos.index_select(2, idx),
        input_ids=inputs.input_ids.index_select(1, idx),
        vis_span=(s, s + len(kept)),
    )
    n_text_before, n_text_after = T - n_vis, out.embeds.shape[1] - len(kept)
    assert n_text_before == n_text_after, f"text tokens changed: {n_text_before} -> {n_text_after}"
    return out, out.vis_span
