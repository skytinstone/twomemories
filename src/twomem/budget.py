"""Memory accounting (a bookkeeping formula, not a measurement).

    MB = KV(n_vis + n_text, kv_bits) + State(state_bits)

KV(T, b)  = b * 2 * L_attn * H_kv * head_dim * T bits
State(b)  = b * L_gdn * H_v * d_k * d_v bits, independent of the token count
MB = 1e6 bytes. Element counts come from ``hybridquant.budget.cache_sizes``
(conv1d state and quantization scales/zero points are not counted).

Bits per level: KV full = 16 (bf16 cache), state full = 32 (the cache stores
fp32, premise 3), quantized = its bit-width, noise = the matched bit-width
(4 for KV, 8 for state), zero / drop / zeroconv = 0. All KV rows are charged at kv_bits, also
under scope="image" where the text rows are actually left at 16 bits; this is
the deployment accounting used for the iso-budget comparison (Table 5).
"""
from __future__ import annotations

from hybridquant.budget import cache_sizes

from .quant import KV_NOISE_BITS, STATE_NOISE_BITS, Condition

__all__ = ["KV_FULL_BITS", "STATE_FULL_BITS", "kv_bits", "state_bits", "mb"]

KV_FULL_BITS = 16
STATE_FULL_BITS = 32


def kv_bits(level: str) -> int:
    named = {"full": KV_FULL_BITS, "noise": KV_NOISE_BITS, "zero": 0, "drop": 0}
    return named[level] if level in named else int(level)


def state_bits(level: str) -> int:
    if level in ("full", "noise", "zero", "zeroconv"):
        return {"full": STATE_FULL_BITS, "noise": STATE_NOISE_BITS, "zero": 0, "zeroconv": 0}[level]
    return int(level[0])


def mb(cond: Condition, n_vis: int, n_text: int, model_cfg) -> dict:
    """``n_vis`` is the number of visual tokens actually kept. ``model_cfg`` is the text config."""
    sz = cache_sizes(model_cfg)
    kv = kv_bits(cond.kv) * sz["kv_per_token"] * (n_vis + n_text) / 8 / 1e6
    st = state_bits(cond.state) * sz["state_elems"] / 8 / 1e6
    return {"kv_mb": kv, "state_mb": st, "total_mb": kv + st}
