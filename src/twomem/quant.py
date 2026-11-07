"""Interventions on the two memories, built on hybridquant (imported, never copied).

scope="image": the image is read with clean compute (stage-1 prefill), then the
memory it left behind is altered once, right after the last image token:
  * KV:    the rows of the visual tokens in every attention layer's cache.
  * state: the recurrent state of every GDN layer.
The question (stage-2 prefill) and decoding read the altered memory, unhooked.
``8ef``/``6ef`` split the image into chunks of ``EF_CHUNK`` visual tokens and
store Q(S + r) at every chunk boundary, carrying r = (S + r) - Q(S + r) to the
next boundary (premise 5: the library writes the state once per forward, so
chunks are separate forward calls; see docs/assumptions.md).

``drop`` (KV) removes the visual rows from every attention layer's cache instead of zeroing them, so no
softmax mass goes to them; the session's token count is reduced by n_vis so the causal mask stays aligned,
and M-RoPE positions of later tokens are unchanged. ``zeroconv`` (state) zeroes the recurrent state and the
conv1d cache of every GDN layer, so no image information survives in the GDN layers at all.

scope="whole": hybridquant's store-time hooks on every write of the whole run
(prefill and every decode step), i.e. the setting of the first paper.

The conv1d state (last ``kernel_size`` inputs) is only touched by ``zeroconv``; elsewhere it is kept, as in
hybridquant.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass

import torch
from hybridquant import HybridQuantConfig, QuantSpec, QuantStats, fake_quant, quantized_cache

__all__ = ["Condition", "KV_LEVELS", "STATE_LEVELS", "EF_CHUNK", "GROUP", "NOISE_SEED", "Intervention", "apply"]

KV_LEVELS = ("full", "4", "noise", "zero", "drop")
STATE_LEVELS = ("full", "8", "8ef", "6", "6ef", "noise", "zero", "zeroconv")
EF_CHUNK = 256      # visual tokens per prefill chunk for error feedback
GROUP = 128         # quantization group along the last dim (hybridquant default)
NOISE_SEED = 0
KV_NOISE_BITS = 4   # noise matches the RTN error RMS at these bits
STATE_NOISE_BITS = 8


@dataclass(frozen=True)
class Condition:
    kv: str = "full"
    state: str = "full"
    token_keep: float = 1.0
    token_method: str = "none"
    scope: str = "image"

    def __post_init__(self):
        if self.kv not in KV_LEVELS:
            raise ValueError(f"kv={self.kv!r} not in {KV_LEVELS}")
        if self.state not in STATE_LEVELS:
            raise ValueError(f"state={self.state!r} not in {STATE_LEVELS}")
        if self.token_keep not in (1.0, 0.5, 0.25):
            raise ValueError(f"token_keep={self.token_keep} not in (1.0, 0.5, 0.25)")
        if self.token_method not in ("none", "uniform", "random"):
            raise ValueError(f"token_method={self.token_method!r}")
        if (self.token_keep == 1.0) != (self.token_method == "none"):
            raise ValueError("token_method must be 'none' exactly when token_keep == 1.0")
        if self.scope not in ("image", "whole"):
            raise ValueError(f"scope={self.scope!r}")
        if self.scope == "whole" and {self.kv, self.state} & {"zero", "drop", "zeroconv"}:
            raise ValueError("scope='whole' supports bit-width and noise conditions only")

    @property
    def is_full(self) -> bool:
        return self.kv == "full" and self.state == "full" and self.token_keep == 1.0

    def label(self) -> str:
        tok = "" if self.token_keep == 1.0 else f"_T{int(self.token_keep * 100)}{self.token_method[0]}"
        return f"KV{self.kv}_S{self.state}{tok}_{self.scope}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Condition":
        d = dict(d)
        d["kv"], d["state"] = str(d.get("kv", "full")), str(d.get("state", "full"))
        d["token_keep"] = float(d.get("token_keep", 1.0))
        return cls(**d)


def _state_spec(level: str) -> QuantSpec:
    if level in ("full", "zero", "zeroconv"):
        return QuantSpec()
    if level == "noise":
        return QuantSpec(STATE_NOISE_BITS, GROUP, noise_match=True)
    return QuantSpec(int(level[0]), GROUP, error_feedback=level.endswith("ef"))


def _kv_spec(level: str) -> QuantSpec:
    if level == "4":
        return QuantSpec(4, GROUP)
    if level == "noise":
        return QuantSpec(KV_NOISE_BITS, GROUP, noise_match=True)
    return QuantSpec()


def _whole_config(cond: Condition) -> HybridQuantConfig:
    kv, st = _kv_spec(cond.kv), _state_spec(cond.state)
    return HybridQuantConfig(key=kv, value=kv, state=st)


def _gen(device, seed: int):
    return torch.Generator(device=torch.device(device).type).manual_seed(seed)


class Intervention:
    """Drives stage-1 prefill and the one-shot memory edit for one sample."""

    def __init__(self, model, cond: Condition, vis_span: tuple[int, int] | None, hq_stats: QuantStats):
        self.model = model
        self.cond = cond
        self.vis_span = vis_span
        self.hq = hq_stats
        self.n_chunks = 0
        self.edits = 0      # number of tensors this class altered in the cache

    # -- stage 1 -----------------------------------------------------------
    def prefill_image(self, session, embeds: torch.Tensor, pos: torch.Tensor) -> None:
        """Feed tokens [0, vis_end) and apply the image-scope edit right after the last image token."""
        s, e = self.vis_span
        ef = self.cond.scope == "image" and self.cond.state.endswith("ef")
        if not ef:
            session.feed(embeds[:, :e], pos[:, :, :e])
            self.n_chunks = 1
            self._edit_after_image(session)
            return
        spec = _state_spec(self.cond.state)
        residuals: dict[int, torch.Tensor] = {}
        if s > 0:
            session.feed(embeds[:, :s], pos[:, :, :s])
        for c in range(s, e, EF_CHUNK):
            c_end = min(c + EF_CHUNK, e)
            session.feed(embeds[:, c:c_end], pos[:, :, c:c_end])
            self._quant_states_ef(session.cache, spec, residuals)
            self.n_chunks += 1
        self._apply_kv(session)

    # -- edits ---------------------------------------------------------------
    def _edit_after_image(self, session) -> None:
        if self.cond.scope != "image":
            return
        self._apply_kv(session)
        self._edit_states(session.cache)

    def _apply_kv(self, session) -> None:
        if self.cond.kv == "drop":
            self._drop_kv(session)
        else:
            self._edit_kv(session.cache)

    def _attn_layers(self, cache):
        from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin

        return [l for l in cache.layers if isinstance(l, DynamicLayer) and not isinstance(l, LinearAttentionCacheLayerMixin)]

    def _gdn_layers(self, cache):
        from transformers.cache_utils import LinearAttentionCacheLayerMixin

        return [l for l in cache.layers if isinstance(l, LinearAttentionCacheLayerMixin)]

    @torch.no_grad()
    def _edit_kv(self, cache) -> None:
        if self.cond.scope != "image" or self.cond.kv == "full":
            return
        s, e = self.vis_span
        spec = _kv_spec(self.cond.kv)
        for layer in self._attn_layers(cache):
            for name in ("keys", "values"):
                t = getattr(layer, name)
                rows = t[:, :, s:e, :]
                if self.cond.kv == "zero":
                    new = torch.zeros_like(rows)
                else:
                    new = fake_quant(rows, spec, _gen(rows.device, NOISE_SEED) if spec.noise_match else None)
                t[:, :, s:e, :] = new
                self.edits += 1

    @torch.no_grad()
    def _drop_kv(self, session) -> None:
        if self.cond.scope != "image":
            return
        s, e = self.vis_span
        for layer in self._attn_layers(session.cache):
            if layer.keys.shape[-2] != session.n_fed:
                raise RuntimeError(f"cache length {layer.keys.shape[-2]} != tokens fed {session.n_fed}")
            for name in ("keys", "values"):
                t = getattr(layer, name)
                setattr(layer, name, torch.cat([t[:, :, :s, :], t[:, :, e:, :]], dim=2))
                self.edits += 1
        session.n_fed -= e - s   # text row of the position ids (causal mask) follows the shorter cache

    @torch.no_grad()
    def _edit_states(self, cache) -> None:
        level = self.cond.state
        if level == "full":
            return
        spec = _state_spec(level)
        for layer in self._gdn_layers(cache):
            st = layer.recurrent_states[0]
            if level == "zeroconv":
                st.zero_()
                layer.conv_states[0].zero_()
                self.edits += 1
            elif level == "zero":
                st.zero_()
            else:
                st.copy_(fake_quant(st, spec, _gen(st.device, NOISE_SEED) if spec.noise_match else None))
            self.edits += 1

    @torch.no_grad()
    def _quant_states_ef(self, cache, spec: QuantSpec, residuals: dict) -> None:
        for i, layer in enumerate(self._gdn_layers(cache)):
            st = layer.recurrent_states[0]
            x = st.float() + residuals.get(i, torch.zeros_like(st, dtype=torch.float32))
            q = fake_quant(x, spec)
            residuals[i] = x - q
            st.copy_(q.to(st.dtype))
            self.edits += 1

    def stats(self) -> dict:
        return {"state_calls": self.hq.state_calls, "kv_calls": self.hq.kv_calls, "n_chunks": self.n_chunks,
                "edits": self.edits}


@contextlib.contextmanager
def apply(model, cond: Condition, vis_span: tuple[int, int] | None):
    """Context for one sample. Everything run inside is counted by hybridquant's hooks.

    scope="image": hooks are pass-through counters; the edit happens in
    ``Intervention.prefill_image``. scope="whole": hooks quantize every write.
    Hooks are restored on exit, also when an exception is raised.
    """
    stats = QuantStats()
    cfg = _whole_config(cond) if cond.scope == "whole" else HybridQuantConfig()
    seed = NOISE_SEED if cond.scope == "whole" and "noise" in (cond.kv, cond.state) else None
    with quantized_cache(cfg, stats, noise_seed=seed):
        yield Intervention(model, cond, vis_span, stats)
