"""Model loading, prompt encoding and a manual two-stage forward for Qwen3.5 VL.

The runner never calls ``model(**inputs)`` or ``generate``. It builds the input
embeddings and the 4-row position ids (text row + 3 M-RoPE rows) once per
sample, then feeds slices of them to the language model with an explicit
cache. This is what lets us stop right after the image span, intervene on the
cache, and continue with the question (see docs/assumptions.md, premises 1-2).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["load", "pick_device", "Encoded", "encode", "encode_blind", "Prepared", "prepare", "Session",
           "lm_forward", "image_token_id", "text_config"]


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_id: str, device: str | None = None, dtype=torch.bfloat16, max_pixels: int | None = None):
    """Return (model, processor). bf16 by default; device order cuda > mps > cpu."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = pick_device(device)
    processor = AutoProcessor.from_pretrained(model_id)
    if max_pixels is not None:
        size = dict(processor.image_processor.size)
        processor.image_processor.size = {"shortest_edge": size.get("shortest_edge"), "longest_edge": int(max_pixels)}
    # Load on CPU and move: device_map="auto" on MPS has crashed before (see hybridquant.eval.load_model).
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    model = model.to(device).eval()
    return model, processor


def text_config(model):
    return model.config.get_text_config(decoder=True)


def image_token_id(model) -> int:
    return int(model.config.image_token_id)


@dataclass
class Encoded:
    input_ids: torch.Tensor            # [1, T]
    attention_mask: torch.Tensor       # [1, T]
    mm_token_type_ids: torch.Tensor    # [1, T]
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    vis_span: tuple[int, int] | None   # half-open [start, end) of image placeholder tokens
    n_vis: int
    n_text: int


def _messages(question: str, image=None, system: str | None = None) -> list[dict]:
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
    content = [{"type": "image", "image": image}] if image is not None else []
    content.append({"type": "text", "text": question})
    msgs.append({"role": "user", "content": content})
    return msgs


def _span_from_mask(mask: torch.Tensor) -> tuple[int, int] | None:
    idx = mask.nonzero().flatten()
    if idx.numel() == 0:
        return None
    s, e = int(idx[0]), int(idx[-1]) + 1
    if e - s != idx.numel():
        raise ValueError(f"image placeholder tokens are not contiguous: {idx.numel()} tokens over [{s}, {e})")
    return s, e


def _encode(processor, msgs, img_tok: int, merge: int) -> Encoded:
    # Qwen3.5 4B/9B templates default to thinking (prompt ends in an open "<think>"), 0.8B to non-thinking.
    # Answers are read right after the prompt, so thinking is always off.
    out = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                        return_tensors="pt", enable_thinking=False)
    ids = out["input_ids"]
    if ids.shape[0] != 1:
        raise ValueError("encode handles one sample at a time")
    tail = processor.tokenizer.decode(ids[0, -8:])
    if tail.rstrip().endswith("<think>"):
        raise ValueError(f"prompt ends in an open <think> block: {tail!r}")
    span = _span_from_mask(ids[0] == img_tok)
    n_vis = 0 if span is None else span[1] - span[0]
    grid = out.get("image_grid_thw")
    if grid is not None:
        expected = int(grid.prod(-1).sum()) // merge**2   # premise 2: prod(grid_thw) / merge^2
        assert n_vis == expected, f"n_vis={n_vis} but prod(grid_thw)/merge^2={expected}"
    return Encoded(
        input_ids=ids,
        attention_mask=out["attention_mask"],
        mm_token_type_ids=out.get("mm_token_type_ids", torch.zeros_like(ids)),
        pixel_values=out.get("pixel_values"),
        image_grid_thw=grid,
        vis_span=span,
        n_vis=n_vis,
        n_text=ids.shape[1] - n_vis,
    )


def encode(processor, image, question: str, system: str | None = None, *, img_tok: int, merge: int) -> Encoded:
    enc = _encode(processor, _messages(question, image, system), img_tok, merge)
    if enc.vis_span is None:
        raise ValueError("no image placeholder tokens in the encoded prompt")
    return enc


def encode_blind(processor, question: str, system: str | None = None, *, img_tok: int, merge: int) -> Encoded:
    """Same system/question, no image."""
    return _encode(processor, _messages(question, None, system), img_tok, merge)


@dataclass
class Prepared:
    embeds: torch.Tensor     # [1, T, D]
    pos: torch.Tensor        # [3, 1, T] M-RoPE positions (text row is rebuilt per forward)
    input_ids: torch.Tensor  # [1, T]
    vis_span: tuple[int, int] | None


@torch.no_grad()
def prepare(model, enc: Encoded) -> Prepared:
    """Input embeddings with image features scattered in, and M-RoPE positions, computed once."""
    dev = model.device
    ids = enc.input_ids.to(dev)
    core = model.model
    embeds = core.get_input_embeddings()(ids)
    if enc.pixel_values is not None:
        grid = enc.image_grid_thw.to(dev)
        feats = core.get_image_features(enc.pixel_values.to(dev), grid, return_dict=True).pooler_output
        feats = torch.cat(feats, dim=0).to(embeds.dtype)
        mask, _ = core.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=feats)
        embeds = embeds.masked_scatter(mask, feats)
        pos, _ = core.get_rope_index(ids, mm_token_type_ids=enc.mm_token_type_ids.to(dev), image_grid_thw=grid)
    else:
        pos = torch.arange(ids.shape[1], device=dev).view(1, 1, -1).expand(3, 1, -1)
    return Prepared(embeds=embeds, pos=pos.long(), input_ids=ids, vis_span=enc.vis_span)


class Session:
    """A cache plus the bookkeeping needed to feed it slice by slice.

    ``n_fed`` is the number of tokens already in the cache (drives the text row
    of position ids, which only the causal mask uses). ``next_pos`` is the next
    M-RoPE position for a plain text token (max position seen + 1).
    """

    def __init__(self, model, cache=None):
        from hybridquant import make_cache

        self.model = model
        self.cache = cache if cache is not None else make_cache(model)
        self.n_fed = 0
        self.next_pos = 0

    @torch.no_grad()
    def feed(self, embeds: torch.Tensor, pos3: torch.Tensor) -> torch.Tensor:
        """Run embeds [1, t, D] with M-RoPE positions [3, 1, t]; returns logits [1, t, V]."""
        t = embeds.shape[1]
        text_row = torch.arange(self.n_fed, self.n_fed + t, device=embeds.device).view(1, 1, -1)
        logits = lm_forward(self.model, embeds, torch.cat([text_row, pos3], dim=0), self.cache)
        self.n_fed += t
        self.next_pos = max(self.next_pos, int(pos3.max()) + 1)
        return logits

    @torch.no_grad()
    def feed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Feed plain text tokens [1, t] at the next positions."""
        emb = self.model.model.get_input_embeddings()(token_ids.to(self.model.device))
        t = token_ids.shape[1]
        pos = torch.arange(self.next_pos, self.next_pos + t, device=emb.device).view(1, 1, -1).expand(3, 1, -1)
        return self.feed(emb, pos)

    def fork(self) -> "Session":
        import copy

        s = Session.__new__(Session)
        s.model, s.cache, s.n_fed, s.next_pos = self.model, copy.deepcopy(self.cache), self.n_fed, self.next_pos
        return s


@torch.no_grad()
def lm_forward(model, embeds: torch.Tensor, pos4: torch.Tensor, cache) -> torch.Tensor:
    out = model.model.language_model(inputs_embeds=embeds, position_ids=pos4, past_key_values=cache, use_cache=True)
    return model.lm_head(out.last_hidden_state)
