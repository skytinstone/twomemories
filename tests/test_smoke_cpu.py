"""CPU smoke tests on a tiny randomly initialised Qwen3.5 VL model (fp32).

Uses the real Qwen3.5-0.8B processor (tokenizer + image processor, a few MB from
the HF cache) so that encode() sees the real chat template and token ids; the
weights are random and tiny.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from analysis.schema import SchemaError, validate
from twomem import budget, data, metrics, quant, tokens
from twomem.model import Session, encode, encode_blind, image_token_id, prepare

PROC_ID = "Qwen/Qwen3.5-0.8B"


def _tiny_model():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    text = dict(
        vocab_size=248320, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, full_attention_interval=4, attn_output_gate=True, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16,
                  spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=2304, in_channels=3,
                  deepstack_visual_indexes=[])
    cfg = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=248056, video_token_id=248057,
                        vision_start_token_id=248053, vision_end_token_id=248054, tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(cfg).eval()


@pytest.fixture(scope="module")
def model():
    return _tiny_model()


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(PROC_ID)


@pytest.fixture(scope="module")
def enc(model, processor):
    img = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(img).text((20, 120), "TWO MEMORIES 2026", fill="black")
    return encode(processor, img, "What text is written in the image?", img_tok=image_token_id(model), merge=2)


def _run(model, enc, cond, span=None, prep=None):
    prep = prep or prepare(model, enc)
    span = span or enc.vis_span
    with quant.apply(model, cond, span) as iv:
        sess = Session(model)
        iv.prefill_image(sess, prep.embeds, prep.pos)
        logits = sess.feed(prep.embeds[:, span[1]:], prep.pos[:, :, span[1]:])
    return sess, logits, iv.stats()


# 1
def test_image_forward_runs(model, enc):
    sess, logits, st = _run(model, enc, quant.Condition())
    assert logits.shape[1] == enc.input_ids.shape[1] - enc.vis_span[1]
    assert torch.isfinite(logits).all() and st["state_calls"] > 0 and st["kv_calls"] > 0


# 2
def test_vis_span_matches_placeholder_mask(model, enc):
    mask = (enc.input_ids[0] == image_token_id(model)).nonzero().flatten()
    assert (int(mask[0]), int(mask[-1]) + 1) == enc.vis_span
    assert enc.n_vis == len(mask) == int(enc.image_grid_thw.prod()) // 4


# 3
def test_full_condition_matches_unpatched(model, enc):
    prep = prepare(model, enc)
    s, e = enc.vis_span
    plain = Session(model)
    plain.feed(prep.embeds[:, :e], prep.pos[:, :, :e])
    ref = plain.feed(prep.embeds[:, e:], prep.pos[:, :, e:])
    _, logits, _ = _run(model, enc, quant.Condition(), prep=prep)
    torch.testing.assert_close(logits, ref, atol=1e-5, rtol=0)


# 4
def test_blind_has_no_image_tokens(model, processor):
    b = encode_blind(processor, "What text is written in the image?", img_tok=image_token_id(model), merge=2)
    assert int((b.input_ids == image_token_id(model)).sum()) == 0 and b.vis_span is None and b.n_vis == 0


# 5
def test_state_zero(model, enc):
    # check right after stage 1: the question tokens fed afterwards write a new state on top
    prep = prepare(model, enc)
    with quant.apply(model, quant.Condition(state="zero"), enc.vis_span) as iv:
        s = Session(model)
        iv.prefill_image(s, prep.embeds, prep.pos)
    gdn = [l for l in s.cache.layers if hasattr(l, "recurrent_states")]
    assert len(gdn) == 3 and all(float(l.recurrent_states[0].abs().max()) == 0.0 for l in gdn)


# 6
def test_kv_zero_only_visual_rows(model, enc):
    prep = prepare(model, enc)
    s, e = enc.vis_span
    clean = Session(model)
    clean.feed(prep.embeds[:, :e], prep.pos[:, :, :e])
    with quant.apply(model, quant.Condition(kv="zero"), enc.vis_span) as iv:
        z = Session(model)
        iv.prefill_image(z, prep.embeds, prep.pos)
    k_clean, k_zero = clean.cache.layers[3].keys, z.cache.layers[3].keys
    v_zero = z.cache.layers[3].values
    assert float(k_zero[:, :, s:e].abs().max()) == 0.0 and float(v_zero[:, :, s:e].abs().max()) == 0.0
    torch.testing.assert_close(k_zero[:, :, :s], k_clean[:, :, :s], atol=0, rtol=0)


# 7
def test_token_half(model, enc):
    prep = prepare(model, enc)
    out, span = tokens.reduce(prep, enc.vis_span, 0.5, "uniform")
    assert span[1] - span[0] == enc.n_vis // 2
    assert out.embeds.shape[1] - (span[1] - span[0]) == enc.n_text
    # kept tokens keep their original M-RoPE positions
    kept = tokens.keep_indices(enc.n_vis, 0.5, "uniform")
    s = enc.vis_span[0]
    torch.testing.assert_close(out.pos[:, :, span[0]:span[1]], prep.pos[:, :, s + torch.as_tensor(kept)])


# 8
def test_token_methods_deterministic():
    assert np.array_equal(tokens.keep_indices(196, 0.5, "uniform"), tokens.keep_indices(196, 0.5, "uniform"))
    a, b = tokens.keep_indices(196, 0.25, "random", seed=0), tokens.keep_indices(196, 0.25, "random", seed=0)
    assert np.array_equal(a, b) and len(a) == 49 and len(np.unique(a)) == 49
    assert not np.array_equal(a, tokens.keep_indices(196, 0.25, "random", seed=1))


# 9
def test_budget_by_hand(model):
    tc = model.config.get_text_config(decoder=True)
    out = budget.mb(quant.Condition(), n_vis=64, n_text=22, model_cfg=tc)
    kv_elems_per_tok = 2 * 1 * 1 * 32          # K+V, 1 attn layer, 1 kv head, head_dim 32
    state_elems = 3 * 2 * 16 * 16              # 3 GDN layers, 2 heads, 16 x 16
    assert out["kv_mb"] == pytest.approx(16 * kv_elems_per_tok * 86 / 8 / 1e6)
    assert out["state_mb"] == pytest.approx(32 * state_elems / 8 / 1e6)
    assert out["total_mb"] == pytest.approx(out["kv_mb"] + out["state_mb"])
    q = budget.mb(quant.Condition(kv="4", state="8ef"), 64, 22, tc)
    assert q["kv_mb"] == pytest.approx(out["kv_mb"] / 4) and q["state_mb"] == pytest.approx(out["state_mb"] / 4)


# 10
def test_state8_on_grid(model, enc):
    from hybridquant import QuantSpec, fake_quant

    prep = prepare(model, enc)
    with quant.apply(model, quant.Condition(state="8"), enc.vis_span) as iv:
        s = Session(model)
        iv.prefill_image(s, prep.embeds, prep.pos)
    st = s.cache.layers[0].recurrent_states[0]
    spec = QuantSpec(8, quant.GROUP)
    torch.testing.assert_close(fake_quant(st, spec), st, atol=1e-6, rtol=1e-5)   # idempotent = on the grid
    assert max(len(torch.unique(r)) for r in st.reshape(-1, st.shape[-1])) <= 256


# 11
def test_hooks_restored_on_exception(model, enc):
    from transformers import cache_utils

    orig_u, orig_s = cache_utils.DynamicLayer.update, cache_utils.LinearAttentionLayer.update_recurrent_state
    with pytest.raises(RuntimeError):
        with quant.apply(model, quant.Condition(kv="4", state="8", scope="whole"), enc.vis_span):
            assert cache_utils.DynamicLayer.update is not orig_u
            raise RuntimeError("boom")
    assert cache_utils.DynamicLayer.update is orig_u
    assert cache_utils.LinearAttentionLayer.update_recurrent_state is orig_s


# 12
def test_subset_order_preserved(tmp_path, monkeypatch):
    import io as _io

    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = _io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    img = {"bytes": buf.getvalue(), "path": None}
    ids = ["q5", "q1", "q9", "q3"]
    table = pa.table({"id": ids, "question": [f"Is {i}?" for i in ids], "answer": ["yes", "no", "yes", "no"],
                      "category": ["random"] * 4, "image": [img] * 4})
    pq.write_table(table, tmp_path / "shard.parquet")
    order = [2, 0, 3]
    sub = {"seed": 0, "n": 3, "items": [{"sample_id": ids[r], "index": r, "file": "shard.parquet", "row": r}
                                        for r in order]}
    (tmp_path / "pope.json").write_text(json.dumps(sub))
    monkeypatch.setitem(data.DATASETS, "pope", {**data.DATASETS["pope"], "id_col": "id"})
    monkeypatch.setattr(data, "_shard_path", lambda task, f: str(tmp_path / f))
    got = [s["sample_id"] for s in data.iter_samples("pope", subset_dir=tmp_path)]
    assert got == [ids[r] for r in order]
    # a truncated cached shard (e.g. an interrupted write) is downloaded again with force_download
    (tmp_path / "bad.parquet").write_bytes((tmp_path / "shard.parquet").read_bytes()[:100])
    monkeypatch.setattr(data, "_shard_path",
                        lambda task, f, force=False: str(tmp_path / ("shard.parquet" if force else "bad.parquet")))
    got = [s["sample_id"] for s in data.iter_samples("pope", subset_dir=tmp_path)]
    assert got == [ids[r] for r in order]
    with pytest.raises(FileNotFoundError):
        data.load_subset("docvqa", subset_dir=tmp_path)


# 13
def test_schema_catches_missing_field():
    from analysis.schema import RESULT_FIELDS

    row = {k: (t() if t is not dict else {}) for k, t in RESULT_FIELDS.items()}
    row["cond"] = {"kv": "full", "state": "full", "token_keep": 1.0, "token_method": "none", "scope": "image"}
    row["scope"] = "image"
    validate(row)
    del row["gold_nll"]
    with pytest.raises(SchemaError, match="gold_nll"):
        validate(row)


# 14
def test_retained_known_value():
    assert metrics.retained(2.0, 1.0, 1.5) == pytest.approx(0.5)
    ci = metrics.paired_bootstrap([2.0, 2.0], [1.0, 1.0], [1.5, 1.5], n=100)
    assert ci["retained"] == pytest.approx(0.5) and ci["lo"] == pytest.approx(0.5)


def test_resume_after_killed_write(tmp_path):
    """A session killed mid-write leaves a partial line: reads skip it, the next append starts a fresh line,
    and the monitor's incremental reader only returns complete lines."""
    import sys

    from twomem.runner import append_jsonl, read_jsonl

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from status import _Tail

    p = tmp_path / "cell.jsonl"
    append_jsonl(p, {"sample_id": "a"})
    tail = _Tail(p)
    assert [r["sample_id"] for r in tail.new_rows()] == ["a"]
    with p.open("a") as f:
        f.write('{"sample_id": "b", "gold')        # killed here
    assert [r["sample_id"] for r in read_jsonl(p)] == ["a"]
    assert tail.new_rows() == []
    append_jsonl(p, {"sample_id": "c"})
    assert [r["sample_id"] for r in read_jsonl(p)] == ["a", "c"]
    assert [r["sample_id"] for r in tail.new_rows()] == ["c"]


def test_kv_drop_equals_masking_visual_columns(model, enc):
    """Dropping the visual rows from the cache must equal keeping them and masking their attention columns."""
    from transformers.masking_utils import create_causal_mask

    prep = prepare(model, enc)
    s, e = enc.vis_span
    _, dropped, st = _run(model, enc, quant.Condition(kv="drop"))
    assert st["edits"] == 2 * sum(t == "full_attention" for t in model.config.text_config.layer_types)

    with quant.apply(model, quant.Condition(), enc.vis_span):
        sess = Session(model)
        sess.feed(prep.embeds[:, :e], prep.pos[:, :, :e])
        q_emb, q_pos = prep.embeds[:, e:], prep.pos[:, :, e:]
        t = q_emb.shape[1]
        text_row = torch.arange(e, e + t).view(1, 1, -1)
        pos4 = torch.cat([text_row, q_pos], dim=0)
        lm = model.model.language_model
        mask = create_causal_mask(config=lm.config, inputs_embeds=q_emb, attention_mask=None,
                                  past_key_values=sess.cache, position_ids=text_row[0])
        if mask is None:   # sdpa may return None for a plain causal mask: build it explicitly
            kv = torch.arange(e + t)
            mask = (kv[None, :] <= (e + torch.arange(t))[:, None])[None, None]
        if mask.dtype == torch.bool:
            mask = mask.clone()
            mask[..., s:e] = False
        else:
            mask = mask.clone()
            mask[..., s:e] = torch.finfo(mask.dtype).min
        out = lm(inputs_embeds=q_emb, position_ids=pos4, past_key_values=sess.cache, use_cache=True,
                 attention_mask={"full_attention": mask, "linear_attention": None})
        masked = model.lm_head(out.last_hidden_state)
    assert torch.allclose(dropped, masked, atol=1e-4), (dropped - masked).abs().max()


def test_state_zeroconv_clears_conv_cache(model, enc):
    sess, logits, st = _run(model, enc, quant.Condition(state="zeroconv"))
    _, logits_zero, _ = _run(model, enc, quant.Condition(state="zero"))
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits, logits_zero)   # the conv cache carried the last visual tokens
    with quant.apply(model, quant.Condition(state="zeroconv"), enc.vis_span) as iv:
        s2 = Session(model)
        iv.prefill_image(s2, prepare(model, enc).embeds, prepare(model, enc).pos)
        from transformers.cache_utils import LinearAttentionCacheLayerMixin
        gdn = [l for l in s2.cache.layers if isinstance(l, LinearAttentionCacheLayerMixin)]
        assert all(l.recurrent_states[0].abs().max() == 0 and l.conv_states[0].abs().max() == 0 for l in gdn)


def test_prompt_never_opens_thinking(model):
    """4B/9B templates default to thinking; encode must close the block like the 0.8B default does."""
    import copy

    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor

    proc = copy.deepcopy(AutoProcessor.from_pretrained(PROC_ID))
    proc.chat_template = Path(hf_hub_download("Qwen/Qwen3.5-4B", "chat_template.jinja")).read_text()
    img = Image.new("RGB", (64, 64), "white")
    for e in (encode(proc, img, "Q?", img_tok=image_token_id(model), merge=2),
              encode_blind(proc, "Q?", img_tok=image_token_id(model), merge=2)):
        assert proc.tokenizer.decode(e.input_ids[0, -6:]).endswith("<think>\n\n</think>\n\n")
