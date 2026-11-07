"""L1 check on the real 0.8B model (MPS, CPU fallback). Records numbers only, no interpretation.

(a) encode output  (b) unhooked two-stage gold NLL  (c) hooked full NLL vs (b), hook counts
(d) blind run: state_calls vs n_vis (premise 5 evidence), plus an 8ef run on a larger image
(e) kv zero / state zero NLL  (f) markdown table, written into docs/assumptions.md "L1 결과".
"""
from __future__ import annotations

import argparse
import re
import time
import traceback

import torch
from PIL import Image, ImageDraw, ImageFont

from twomem import metrics, quant
from twomem.model import Session, encode, image_token_id, load, prepare, text_config
from twomem.runner import REPO, eval_sample, git_commit

MODEL = "Qwen/Qwen3.5-0.8B"
QUESTION = "What text is written in the image?"
GOLD = "TWO MEMORIES 2026"
MARK_START, MARK_END = "<!-- L1:start -->", "<!-- L1:end -->"


def synth_image(size: int = 448) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size // 14)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([size * 0.08, size * 0.1, size * 0.4, size * 0.35], outline="black", width=4, fill="#88c")
    d.rectangle([size * 0.55, size * 0.6, size * 0.92, size * 0.9], outline="black", width=4, fill="#c88")
    d.text((size * 0.08, size * 0.45), GOLD, fill="black", font=font)
    return img


@torch.no_grad()
def unhooked_nll(model, processor, image) -> float:
    """Two-stage gold NLL with no hybridquant hooks installed at all."""
    tok = processor.tokenizer
    enc = encode(processor, image, QUESTION, img_tok=image_token_id(model), merge=2)
    prep = prepare(model, enc)
    e = enc.vis_span[1]
    s = Session(model)
    s.feed(prep.embeds[:, :e], prep.pos[:, :, :e])
    last = s.feed(prep.embeds[:, e:], prep.pos[:, :, e:])[:, -1]
    gold = torch.tensor(tok(GOLD, add_special_tokens=False)["input_ids"])
    rest = s.feed_tokens(gold[None, :-1])[0]
    return metrics.gold_nll(torch.cat([last, rest]), gold)[0] / gold.numel()


@torch.no_grad()
def single_pass_nll(model, processor, image) -> float:
    """Reference: the stock model(**inputs) forward on prompt + gold (one pass, no cache split)."""
    tok = processor.tokenizer
    enc = encode(processor, image, QUESTION, img_tok=image_token_id(model), merge=2)
    gold = torch.tensor(tok(GOLD, add_special_tokens=False)["input_ids"])
    ids = torch.cat([enc.input_ids[0], gold])[None]
    out = model(input_ids=ids.to(model.device), attention_mask=torch.ones_like(ids).to(model.device),
                mm_token_type_ids=torch.cat([enc.mm_token_type_ids[0], torch.zeros_like(gold)])[None].to(model.device),
                pixel_values=enc.pixel_values.to(model.device), image_grid_thw=enc.image_grid_thw.to(model.device))
    T = enc.input_ids.shape[1]
    return metrics.gold_nll(out.logits[0, T - 1:T - 1 + gold.numel()], gold)[0] / gold.numel()


@torch.no_grad()
def prefill_state_calls(model, processor, image) -> tuple[int, int]:
    """(n_vis, state writes) for ONE forward over the whole prompt, hooks counting only."""
    from hybridquant import HybridQuantConfig, QuantStats, quantized_cache
    from twomem.model import encode_blind

    if image is None:
        enc = encode_blind(processor, QUESTION, img_tok=image_token_id(model), merge=2)
    else:
        enc = encode(processor, image, QUESTION, img_tok=image_token_id(model), merge=2)
    prep = prepare(model, enc)
    st = QuantStats()
    with quantized_cache(HybridQuantConfig(), st):
        Session(model).feed(prep.embeds, prep.pos)
    return enc.n_vis, st.state_calls


def run(device: str) -> tuple[list[str], dict]:
    model, processor = load(MODEL, device)
    tc = text_config(model)
    img = synth_image()
    sample = {"prompt": QUESTION, "image": img, "gold": GOLD, "answers": [GOLD]}
    lines, info = [], {"device": device}

    enc = encode(processor, img, QUESTION, img_tok=image_token_id(model), merge=2)
    lines.append(f"(a) input_ids={tuple(enc.input_ids.shape)} vis_span={enc.vis_span} n_vis={enc.n_vis} "
                 f"n_text={enc.n_text} image_grid_thw={enc.image_grid_thw.tolist()} "
                 f"pixel_values={tuple(enc.pixel_values.shape)}")
    t0 = time.perf_counter()
    nll_b = unhooked_nll(model, processor, img)
    nll_sp = single_pass_nll(model, processor, img)
    rows = []

    def add(name, cond, smp=sample):
        t = time.perf_counter()
        r = eval_sample(model, processor, "docvqa", smp, cond)   # "docvqa" -> ANLS against GOLD
        r["sec"] = time.perf_counter() - t
        r["nll_tok"] = r["gold_nll"] / r["gold_ntok"]
        rows.append((name, r))
        return r

    full = add("full (hooks on, pass-through)", quant.Condition())
    blind = add("blind", None)
    add("kv zero", quant.Condition(kv="zero"))
    add("state zero", quant.Condition(state="zero"))
    add("kv4 / state8ef", quant.Condition(kv="4", state="8ef"))
    big = {**sample, "image": synth_image(1024)}
    add("full, 1024px image", quant.Condition(), big)
    add("state 8ef, 1024px image", quant.Condition(state="8ef"), big)
    info["sec_total"] = time.perf_counter() - t0

    diff = abs(full["nll_tok"] - nll_b)
    lines.append(f"(b) unhooked two-stage gold NLL/token = {nll_b:.6f}; stock single-pass model(**inputs) = {nll_sp:.6f} "
                 f"(|diff| {abs(nll_b - nll_sp):.2e}, bf16 split vs one pass)")
    lines.append(f"(c) hooked full NLL/token = {full['nll_tok']:.6f}, |(c)-(b)| = {diff:.2e} "
                 f"({'PASS' if diff <= 1e-4 else 'FAIL'} at 1e-4); state_calls={full['stats']['state_calls']} "
                 f"kv_calls={full['stats']['kv_calls']}")
    gdn = list(tc.layer_types).count("linear_attention")
    pf = [prefill_state_calls(model, processor, im) for im in (None, img, synth_image(1024))]
    lines.append("(d) one prefill forward, state writes: " + ", ".join(f"n_vis={n} -> {c}" for n, c in pf)
                 + f"; GDN layers={gdn}. Whole-cell totals (with decode steps) are in the table: "
                 f"blind {blind['stats']['state_calls']}, image {full['stats']['state_calls']}")
    lines.append("")
    lines.append("| run | n_vis | gold NLL/token | pred | score | state_calls | kv_calls | n_chunks | sec |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, r in rows:
        st = r["stats"]
        lines.append(f"| {name} | {r['n_vis_kept']} | {r['nll_tok']:.4f} | {r['pred']!r} | {r['score']:.2f} | "
                     f"{st['state_calls']} | {st['kv_calls']} | {st['n_chunks']} | {r['sec']:.2f} |")
    info["pass_c"] = diff <= 1e-4
    return lines, info


def write_section(lines: list[str], info: dict, failures: list[str]) -> None:
    path = REPO / "docs" / "assumptions.md"
    text = path.read_text()
    body = [MARK_START, "## L1 결과", "",
            f"`scripts/check_l1.py`, 모델 {MODEL}, device={info['device']}, commit {git_commit()}, "
            f"총 {info['sec_total']:.1f}s. 값은 기록만 하고 해석하지 않는다.", ""]
    if failures:
        body += ["MPS 실패 기록:", "", "```", *failures, "```", ""]
    body += [l if l.startswith("|") or not l else f"- {l}" for l in lines]
    body += ["", MARK_END]
    block = "\n".join(body)
    if MARK_START in text:
        text = re.sub(re.escape(MARK_START) + ".*?" + re.escape(MARK_END), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    failures = []
    try:
        lines, info = run(args.device)
    except Exception as e:  # noqa: BLE001  (record the failing op, retry on CPU)
        if args.device == "cpu":
            raise
        failures.append(f"{type(e).__name__}: {e}")
        failures.append(traceback.format_exc().strip().splitlines()[-3])
        print("MPS failed, retrying on cpu:", failures[0])
        lines, info = run("cpu")
    print("\n".join(lines))
    if not args.no_write:
        write_section(lines, info, failures)
    return 0 if info["pass_c"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
