"""Two-stage vs single-pass check on real task samples (D-32).

For the first --n samples of each task, recompute the gold NLL with the stock single-pass forward
(model(**inputs) on prompt + gold, no cache split, no hooks) and compare it with the two-stage gold NLL
recorded in the unaltered cell (KVfull_Sfull_image) of --results. Writes one row per sample to --out.
Run only when no phase is running (it shares the MPS device).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml

from twomem import data, metrics
from twomem.model import encode, image_token_id, load
from twomem.runner import REPO, git_commit, read_jsonl

TASKS = ("docvqa", "chartqa", "textvqa", "pope")


@torch.no_grad()
def single_pass_nll(model, processor, sample: dict) -> tuple[float, int]:
    """Sum of gold NLL and token count from one stock forward over prompt + gold."""
    tok = processor.tokenizer
    enc = encode(processor, sample["image"], sample["prompt"], img_tok=image_token_id(model),
                 merge=model.config.vision_config.spatial_merge_size)
    gold = torch.tensor(tok(sample["gold"], add_special_tokens=False)["input_ids"])
    ids = torch.cat([enc.input_ids[0], gold])[None]
    dev = model.device
    out = model(input_ids=ids.to(dev), attention_mask=torch.ones_like(ids).to(dev),
                mm_token_type_ids=torch.cat([enc.mm_token_type_ids[0], torch.zeros_like(gold)])[None].to(dev),
                pixel_values=enc.pixel_values.to(dev), image_grid_thw=enc.image_grid_thw.to(dev))
    T = enc.input_ids.shape[1]
    return metrics.gold_nll(out.logits[0, T - 1:T - 1 + gold.numel()], gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--results", default="results/mac")
    ap.add_argument("--out", default="results/checks/twostage_0.8B.jsonl")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    mp = yaml.safe_load(open(REPO / "configs/phase1_probe.yaml"))["max_pixels"]
    model, processor = load(args.model, max_pixels=mp)
    commit = git_commit()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")
    for task in TASKS:
        cell = Path(args.results) / args.model.replace("/", "__") / task / "KVfull_Sfull_image.jsonl"
        two = {r["sample_id"]: r for r in read_jsonl(cell)}
        diffs = []
        for s in data.iter_samples(task, n_limit=args.n):
            nll, ntok = single_pass_nll(model, processor, s)
            r = two[s["sample_id"]]
            row = {"model_id": args.model, "task": task, "sample_id": s["sample_id"], "n_vis": r["n_vis"],
                   "single_nll_tok": nll / ntok, "twostage_nll_tok": r["gold_nll"] / r["gold_ntok"],
                   "twostage_commit": r["commit"], "commit": commit}
            diffs.append(row["single_nll_tok"] - row["twostage_nll_tok"])
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
        absd = [abs(d) for d in diffs]
        print(f"{task}: n={len(diffs)} mean diff {statistics.mean(diffs):+.4f} mean |diff| {statistics.mean(absd):.4f} "
              f"max |diff| {max(absd):.4f} (NLL/token, single - two-stage)")


if __name__ == "__main__":
    main()
