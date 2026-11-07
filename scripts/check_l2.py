"""L2-local extras (0.8B, MPS): greedy determinism and batch-1 vs batch-2 gold NLL.

The runner only implements batch 1. The batch check here duplicates one sample
into a batch of 2 (same length, no padding) and compares the gold NLL of each
row with the batch-1 value; it tests batched numerics only, not padded batches
of different samples.
"""
from __future__ import annotations

import argparse
import json

import torch

from hybridquant import make_cache
from twomem import data, metrics, quant
from twomem.model import encode, image_token_id, lm_forward, load, prepare
from twomem.runner import eval_sample


@torch.no_grad()
def batch2_nll(model, processor, sample) -> list[float]:
    tok = processor.tokenizer
    enc = encode(processor, sample["image"], sample["prompt"], img_tok=image_token_id(model), merge=2)
    p = prepare(model, enc)
    e = enc.vis_span[1]
    gold = torch.tensor(tok(sample["gold"], add_special_tokens=False)["input_ids"])
    B = 2
    emb = p.embeds.expand(B, -1, -1).contiguous()
    pos = p.pos.expand(3, B, -1).contiguous()
    cache = make_cache(model)

    def fwd(x, p3, start):
        t = x.shape[1]
        text = torch.arange(start, start + t, device=x.device).view(1, 1, -1).expand(1, B, -1)
        return lm_forward(model, x, torch.cat([text, p3], 0), cache)

    fwd(emb[:, :e], pos[:, :, :e], 0)
    last = fwd(emb[:, e:], pos[:, :, e:], e)[:, -1:]
    nxt = int(p.pos.max()) + 1
    g = gold[None, :-1].expand(B, -1).to(model.device)
    gemb = model.model.get_input_embeddings()(g)
    gpos = torch.arange(nxt, nxt + g.shape[1], device=gemb.device).view(1, 1, -1).expand(3, B, -1)
    rest = fwd(gemb, gpos, emb.shape[1])
    logits = torch.cat([last, rest], 1)
    return [metrics.gold_nll(logits[b], gold)[0] for b in range(B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="docvqa")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    model, processor = load("Qwen/Qwen3.5-0.8B", max_pixels=1310720)
    out = {"greedy_identical": True, "batch_max_diff": 0.0, "rows": []}
    for s in data.iter_samples(args.task, args.limit):
        a = eval_sample(model, processor, args.task, s, quant.Condition())
        b = eval_sample(model, processor, args.task, s, quant.Condition())
        same = a["pred"] == b["pred"] and a["gold_nll"] == b["gold_nll"]
        nb = batch2_nll(model, processor, s)
        diff = max(abs(x - a["gold_nll"]) for x in nb)
        out["greedy_identical"] &= same
        out["batch_max_diff"] = max(out["batch_max_diff"], diff)
        out["rows"].append({"sample_id": s["sample_id"], "n_vis": a["n_vis"], "nll_b1": a["gold_nll"],
                            "nll_b2": nb, "diff": diff, "greedy_same": same, "pred": a["pred"]})
        print(json.dumps(out["rows"][-1]), flush=True)
    out["batch_pass_1e-3"] = out["batch_max_diff"] <= 1e-3
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main()
