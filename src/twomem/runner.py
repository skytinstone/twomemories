"""One experiment cell = (model, task, condition, subset). One jsonl line per sample.

Per sample:
  encode -> tokens.reduce -> stage-1 prefill (prefix + image) -> intervention
  -> stage-2 prefill (question) -> gold NLL (one forward on a forked cache)
  -> greedy decode (max_new_tokens 16) on the original cache -> score.
The blind cell (same prompt, no image) is a separate cell per task, independent
of the condition, and is skipped when already present (resume).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from analysis.schema import validate, validate_compute

from . import budget, data, metrics, quant, tokens
from .model import Session, encode, encode_blind, image_token_id, load, prepare, text_config

__all__ = ["MAX_NEW_TOKENS", "REPO", "RESULTS", "COMPUTE_LOG", "git_commit", "gpu_name", "units_per_hour",
           "subset_id_of", "cell_hash", "cell_path", "read_jsonl", "append_jsonl", "cell_done", "run_cell",
           "run_blind", "eval_sample"]

MAX_NEW_TOKENS = 16
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
COMPUTE_LOG = RESULTS / "compute_log.jsonl"
BLIND = "blind"


def git_commit() -> str:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    h = run("rev-parse", "--short=10", "HEAD") or "unknown"
    dirty = bool(run("status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "configs", "analysis"))
    return h + ("-dirty" if dirty else "")


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def units_per_hour(gpu: str) -> float:
    import yaml

    table = yaml.safe_load((REPO / "configs" / "compute.yaml").read_text())["units_per_hour"]
    for key, val in table.items():
        if key.lower() in gpu.lower():
            return float(val)
    return float(table.get("default", 0.0))


def subset_id_of(task: str) -> str:
    raw = (data.SUBSET_DIR / f"{task}.json").read_bytes()
    sub = json.loads(raw)
    return f"{task}-s{sub['seed']}-n{sub['n']}-{hashlib.sha1(raw).hexdigest()[:8]}"


def cell_hash(model_id: str, task: str, cond: quant.Condition | None, sub_id: str, max_pixels: int | None) -> str:
    key = {"model_id": model_id, "task": task, "cond": cond.to_dict() if cond else BLIND, "subset_id": sub_id,
           "max_pixels": max_pixels, "prompt": data.PROMPTS[task], "max_new_tokens": MAX_NEW_TOKENS,
           "thinking": False}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def cell_path(out_dir: Path, model_id: str, task: str, cond: quant.Condition | None) -> Path:
    name = BLIND if cond is None else cond.label()
    return Path(out_dir) / model_id.replace("/", "__") / task / f"{name}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    """All parseable lines of a jsonl file. A line cut off by a killed session is skipped, not fatal."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, obj: dict) -> None:
    """Append one line; starts on a fresh line if the file ends in a partial line from a killed session."""
    path = Path(path)
    lead = ""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, 2)
            lead = "" if f.read(1) == b"\n" else "\n"
    with path.open("a") as f:
        f.write(lead + json.dumps(obj) + "\n")


def _done(path: Path, h: str) -> set[str]:
    return {r["sample_id"] for r in read_jsonl(path) if r.get("cell_hash") == h and "sample_id" in r}


def cell_done(out_dir: Path, model_id: str, task: str, cond: quant.Condition | None, max_pixels: int | None) -> int:
    """Number of finished samples of one cell (no model needed)."""
    h = cell_hash(model_id, task, cond, subset_id_of(task), max_pixels)
    return len(_done(cell_path(out_dir, model_id, task, cond), h))


def _eos_ids(model, processor) -> set[int]:
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, (list, tuple)) else [ids])
    ids.add(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return {int(i) for i in ids if i is not None}


@torch.no_grad()
def _greedy(sess: Session, last_logits: torch.Tensor, eos: set[int], max_new: int) -> list[int]:
    out = []
    logits = last_logits
    for _ in range(max_new):
        t = int(logits[0].argmax())
        if t in eos:
            break
        out.append(t)
        if len(out) == max_new:
            break
        logits = sess.feed_tokens(torch.tensor([[t]]))[:, -1]
    return out


@torch.no_grad()
def eval_sample(model, processor, task: str, sample: dict, cond: quant.Condition | None) -> dict:
    """Evaluate one sample under ``cond`` (None = blind). Returns the per-sample measurements."""
    tok = processor.tokenizer
    merge = model.config.vision_config.spatial_merge_size
    img_tok = image_token_id(model)
    gold_ids = torch.tensor(tok(sample["gold"], add_special_tokens=False)["input_ids"])
    run_cond = cond or quant.Condition()
    if cond is None:
        enc = encode_blind(processor, sample["prompt"], img_tok=img_tok, merge=merge)
        prep, span = prepare(model, enc), None
    else:
        enc = encode(processor, sample["image"], sample["prompt"], img_tok=img_tok, merge=merge)
        prep, span = tokens.reduce(prepare(model, enc), enc.vis_span, cond.token_keep, cond.token_method, seed=0)
    n_vis_kept = 0 if span is None else span[1] - span[0]
    with quant.apply(model, run_cond, span) as iv:
        sess = Session(model)
        if span is None:
            last = sess.feed(prep.embeds, prep.pos)[:, -1]
        else:
            iv.prefill_image(sess, prep.embeds, prep.pos)
            e = span[1]
            last = sess.feed(prep.embeds[:, e:], prep.pos[:, :, e:])[:, -1]
        fork = sess.fork()
        rows = [last]
        if gold_ids.numel() > 1:
            rows.append(fork.feed_tokens(gold_ids[None, :-1])[0])
        del fork
        nll, ntok = metrics.gold_nll(torch.cat(rows, 0), gold_ids)
        pred = tok.decode(_greedy(sess, last, _eos_ids(model, processor), MAX_NEW_TOKENS),
                          skip_special_tokens=True).strip()
        st = iv.stats()
    return {"n_vis": enc.n_vis, "n_vis_kept": n_vis_kept, "n_text": enc.n_text, "gold_nll": nll, "gold_ntok": ntok,
            "pred": pred, "score": metrics.score(task, pred, sample["answers"]), "stats": st}


def _run(model, processor, model_id, task, cond, n_limit, out_dir, max_pixels, extra_meta=None,
         compute_log: Path | None = None, phase: str | None = None) -> dict:
    sub_id = subset_id_of(task)
    h = cell_hash(model_id, task, cond, sub_id, max_pixels)
    path = cell_path(out_dir, model_id, task, cond)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(path, h)
    gpu, uph, commit = gpu_name(), units_per_hour(gpu_name()), git_commit()
    tcfg = text_config(model)
    run_cond = cond or quant.Condition()
    n_new, t_cell = 0, 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()   # peak VRAM is per cell
    from hybridquant import run_metadata

    hq_meta = run_metadata()
    for sample in data.iter_samples(task, n_limit):
        if sample["sample_id"] in done:
            continue
        t0 = time.perf_counter()
        r = eval_sample(model, processor, task, sample, cond)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        mb = budget.mb(run_cond, r["n_vis_kept"], r["n_text"], tcfg)
        row = {
            "cell_hash": h, "model_id": model_id, "task": task, "subset_id": sub_id, "sample_id": sample["sample_id"],
            "cond": run_cond.to_dict(), "scope": run_cond.scope, "n_vis": r["n_vis"], "n_vis_kept": r["n_vis_kept"],
            "n_text": r["n_text"], **mb, "gold_nll": r["gold_nll"], "gold_ntok": r["gold_ntok"], "pred": r["pred"],
            "score": r["score"], "state_calls": r["stats"]["state_calls"], "kv_calls": r["stats"]["kv_calls"],
            "n_chunks": r["stats"]["n_chunks"], "gpu": gpu, "seconds": sec, "units_est": sec / 3600 * uph,
            "commit": commit, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "blind": cond is None,
            "meta": {"gold": sample["gold"], "edits": r["stats"]["edits"], "max_pixels": max_pixels,
                     "inject_layers": [0], "hybridquant_commit": hq_meta["git_commit"],
                     "transformers": hq_meta["transformers"], "torch": hq_meta["torch"],
                     "peak_vram_gb": (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None,
                     **sample["meta"], **(extra_meta or {})},
        }
        append_jsonl(path, validate(row))
        n_new += 1
        t_cell += sec
    if n_new:
        rec = validate_compute({"cell_hash": h, "gpu": gpu, "seconds": t_cell, "units_est": t_cell / 3600 * uph,
                                "model_id": model_id, "task": task, "label": BLIND if cond is None else cond.label(),
                                "n_samples": n_new, "commit": commit,
                                "peak_vram_gb": (torch.cuda.max_memory_allocated() / 2**30)
                                if torch.cuda.is_available() else None})
        if phase:
            rec["phase"] = phase
        append_jsonl(compute_log or COMPUTE_LOG, rec)
    return {"cell_hash": h, "path": str(path), "n_new": n_new, "seconds": t_cell, "n_skipped": len(done)}


def run_cell(model_id: str, task: str, cond: quant.Condition, subset_id: str | None = None, n_limit: int | None = None,
             out_dir: Path = RESULTS, *, model=None, processor=None, max_pixels: int | None = None, batch: int = 1,
             compute_log: Path | None = None, phase: str | None = None) -> dict:
    """Run one cell (resumable). ``subset_id``, if given, must match the committed subset file."""
    if batch != 1:
        raise NotImplementedError("batch > 1 is disabled (docs/decision_log.md, D-batch)")
    if subset_id is not None and subset_id != subset_id_of(task):
        raise ValueError(f"subset_id {subset_id} does not match configs/subsets/{task}.json")
    if model is None:
        model, processor = load(model_id, max_pixels=max_pixels)
    return _run(model, processor, model_id, task, cond, n_limit, out_dir, max_pixels, compute_log=compute_log,
                phase=phase)


def run_blind(model_id: str, task: str, n_limit: int | None = None, out_dir: Path = RESULTS, *, model=None,
              processor=None, max_pixels: int | None = None, compute_log: Path | None = None,
              phase: str | None = None) -> dict:
    if model is None:
        model, processor = load(model_id, max_pixels=max_pixels)
    return _run(model, processor, model_id, task, None, n_limit, out_dir, max_pixels, compute_log=compute_log,
                phase=phase)
