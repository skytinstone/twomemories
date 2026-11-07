"""Task data: four VQA sets from lmms-lab parquet shards, pinned to a revision.

Subsets are fixed once by ``scripts/make_subsets.py`` (seed 0) and committed as
configs/subsets/{task}.json. ``iter_samples`` only reads those files and never
re-draws a subset. Only the parquet shards that hold subset rows are downloaded.
"""
from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

__all__ = ["TASKS", "DATASETS", "PROMPTS", "SUBSET_DIR", "SUBSET_SIZES", "prompt", "gold_answer", "load_subset",
           "iter_samples", "shard_files"]

TASKS = ("docvqa", "chartqa", "textvqa", "pope")

# repo id, pinned revision, shard prefix (split), id column (None -> row index)
DATASETS = {
    "docvqa": {"repo": "lmms-lab/DocVQA", "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
               "prefix": "DocVQA/validation-", "id_col": "questionId"},
    "chartqa": {"repo": "lmms-lab/ChartQA", "revision": "9e63b7df1592a1c2158e735cc1725454aef0d6d9",
                "prefix": "data/test-", "id_col": None},
    "textvqa": {"repo": "lmms-lab/textvqa", "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
                "prefix": "data/validation-", "id_col": "question_id"},
    "pope": {"repo": "lmms-lab/POPE", "revision": "4db1276663dfa5eb8ad16a52d24c31a09e470896",
             "prefix": "data/test-", "id_col": "id"},
}
SUBSET_SIZES = {"docvqa": 200, "chartqa": 200, "textvqa": 200, "pope": 300}

# One instruction line per task, appended after the question (lmms-eval style).
PROMPTS = {
    "docvqa": "{question}\nAnswer the question using a single word or phrase.",
    "chartqa": "{question}\nAnswer the question using a single word or phrase.",
    "textvqa": "{question}\nAnswer the question using a single word or phrase.",
    "pope": "{question}\nAnswer the question using a single word or phrase.",
}

SUBSET_DIR = Path(__file__).resolve().parents[2] / "configs" / "subsets"


def prompt(task: str, question: str) -> str:
    return PROMPTS[task].format(question=question)


def gold_answer(task: str, row: dict) -> tuple[str, list[str]]:
    """(gold string used for teacher-forced NLL, all reference answers used for scoring)."""
    if task == "docvqa":
        return row["answers"][0], list(row["answers"])
    if task == "chartqa":
        return str(row["answer"]), [str(row["answer"])]
    if task == "textvqa":
        answers = list(row["answers"])
        return Counter(answers).most_common(1)[0][0], answers
    if task == "pope":
        return row["answer"], [row["answer"]]
    raise KeyError(task)


def shard_files(task: str) -> list[str]:
    from huggingface_hub import HfFileSystem

    spec = DATASETS[task]
    folder, stem = spec["prefix"].split("/")
    fs = HfFileSystem()
    base = f"datasets/{spec['repo']}@{spec['revision']}/{folder}"
    names = sorted(p.split("/")[-1] for p in fs.ls(base, detail=False))
    return [f"{folder}/{n}" for n in names if n.startswith(stem) and n.endswith(".parquet")]


def load_subset(task: str, subset_dir: Path | None = None) -> dict:
    path = Path(subset_dir or SUBSET_DIR) / f"{task}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Subsets are drawn once by scripts/make_subsets.py and committed; "
                                "they are never re-drawn at run time.")
    return json.loads(path.read_text())


def _shard_path(task: str, file: str, force: bool = False) -> str:
    from huggingface_hub import hf_hub_download

    spec = DATASETS[task]
    return hf_hub_download(spec["repo"], file, repo_type="dataset", revision=spec["revision"], force_download=force)


def _read_shard(task: str, file: str):
    """Read one parquet shard; a cached copy that fails to parse (truncated write) is downloaded once more."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        return pq.read_table(_shard_path(task, file))
    except (pa.ArrowInvalid, OSError):
        return pq.read_table(_shard_path(task, file, force=True))


def iter_samples(task: str, n_limit: int | None = None, subset_dir: Path | None = None):
    """Yield samples in the subset file's order: dict(sample_id, image, question, prompt, gold, answers, meta)."""
    from PIL import Image

    sub = load_subset(task, subset_dir)
    items = sub["items"][:n_limit] if n_limit else sub["items"]
    cur_file, table = None, None
    for it in items:
        if it["file"] != cur_file:
            table = _read_shard(task, it["file"])
            cur_file = it["file"]
        row = table.slice(it["row"], 1).to_pylist()[0]
        rid = row[DATASETS[task]["id_col"]] if DATASETS[task]["id_col"] else f"{task}-{it['index']}"
        if str(rid) != it["sample_id"]:
            raise ValueError(f"{task}: subset row {it} resolves to id {rid}; dataset revision changed?")
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        gold, answers = gold_answer(task, row)
        meta = {"category": row.get("category")} if task == "pope" else {}
        yield {"sample_id": it["sample_id"], "image": image, "question": row["question"],
               "prompt": prompt(task, row["question"]), "gold": gold, "answers": answers, "meta": meta}
