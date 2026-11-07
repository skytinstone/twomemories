"""Draw the fixed evaluation subsets once (seed 0) and write configs/subsets/{task}.json.

Run once and commit the output. Refuses to overwrite existing files unless --force.
Reads only parquet footers and the id column, no images.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from twomem.data import DATASETS, SUBSET_DIR, SUBSET_SIZES, TASKS, shard_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        out = SUBSET_DIR / f"{task}.json"
        if out.exists() and not args.force:
            print(f"{out} exists, skipping (use --force to redraw)")
            continue
        spec = DATASETS[task]
        rows = []   # (file, local_row, sample_id)
        for f in shard_files(task):
            pf = pq.ParquetFile(fs.open(f"datasets/{spec['repo']}@{spec['revision']}/{f}"))
            n = pf.metadata.num_rows
            ids = pf.read(columns=[spec["id_col"]]).column(0).to_pylist() if spec["id_col"] else [None] * n
            rows += [(f, r, ids[r]) for r in range(n)]
        rng = np.random.default_rng(args.seed)
        picked = np.sort(rng.choice(len(rows), size=SUBSET_SIZES[task], replace=False))
        items = []
        for g in picked.tolist():
            f, r, rid = rows[g]
            items.append({"sample_id": str(rid) if rid is not None else f"{task}-{g}", "index": g, "file": f, "row": r})
        assert len({it["sample_id"] for it in items}) == len(items), f"{task}: duplicate sample ids"
        doc = {"task": task, "dataset": spec["repo"], "revision": spec["revision"], "split_prefix": spec["prefix"],
               "seed": args.seed, "n_total": len(rows), "n": len(items), "items": items}
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{task}: {len(items)} of {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
