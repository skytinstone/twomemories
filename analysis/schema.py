"""Schema of one result line in results/**/*.jsonl. Every line is validated before it is written."""
from __future__ import annotations

__all__ = ["RESULT_FIELDS", "COND_FIELDS", "COMPUTE_FIELDS", "SchemaError", "validate", "validate_compute"]

RESULT_FIELDS = {
    "cell_hash": str, "model_id": str, "task": str, "subset_id": str, "sample_id": str, "cond": dict,
    "scope": str, "n_vis": int, "n_vis_kept": int, "n_text": int, "kv_mb": float, "state_mb": float,
    "total_mb": float, "gold_nll": float, "gold_ntok": int, "pred": str, "score": float, "state_calls": int,
    "kv_calls": int, "n_chunks": int, "gpu": str, "seconds": float, "units_est": float, "commit": str,
    "timestamp": str,
}
COND_FIELDS = {"kv": str, "state": str, "token_keep": float, "token_method": str, "scope": str}
COMPUTE_FIELDS = {"cell_hash": str, "gpu": str, "seconds": float, "units_est": float}


class SchemaError(ValueError):
    pass


def _check(obj: dict, fields: dict, where: str) -> None:
    missing = [k for k in fields if k not in obj]
    if missing:
        raise SchemaError(f"{where}: missing fields {missing}")
    for k, typ in fields.items():
        v = obj[k]
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) if typ is float else isinstance(v, typ)
        if typ is int and isinstance(v, bool):
            ok = False
        if not ok:
            raise SchemaError(f"{where}: field {k!r} should be {typ.__name__}, got {type(v).__name__}")


def validate(row: dict) -> dict:
    """Raise SchemaError if ``row`` lacks a field or has a wrong type. Extra fields (e.g. meta) are allowed."""
    _check(row, RESULT_FIELDS, "result")
    _check(row["cond"], COND_FIELDS, "result.cond")
    if row["cond"]["scope"] != row["scope"]:
        raise SchemaError("result: scope differs from cond.scope")
    return row


def validate_compute(row: dict) -> dict:
    _check(row, COMPUTE_FIELDS, "compute")
    return row
