"""Scoring of model outputs for ASTERISM items."""
import collections
import json

from .judge import judge


def contract_of(record):
    """(answer_format, public contract, gold contract) of an item record.

    The record's "scoring" field holds {"public": {...}, "gold": {...}}, as a dict or as the JSON string
    of the dataset files. If "options" is not repeated inside "public", the record's top-level "options"
    are used.
    """
    kind = record.get("answer_format")
    sc = record.get("scoring")
    if isinstance(sc, str):  # as stored in the dataset files
        sc = json.loads(sc)
    if not kind or not isinstance(sc, dict) or not ("public" in sc or "gold" in sc):
        raise ValueError(f"{record.get('id')}: no scoring contract (items of identifier-only sources must be "
                         f"assembled from their public release first)")
    pub = dict(sc.get("public") or {})
    gold = dict(sc.get("gold") or {})
    if kind == "mcq":
        if "options" not in pub:
            if not record.get("options"):
                raise ValueError(f"{record.get('id')}: multiple-choice item without options")
            pub["options"] = record["options"]
        if "answer" not in gold:
            if record.get("answer") not in pub["options"]:
                raise ValueError(f"{record.get('id')}: multiple-choice item without an option id as answer")
            gold["answer"] = record["answer"]
    return kind, pub, gold


def score_output(record, output):
    """Judge one output against one item. Returns the judge's result dict (score, valid, how, ...)."""
    kind, pub, gold = contract_of(record)
    return judge(kind, pub, gold, output)


def score_generations(records, generations):
    """Score a list of generation rows {"id", "output"} against the item records.

    Returns (results, errors). results maps item id to the judge's result. Duplicate rows, rows of
    unknown items, items without a row and judge exceptions are errors; any error means the run must
    not be reported.
    """
    by_id = {r["id"]: r for r in records}
    errors = []
    counts = collections.Counter(g["id"] for g in generations)
    dup = sorted(i for i, c in counts.items() if c > 1)
    unknown = sorted(i for i in counts if i not in by_id)
    missing = sorted(i for i in by_id if i not in counts)
    if dup:
        errors.append(f"duplicate rows: {len(dup)} (e.g. {dup[0]})")
    if unknown:
        errors.append(f"rows of unknown items: {len(unknown)} (e.g. {unknown[0]})")
    if missing:
        errors.append(f"items without a row: {len(missing)} (e.g. {missing[0]})")
    results = {}
    for g in generations:
        i = g["id"]
        if i not in by_id or i in results:
            continue
        try:
            results[i] = score_output(by_id[i], g.get("output", ""))
        except Exception as e:
            errors.append(f"judge error on {i}: {type(e).__name__}: {e}")
    return results, errors
