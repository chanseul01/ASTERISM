"""Per-category accuracy and the ASTERISM score.

The ASTERISM score is the macro average of the accuracy over the seven task categories. Every item
counts once in its category, and a rewritten item keeps the crosswalk category of the item it was rewritten from.
"""
CATEGORIES = ("aggregate", "compare_change", "reasoning_verification", "spatial_correspondence", "temporal_action", "text_web", "retrieval")
SHORT_NAMES = {"aggregate": "Agg", "compare_change": "Cmp", "reasoning_verification": "Rsn", "spatial_correspondence": "Spa",
               "temporal_action": "Tmp", "text_web": "Txt", "retrieval": "Ret"}


def summarize(records, results):
    """Accuracy per category (in %), their macro average and the share of outputs without a valid answer.

    records: item records (each with "id" and "category"); results: item id -> judge result
    ({"score": 0/1, "valid": bool, ...}). Every record needs a result.
    """
    missing = [r["id"] for r in records if r["id"] not in results]
    if missing:
        raise ValueError(f"{len(missing)} items have no result (e.g. {missing[0]})")
    per = {c: [] for c in CATEGORIES}
    for r in sorted(records, key=lambda x: x["id"]):
        c = r["category"]
        if c not in per:
            raise ValueError(f"{r['id']}: unknown category {c!r}")
        per[c].append(results[r["id"]])
    empty = [c for c in CATEGORIES if not per[c]]
    if empty:
        raise ValueError(f"no items in categories {empty}")
    acc = {c: 100 * sum(float(x["score"]) for x in per[c]) / len(per[c]) for c in CATEGORIES}
    n = sum(len(v) for v in per.values())
    invalid = 100 * sum(1 for v in per.values() for x in v if not x.get("valid", True)) / n
    return {"score": sum(acc[c] for c in CATEGORIES) / len(CATEGORIES),
            "categories": acc,
            "n": n,
            "n_per_category": {c: len(per[c]) for c in CATEGORIES},
            "invalid": invalid}


def format_summary(summary):
    """Header and one row: the seven category accuracies and the ASTERISM score, one decimal."""
    head = " ".join(f"{SHORT_NAMES[c]:>5}" for c in CATEGORIES) + "   Avg"
    vals = " ".join(f"{summary['categories'][c]:5.1f}" for c in CATEGORIES) + f" {summary['score']:5.1f}"
    return head + "\n" + vals
