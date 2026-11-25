#!/usr/bin/env python
"""Score a generations file on ASTERISM or ASTERISM-Lite with the rule-based scorer.

The generations file has one JSON object per line with "id" and "output" (and optionally "finished",
false when the output hit the token limit). Every selected item needs exactly one row.

Example:
  python scripts/score.py --data data/ASTERISM/data/asterism_full.jsonl --generations outputs/qwen2.5-vl-7b-instruct/generations.jsonl
  python scripts/score.py --data data/ASTERISM/data/asterism_full.jsonl --generations ... --lite
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asterism import __version__  # noqa: E402
from asterism.data import check_complete, load_records, read_jsonl, write_jsonl  # noqa: E402
from asterism.metrics import format_summary, summarize  # noqa: E402
from asterism.scoring import score_generations  # noqa: E402


def score_run(records, generations, out_dir, lite=False):
    """Score, write scores(.lite).jsonl and summary(.lite).json to out_dir, and return the summary.
    Exits with status 1 if the run is incomplete or a row cannot be judged."""
    ids = {r["id"] for r in records}
    if lite:  # a full run can be scored on ASTERISM-Lite
        generations = [g for g in generations if g["id"] in ids]
    results, errors = score_generations(records, generations)
    sfx = ".lite" if lite else ""
    os.makedirs(out_dir, exist_ok=True)
    if errors:
        for e in errors[:20]:
            print("error:", e, file=sys.stderr)
        print(f"{len(errors)} errors: no score is reported", file=sys.stderr)
        sys.exit(1)
    write_jsonl(os.path.join(out_dir, f"scores{sfx}.jsonl"), [dict(id=i, **results[i]) for i in sorted(results)])
    summary = summarize(records, results)
    gen_by_id = {g["id"]: g for g in generations}
    summary["unfinished"] = sum(1 for i in ids if gen_by_id[i].get("finished") is False)
    summary["benchmark"] = "ASTERISM-Lite" if lite else "ASTERISM"
    expected = 400 if lite else 18696
    if len(records) != expected:  # the score of the benchmark needs all of its items
        print(f"warning: {len(records)} of the {expected} items of {summary['benchmark']}", file=sys.stderr)
        summary["benchmark"] += f" (subset: {len(records)} of {expected} items)"
    summary["scorer_version"] = __version__
    with open(os.path.join(out_dir, f"summary{sfx}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(format_summary(summary))
    print(f"{summary['n']} items, no valid answer {summary['invalid']:.2f}%, token limit reached {summary['unfinished']}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a generations file on ASTERISM or ASTERISM-Lite.")
    ap.add_argument("--data", required=True, help="item records (JSON Lines)")
    ap.add_argument("--generations", required=True, help="model outputs, one row {\"id\", \"output\"} per item (JSON Lines)")
    ap.add_argument("--lite", action="store_true", help="score the 400 items of ASTERISM-Lite")
    ap.add_argument("--out", default=None, help="output folder (default: the folder of --generations)")
    a = ap.parse_args(argv)
    for p in (a.data, a.generations):
        if not os.path.exists(p) or os.path.isdir(p):
            sys.exit(f"error: no file {p}")
    records = load_records(a.data, lite=a.lite)
    a.lite = a.lite or all(r.get("in_lite") for r in records)  # a file of ASTERISM-Lite records
    try:
        check_complete(records, fields=("scoring",))
    except ValueError as e:
        sys.exit(f"error: {e}")
    score_run(records, read_jsonl(a.generations), a.out or os.path.dirname(os.path.abspath(a.generations)), lite=a.lite)


if __name__ == "__main__":
    main()
