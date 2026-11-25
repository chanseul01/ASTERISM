"""Reading item records and generation files (JSON Lines)."""
import collections
import json


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_records(path, lite=False):
    """Item records of ASTERISM; with lite=True only the 400 items of ASTERISM-Lite.
    The "scoring" field, stored as a JSON string, is parsed into a dict."""
    records = read_jsonl(path)
    if lite:
        records = [r for r in records if r.get("in_lite")]
    for r in records:
        if isinstance(r.get("scoring"), str):
            r["scoring"] = json.loads(r["scoring"])
    dup = [i for i, c in collections.Counter(r["id"] for r in records).items() if c > 1]
    if dup:
        raise ValueError(f"{path}: duplicate ids, e.g. {dup[0]}")
    if not records:
        raise ValueError(f"{path}: no items selected")
    return records


def check_complete(records, fields=("prompt", "images", "scoring")):
    """Fail if an item still lacks fields that only the build scripts fill in for identifier-only sources."""
    ok = {"prompt": lambda r: isinstance(r.get("prompt"), str), "images": lambda r: bool(r.get("images")),
          "scoring": lambda r: isinstance(r.get("scoring"), dict)}
    bad = [r["id"] for r in records if not all(ok[f](r) for f in fields)]
    if bad:
        raise ValueError(f"{len(bad)} items lack one of {', '.join(fields)} (e.g. {bad[0]}); "
                         f"assemble the items of identifier-only sources with the build scripts first")
