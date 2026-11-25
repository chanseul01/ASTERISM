"""ASTERISM records: reading, writing, the written-out gold answer and checksums."""
import ast
import hashlib
import json
import os
import re

FIELDS = ["id", "source", "source_subset", "source_id", "category", "images", "num_images", "prompt", "options", "answer",
          "answer_format", "scoring", "is_rewrite", "rewrite_of", "corrected", "labels", "release", "license",
          "image_origin", "use_terms", "in_lite"]


def read_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_records(path, records):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _first_number(s):
    m = _NUMBER.search(s)
    return float(m.group(0).replace(",", "")) if m else None


def _as_list(s):
    s = s.strip()
    if s.startswith("["):
        try:
            return [str(v).strip() for v in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            pass
    return [v.strip() for v in s.strip("()[]").split(",") if v.strip()]


def answer_text(label, kind, public, gold):
    """Gold answer as a string: the label of the source when it agrees with the scoring entry,
    otherwise the gold of the scoring entry written out."""
    g = str(label).strip()
    if kind == "mcq":
        return gold["answer"]
    if gold.get("unanswerable"):
        a = (gold.get("accepted") or ["Not answerable"])[0]
        return g if g == a else a
    if kind == "polar":
        v = bool(gold["value"])
        low = g.lower()
        if (v and low in ("yes", "true", "1", "t")) or (not v and low in ("no", "false", "0", "f")):
            return g
        yes, no = (public.get("polar_as") or "Yes/No").split("/")
        return yes if v else no
    if kind in ("number", "int"):
        n, v = _first_number(g), float(gold["value"])
        if n is not None and abs(n - v) <= 1e-9 * max(1.0, abs(v)):
            return g
        return str(int(v)) if v == int(v) else repr(v)
    if kind == "text":
        return g if g in gold["accepted"] else gold["accepted"][0]
    if kind == "identifier":
        return str(gold["value"])
    if kind in ("anyof", "multiset", "set"):
        mine, want = _as_list(g), [str(v) for v in gold["items"]]
        same = sorted(mine) == sorted(want) if kind == "multiset" else set(mine) == set(want)
        return g if same else ", ".join(want)
    if kind == "tuple":
        want = [str(v) for v in gold["fields"]]
        return g if _as_list(g) == want else "(" + ", ".join(want) + ")"
    if kind == "ipair":
        want = list(gold["pair"])
        try:
            same = [int(v) for v in _as_list(g)] == want
        except ValueError:
            same = False
        return g if same else f"({want[0]}, {want[1]})"
    raise ValueError(f"unknown answer format {kind}")


def text_digest(record):
    """sha256 of the canonical JSON of prompt, options, answer and scoring."""
    scoring = json.loads(record["scoring"]) if record.get("scoring") else None
    payload = {"prompt": record.get("prompt"), "options": record.get("options"), "answer": record.get("answer"), "scoring": scoring}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
