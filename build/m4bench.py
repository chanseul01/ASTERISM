"""M4Bench: the task files <task>.tsv of the public release, with the two images as base64."""
import base64
import csv
import math
import os
import re
import sys

from . import images, scoring, text

NAME = "M4Bench"
SLUG = "m4bench"
SUFFIX = "\nAnswer directly with the expected option or short answer."
COLUMNS = {"image1": ("image1", "image_1", "image_0", "image"), "image2": ("image2", "image_2", "image_b")}
_REF = re.compile(r"([^/#]+)\.tsv#index=(.+)&column=(image[12])")


def _text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _decode(value):
    encoded = value.strip()
    if encoded.startswith("data:image"):
        encoded = encoded.split(",", 1)[-1]
    encoded = re.sub(r"\s+", "", encoded)
    encoded += "=" * (-len(encoded) % 4)
    return base64.b64decode(encoded, validate=True)


class Release:
    official_prompt = False
    layout = {"layout": "direct", "answer_line": True}

    def __init__(self, root, records):
        wanted = {}
        for r in records:
            wanted.setdefault(r["source_subset"], set()).add(r["source_id"])
        self.rows = {}
        csv.field_size_limit(sys.maxsize)
        for task, ids in wanted.items():
            with open(os.path.join(root, f"{task}.tsv"), encoding="utf-8", newline="") as f:
                for n, row in enumerate(csv.DictReader(f, delimiter="\t"), 1):
                    index = _text(row.get("index") or n).strip()
                    if index in ids:
                        self.rows[(task, index)] = row
        missing = [(t, i) for t, ids in wanted.items() for i in ids if (t, i) not in self.rows]
        if missing:
            raise KeyError(f"{len(missing)} M4Bench items not in the release, e.g. {missing[0]}")

    def item(self, record):
        row = self.rows[(record["source_subset"], record["source_id"])]
        answer = _text(row.get("response") or row.get("answer")).strip()
        low = answer.casefold()
        if re.fullmatch(r"[a-d]", low):
            label = answer.upper()
        elif low in ("yes", "true"):
            label = "True"
        elif low in ("no", "false"):
            label = "False"
        else:
            label = answer
        return {"text": _text(row.get("query") or row.get("question")).strip() + SUFFIX, "label": label}

    def entry(self, item):
        return scoring.label_entry(item["text"], item["label"])

    def format_line(self, item, label):
        return text.free_format(label)

    def image(self, ref, out_root, link):
        task, index, column = _REF.fullmatch(ref).groups()
        row = self.rows[(task, index)]
        value = next(row[c] for c in COLUMNS[column] if row.get(c) and row[c].strip())
        data = _decode(value)
        rel = f"images/{SLUG}/{task}/{index}_{column}{images.suffix_of(data)}"
        images.write_bytes(os.path.join(out_root, rel), data)
        return rel
