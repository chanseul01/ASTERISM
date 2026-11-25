"""VLM2-Bench: VLM2Bench_img.tsv of the public release, with the images as base64 lists."""
import ast
import base64
import csv
import os
import re
import sys

from . import images, scoring, text

NAME = "VLM2-Bench"
SLUG = "vlm2-bench"
TRUE_FALSE = {"gc-mat", "gc-trk", "oc-cpr", "pc-cpr"}
COUNT = {"oc-cnt", "pc-cnt"}
GROUP = {"oc-grp", "pc-grp"}
_REF = re.compile(r"VLM2Bench_img\.tsv#index=(.+)&image=(\d+)")


class Release:
    official_prompt = False
    layout = {"layout": "direct", "answer_line": True}

    def __init__(self, root, records):
        wanted = {r["source_id"] for r in records}
        self.rows = {}
        csv.field_size_limit(sys.maxsize)
        with open(os.path.join(root, "VLM2Bench_img.tsv"), encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                index = row["index"].strip()
                if index in wanted:
                    self.rows[index] = row
        missing = wanted - set(self.rows)
        if missing:
            raise KeyError(f"{len(missing)} VLM2-Bench items not in the release, e.g. {sorted(missing)[0]}")

    def item(self, record):
        row = self.rows[record["source_id"]]
        category = row["category"].strip()
        raw = row["answer"].strip().strip("'\"")
        if category in TRUE_FALSE:
            label, style = {"T": "True", "F": "False"}.get(raw, raw), "true_false"
        elif category in GROUP:
            label, style = raw.split(")")[0].strip(), "letter"
        elif category in COUNT:
            label, style = raw, "free"
        else:
            raise ValueError(f"unknown VLM2-Bench category {category}")
        return {"text": row["question"], "label": label, "style": style}

    def entry(self, item):
        return scoring.label_entry(item["text"], item["label"], true_false_letters=True)

    def format_line(self, item, label):
        if item["style"] == "true_false":
            return text.true_false_format(label)
        return text.LETTER if item["style"] == "letter" else text.free_format(label)

    def image(self, ref, out_root, link):
        m = _REF.fullmatch(ref)
        index, k = m.group(1), int(m.group(2))
        encoded = ast.literal_eval(self.rows[index]["image"])
        if isinstance(encoded, str):
            encoded = [encoded]
        data = base64.b64decode(encoded[k - 1])
        rel = f"images/{SLUG}/{index}_{k}{images.suffix_of(data)}"
        images.write_bytes(os.path.join(out_root, rel), data)
        return rel
