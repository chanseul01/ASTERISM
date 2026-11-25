"""MaRVL: the parquet files in data/ of the public release, with the two images stored in the files."""
import glob
import math
import os
import re

from . import images, scoring, text

NAME = "MaRVL"
SLUG = "marvl"
SUFFIX = "\nAnswer True or False directly."
_REF = re.compile(r"(data/[^#]+\.parquet)#id=(.+)&column=(left_img|right_img)")


def _text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


class Release:
    official_prompt = False
    layout = {"layout": "direct", "answer_line": True}

    def __init__(self, root, records):
        import pyarrow.parquet as pq
        wanted = {r["source_id"] for r in records}
        self.rows = {}
        for path in sorted(glob.glob(os.path.join(root, "data", "*.parquet"))):
            pf = pq.ParquetFile(path)
            for g in range(pf.num_row_groups):
                ids = pf.read_row_group(g, columns=["id"]).column("id").to_pylist()
                hits = [k for k, i in enumerate(ids) if i in wanted]
                if not hits:
                    continue
                table = pf.read_row_group(g, columns=["id", "hypothesis", "label", "left_img", "right_img"])
                for k in hits:
                    row = table.slice(k, 1).to_pylist()[0]
                    self.rows[row["id"]] = row
        missing = wanted - set(self.rows)
        if missing:
            raise KeyError(f"{len(missing)} MaRVL items not in the release, e.g. {sorted(missing)[0]}")

    def item(self, record):
        row = self.rows[record["source_id"]]
        return {"text": _text(row["hypothesis"]).strip() + SUFFIX, "label": "True" if row["label"] else "False"}

    def entry(self, item):
        return scoring.label_entry(item["text"], item["label"])

    def format_line(self, item, label):
        return text.true_false_format(label)

    def image(self, ref, out_root, link):
        _, item_id, column = _REF.fullmatch(ref).groups()
        data = bytes(self.rows[item_id][column]["bytes"])
        side = column.split("_")[0]
        rel = f"images/{SLUG}/{item_id}_{side}{images.suffix_of(data)}"
        images.write_bytes(os.path.join(out_root, rel), data)
        return rel
