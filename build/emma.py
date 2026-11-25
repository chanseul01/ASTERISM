"""EMMA: <subject>/test-00000-of-00001.parquet of the public release.

The prompt is the official EMMA multiple-choice prompt with its chain-of-thought instruction, and
the images are saved as RGB JPEG (quality 92).
"""
import os
import re
import string

from . import images, scoring

NAME = "EMMA"
SLUG = "emma"
MCQ = "{question}\n\n{options}\n\nAnswer with the option's letter from the given choices. "
MCQ_CONTEXT = "{context}\n\n" + MCQ
OPEN = "{question}\n\n"
OPEN_CONTEXT = "{context}\n\n" + OPEN
STEPS = "Please solve the problem step by step. "
_REF = re.compile(r"([^/]+)/test-00000-of-00001\.parquet#pid=(.+)&column=(image_[1-5])")


class Release:
    official_prompt = True
    layout = {"layout": "cot", "answer_line": False}

    def __init__(self, root, records):
        import pyarrow.parquet as pq
        self.rows = {}
        for subject in sorted({r["source_subset"] for r in records}):
            for row in pq.read_table(os.path.join(root, subject, "test-00000-of-00001.parquet")).to_pylist():
                self.rows[row["pid"]] = row

    def item(self, record):
        row = self.rows[record["source_id"]]
        context = row.get("context")
        has_context = context is not None and str(context) not in ("", "nan")
        options = row.get("options")
        options = list(options) if options is not None and not isinstance(options, str) else []
        listed = "".join(f"{ch}: {o}\n" for ch, o in zip(string.ascii_uppercase, options))
        if str(row.get("type", "")).startswith("Multiple") or options:
            template = MCQ_CONTEXT if has_context else MCQ
            body = template.format(context=context, question=str(row["question"]), options=listed)
        else:
            template = OPEN_CONTEXT if has_context else OPEN
            body = template.format(context=context, question=str(row["question"]))
        return {"text": body + STEPS, "label": str(row["answer"]).strip()}

    def entry(self, item):
        return scoring.label_entry(item["text"], item["label"])

    def format_line(self, item, label):
        return None

    def image(self, ref, out_root, link):
        _, pid, column = _REF.fullmatch(ref).groups()
        data = images.jpeg(self.rows[pid][column]["bytes"], quality=92)
        rel = f"images/{SLUG}/{pid}_{column}.jpg"
        images.write_bytes(os.path.join(out_root, rel), data)
        return rel
