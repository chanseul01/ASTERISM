"""Q-Bench2: q-bench2-a1-dev.jsonl and all_single_images.zip of the public release.

source_id is the line of an item in q-bench2-a1-dev.jsonl, counted from 0.
"""
import json
import math
import os
import re
import zipfile

from . import images, scoring, text

NAME = "Q-Bench2"
SLUG = "q-bench2"
LETTER_PROMPT = "Answer with the option's letter from the given choices directly."


def _text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _norm(value):
    return re.sub(r"\s+", " ", _text(value).strip()).casefold()


def _letter(answer, options):
    value = _text(answer).strip()
    if re.fullmatch(r"[A-Da-d]", value):
        return value.upper()
    if isinstance(answer, int) or value.isdigit():
        if 0 <= int(answer) < len(options):
            return chr(65 + int(answer))
    for k, option in enumerate(options):
        if _norm(option) == _norm(answer):
            return chr(65 + k)
    raise ValueError(f"answer {answer!r} is not one of the options")


class Release:
    official_prompt = False
    layout = {"layout": "direct", "answer_line": True}

    def __init__(self, root, records):
        with open(os.path.join(root, "q-bench2-a1-dev.jsonl"), encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.archive = zipfile.ZipFile(os.path.join(root, "all_single_images.zip"))

    def item(self, record):
        row = self.rows[int(record["source_id"])]
        options = list(row["candidates"])
        choices = "\n".join(f"{chr(65 + k)}. {_text(o).strip()}" for k, o in enumerate(options))
        body = f"{_text(row['question']).strip()}\nChoices:\n{choices}\n{LETTER_PROMPT}"
        return {"text": body, "label": _letter(row["correct_choice"], options)}

    def entry(self, item):
        return scoring.label_entry(item["text"], item["label"])

    def format_line(self, item, label):
        return text.LETTER

    def image(self, ref, out_root, link):
        data = self.archive.read(ref)
        rel = f"images/{SLUG}/{ref.split('/', 1)[1]}"
        images.write_bytes(os.path.join(out_root, rel), data)
        return rel
