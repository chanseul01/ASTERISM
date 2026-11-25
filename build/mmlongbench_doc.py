"""MMLongBench-Doc: data/train-00000-of-00001.parquet and the PDF files in documents/.

Pages are rendered at 72 dpi and saved as JPEG (quality 90); each record lists the pages of its item (at
most 60).
"""
import os
import re

from . import images, scoring, text

NAME = "MMLongBench-Doc"
SLUG = "mmlongbench-doc"
QUESTION = ("Answer the question using a single word or phrase. If the question cannot be answered from the given pages, "
            "answer \"Not answerable\".\nQuestion: {q}")
_REF = re.compile(r"documents/(.+)#page=(\d+)")


class Release:
    official_prompt = False
    layout = {"layout": "direct", "answer_line": True}

    def __init__(self, root, records):
        import pyarrow.parquet as pq
        self.root = root
        self.rows = pq.read_table(os.path.join(root, "data", "train-00000-of-00001.parquet")).to_pylist()
        self._documents = {}
        self._written = set()

    def item(self, record):
        row = self.rows[int(record["source_id"])]
        return {"text": QUESTION.format(q=str(row["question"])), "label": str(row["answer"]),
                "answer_format": str(row.get("answer_format", ""))}

    def entry(self, item):
        return scoring.document_entry(item["text"], item["label"], item["answer_format"])

    def format_line(self, item, label):
        return text.free_format(label)

    def _document(self, name):
        if name not in self._documents:
            try:
                import pymupdf
            except ImportError:
                import fitz as pymupdf
            self._documents[name] = pymupdf.open(os.path.join(self.root, "documents", name))
        return self._documents[name]

    def image(self, ref, out_root, link):
        name, page = _REF.fullmatch(ref).groups()
        stem = name[:-4] if name.lower().endswith(".pdf") else name
        rel = f"images/{SLUG}/{stem.replace('/', '_')}_p{page}.jpg"
        if rel not in self._written:
            images.write_bytes(os.path.join(out_root, rel), images.render_page(self._document(name), int(page)))
            self._written.add(rel)
        return rel
