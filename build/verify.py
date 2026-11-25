"""Check assembled items against the checksums of the items that ASTERISM evaluates.

    python -m build.verify --records data/ASTERISM/data/asterism_full.jsonl

For each item of an identifier source, the sha256 of the canonical JSON of prompt, options, answer
and scoring is compared with build/checksums/<source>.jsonl, and so is the sha256 of each image file.
Rendered PDF pages (MMLongBench-Doc) and re-encoded JPEG images (EMMA) depend on the PyMuPDF and
Pillow versions, so a different digest of these files is reported but does not make the check fail.
A missing image file or a different number of images fails the check for every source, and so do a
repeated item id, an item of the checksums that ASTERISM records lack, and a run that checks no item.
"""
import argparse
import collections
import glob
import json
import os
import sys

from .assemble import SLUGS, SOURCES
from .records import file_digest, read_records, text_digest

VERSION_DEPENDENT = {"MMLongBench-Doc": "rendered PDF pages", "EMMA": "re-encoded JPEG images"}


def load_checksums(folder):
    """Checksum rows by item id, and the item ids of each source."""
    sums, ids = {}, collections.defaultdict(set)
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        name = SLUGS[os.path.splitext(os.path.basename(path))[0]].NAME
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    sums[row["id"]] = row
                    ids[name].add(row["id"])
    return sums, ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True, help="assembled records, e.g. data/asterism_full.jsonl")
    ap.add_argument("--root", help="dataset root that holds images/ (default: the parent of the records folder)")
    ap.add_argument("--checksums", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checksums"))
    ap.add_argument("--source", action="append", help="check only this source (name as in build/assemble.py); repeatable")
    args = ap.parse_args(argv)
    only = set(args.source or [])
    unknown = only - set(SOURCES)
    if unknown:
        sys.exit(f"unknown source {sorted(unknown)} (one of: {', '.join(SOURCES)})")
    root = args.root or os.path.dirname(os.path.dirname(os.path.abspath(args.records)))
    sums, source_ids = load_checksums(args.checksums)
    records = read_records(args.records)
    seen = collections.Counter(r["id"] for r in records)
    full_set = any(not r["in_lite"] for r in records)  # ASTERISM, not ASTERISM-Lite: every checksum item is expected

    count = collections.defaultdict(collections.Counter)
    for r in records:
        if r["release"] != "identifiers" or (only and r["source"] not in only):
            continue
        c = count[r["source"]]
        want = sums.get(r["id"])
        if want is None:
            c["unknown"] += 1
            continue
        c["items"] += 1
        c["text"] += r["prompt"] is not None and text_digest(r) == want["text"]
        if len(r["images"]) != len(want["images"]):
            c["images"] += len(want["images"])
            c["count_mismatch"] += 1
            continue
        for path, digest in zip(r["images"], want["images"]):
            c["images"] += 1
            full = os.path.join(root, path)
            if not os.path.isfile(full):
                c["missing"] += 1
                continue
            c["images_ok"] += file_digest(full) == digest

    failed = False
    for name in SOURCES:
        if only and name not in only:
            continue
        c = count[name]
        absent = len(source_ids[name] - seen.keys()) if full_set else 0
        if not c and not absent:
            continue
        line = f"{name:16s} text {c['text']}/{c['items']}  images {c['images_ok']}/{c['images']}"
        if name in VERSION_DEPENDENT:
            line += f"  ({VERSION_DEPENDENT[name]}, depend on library versions)"
        print(line)
        if c["missing"] or c["count_mismatch"]:
            print(f"{'':16s} {c['missing']} image files missing, {c['count_mismatch']} items with a different number of images")
        if absent:
            print(f"{'':16s} {absent} items of the checksums missing from the records")
        if (c["text"] != c["items"] or c["unknown"] or c["missing"] or c["count_mismatch"] or absent
                or (name not in VERSION_DEPENDENT and c["images_ok"] != c["images"])):
            failed = True
        if c["unknown"]:
            print(f"{'':16s} {c['unknown']} items without a checksum")
    total = sum(c["items"] for c in count.values())
    print(f"{'all':16s} text {sum(c['text'] for c in count.values())}/{total}")
    repeated = sum(n > 1 for n in seen.values())
    if repeated:
        print(f"{repeated} item ids appear more than once in the records")
        failed = True
    if total == 0:
        print("no item of an identifier source was checked")
        failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
