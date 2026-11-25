"""Check assembled ASTERISM-Train items against the checksums of the released items.

    python -m build.train.verify --records data/ASTERISM-Train/data/asterism_train_full.jsonl

For each identifier item with an entry in build/train/checksums.jsonl, the sha256 of the canonical JSON of prompt,
options and answer is compared with the entry, and so is the sha256 of each image file. Images that the scripts
store as JPEG (entries marked reencoded) depend on the Pillow and libjpeg versions and are counted on their own
line. The check fails when an expected item has no text, when the number of images differs, when text or an image
that should match does not, when an item of the checksums is missing from the records or an identifier record has
no checksum, when an item id repeats, or when no item is checked.
"""
import argparse
import collections
import hashlib
import json
import os
import sys


def text_digest(r):
    s = json.dumps({"prompt": r["prompt"], "options": r["options"], "answer": r["answer"]}, ensure_ascii=False,
                   sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True, help="data/asterism_train_full.jsonl written by build.train.assemble")
    ap.add_argument("--root", help="dataset root that holds images/ (default: the parent of the records folder)")
    ap.add_argument("--checksums", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checksums.jsonl"))
    args = ap.parse_args(argv)
    root = args.root or os.path.dirname(os.path.dirname(os.path.abspath(args.records)))
    sums = {}
    with open(args.checksums, encoding="utf-8") as f:
        for row in (json.loads(line) for line in f if line.strip()):
            if row["id"] in sums:
                sys.exit(f"{args.checksums}: checksum id {row['id']} repeated")
            sums[row["id"]] = row

    count = collections.defaultdict(collections.Counter)
    skipped, failed, seen = collections.Counter(), collections.Counter(), collections.Counter()
    identifier_ids = set()
    with open(args.records, encoding="utf-8") as f:
        for r in (json.loads(line) for line in f if line.strip()):
            seen[r["id"]] += 1
            want = sums.get(r["id"])
            if r["release"] != "identifiers":
                skipped["released with images"] += 1
                continue
            identifier_ids.add(r["id"])
            if want is None:
                failed["identifier item without a checksum entry"] += 1
                continue
            name = r["source"] if r["group"] != "synthetic" else "synthetic"
            c = count[name]
            if r["prompt"] is None:
                c["not assembled"] += 1
                failed["expected item without text"] += 1
                continue
            c["items"] += 1
            if text_digest(r) == want["text"]:
                c["text"] += 1
            else:
                failed["text differs"] += 1
            if len(r["images"]) != len(want["images"]):
                c["image list length differs"] += 1
                failed["image list length differs"] += 1
                continue
            for path, digest in zip(r["images"], want["images"]):
                full = os.path.join(root, path)
                ok = os.path.isfile(full) and file_digest(full) == digest
                if not os.path.isfile(full):
                    c["image missing"] += 1
                    failed["image missing"] += 1
                elif want["reencoded"]:
                    c["jpeg"] += 1
                    c["jpeg_ok"] += ok
                else:
                    c["images"] += 1
                    c["images_ok"] += ok
                    if not ok:
                        failed["image differs"] += 1
    checked = sum(c["items"] for c in count.values())
    if sums.keys() - identifier_ids:
        failed["expected item missing from the records"] += len(sums.keys() - identifier_ids)
    if any(n > 1 for n in seen.values()):
        failed["item id repeated"] += sum(n > 1 for n in seen.values())
    if not checked:
        failed["no item checked"] += 1
    for name, c in sorted(count.items()):
        line = f"{name:24s} text {c['text']}/{c['items']}  images {c['images_ok']}/{c['images']}"
        if c["jpeg"]:
            line += f"  stored as JPEG {c['jpeg_ok']}/{c['jpeg']} (depend on the Pillow version)"
        for key in ("not assembled", "image list length differs", "image missing"):
            if c[key]:
                line += f"  {key} {c[key]}"
        print(line)
    print(f"checked {checked} items; skipped " + (", ".join(f"{v} {k}" for k, v in sorted(skipped.items())) or "none"))
    if failed:
        print("failed: " + ", ".join(f"{v} {k}" for k, v in sorted(failed.items())))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
