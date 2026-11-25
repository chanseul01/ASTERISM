"""Assemble the ASTERISM-Train items of the identifier sources.

    python -m build.train.assemble --records data/ASTERISM-Train/data/asterism_train.jsonl --out data/ASTERISM-Train \
        --m4 $ROOT/m4-instruct --mantis $ROOT/mantis-instruct --leopard $ROOT/leopard-instruct \
        --mgrounding $ROOT/mgrounding --spair $ROOT/spair --parquet-images $ROOT/parquet-images \
        --nlvr2 $ROOT/nlvr2 --star $ROOT/star --spot-the-diff $ROOT/spot-the-diff --clevr-change $ROOT/clevr-change

Each folder holds a release as described in README.md, with its image archives extracted (for Mantis-Instruct,
the train_images.zip archives can also stay packed). Items of public datasets and instruction collections are
written from the `build` recipe of their record: our own text and slices of the source record. Synthetic items
keep their prompt, options and answer, and their images are made from the source images by the operations in
their references. A release whose option is left out is skipped. The output
<out>/data/asterism_train_full.jsonl holds all records, with the assembled items filled in, and the images go to
<out>/images/<group-source>/<id>/<k>.<ext>.
"""
import argparse
import collections
import json
import os
import re
import sys
import time

from .releases import NLVR2, STAR, Files, ParquetImages, LeopardInstruct, M4Instruct, MantisInstruct, MGrounding, SentencePairs

FIELDS = ["id", "group", "source", "source_id", "images", "num_images", "prompt", "options", "answer", "answer_format",
          "release", "license", "build"]
OPTION = re.compile(r"(?m)^\s*\(?([A-J])[\.\):]\s+(.*?)\s*$")


def release_of(record):
    """Which release a record of an identifier source is read from (None: not covered by these scripts)."""
    if record["group"] == "synthetic":
        for name, prefix in (("mgrounding", "MGrounding-630k/"), ("spair", "JPEGImages/"), ("parquet-images", ("genimage/", "wikiart/"))):
            if all(x.startswith(prefix) for x in record["images"]):
                return name
        return None
    if not record["build"]:
        return None
    if record["group"] == "M4-Instruct" or record["source"] == "WebQA":
        return "m4"
    if record["group"] == "training split":
        return {"NLVR2": "nlvr2", "STAR": "star", "Spot-the-Diff": "spot-the-diff", "CLEVR-Change": "clevr-change"}.get(record["source"])
    if record["group"] in ("Mantis-Instruct", "other"):
        return "mantis"
    if record["group"] == "LEOPARD":
        return "leopard"
    return None


def fill(segments, fields):
    return "".join(s if isinstance(s, str) else fields[s[0]][s[1]:s[2]] for s in segments)


def slug(record):
    s = re.sub(r"[^a-z0-9]+", "-", f"{record['group']} {record['source']}".lower()).strip("-")
    return s.replace("training-split-", "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True, help="data/asterism_train.jsonl of the dataset")
    ap.add_argument("--out", required=True, help="dataset root that receives images/ and data/asterism_train_full.jsonl")
    ap.add_argument("--m4", help="folder of M4-Instruct-Data")
    ap.add_argument("--mantis", help="folder of Mantis-Instruct")
    ap.add_argument("--leopard", help="folder of Leopard-Instruct")
    ap.add_argument("--mgrounding", help="folder that holds MGrounding-630k/")
    ap.add_argument("--spair", help="folder SPair-71k (JPEGImages/, PairAnnotation/)")
    ap.add_argument("--parquet-images", help="folder with genimage/ (TheKernel01/Tiny-GenImage data/train-*.parquet) and "
                    "wikiart/ (Artificio/WikiArt data/train-*.parquet)")
    ap.add_argument("--nlvr2", help="folder with train.json of NLVR2 and data/train-*.parquet of pingzhili/nlvr2-resized")
    ap.add_argument("--star", help="folder with STAR_train.json and the Charades_v1_480/ videos")
    ap.add_argument("--spot-the-diff", help="folder with train.json of Spot-the-Diff and resized_images/")
    ap.add_argument("--clevr-change", help="folder with data/ of CLEVR-Change (images/, sc_images/, nsc_images/), sentences.json "
                    "(the sentences column of the train split of Lancelot53/clevr-change) and the train-*.parquet files of "
                    "Lancelot53/clevr-change")
    args = ap.parse_args(argv)

    with open(args.records, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    readers = {"m4": (M4Instruct, args.m4), "mantis": (MantisInstruct, args.mantis),
               "leopard": (LeopardInstruct, args.leopard), "mgrounding": (MGrounding, args.mgrounding),
               "spair": (Files, args.spair), "parquet-images": (ParquetImages, args.parquet_images), "nlvr2": (NLVR2, args.nlvr2), "star": (STAR, args.star),
               "spot-the-diff": (lambda root, recs: SentencePairs(root, recs, "Spot-the-Diff"), args.spot_the_diff),
               "clevr-change": (lambda root, recs: SentencePairs(root, recs, "CLEVR-Change"), args.clevr_change)}
    groups = collections.defaultdict(list)
    for k, r in enumerate(records):
        if r["release"] == "identifiers":
            groups[release_of(r)].append(k)
    out, failures = list(records), 0
    for name, positions in groups.items():
        if name is None:
            print(f"not covered by these scripts: {len(positions)} items")
            continue
        cls, root = readers[name]
        if not root:
            print(f"{name}: no folder given, {len(positions)} items skipped")
            continue
        t0 = time.time()
        release = cls(root, [records[k] for k in positions])
        done = 0
        for k in positions:
            r = records[k]
            try:
                full = dict(r)
                if r["build"]:
                    recipe = json.loads(r["build"])
                    fields = release.fields(r)
                    full["prompt"] = fill(recipe["prompt"], fields)
                    full["answer"] = fill(recipe["answer"], fields)
                    if r["answer_format"] == "mcq":
                        full["options"] = {a: t for a, t in OPTION.findall(full["prompt"])}
                paths = []
                for j, ref in enumerate(r["images"]):
                    data, ext = release.image(r, j, ref)
                    rel = f"images/{slug(r)}/{r['id']}/{j}{ext}"
                    dst = os.path.join(args.out, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(dst + ".part", "wb") as f:
                        f.write(data)
                    os.replace(dst + ".part", dst)
                    paths.append(rel)
                full["images"] = paths
                out[k] = full
                done += 1
            except Exception as e:  # report every failing item, then stop with an error
                failures += 1
                print(f"{r['id']}: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"{name}: {done}/{len(positions)} items in {time.time() - t0:.0f}s")

    path = os.path.join(args.out, "data", "asterism_train_full.jsonl")
    if failures:
        sys.exit(f"{failures} items failed; {path} is left as it was")
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n")
    os.replace(path + ".part", path)
    print(f"wrote {path} ({len(out)} records)")


if __name__ == "__main__":
    main()
