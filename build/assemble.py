"""Assemble the ASTERISM items of the identifier sources from their public releases.

    python -m build.assemble --source all --release-root /path/to/releases \\
        --records data/ASTERISM/data/asterism.jsonl --out data/ASTERISM

For each item of a selected source, the item is read from the public release by its source id,
the changes recorded in the ASTERISM record are applied (corrected text, label or answer format,
exchanged options, our scoring entries), the prompt is written as the models received it, and the
images are written under <out>/images/<folder>/. The output <out>/data/<records>_full.jsonl holds
all records of the input, with the items of the selected sources filled in; items of other sources that an earlier
run filled in are kept. If an item fails, the output is not written.
"""
import argparse
import json
import os
import sys
import time

from . import emma, m4bench, marvl, milebench, mmlongbench_doc, q_bench2, vlm2_bench
from . import scoring, text
from .records import answer_text, read_records, write_records

MODULES = [milebench, vlm2_bench, m4bench, mmlongbench_doc, marvl, q_bench2, emma]
SOURCES = {m.NAME: m for m in MODULES}
SLUGS = {m.SLUG: m for m in MODULES}


def fill(record, release, out_root, link):
    """Full record of one item of an identifier source."""
    item = release.item(record)
    if record["prompt"] is not None:
        prompt, options, answer, entry = record["prompt"], record["options"], record["answer"], record["scoring"]
    else:
        label = item["label"]
        kind, public, gold = release.entry(item)
        body = item["text"]
        ours = record["answer"]
        if kind == "mcq":
            if record["corrected"]["label"] and ours is not None:
                gold = {"answer": ours}
            elif ours is not None and ours != gold["answer"]:
                a, b = gold["answer"], ours
                options = dict(public["options"])
                body = text.exchange_options(body, a, options[a], b, options[b])
                options[a], options[b] = options[b], options[a]
                public, gold = dict(public, options=options), {"answer": b}
        elif record["corrected"]["label"] and ours is not None:
            label = ours
            kind, public, gold = release.entry(dict(item, label=label))
        kind, public, gold = scoring.finish(kind, public, gold, body, **release.layout)
        prompt = body if release.official_prompt else text.direct_prompt(body, release.format_line(item, label))
        if record["scoring"] is not None:
            entry = record["scoring"]
            parsed = json.loads(entry)
            public, gold = parsed["public"], parsed["gold"]
        elif kind != record["answer_format"]:
            raise ValueError(f"answer format {kind} instead of {record['answer_format']}")
        else:
            entry = json.dumps({"public": public, "gold": gold}, ensure_ascii=False)
        options = public.get("options") if record["answer_format"] == "mcq" else None
        answer = ours if ours is not None else answer_text(label, record["answer_format"], public, gold)
    paths = [release.image(ref, out_root, link) for ref in record["images"]]
    return dict(record, images=paths, prompt=prompt, options=options, answer=answer, scoring=entry)


FILLED = ("images", "prompt", "options", "answer", "scoring")


def keep_earlier(records, path, names):
    """The records to fill in: the input records, where the items of identifier sources outside `names` that an
    earlier run wrote to `path` (same records, same order, same annotations) are taken from that output."""
    out = list(records)
    try:
        earlier = read_records(path)
        if [e.get("id") for e in earlier] != [r["id"] for r in records]:
            return out
    except (OSError, ValueError, AttributeError, TypeError):
        return out
    kept = 0
    for k, (r, e) in enumerate(zip(records, earlier)):
        same = (all(f in e for f in FILLED) and all(e.get(f) == v for f, v in r.items() if f not in FILLED)
                and all(r[f] is None or e[f] == r[f] for f in ("prompt", "options", "answer", "scoring")))
        if r["release"] == "identifiers" and r["source"] not in names and same and e.get("images") != r["images"]:
            out[k] = dict(r, **{f: e[f] for f in FILLED})
            kept += 1
    if kept:
        print(f"kept {kept} items of other sources from {path}")
    return out


def release_dirs(args, names):
    dirs = {}
    for value in args.release_dir or []:
        if "=" in value:
            key, path = value.split("=", 1)
            module = SOURCES.get(key) or SLUGS.get(key)
            if module is None:
                sys.exit(f"unknown source {key}")
            dirs[module.NAME] = path
        elif len(names) == 1:
            dirs[names[0]] = value
        else:
            sys.exit("with several sources, give --release-root or --release-dir SOURCE=PATH")
    for name in names:
        if name not in dirs:
            if not args.release_root:
                sys.exit(f"no release folder for {name}")
            dirs[name] = os.path.join(args.release_root, SOURCES[name].SLUG)
    return dirs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", action="append", required=True,
                    help="source name or folder name (" + ", ".join(SOURCES) + "), or 'all'; repeatable")
    ap.add_argument("--release-dir", action="append",
                    help="folder of the public release: PATH for a single source, or SOURCE=PATH; repeatable")
    ap.add_argument("--release-root", help="folder that holds one subfolder per source, named " + ", ".join(SLUGS))
    ap.add_argument("--records", required=True, help="asterism.jsonl or asterism_lite.jsonl of the dataset")
    ap.add_argument("--out", required=True, help="dataset root that receives images/ and data/<records>_full.jsonl")
    ap.add_argument("--link", action="store_true", help="hard-link image files that the release ships as files instead of copying them")
    args = ap.parse_args(argv)

    names = []
    for s in args.source:
        if s == "all":
            names += [m.NAME for m in MODULES]
        else:
            module = SOURCES.get(s) or SLUGS.get(s)
            if module is None:
                sys.exit(f"unknown source {s}")
            names.append(module.NAME)
    names = list(dict.fromkeys(names))
    dirs = release_dirs(args, names)

    records = read_records(args.records)
    stem = os.path.splitext(os.path.basename(args.records))[0]
    path = os.path.join(args.out, "data", f"{stem}_full.jsonl")
    out_records = keep_earlier(records, path, names)
    failures = 0
    for name in names:
        t0 = time.time()
        positions = [k for k, r in enumerate(records) if r["source"] == name and r["release"] == "identifiers"]
        release = SOURCES[name].Release(dirs[name], [records[k] for k in positions])
        done = 0
        for k in positions:
            try:
                out_records[k] = fill(records[k], release, args.out, args.link)
                done += 1
            except Exception as e:  # report every failing item, then stop with an error
                failures += 1
                print(f"{name} {records[k]['id']}: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"{name}: {done}/{len(positions)} items in {time.time() - t0:.0f}s")

    if failures:
        sys.exit(f"{failures} items failed; {path} is left as it was")
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    write_records(path, out_records)
    print(f"wrote {path} ({len(out_records)} records)")


if __name__ == "__main__":
    main()
