#!/usr/bin/env python
"""Generate with vLLM and score a model on ASTERISM or ASTERISM-Lite.

Writes to --out: generations.jsonl (resumable for the same model, settings and inputs: items with a row are skipped), run_info.json
(settings, library versions, GPUs), scores.jsonl and summary.json (.lite for ASTERISM-Lite).

Images are read relative to the root of the dataset repository, by default the folder above the one that
holds the records file (data/ASTERISM/data/asterism_full.jsonl -> data/ASTERISM/).

Examples:
  python scripts/evaluate.py --config configs/eval/qwen2.5-vl-7b-instruct.yaml \
      --data data/ASTERISM/data/asterism_full.jsonl --out outputs/qwen2.5-vl-7b-instruct
  python scripts/evaluate.py --config configs/eval/cica-qwen3.5-9b.yaml --model path/to/CICA-Qwen3.5-9B \
      --data data/ASTERISM-Lite/data/asterism_lite_full.jsonl --out outputs/cica-qwen3.5-9b --lite
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from asterism.data import check_complete, load_records, read_jsonl  # noqa: E402
from asterism.runner import environment, load_config, read_rows  # noqa: E402
from score import score_run  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Generate with vLLM and score a model on ASTERISM or ASTERISM-Lite.")
    ap.add_argument("--config", required=True, help="model config (configs/eval/*.yaml)")
    ap.add_argument("--data", required=True, help="item records (JSON Lines)")
    ap.add_argument("--image-root", default=None,
                    help="root of the dataset repository, where relative image paths start (default: the folder above the one of --data)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--lite", action="store_true", help="evaluate the 400 items of ASTERISM-Lite")
    ap.add_argument("--model", default=None, help="checkpoint path or Hugging Face id (overrides the config)")
    ap.add_argument("--processor", default=None, help="processor and tokenizer path or id (overrides the config)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="items per vLLM call (overrides the config)")
    return ap.parse_args(argv)


def input_digest(record):
    """Digest of what the model receives for an item: prompt, options and image paths."""
    s = json.dumps([record.get("prompt"), record.get("options"), record.get("images")], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main(argv=None):
    a = parse_args(argv)
    cfg = load_config(a.config)
    if a.model:
        if cfg["processor"] == cfg["model"]:  # the processor follows the model unless the config names its own
            cfg["processor"], cfg["processor_revision"] = a.model, None
        cfg["model"], cfg["revision"] = a.model, None
    if a.processor:
        cfg["processor"], cfg["processor_revision"] = a.processor, None
    if a.gpu_memory_utilization is not None:
        cfg["gpu_memory_utilization"] = a.gpu_memory_utilization
    if a.batch_size is not None:
        cfg["batch_size"] = a.batch_size
    for key in ("model", "processor"):  # a local checkpoint is recorded and compared by its resolved path
        if cfg[key] and os.path.isdir(cfg[key]):
            cfg[key] = os.path.realpath(cfg[key])
    records = load_records(a.data, lite=a.lite)
    a.lite = a.lite or all(r.get("in_lite") for r in records)  # a file of ASTERISM-Lite records
    try:
        check_complete(records)
    except ValueError as e:
        sys.exit(f"error: {e}")
    image_root = a.image_root or os.path.dirname(os.path.dirname(os.path.abspath(a.data)))
    os.makedirs(a.out, exist_ok=True)
    gen_path, info_path = os.path.join(a.out, "generations.jsonl"), os.path.join(a.out, "run_info.json")
    inputs = {r["id"]: input_digest(r) for r in records}
    if os.path.isfile(gen_path) and os.path.getsize(gen_path):  # resume only for the same model, settings and inputs
        runtime = ("gpu_memory_utilization", "batch_size")
        try:
            with open(info_path) as f:
                before_info = json.load(f)
            before, before_inputs = before_info["config"], dict(before_info["inputs"])
            differ = sorted(k for k in set(before) | set(cfg) if k not in runtime and before.get(k) != cfg.get(k))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            sys.exit(f"error: {a.out} holds generations without a readable run_info.json; choose another --out")
        if differ:
            sys.exit(f"error: {a.out} holds generations of another model or setting (differs in {', '.join(differ)}); "
                     f"choose another --out")
        changed = sorted(r["id"] for r in read_rows(gen_path) if r["id"] in inputs and before_inputs.get(r["id"]) != inputs[r["id"]])
        if changed:
            sys.exit(f"error: the inputs of {len(changed)} items in {a.out} differ from the current records "
                     f"(e.g. {changed[0]}); choose another --out")
        before_inputs.update(inputs)
        inputs = before_inputs
    info = {"config": cfg, "config_file": os.path.abspath(a.config), "data": os.path.abspath(a.data), "lite": a.lite,
            "items": len(records), "environment": environment(), "inputs": inputs}
    with open(info_path + ".part", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(info_path + ".part", info_path)
    from asterism.runner import Runner
    stats = Runner(cfg).run(records, image_root, gen_path)
    print(json.dumps(stats), flush=True)
    score_run(records, read_jsonl(gen_path), a.out, lite=a.lite)


if __name__ == "__main__":
    main()
