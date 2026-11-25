#!/usr/bin/env python
"""Generate with vLLM and score a model on ASTERISM or ASTERISM-Lite.

Writes to --out: generations.jsonl (resumable: items with a row are skipped), run_info.json
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
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from asterism.data import check_complete, load_records, read_jsonl  # noqa: E402
from asterism.runner import environment, load_config  # noqa: E402
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
    records = load_records(a.data, lite=a.lite)
    try:
        check_complete(records)
    except ValueError as e:
        sys.exit(f"error: {e}")
    image_root = a.image_root or os.path.dirname(os.path.dirname(os.path.abspath(a.data)))
    os.makedirs(a.out, exist_ok=True)
    gen_path, info_path = os.path.join(a.out, "generations.jsonl"), os.path.join(a.out, "run_info.json")
    if os.path.isfile(gen_path) and os.path.getsize(gen_path) and os.path.isfile(info_path):
        with open(info_path) as f:  # generations are resumed only for the same model and settings
            before = json.load(f)["config"]
        runtime = ("gpu_memory_utilization", "batch_size")
        if {k: v for k, v in before.items() if k not in runtime} != {k: v for k, v in cfg.items() if k not in runtime}:
            sys.exit(f"error: {a.out} holds generations of another model or setting; choose another --out")
    info = {"config": cfg, "config_file": os.path.abspath(a.config), "data": os.path.abspath(a.data), "lite": a.lite,
            "items": len(records), "environment": environment()}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=1)
    from asterism.runner import Runner
    stats = Runner(cfg).run(records, image_root, gen_path)
    print(json.dumps(stats), flush=True)
    score_run(records, read_jsonl(gen_path), a.out, lite=a.lite)


if __name__ == "__main__":
    main()
