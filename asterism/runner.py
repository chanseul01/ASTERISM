"""Generation with vLLM for ASTERISM and ASTERISM-Lite, with the settings of the paper.

A model config (configs/eval/*.yaml) sets the checkpoint, the processor, the decoding and the
context. Qwen2.5-VL-7B, GLM-4.1V-9B-Thinking and the models trained from them decode greedily, and
GLM-4.1V-9B-Thinking reasons before every answer through its own chat template. Qwen3.5-9B and CICA-Qwen3.5-9B run in thinking mode with a sampling that the model card gives for thinking
mode (temperature 0.6, top-p 0.95, top-k 20, no penalties) and a seed per item derived from the item id.
"""
import hashlib
import json
import math
import os
import platform
import subprocess
import time

from PIL import Image

from .prompts import MAX_IMAGES, build_messages

# Images are downscaled before the processor so that an item fits the context: each image keeps at
# most MAX_PIXELS, an item at most TOTAL_PIXELS (each image at least MIN_PIXELS), and both sides are
# cut to multiples of 28.
MAX_PIXELS = 451584
TOTAL_PIXELS = 22 * MAX_PIXELS
MIN_PIXELS = 28 * 28 * 64

SEED_OFFSET = 20261021
THINK_OPEN = "<think>\n"

DEFAULTS = {"processor": None, "revision": None, "processor_revision": None, "enable_thinking": False, "per_item_seed": False,
            "max_model_len": 32768, "max_images": MAX_IMAGES, "gpu_memory_utilization": 0.85, "batch_size": 16, "engine_args": {}}


def pixel_budget(n):
    """Pixel budget of one image when an item has n images."""
    return int(min(MAX_PIXELS, max(TOTAL_PIXELS // max(n, 1), MIN_PIXELS)))


def cap_pixels(im, budget):
    """Downscale (keeping the aspect ratio) to at most budget pixels, then cut both sides to multiples of 28."""
    w, h = im.size
    if w * h > budget:
        s = math.sqrt(budget / (w * h))
        w, h = max(28, int(w * s)), max(28, int(h * s))
    w, h = max(28, (w // 28) * 28), max(28, (h // 28) * 28)
    if (w, h) == im.size:
        return im
    return im.resize((w, h))


def load_images(paths):
    budget = pixel_budget(len(paths))
    return [cap_pixels(Image.open(p).convert("RGB"), budget) for p in paths]


def item_seed(item_id):
    """Sampling seed of an item in thinking mode."""
    return (int(hashlib.md5(str(item_id).encode()).hexdigest()[:8], 16) + SEED_OFFSET) % (2 ** 31)


def load_config(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    unknown = set(cfg) - set(DEFAULTS) - {"name", "model", "sampling"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    if not cfg.get("model") or not isinstance(cfg.get("sampling"), dict):
        raise ValueError(f"{path}: 'model' and 'sampling' are required")
    out = dict(DEFAULTS)
    out.update(cfg)
    if out["processor"] is None:
        out["processor"] = out["model"]
        if out["processor_revision"] is None:
            out["processor_revision"] = out["revision"]
    return out


def engine_kwargs(cfg):
    """Arguments of vllm.LLM for a model config."""
    kw = dict(model=cfg["model"], tokenizer=cfg["processor"], limit_mm_per_prompt={"image": cfg["max_images"]},
              max_model_len=cfg["max_model_len"], gpu_memory_utilization=cfg["gpu_memory_utilization"],
              trust_remote_code=True, mm_processor_cache_gb=0, enable_prefix_caching=False)
    if cfg["revision"]:
        kw["revision"] = cfg["revision"]
    if cfg["processor_revision"]:
        kw["tokenizer_revision"] = cfg["processor_revision"]
    kw.update(cfg["engine_args"] or {})
    return kw


def sampling_kwargs(cfg, item_id):
    """Arguments of vllm.SamplingParams for one item."""
    kw = dict(cfg["sampling"])
    if cfg["per_item_seed"]:
        kw["seed"] = item_seed(item_id)
    return kw


def chat_kwargs(cfg):
    """Extra arguments of apply_chat_template: only the thinking switch of Qwen3.5."""
    return {"enable_thinking": True} if cfg["enable_thinking"] else {}


def resolve_images(record, image_root):
    """Image paths of an item (relative paths are taken from the dataset root)."""
    return [p if os.path.isabs(p) else os.path.join(image_root, p) for p in record["images"]]


def read_rows(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    return rows


class Runner:
    """Loads one model with vLLM and generates one output per item."""

    def __init__(self, cfg):
        from transformers import AutoProcessor
        from vllm import LLM
        self.cfg = cfg
        self.llm = LLM(**engine_kwargs(cfg))
        pkw = {"revision": cfg["processor_revision"]} if cfg["processor_revision"] else {}
        self.processor = AutoProcessor.from_pretrained(cfg["processor"], trust_remote_code=True, **pkw)

    def prompt_text(self, record, n_img):
        msgs = build_messages(record, n_img)
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs(self.cfg))
        if self.cfg["enable_thinking"] and not text.endswith(THINK_OPEN):
            raise RuntimeError(f"thinking-mode prompt does not end with {THINK_OPEN!r}: {text[-60:]!r}")
        return text

    def sampling_params(self, record):
        from vllm import SamplingParams
        return SamplingParams(**sampling_kwargs(self.cfg, record["id"]))

    def _row(self, record, out):
        text = out.outputs[0].text if out.outputs else ""
        if self.cfg["enable_thinking"]:  # the prompt opened the reasoning channel; the scorer needs the opening tag
            text = THINK_OPEN + text
        finished = (out.outputs[0].finish_reason == "stop") if out.outputs else False
        return {"id": record["id"], "output": text, "finished": finished}

    def run(self, records, image_root, out_path):
        """Generate for every record and append rows {"id", "output", "finished"} to out_path.
        Items that already have a row in out_path are skipped."""
        done = {r["id"] for r in read_rows(out_path)}
        todo = [r for r in records if r["id"] not in done]
        n_cut = sum(len(r["images"]) > self.cfg["max_images"] for r in todo)
        t0 = time.time()
        with open(out_path, "a", encoding="utf-8") as w:
            for b0 in range(0, len(todo), self.cfg["batch_size"]):
                chunk = todo[b0:b0 + self.cfg["batch_size"]]
                reqs = []
                for rec in chunk:
                    imgs = load_images(resolve_images(rec, image_root)[:self.cfg["max_images"]])
                    text = self.prompt_text(rec, len(imgs))
                    reqs.append({"prompt": text, "multi_modal_data": {"image": imgs}} if imgs else {"prompt": text})
                params = [self.sampling_params(rec) for rec in chunk]
                try:
                    pairs = list(zip(chunk, self.llm.generate(reqs, params, use_tqdm=False)))
                    rows = [self._row(rec, o) for rec, o in pairs]
                except ValueError:  # a prompt longer than the context; run the items one by one
                    rows = []
                    for rec, rq, sp in zip(chunk, reqs, params):
                        try:
                            rows.append(self._row(rec, self.llm.generate([rq], sp, use_tqdm=False)[0]))
                        except ValueError as e:
                            rows.append({"id": rec["id"], "output": "", "finished": True, "note": "too_long: " + str(e)[:80]})
                for r in rows:
                    w.write(json.dumps(r, ensure_ascii=False) + "\n")
                w.flush()
                print(f"generated {min(b0 + len(chunk), len(todo))}/{len(todo)} items ({time.time() - t0:.0f} s)", flush=True)
        return {"generated": len(todo), "skipped_existing": len(done), "images_cut_to_limit": n_cut, "seconds": round(time.time() - t0)}


def environment():
    """Library versions and GPUs, recorded next to the generations."""
    info = {"python": platform.python_version()}
    for name in ("vllm", "transformers", "torch", "PIL"):
        try:
            info[name] = __import__(name).__version__
        except Exception:
            info[name] = None
    try:  # nvidia-smi rather than torch.cuda, so that CUDA is not initialized before vLLM starts
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        info["gpus"] = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    except Exception:
        info["gpus"] = None
    info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return info
