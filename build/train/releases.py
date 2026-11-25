"""Readers of the releases that ASTERISM-Train items are assembled from.

Each reader returns, for a record, the text fields that its `build` recipe slices, and writes its images.
  M4-Instruct      m4_instruct_annotations.json and the extracted image archives (also WebQA)
  Mantis-Instruct  <subset>/train-00000-of-00001.parquet and <subset>/train_images.zip (or its extracted folder)
  Leopard-Instruct <subset>/*.parquet with the images inside the rows
  MGrounding-630k  the image folders of MGrounding-630k (synthetic items)
"""
import glob
import io
import json
import os
import re
import zipfile

from PIL import Image


def _jpeg(im, quality):
    out = io.BytesIO()
    im.convert("RGB").save(out, format="JPEG", quality=quality)
    return out.getvalue()


def _shrink(im, max_side):
    """Reduce the long side to max_side (bicubic)."""
    im = im.convert("RGB")
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    return im


def _records(path):
    """The objects of a JSON array file, one at a time, with their index."""
    dec = json.JSONDecoder()
    space = re.compile(r"[\s,]*")
    with open(path, encoding="utf-8") as f:
        buf = f.read(1 << 24)
        pos = buf.index("[") + 1
        i = 0
        while True:
            pos = space.match(buf, pos).end()
            while pos >= len(buf):
                more = f.read(1 << 24)
                if not more:
                    return
                buf, pos = buf[pos:] + more, 0
                pos = space.match(buf, pos).end()
            if buf[pos] == "]":
                return
            while True:
                try:
                    obj, end = dec.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    more = f.read(1 << 24)
                    if not more:
                        raise
                    buf, pos = buf[pos:] + more, 0
            yield i, obj
            i += 1
            pos = end
            if pos > (1 << 24):
                buf, pos = buf[pos:], 0


class M4Instruct:
    """lmms-lab/M4-Instruct-Data: records by their index in m4_instruct_annotations.json."""

    def __init__(self, root, records):
        self.root = root
        want = {int(r["source_id"]) for r in records}
        self.rows = {i: x for i, x in _records(os.path.join(root, "m4_instruct_annotations.json")) if i in want}
        missing = want - set(self.rows)
        if missing:
            raise KeyError(f"{len(missing)} records not in m4_instruct_annotations.json, e.g. {min(missing)}")

    def fields(self, record):
        x = self.rows[int(record["source_id"])]
        turns = x["conversations"]
        f = {"user": next(c["value"] for c in turns if c["from"] == "human"),
             "assistant": next(c["value"] for c in turns if c["from"] != "human")}
        for k, c in enumerate(x.get("choice_list") or []):
            f[f"choice_list.{k}"] = c
        return f

    def image(self, record, k, ref):
        if self.rows[int(record["source_id"])]["image"][k] != ref:
            raise ValueError(f"image {k} of {record['id']} is not {ref}")
        data = open(os.path.join(self.root, ref), "rb").read()
        if record["source"] == "WebQA":
            return data, os.path.splitext(ref)[1].lower()            # copied as they are
        return _jpeg(_shrink(Image.open(io.BytesIO(data)), 768), 90), ".jpg"


class MantisInstruct:
    """TIGER-Lab/Mantis-Instruct: records by id."""

    def __init__(self, root, records):
        import pyarrow.parquet as pq
        self.root = root
        self.rows = {}
        for sub in sorted({r["images"][0].split("/", 1)[0] for r in records}):
            for x in pq.read_table(os.path.join(root, sub, "train-00000-of-00001.parquet"),
                                   columns=["id", "conversation"]).to_pylist():
                self.rows[x["id"]] = x
        self._zips = {}

    def fields(self, record):
        conv = self.rows[record["source_id"]]["conversation"]
        return {"user": next(c["content"] for c in conv if c["role"] in ("user", "human")),
                "assistant": next(c["content"] for c in conv if c["role"] in ("assistant", "gpt"))}

    def image(self, record, k, ref):
        archive, path = ref.split("#", 1)
        sub = archive.split("/", 1)[0]
        zp = os.path.join(self.root, archive)
        if os.path.isfile(zp):
            if zp not in self._zips:
                self._zips[zp] = zipfile.ZipFile(zp)
            z = self._zips[zp]
            name = path if path in z.NameToInfo else f"train_images/{path}"
            data = z.read(name)
        else:
            cands = [os.path.join(self.root, sub, path), os.path.join(self.root, sub, "train_images", path)]
            data = open(next(p for p in cands if os.path.isfile(p)), "rb").read()
        return data, os.path.splitext(path)[1].lower()


class LeopardInstruct:
    """wyu1/Leopard-Instruct: records by id, with the images inside the rows."""

    def __init__(self, root, records):
        import pyarrow.parquet as pq
        want = {r["source_id"] for r in records}
        self.rows = {}
        for sub in sorted({r["images"][0].split("#", 1)[0] for r in records}):
            for fn in sorted(glob.glob(os.path.join(root, sub, "*.parquet"))):
                pf = pq.ParquetFile(fn)
                for g in range(pf.num_row_groups):       # read only the row groups that hold wanted records
                    ids = pf.read_row_group(g, columns=["id"]).column("id").to_pylist()
                    if want.intersection(ids):
                        for x in pf.read_row_group(g).to_pylist():
                            if x["id"] in want:
                                self.rows[x["id"]] = x

    def fields(self, record):
        t = self.rows[record["source_id"]]["texts"]
        return {"user": next(x["content"] for x in t if x["from"] in ("human", "user")),
                "assistant": next(x["content"] for x in t if x["from"] in ("gpt", "assistant"))}

    def image(self, record, k, ref):
        m = re.fullmatch(r"[^#]+#id=(.+)&image=(\d+)", ref)
        im = self.rows[m.group(1)]["images"][int(m.group(2)) - 1]
        return _jpeg(Image.open(io.BytesIO(im["bytes"])), 92), ".jpg"


class MGrounding:
    """Michael4933/MGrounding-630k: image files, taken as they are or with the operations of the reference."""

    def __init__(self, root, records):
        self.files = Files(root, records)

    def image(self, record, k, ref):
        return self.files.image(record, k, ref)

def apply_ops(im, ops):
    """Operations of an image reference, in order: max=N (long side reduced to N, bicubic), crop=x0,y0,x1,y1 (box in
    fractions of width and height), box=x0,y0,x1,y1 (box in pixels), blank=x0,y0,x1,y1 (box filled black),
    fit=N (long side reduced to N, Lanczos), warp=h00,h01,...,h22 (perspective warp by that 3x3 matrix with OpenCV,
    same output size, borders reflected), mark=x,y and marks=x,y;x,y;... (red rings, the second form labelled
    Point A, Point B, ... in DejaVu Sans Bold, or with font=default in the default font of Pillow).
    Returns the image and the JPEG quality (q=N)."""
    q = None
    parts = ops.split("&") if ops else []
    default_font = "font=default" in parts
    for op in parts:
        key, _, val = op.partition("=")
        nums = [float(v) for v in re.split(r"[,;]", val)] if key not in ("q", "font") else []
        if key == "max":
            w, h = im.size
            s = min(nums[0] / max(w, h), 1)
            im = im.resize((int(w * s), int(h * s)), Image.BICUBIC) if s < 1 else im
        elif key == "crop":
            w, h = im.size
            im = im.crop((int(nums[0] * w), int(nums[1] * h), int(nums[2] * w), int(nums[3] * h)))
        elif key == "box":
            im = im.crop(tuple(int(v) for v in nums))
        elif key == "blank":
            im = im.copy()
            im.paste((0, 0, 0), tuple(int(v) for v in nums))
        elif key == "fit":
            if max(im.size) > nums[0]:
                s = nums[0] / max(im.size)
                im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.LANCZOS)
        elif key == "warp":
            import cv2
            import numpy as np
            m = np.array(nums, dtype=np.float64).reshape(3, 3)
            im = Image.fromarray(cv2.warpPerspective(np.array(im), m, im.size, borderMode=cv2.BORDER_REFLECT))
        elif key in ("mark", "marks"):
            pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
            im = _rings(im, pts, [f"Point {'ABCD'[i]}" for i in range(len(pts))] if key == "marks" else None, default_font)
        elif key == "q":
            q = int(val)
    return im, q


def _font(size):
    from PIL import ImageFont
    for path in sorted(glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True)):
        return ImageFont.truetype(path, size)
    import matplotlib
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans-Bold.ttf"), size)


def _rings(im, pts, labels, default_font=False):
    from PIL import ImageDraw, ImageFont
    im = im.copy()
    d = ImageDraw.Draw(im)
    r = max(6, min(im.size) // 40)
    f = (ImageFont.load_default() if default_font else _font(max(14, r * 2))) if labels else None
    for i, (x, y) in enumerate(pts):
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=max(2, r // 3))
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))
        if labels:
            d.text((x + r + 2, y - r - 2), labels[i], fill=(255, 0, 0), font=f)
    return im


class Files:
    """A release read as image files under a folder, with the operations of the reference applied."""

    def __init__(self, root, records):
        self.root = root

    def image(self, record, k, ref):
        path, _, ops = ref.partition("#")
        src = os.path.join(self.root, path)
        if not ops:
            return open(src, "rb").read(), os.path.splitext(path)[1].lower()
        im, q = apply_ops(Image.open(src).convert("RGB"), ops)
        return _jpeg(im, q or 90), ".jpg"


class NLVR2:
    """NLVR2: train.json of github.com/lil-lab/nlvr for the text, and the train split of pingzhili/nlvr2-resized
    (data/train-*.parquet) for the images, read by identifier: train#identifier=<id>&column=image0|image1."""

    def __init__(self, root, records):
        import pyarrow.parquet as pq
        want = {r["source_id"] for r in records}
        self.rows = {}
        with open(os.path.join(root, "train.json"), encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                if x["identifier"] in want:
                    self.rows[x["identifier"]] = x
        self.images = {}
        for fn in sorted(glob.glob(os.path.join(root, "data", "train-*.parquet"))):
            pf = pq.ParquetFile(fn)
            for g in range(pf.num_row_groups):
                ids = pf.read_row_group(g, columns=["identifier"]).column("identifier").to_pylist()
                if want.intersection(ids):
                    for x in pf.read_row_group(g, columns=["identifier", "image0", "image1"]).to_pylist():
                        if x["identifier"] in want:
                            self.images[x["identifier"]] = {"image0": x["image0"]["bytes"], "image1": x["image1"]["bytes"]}
        missing = want - set(self.images)
        if missing:
            raise KeyError(f"{len(missing)} items not in data/train-*.parquet, e.g. {min(missing)}")

    def fields(self, record):
        x = self.rows[record["source_id"]]
        return {"sentence": x["sentence"], "label": x["label"]}

    def image(self, record, k, ref):
        m = re.fullmatch(r"train#identifier=(.+)&column=(image[01])", ref)
        data = self.images[m.group(1)][m.group(2)]
        return data, ".png" if data.startswith(b"\x89PNG") else ".jpg"


class STAR:
    """STAR: STAR_train.json and the Charades_v1_480 videos; one frame per second from the first frame, long side 480."""

    def __init__(self, root, records):
        self.root = root
        want = {r["source_id"] for r in records}
        with open(os.path.join(root, "STAR_train.json"), encoding="utf-8") as f:
            self.rows = {x["question_id"]: x for x in json.load(f) if x["question_id"] in want}
        self._frames = {}

    def fields(self, record):
        x = self.rows[record["source_id"]]
        f = {"question": x["question"], "answer": x["answer"]}
        for k, c in enumerate(x["choices"]):
            f[f"choices.{k}"] = c["choice"]
        return f

    def image(self, record, k, ref):
        import cv2
        video, _, sec = ref.partition("#second=")
        if video not in self._frames:
            self._frames.clear()
            cap = cv2.VideoCapture(os.path.join(self.root, video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, int(round(fps)))
            frames = []
            for i, fi in enumerate(range(0, n, step)):
                if i >= 60:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if not ok:
                    break
                h, w = fr.shape[:2]
                if max(h, w) > 480:
                    s = 480 / max(h, w)
                    fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                frames.append(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
            cap.release()
            self._frames[video] = frames
        return self._frames[video][int(sec)], ".jpg"


class SentencePairs(Files):
    """Spot-the-Diff (train.json and resized_images/ of github.com/harsh19/spot-the-diff) and CLEVR-Change (data/ of
    github.com/Seth-Park/RobustChangeCaptioning, and sentences.json, the sentences of the train split of
    Lancelot53/clevr-change). Sentences of another row are named sentences.<k>@<row>."""

    def __init__(self, root, records, name):
        self.root = root
        with open(os.path.join(root, "train.json" if name == "Spot-the-Diff" else "sentences.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.rows = [x["sentences"] if isinstance(x, dict) else x for x in data]

    def image(self, record, k, ref):
        if ref.startswith("train#row="):        # images read from the train split of Lancelot53/clevr-change
            return _hf_row_image(self.root, ref)
        return Files.image(self, record, k, ref)

    def fields(self, record):
        recipe = json.loads(record["build"])
        f = {}
        for seg in recipe["prompt"] + recipe["answer"]:
            if isinstance(seg, list):
                name, _, row = seg[0].partition("@")
                k = int(name.split(".")[1])
                f[seg[0]] = self.rows[int(row or record["source_id"])][k]
        return f


def _hf_row_image(root, ref):
    """train#row=<n>&column=image1|image2: an image of the train split of a Hugging Face dataset (train-*.parquet under root)."""
    import pyarrow.parquet as pq
    m = re.fullmatch(r"train#row=(\d+)&column=(image[12])", ref)
    row = int(m.group(1))
    for fn in sorted(glob.glob(os.path.join(root, "train-*.parquet"))):
        pf = pq.ParquetFile(fn)
        if row < pf.metadata.num_rows:
            data = pf.read(columns=[m.group(2)]).column(0)[row].as_py()["bytes"]
            return data, ".png" if data.startswith(b"\x89PNG") else ".jpg"
        row -= pf.metadata.num_rows
    raise KeyError(ref)


class ParquetImages:
    """Images stored in the rows of parquet files (Tiny-GenImage, WikiArt): <folder>/<file>.parquet#row=<n>&<operations>.
    Besides the operations of apply_ops: jpeg=N stores the image as JPEG of quality N and reads it back, and sq=N cuts
    the centre square and resizes it to N x N (bicubic)."""

    def __init__(self, root, records):
        import pyarrow.parquet as pq
        self.root = root
        need = {}
        for r in records:
            for ref in r["images"]:
                path, _, ops = ref.partition("#")
                need.setdefault(path, set()).add(int(ops.split("&")[0][len("row="):]))
        self._bytes = {}                      # each parquet file is read once, keeping only the rows the records use
        for path, rows in sorted(need.items()):
            if not os.path.isfile(os.path.join(root, path)):
                continue                      # the items of a missing file fail one by one
            column = pq.read_table(os.path.join(root, path), columns=["image"]).column("image")
            for row in rows:
                self._bytes[(path, row)] = column[row].as_py()["bytes"]
            del column

    def image(self, record, k, ref):
        path, _, ops = ref.partition("#")
        parts = ops.split("&")
        row = int(parts[0][len("row="):])
        im = Image.open(io.BytesIO(self._bytes[(path, row)])).convert("RGB")
        rest = []
        for op in parts[1:]:
            key, _, val = op.partition("=")
            if key == "jpeg":
                im = Image.open(io.BytesIO(_jpeg(im, int(val)))).convert("RGB")
            elif key == "sq":
                w, h = im.size
                m = min(w, h)
                l, t = (w - m) // 2, (h - m) // 2
                im = im.crop((l, t, l + m, t + m)).resize((int(val), int(val)), Image.BICUBIC)
            else:
                rest.append(op)
        im, q = apply_ops(im, "&".join(rest))
        return _jpeg(im, q or 90), ".jpg"
