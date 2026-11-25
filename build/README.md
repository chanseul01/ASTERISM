# Assembling the items of the identifier sources

MileBench, VLM2-Bench, M4Bench, MMLongBench-Doc, MaRVL, Q-Bench2 and EMMA carry non-commercial terms or
no stated license. For their 6,947 items (549 of them rewrites), the ASTERISM dataset holds the source
identifiers, the image references and our annotations: the rewritten items, the corrected labels,
answer formats and text, the new answer letters of items whose correct option we exchanged with
another option to balance the answer positions, our scoring entries and the per-item labels. The
scripts in this folder read each item from the public release of its source, apply these changes
and write each item with its images.

## Quick start

With the ASTERISM dataset (or ASTERISM-Lite) downloaded to `$DATA`, one command downloads the seven
public releases at the revisions we used (about 28 GB to download, about 45 GB on disk once MileBench is
extracted, no login needed), assembles the items (about 6 GB of images for ASTERISM) and checks them against
our checksums:

```bash
pip install -r build/requirements.txt
bash build/get_asterism.sh "$DATA"
```

The releases go to `./asterism_sources` (a second argument chooses another folder), and a second run
skips completed downloads. The records with every item filled in are written to
`$DATA/data/asterism_full.jsonl` (`asterism_lite_full.jsonl` for ASTERISM-Lite), and the images under
`$DATA/images/`. `LINK=1 bash build/get_asterism.sh "$DATA"` hard-links the MileBench image files instead of
copying them (same file system only), which saves about 4 GB and much of the assembly time. The sections
below describe the same steps one by one.

## 1. Download the public releases

Put each release in its own folder under one root folder, using the folder names below.

| Folder | Source | Dataset repository and the revision we used | Files read |
|---|---|---|---|
| `milebench` | MileBench | [FreedomIntelligence/MileBench](https://huggingface.co/datasets/FreedomIntelligence/MileBench), `53c7a58051ef88bacf76541d91f03f5ba2d71e7d` | `MileBench_part0.tar.gz` to `MileBench_part5.tar.gz`, extracted in place |
| `vlm2-bench` | VLM2-Bench | [Sterzhang/vlm2-bench](https://huggingface.co/datasets/Sterzhang/vlm2-bench), `188a97b9f52ef44316a260a0c3e90bf49d93ca5f` | `VLM2Bench_img.tsv` |
| `m4bench` | M4Bench | [Anonymous8976/M4Bench](https://huggingface.co/datasets/Anonymous8976/M4Bench), `65ca74a18edd43921a479053265a3490157f612c` | `Detailed_Difference.tsv`, `Instance_Comparison.tsv`, `State_Comparison.tsv`, `State_Invariance.tsv` |
| `mmlongbench-doc` | MMLongBench-Doc | [yubo2333/MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc), `2ff6aa9237fc777b6627dc57a486e9225ac5fb86` | `data/train-00000-of-00001.parquet` and the `documents/*.pdf` that the records name (132 for ASTERISM) |
| `marvl` | MaRVL | [floschne/marvl](https://huggingface.co/datasets/floschne/marvl), `d25b0dc0bce9d8e9b56bf666d3442dc907180667` | `data/*.parquet` |
| `q-bench2` | Q-Bench2 | [q-future/q-bench2](https://huggingface.co/datasets/q-future/q-bench2), `c766b14f9a54205d8e06cf644441d745dc16f4e3` | `q-bench2-a1-dev.jsonl`, `all_single_images.zip` |
| `emma` | EMMA | [luckychao/EMMA](https://huggingface.co/datasets/luckychao/EMMA), `6c87aec9048c489108170088e50b768646b5bcf9` | `Chemistry/`, `Coding/` and `Physics/test-00000-of-00001.parquet` |

The MileBench repository is the one linked from the MileBench code repository
(github.com/MileBench/MileBench). Other revisions may work but are not checked.

With `huggingface_hub` (0.34 or later, which provides the `hf` command), in bash:

```bash
ROOT=/path/to/releases
DATA=/path/to/ASTERISM   # the downloaded ASTERISM or ASTERISM-Lite dataset
# get <repository> <revision> <folder> [<file pattern> ...]: one hf call per pattern, or the whole repository
get() {
  local p
  if [ $# -le 3 ]; then hf download "$1" --repo-type dataset --revision "$2" --local-dir "$ROOT/$3"; return; fi
  for p in "${@:4}"; do hf download "$1" --repo-type dataset --revision "$2" --local-dir "$ROOT/$3" --include "$p" || return; done
}
get FreedomIntelligence/MileBench 53c7a58051ef88bacf76541d91f03f5ba2d71e7d milebench "MileBench_part*.tar.gz"
for f in "$ROOT"/milebench/MileBench_part*.tar.gz; do tar xzf "$f" -C "$ROOT/milebench"; done
get Sterzhang/vlm2-bench 188a97b9f52ef44316a260a0c3e90bf49d93ca5f vlm2-bench VLM2Bench_img.tsv
get Anonymous8976/M4Bench 65ca74a18edd43921a479053265a3490157f612c m4bench "*.tsv"
# MMLongBench-Doc: the QA table and the documents that the records name
REC=$DATA/data/asterism.jsonl; [ -f "$REC" ] || REC=$DATA/data/asterism_lite.jsonl
DOCS=$(python -c 'import json, sys; print(" ".join(sorted({x.split("#")[0] for l in open(sys.argv[1], encoding="utf-8") if l.strip() for r in [json.loads(l)] if r["source"] == "MMLongBench-Doc" and r["release"] == "identifiers" for x in r["images"]})))' "$REC") &&
  hf download yubo2333/MMLongBench-Doc data/train-00000-of-00001.parquet $DOCS \
    --repo-type dataset --revision 2ff6aa9237fc777b6627dc57a486e9225ac5fb86 --local-dir "$ROOT/mmlongbench-doc"
get floschne/marvl d25b0dc0bce9d8e9b56bf666d3442dc907180667 marvl "data/*.parquet"
get q-future/q-bench2 c766b14f9a54205d8e06cf644441d745dc16f4e3 q-bench2 q-bench2-a1-dev.jsonl all_single_images.zip
get luckychao/EMMA 6c87aec9048c489108170088e50b768646b5bcf9 emma "*/test-00000-of-00001.parquet"
```

Expected layout:

```
$ROOT/milebench/<task>/<task>.json, $ROOT/milebench/<task>/images/...
$ROOT/vlm2-bench/VLM2Bench_img.tsv
$ROOT/m4bench/<task>.tsv
$ROOT/mmlongbench-doc/data/train-00000-of-00001.parquet, $ROOT/mmlongbench-doc/documents/*.pdf
$ROOT/marvl/data/*.parquet
$ROOT/q-bench2/q-bench2-a1-dev.jsonl, $ROOT/q-bench2/all_single_images.zip
$ROOT/emma/<subject>/test-00000-of-00001.parquet
```

## 2. Assemble

Run from the root of this repository, with the ASTERISM dataset downloaded to `$DATA`:

```bash
python -m build.assemble --source all --release-root "$ROOT" --records "$DATA/data/asterism.jsonl" --out "$DATA"
python -m build.verify --records "$DATA/data/asterism_full.jsonl"
```

`build.assemble` writes the images under `$DATA/images/<folder>/` and all 18,696 records, with the
identifier items filled in, to `$DATA/data/asterism_full.jsonl`, in the format of the released records.
The evaluation code reads this file like the released one. For ASTERISM-Lite, pass its
`data/asterism_lite.jsonl` and its dataset folder. One source can be assembled alone (items of other sources that
an earlier run filled in are kept), for example
`--source MaRVL --release-dir "$ROOT/marvl"`. With `--link`, the MileBench image files are hard-linked
instead of copied. The images take about 6 GB, of which 4 GB are MileBench files. Assembling all seven
sources took about 20 minutes on our storage (about a minute for ASTERISM-Lite), most of it for copying the
MileBench files and rendering the MMLongBench-Doc pages; extracting the MileBench archives on the first run
took about 20 minutes more. While rendering, MuPDF prints lines such as
"MuPDF error: syntax error: could not parse color space"; with the versions in build/requirements.txt, the
rendered pages still match our checksums.

For each item, the scripts
- read the item by `source_id` (and `source_subset`) and write its prompt as the evaluation did: the
  item text with the official instruction of the source, followed by the answer-format line and the
  final answer line; EMMA items keep the official EMMA prompt;
- take the prompt, options, answer and scoring entry from the record where the text is ours (rewrites,
  corrected questions or options), the answer where the label is ours, and the scoring entry where the
  answer format or a scoring setting is ours;
- exchange the correct option with the option at the recorded answer letter where only the position
  of the correct option changed;
- derive the remaining scoring entries with the rules of the scorer (`asterism/scoring`);
- copy MileBench and Q-Bench2 images, decode the images stored in the VLM2-Bench and M4Bench tables
  and the MaRVL parquet files, re-encode EMMA images as RGB JPEG (quality 92) and render
  the MMLongBench-Doc pages listed in each record (at most 60 per item) at 72 dpi as JPEG (quality 90).

## 3. Check

`build/checksums/<folder>.jsonl` lists, for every item of the identifier sources, the sha256 of the
canonical JSON of its prompt, options, answer and scoring entry (`json.dumps` with sorted keys,
compact separators and UTF-8, the scoring entry parsed), and the sha256 of each of its image files.
`python -m build.verify` compares the assembled items with these values and reports the matches per
source. Rendered PDF pages and re-encoded JPEG images depend on the PyMuPDF and Pillow versions; they
are reported on their own lines and do not make the check fail. A missing image file or a different
number of images fails the check for every source, and so do a repeated item id, an item of the
checksums missing from ASTERISM records (ASTERISM-Lite holds a subset), and a run that checks no item.

## Requirements

`pyarrow`, `Pillow` and `PyMuPDF`, and the `asterism` package of this repository (`build/requirements.txt`). We ran the scripts
with Python 3.11.15, pyarrow 24.0.0, Pillow 12.2.0 (wheel with libjpeg-turbo 3.1.4.1) and PyMuPDF 1.28.2;
with these versions all 6,947 items and all their image files match the checksums in `build/checksums`.
With Pillow 10.4.0 built against libjpeg instead of libjpeg-turbo, the rendered pages and some
re-encoded EMMA images differ.
