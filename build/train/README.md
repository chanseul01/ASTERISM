# Assembling the identifier items of ASTERISM-Train

ASTERISM-Train releases the items of sources with non-commercial terms or without a stated license as
identifiers. For an item of a public dataset or an instruction collection, the record holds the source
identifier, the image references and, in the `build` field, a recipe that writes the training text as our own
text and slices of the source record (a field written `name@row` is read from that row of the same release). For a synthetic item, the record holds our prompt, options and answer,
and each image reference names a source image and the operations that make the item image. The scripts in
this folder read the public releases and write the items as they are used in training, with their images.

## 1. Download the releases

Put each release in its own folder under one root folder, using the folder names below.

| Folder | Release | Repository or site, and the revision or version we used | Files read |
|---|---|---|---|
| `m4-instruct` | M4-Instruct (also the WebQA items) | [lmms-lab/M4-Instruct-Data](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data), `9078d03bb7442cef5a71771a29a2af9ec0f63134` | `m4_instruct_annotations.json`; `nuscenes.zip`, `RecipeQA_ImageCoherence.zip`, `contrastive_caption.zip`, `VISION.zip`, `iconqa.zip`, `TQA.zip`, `WebQA.zip`, extracted |
| `mantis-instruct` | Mantis-Instruct | [TIGER-Lab/Mantis-Instruct](https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct), `01a9edfe0bb8c2582431308c5b2645a9f4796939` | `iconqa/`, `coinstruct/` and `nlvr2/`: `train-00000-of-00001.parquet`, `train_images.zip` |
| `leopard-instruct` | LEOPARD | [wyu1/Leopard-Instruct](https://huggingface.co/datasets/wyu1/Leopard-Instruct), `93317b272c5a9d9c0417fa6ea6e2be89ac9215ea` | `slidevqa/*.parquet`, `tat_dqa/*.parquet` |
| `mgrounding` | MGrounding-630k | [Michael4933/MGrounding-630k](https://huggingface.co/datasets/Michael4933/MGrounding-630k), `1575fac4a9928762479582b9deb683136dc51755` | `Common_Object.zip`, `Difference.zip`, and the split archives `Group_Grounding.z01` to `Group_Grounding.zip` and `Object_Tracking.z01` to `Object_Tracking.zip`, extracted so that `MGrounding-630k/` holds their folders |
| `spair` | SPair-71k | [cvlab.postech.ac.kr/research/SPair-71k](https://cvlab.postech.ac.kr/research/SPair-71k/), `SPair-71k.tar.gz` | `JPEGImages/` |
| `parquet-images` | Tiny-GenImage and WikiArt | [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`; [Artificio/WikiArt](https://huggingface.co/datasets/Artificio/WikiArt), `b9190341f1939b12ce99c0b3120590e9d24033dc` | `genimage/train-*-of-00014.parquet`, `wikiart/train-*-of-00004-*.parquet` |
| `nlvr2` | NLVR2 | [github.com/lil-lab/nlvr](https://github.com/lil-lab/nlvr), `nlvr2/data/train.json` at `18924841aa96d519cfb8584f0945350f07a6d860`; [pingzhili/nlvr2-resized](https://huggingface.co/datasets/pingzhili/nlvr2-resized), `bc8d50a4a18d142abe876b5bf185e15cda2edc85` | `train.json`; `data/train-00000-of-00081.parquet` to `data/train-00009-of-00081.parquet` |
| `star` | STAR with Charades | [star.csail.mit.edu](http://star.csail.mit.edu), `STAR_train.json`; Charades `Charades_v1_480.zip` from [prior.allenai.org/projects/charades](https://prior.allenai.org/projects/charades) under its license | `STAR_train.json`, `Charades_v1_480/<video>.mp4` |
| `spot-the-diff` | Spot-the-Diff | [github.com/harsh19/spot-the-diff](https://github.com/harsh19/spot-the-diff), `data/annotations/train.json` at `e081c9411a069cac86ce325c5975054f2556c803`, and the resized images of that repository, which require the VIRAT data agreement | `train.json`, `resized_images/` |
| `clevr-change` | CLEVR-Change | [github.com/Seth-Park/RobustChangeCaptioning](https://github.com/Seth-Park/RobustChangeCaptioning), the data archive linked there; [Lancelot53/clevr-change](https://huggingface.co/datasets/Lancelot53/clevr-change), `453d02d48a1c11a4d370e382112f8acb9b334b43` | `data/images/`, `data/sc_images/`, `data/nsc_images/`; `train-*.parquet` of the train split and `sentences.json`, its `sentences` column |

STAR frames are decoded with OpenCV, and the frames depend on its video backend.

In the build environment (`pip install -r build/requirements.txt`, which includes `huggingface_hub` and its `hf` command),
starting from an empty root folder:

```bash
ROOT=/path/to/releases
# get <repository> <revision> <folder> <file pattern> ...: one hf call per pattern
get() {
  local p
  for p in "${@:4}"; do hf download "$1" --repo-type dataset --revision "$2" --local-dir "$ROOT/$3" --include "$p" || return; done
}
get lmms-lab/M4-Instruct-Data 9078d03bb7442cef5a71771a29a2af9ec0f63134 m4-instruct m4_instruct_annotations.json \
    nuscenes.zip RecipeQA_ImageCoherence.zip contrastive_caption.zip VISION.zip iconqa.zip TQA.zip WebQA.zip
for z in "$ROOT"/m4-instruct/*.zip; do unzip -qo "$z" -d "$ROOT/m4-instruct" || echo "FAILED: $z"; done
get TIGER-Lab/Mantis-Instruct 01a9edfe0bb8c2582431308c5b2645a9f4796939 mantis-instruct \
    "iconqa/train*" "coinstruct/train*" "nlvr2/train-00000-of-00001.parquet" "nlvr2/train_images.zip"
get wyu1/Leopard-Instruct 93317b272c5a9d9c0417fa6ea6e2be89ac9215ea leopard-instruct "slidevqa/*.parquet" "tat_dqa/*.parquet"
get Michael4933/MGrounding-630k 1575fac4a9928762479582b9deb683136dc51755 mgrounding \
    Common_Object.zip Difference.zip "Group_Grounding.z*" "Object_Tracking.z*"
(cd "$ROOT/mgrounding" && for n in Group_Grounding Object_Tracking; do zip -q -s 0 "$n.zip" --out "$n.joined.zip" || exit 1; done \
    && for z in Common_Object.zip Difference.zip Group_Grounding.joined.zip Object_Tracking.joined.zip; do unzip -qo "$z" -d unpacked || exit 1; done \
    && { [ -d MGrounding-630k ] || mv unpacked/home/*/*/MGrounding-630k .; }) || echo "FAILED: MGrounding-630k archives"
get TheKernel01/Tiny-GenImage 89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69 parquet-images/tg "data/train-*.parquet"
get Artificio/WikiArt b9190341f1939b12ce99c0b3120590e9d24033dc parquet-images/wa "data/train-*.parquet"
[ -d "$ROOT/parquet-images/genimage" ] || mv "$ROOT"/parquet-images/tg/data "$ROOT"/parquet-images/genimage
[ -d "$ROOT/parquet-images/wikiart" ] || mv "$ROOT"/parquet-images/wa/data "$ROOT"/parquet-images/wikiart
get pingzhili/nlvr2-resized bc8d50a4a18d142abe876b5bf185e15cda2edc85 nlvr2 "data/train-0000[0-9]-of-00081.parquet"
get Lancelot53/clevr-change 453d02d48a1c11a4d370e382112f8acb9b334b43 clevr-change "data/train-*.parquet"
mv "$ROOT"/clevr-change/data/train-*.parquet "$ROOT"/clevr-change/
python - "$ROOT/clevr-change" <<'EOF'
import glob, json, sys, pyarrow.parquet as pq
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/train-*.parquet")):
    rows += pq.read_table(f, columns=["sentences"]).column("sentences").to_pylist()
json.dump(rows, open(sys.argv[1] + "/sentences.json", "w"))
EOF
curl -fL --create-dirs -o "$ROOT/nlvr2/train.json" https://raw.githubusercontent.com/lil-lab/nlvr/18924841aa96d519cfb8584f0945350f07a6d860/nlvr2/data/train.json
curl -fL --create-dirs -o "$ROOT/spot-the-diff/train.json" https://raw.githubusercontent.com/harsh19/spot-the-diff/e081c9411a069cac86ce325c5975054f2556c803/data/annotations/train.json
```

The MGrounding-630k archives keep their files under a longer folder path that ends in `MGrounding-630k/`, and the
last command moves that folder to `$ROOT/mgrounding/MGrounding-630k`. `zip -s 0` first joins each split archive into
one file.

`STAR_train.json`, the Charades videos, the Spot-the-Diff images, the CLEVR-Change data archive and SPair-71k
come from the sites in the table, after their request form or agreement where one is required.

Expected layout:

```
$ROOT/m4-instruct/m4_instruct_annotations.json, $ROOT/m4-instruct/<subset>/...
$ROOT/mantis-instruct/<subset>/train-00000-of-00001.parquet, $ROOT/mantis-instruct/<subset>/train_images.zip
$ROOT/leopard-instruct/slidevqa/*.parquet, $ROOT/leopard-instruct/tat_dqa/*.parquet
$ROOT/mgrounding/MGrounding-630k/{Common_Object,Difference,Group_Grounding,Object_Tracking}/...
$ROOT/spair/JPEGImages/<category>/*.jpg
$ROOT/parquet-images/genimage/train-*-of-00014.parquet, $ROOT/parquet-images/wikiart/train-*.parquet
$ROOT/nlvr2/train.json, $ROOT/nlvr2/data/train-*.parquet
$ROOT/star/STAR_train.json, $ROOT/star/Charades_v1_480/<video>.mp4
$ROOT/spot-the-diff/train.json, $ROOT/spot-the-diff/resized_images/*.png
$ROOT/clevr-change/data/{images,sc_images,nsc_images}/*.png, $ROOT/clevr-change/sentences.json, $ROOT/clevr-change/train-*.parquet
```

## 2. Assemble

Run from the root of this repository, with the ASTERISM-Train dataset downloaded to `$DATA`:

```bash
python -m build.train.assemble --records "$DATA/data/asterism_train.jsonl" --out "$DATA" \
    --m4 "$ROOT/m4-instruct" --mantis "$ROOT/mantis-instruct" --leopard "$ROOT/leopard-instruct" \
    --mgrounding "$ROOT/mgrounding" --spair "$ROOT/spair" --parquet-images "$ROOT/parquet-images" \
    --nlvr2 "$ROOT/nlvr2" --star "$ROOT/star" --spot-the-diff "$ROOT/spot-the-diff" --clevr-change "$ROOT/clevr-change"
python -m build.train.verify --records "$DATA/data/asterism_train_full.jsonl"
```

`build.train.assemble` writes the images under `$DATA/images/<group-source>/<id>/` and all 49,105 records, with
the assembled items filled in, to `$DATA/data/asterism_train_full.jsonl`, in the format of the released records.
A release whose option is left out is skipped; each run writes `asterism_train_full.jsonl` anew from the released
records, so give every release option in one run. `build.train.verify` compares the text and the images of the
assembled items with `checksums.jsonl` and prints the numbers of checked and skipped records; images that the
scripts store as JPEG are reported on their own line. It fails when an expected item has no text (so also for the
items of a skipped release), when the number of images differs, when a text or an image that should match does not,
when an item of the checksums is missing from the records or an identifier record has no checksum, when an item id
repeats, or when no item is checked.

## 3. Image operations

An image reference names a file or a parquet row of a release, followed by `#` and operations applied in order.

| Operation | Meaning |
|---|---|
| `row=N` | the image in row N of a parquet file, counted from 0 |
| `max=N` | reduce the long side to N pixels (bicubic) |
| `crop=x0,y0,x1,y1` | cut a box given as fractions of the width and height |
| `box=x0,y0,x1,y1` | cut a box in pixels |
| `blank=x0,y0,x1,y1` | fill a box in pixels with black |
| `fit=N` | reduce the long side to N pixels (Lanczos) |
| `warp=h00,...,h22` | apply the perspective transform given by this 3×3 matrix (OpenCV, same size, borders reflected) |
| `jpeg=N` | store as JPEG of quality N and read back |
| `sq=N` | cut the centre square and resize it to N×N pixels (bicubic) |
| `mark=x,y`, `marks=x,y;...` | draw red rings, the second form labelled Point A, Point B, ... (`font=default`: the default font of Pillow) |
| `q=N` | store the result as JPEG of quality N |

## 4. Versions

The packages are listed in `build/requirements.txt`. Every check passed with Python 3.11.15, Pillow 12.2.0 with
libjpeg-turbo 3.1.4.1, pyarrow 24.0.0, opencv 4.11.0 and numpy 1.26.4. matplotlib is used only as a font fallback. Images stored as
JPEG and frames decoded from video depend on these versions.
