#!/usr/bin/env bash
# One command for the items released as identifiers: download the public releases of the seven
# sources at the revisions we used, assemble the items and check them against build/checksums.
#
#   bash build/get_asterism.sh <dataset folder> [<folder for the source releases>]
#
# <dataset folder> is the downloaded ASTERISM or ASTERISM-Lite dataset (it holds data/asterism.jsonl or
# data/asterism_lite.jsonl). The assembled records are written to data/asterism_full.jsonl (or
# data/asterism_lite_full.jsonl) and the images under images/ of that folder.
# The source releases (about 28 GB) go to <folder for the source releases>, by default ./asterism_sources.
# Downloads that are already complete are skipped, so the command can be run again after an interruption.
# LINK=1 hard-links the MileBench image files instead of copying them (same file system only).
# SOURCES="EMMA M4Bench" limits the run to some sources (names as in build/README.md); the written records
# then hold only the items of these sources filled in, and a run without SOURCES fills in every item.
# PYTHON selects the interpreter of the build environment (default: python).
set -euo pipefail

DATA=${1:?usage: bash build/get_asterism.sh <dataset folder> [<folder for the source releases>]}
ROOT=${2:-asterism_sources}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=$(cd "$DATA" && pwd)
PY=${PYTHON:-python}

if [ -f "$DATA/data/asterism.jsonl" ]; then
  RECORDS=$DATA/data/asterism.jsonl
elif [ -f "$DATA/data/asterism_lite.jsonl" ]; then
  RECORDS=$DATA/data/asterism_lite.jsonl
else
  echo "no data/asterism.jsonl or data/asterism_lite.jsonl under $DATA" >&2
  exit 2
fi

ALL="MileBench VLM2-Bench M4Bench MMLongBench-Doc MaRVL Q-Bench2 EMMA"
SOURCES=${SOURCES:-$ALL}
for s in $SOURCES; do
  case " $ALL " in
    *" $s "*) ;;
    *) echo "unknown source $s (one of: $ALL)" >&2; exit 2 ;;
  esac
done

# check the build environment before downloading
command -v hf > /dev/null || { echo "the hf command is missing: pip install -r build/requirements.txt" >&2; exit 2; }
(cd "$REPO" && "$PY" -c 'import build.assemble, build.verify, PIL, pyarrow.parquet, importlib.util as u
assert u.find_spec("pymupdf") or u.find_spec("fitz"), "PyMuPDF is missing"') \
  || { echo "the build environment lacks a package: pip install -r build/requirements.txt (or set PYTHON)" >&2; exit 2; }

mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)

# get <repository> <revision> <folder> [<file pattern> ...] downloads the files that match the patterns, or the
# whole repository without a pattern. Each pattern gets its own hf call, since huggingface_hub versions differ
# in how --include takes several patterns.
get() {
  local repo=$1 rev=$2 dir=$3 p
  shift 3
  if [ $# -eq 0 ]; then
    hf download "$repo" --repo-type dataset --revision "$rev" --local-dir "$ROOT/$dir"
  fi
  for p in "$@"; do
    hf download "$repo" --repo-type dataset --revision "$rev" --local-dir "$ROOT/$dir" --include "$p"
  done
}

for s in $SOURCES; do
  echo "== download $s"
  case "$s" in
    MileBench)
      get FreedomIntelligence/MileBench 53c7a58051ef88bacf76541d91f03f5ba2d71e7d milebench "MileBench_part*.tar.gz"
      for f in "$ROOT"/milebench/MileBench_part*.tar.gz; do
        [ -e "$f.extracted" ] && continue
        tar xzf "$f" -C "$ROOT/milebench"
        touch "$f.extracted"
      done ;;
    VLM2-Bench)
      get Sterzhang/vlm2-bench 188a97b9f52ef44316a260a0c3e90bf49d93ca5f vlm2-bench VLM2Bench_img.tsv ;;
    M4Bench)
      get Anonymous8976/M4Bench 65ca74a18edd43921a479053265a3490157f612c m4bench "*.tsv" ;;
    MMLongBench-Doc)
      get yubo2333/MMLongBench-Doc 2ff6aa9237fc777b6627dc57a486e9225ac5fb86 mmlongbench-doc ;;
    MaRVL)
      get floschne/marvl d25b0dc0bce9d8e9b56bf666d3442dc907180667 marvl "data/*.parquet" ;;
    Q-Bench2)
      get q-future/q-bench2 c766b14f9a54205d8e06cf644441d745dc16f4e3 q-bench2 q-bench2-a1-dev.jsonl all_single_images.zip ;;
    EMMA)
      get luckychao/EMMA 6c87aec9048c489108170088e50b768646b5bcf9 emma "*/test-00000-of-00001.parquet" ;;
  esac
done

cd "$REPO"
ARGS=()
for s in $SOURCES; do ARGS+=(--source "$s"); done
[ "${LINK:-0}" = 1 ] && ARGS+=(--link)
echo "== assemble"
"$PY" -m build.assemble "${ARGS[@]}" --release-root "$ROOT" --records "$RECORDS" --out "$DATA"
FULL=$DATA/data/$(basename "$RECORDS" .jsonl)_full.jsonl
echo "== check against build/checksums"
VARGS=()
[ "$SOURCES" = "$ALL" ] || for s in $SOURCES; do VARGS+=(--source "$s"); done
"$PY" -m build.verify --records "$FULL" ${VARGS[@]+"${VARGS[@]}"}
[ "$SOURCES" = "$ALL" ] || echo "note: only the items of $SOURCES are filled in; run without SOURCES to fill in every item"
echo "done: $FULL"
