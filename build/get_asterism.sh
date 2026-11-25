#!/usr/bin/env bash
# One command for the items released as identifiers: download the public releases of the seven
# sources at the revisions we used, assemble the items and check them against build/checksums.
#
#   bash build/get_asterism.sh <dataset folder> [<folder for the source releases>]
#
# <dataset folder> is the downloaded ASTERISM or ASTERISM-Lite dataset (it holds data/asterism.jsonl or
# data/asterism_lite.jsonl). The assembled records are written to data/asterism_full.jsonl (or
# data/asterism_lite_full.jsonl) and the images under images/ of that folder (about 6 GB for ASTERISM).
# The source releases (about 28 GB to download, about 45 GB on disk once MileBench is extracted) go to
# <folder for the source releases>, by default ./asterism_sources.
# Downloads that are already complete are skipped, so the command can be run again after an interruption.
# LINK=1 hard-links the MileBench image files instead of copying them (same file system only).
# SOURCES="EMMA M4Bench" limits the run to some sources (names as in build/README.md); items of other sources
# that an earlier run filled in are kept, and only the selected sources are checked.
# PYTHON selects the interpreter of the build environment (default: python); the hf command next to it is used
# when there is one.
set -euo pipefail
unset CDPATH
case "${1:-}" in -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' < "$0"; exit 0 ;; esac

DATA=${1:?usage: bash build/get_asterism.sh <dataset folder> [<folder for the source releases>]}
ROOT=${2:-asterism_sources}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=$(cd "$DATA" && pwd)
PY=${PYTHON:-python}
case "$PY" in
  */*) PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY") ;;  # a relative path keeps working after cd
esac
command -v "$PY" > /dev/null || { echo "PYTHON not found: $PY" >&2; exit 2; }

if [ -f "$DATA/data/asterism.jsonl" ]; then
  RECORDS=$DATA/data/asterism.jsonl
elif [ -f "$DATA/data/asterism_lite.jsonl" ]; then
  RECORDS=$DATA/data/asterism_lite.jsonl
else
  echo "no data/asterism.jsonl or data/asterism_lite.jsonl under $DATA" >&2
  exit 2
fi

ALL="MileBench VLM2-Bench M4Bench MMLongBench-Doc MaRVL Q-Bench2 EMMA"
read -r -d '' -a REQUESTED <<< "${SOURCES:-$ALL}" || true  # split on blanks and newlines, no file-name expansion
SRCS=()
for s in ${REQUESTED[@]+"${REQUESTED[@]}"}; do
  case " $ALL " in
    *" $s "*) ;;
    *) echo "unknown source $s (one of: $ALL)" >&2; exit 2 ;;
  esac
  case " ${SRCS[*]-} " in
    *" $s "*) ;;
    *) SRCS+=("$s") ;;
  esac
done
[ ${#SRCS[@]} -gt 0 ] || { echo "SOURCES names no source (one of: $ALL)" >&2; exit 2; }

# check the build environment before downloading
HF=hf
PYBIN=$(command -v "$PY" || true)
if [ -n "$PYBIN" ] && [ -x "$(dirname "$PYBIN")/hf" ]; then
  HF=$(dirname "$PYBIN")/hf
fi
command -v "$HF" > /dev/null || { echo "the hf command is missing: pip install -r build/requirements.txt" >&2; exit 2; }
(cd "$REPO" && "$PY" -c 'import build.assemble, build.verify, PIL.Image, pyarrow.parquet
try:
    import pymupdf
except ImportError:
    import fitz') \
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
    "$HF" download "$repo" --repo-type dataset --revision "$rev" --local-dir "$ROOT/$dir"
  else
    for p in "$@"; do
      "$HF" download "$repo" --repo-type dataset --revision "$rev" --local-dir "$ROOT/$dir" --include "$p"
    done
  fi
}

# read every record before any download; for MMLongBench-Doc, list the documents that the records name
LIST=$("$PY" -c 'import json, sys
names = set()
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        if r["source"] == "MMLongBench-Doc" and r["release"] == "identifiers":
            names.update(ref.split("#")[0] for ref in r["images"])
for name in sorted(names):
    print(name)' "$RECORDS")
DOCS=()
while IFS= read -r d; do
  if [ -n "$d" ]; then DOCS+=("$d"); fi
done <<< "$LIST"

for s in "${SRCS[@]}"; do
  echo "== download $s"
  case "$s" in
    MileBench)
      get FreedomIntelligence/MileBench 53c7a58051ef88bacf76541d91f03f5ba2d71e7d milebench "MileBench_part*.tar.gz"
      for f in "$ROOT"/milebench/MileBench_part*.tar.gz; do
        [ -e "$f.extracted" ] && continue
        echo "   extract $(basename "$f")"
        tar xzf "$f" -C "$ROOT/milebench"
        touch "$f.extracted"
      done ;;
    VLM2-Bench)
      get Sterzhang/vlm2-bench 188a97b9f52ef44316a260a0c3e90bf49d93ca5f vlm2-bench VLM2Bench_img.tsv ;;
    M4Bench)
      get Anonymous8976/M4Bench 65ca74a18edd43921a479053265a3490157f612c m4bench "*.tsv" ;;
    MMLongBench-Doc)
      # the QA table and the documents that the records use, in one call (positional file names work with every
      # version of hf); a download of the whole repository would also fetch documents that no item uses
      echo "   ${#DOCS[@]} documents"
      "$HF" download yubo2333/MMLongBench-Doc data/train-00000-of-00001.parquet ${DOCS[@]+"${DOCS[@]}"} \
        --repo-type dataset --revision 2ff6aa9237fc777b6627dc57a486e9225ac5fb86 --local-dir "$ROOT/mmlongbench-doc" ;;
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
for s in "${SRCS[@]}"; do ARGS+=(--source "$s"); done
[ "${LINK:-0}" = 1 ] && ARGS+=(--link)
echo "== assemble"
"$PY" -m build.assemble "${ARGS[@]}" --release-root "$ROOT" --records "$RECORDS" --out "$DATA"
FULL=$DATA/data/$(basename "$RECORDS" .jsonl)_full.jsonl
echo "== check against build/checksums"
VARGS=()
[ ${#SRCS[@]} -eq 7 ] || for s in "${SRCS[@]}"; do VARGS+=(--source "$s"); done
"$PY" -m build.verify --records "$FULL" ${VARGS[@]+"${VARGS[@]}"}
[ ${#SRCS[@]} -eq 7 ] || echo "note: checked only ${SRCS[*]}; run without SOURCES to assemble and check every source"
echo "done: $FULL"
