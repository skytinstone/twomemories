#!/usr/bin/env bash
# Run phases 1 -> 4 in order. After each phase: copy results to a git worktree of branch results/<tag>,
# regenerate tables and figures there (paper/), commit and push (merged into main on the Mac).
#   bash scripts/run_all.sh            # from phase 1; finished phases and cells are skipped
#   bash scripts/run_all.sh --from 3   # start at phase 3
# Safe to rerun in a new session: results live on Drive (TWOMEM_RESULTS), run_phase skips finished cells, and the
# worktree is rebuilt on top of the pushed branch, so pushes stay fast-forward. The checkout itself stays at the
# tag, so every result line carries the tag's commit.
# A phase that hits its unit cap (exit 2) stops the chain so the next phase is approved by hand.
set -euo pipefail

FROM=1
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM=$2; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."
OUT=${TWOMEM_RESULTS:-results}
REF=$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)
BRANCH="results/$REF"
WT="$PWD/.results_wt"
if [ -n "${GH_TOKEN:-}" ]; then
  REMOTE="https://x-access-token:${GH_TOKEN}@github.com/skytinstone/twomemories.git"
else
  REMOTE=origin
fi
MODELS=(Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B)
PHASES=(phase1_probe phase2_bits phase3_tokens phase4_scale)

publish() {  # $1 = phase name
  local base=HEAD
  if git fetch -q "$REMOTE" "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" 2>/dev/null; then
    base="refs/remotes/origin/$BRANCH"
  fi
  [ -d "$WT" ] || git worktree add -q --detach "$WT" "$base"
  git -C "$WT" checkout -q --detach "$base"
  git -C "$WT" reset -q --hard "$base"   # Drive is the source of truth; the worktree is only staging
  rsync -a "$OUT/" "$WT/results/"
  for m in "${MODELS[@]}"; do
    local short=${m#*/}
    python analysis/make_tables.py --model "$m" --results "$WT/results" --out "$WT/paper/tables/$short" >/dev/null \
      && python analysis/figures.py --model "$m" --results "$WT/results" --out "$WT/paper/figures/retained_$short.pdf" \
      && python analysis/report.py --model "$m" --results "$WT/results" > "$WT/paper/report_$short.md" \
      || echo "analysis for $m failed (results are still committed)" >&2
  done
  git -C "$WT" add results paper
  git -C "$WT" commit -q -m "results: $1 ($REF)" || echo "nothing to commit"
  git -C "$WT" push -q "$REMOTE" "HEAD:refs/heads/$BRANCH"
  echo "pushed $BRANCH"
}

for i in 1 2 3 4; do
  [ "$i" -lt "$FROM" ] && continue
  phase=${PHASES[$((i - 1))]}
  echo "=== phase $i/4: $phase"
  set +e
  python scripts/run_phase.py --phase "$phase" --out "$OUT"
  rc=$?
  set -e
  publish "$phase"
  if [ "$rc" -ne 0 ]; then
    echo "phase $phase stopped with exit $rc; see its stop_rule in configs/$phase.yaml" >&2
    exit "$rc"
  fi
done
echo "=== all phases done"
