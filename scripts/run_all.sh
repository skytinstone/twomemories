#!/usr/bin/env bash
# Run phases 1 -> 4 in order, committing and pushing results after each phase
# (to branch results/<tag>, merged into main on the Mac).
#   bash scripts/run_all.sh            # from phase 1
#   bash scripts/run_all.sh --from 3   # resume from phase 3 (finished cells are skipped anyway)
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
BRANCH="results/$(git describe --tags --always)"
PHASES=(phase1_probe phase2_bits phase3_tokens phase4_scale)
for i in 1 2 3 4; do
  [ "$i" -lt "$FROM" ] && continue
  phase=${PHASES[$((i - 1))]}
  echo "=== phase $i: $phase"
  set +e
  python scripts/run_phase.py --phase "$phase" --out "$OUT"
  rc=$?
  set -e
  [ "$OUT" != "results" ] && rsync -a "$OUT/" results/
  git add results
  git commit -q -m "results: $phase ($(git rev-parse --short HEAD))" || echo "nothing to commit"
  if [ -n "${GH_TOKEN:-}" ]; then
    git push -q "https://x-access-token:${GH_TOKEN}@github.com/skytinstone/twomemories.git" HEAD:refs/heads/$BRANCH
  else
    git push -q origin "HEAD:refs/heads/$BRANCH"
  fi
  if [ "$rc" -ne 0 ]; then
    echo "phase $phase stopped with exit $rc; see its stop_rule in configs/$phase.yaml" >&2
    exit "$rc"
  fi
done
