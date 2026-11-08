#!/usr/bin/env bash
# Colab setup. Run from a notebook cell after mounting Drive:
#   GH_TOKEN=... bash scripts/colab_setup.sh      (the notebook reads GH_TOKEN from Colab secrets)
# Clones twomemories and hybridquant with GH_TOKEN, installs both editable, puts the HF cache on the local disk,
# makes runs write results straight to Drive (TWOMEM_RESULTS), and records the GPU.
# The HF cache is not on Drive: a parquet shard read back through the Drive mount came back truncated
# (docs/decision_log.md, D-18). Models and shards are re-downloaded once per session instead.
# results/ is not a symlink: git would commit the link instead of the jsonl files. run_all.sh rsyncs
# Drive -> results/ before each commit (docs/decision_log.md, D-7).
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is not set (Colab secrets -> GH_TOKEN)}"
OWNER=skytinstone
DRIVE=/content/drive/MyDrive/twomemories
WORK=/content
REF=${TWOMEM_REF:-v0.4-nothink}
HF_LOCAL=/content/hf_cache

if [ ! -d /content/drive/MyDrive ]; then
  echo "Drive is not mounted (run drive.mount('/content/drive') first)" >&2
  exit 1
fi
mkdir -p "$HF_LOCAL" "$DRIVE/results_v2"

clone() {  # name -> /content/name; the token is used for the fetch only and not stored in .git/config
  local name=$1
  if [ ! -d "$WORK/$name/.git" ]; then
    git clone -q "https://x-access-token:${GH_TOKEN}@github.com/${OWNER}/${name}.git" "$WORK/$name"
    git -C "$WORK/$name" remote set-url origin "https://github.com/${OWNER}/${name}.git"
  else
    git -C "$WORK/$name" fetch -q --tags "https://x-access-token:${GH_TOKEN}@github.com/${OWNER}/${name}.git" \
      '+refs/heads/*:refs/remotes/origin/*'
  fi
}
clone hybridquant
clone twomemories
git -C "$WORK/twomemories" checkout -q "$REF"

# Seed Drive results with the repo's tracked results (never overwrite what Drive already has),
# then bring Drive's results back into the checkout so resume and git see the same files.
cd "$WORK/twomemories"
rsync -a --ignore-existing results/ "$DRIVE/results_v2/"
rsync -a "$DRIVE/results_v2/" results/
git config user.name "twomem-colab"
git config user.email "horcruxtech.ai@gmail.com"

pip install -q -e "$WORK/hybridquant" -e "$WORK/twomemories"

# HF cache on the local disk; hub kernels off so GatedDeltaNet keeps calling cache.update_recurrent_state
# (see hybridquant.check_hook_coverage and docs/assumptions.md, premise 3).
cat > /content/twomem_env.sh <<EOF
export HF_HOME=$HF_LOCAL
export TWOMEM_RESULTS=$DRIVE/results_v2
export USE_HUB_KERNELS=NO
export TOKENIZERS_PARALLELISM=false
EOF

GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "none")
echo "GPU: $GPU"
python - <<'EOF'
import torch, transformers
from twomem.runner import gpu_name, units_per_hour, git_commit
g = gpu_name()
print(f"torch {torch.__version__} transformers {transformers.__version__} gpu={g} "
      f"units_per_hour={units_per_hour(g)} commit={git_commit()}")
EOF
echo "source /content/twomem_env.sh before running scripts"
