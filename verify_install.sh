#!/usr/bin/env bash
#
# CAREER installation verification.
#
# Run this after `uv sync` to confirm the environment is wired up correctly.
# Exercises imports, compiled extensions, CUDA availability, the binarization
# pipeline, and a tiny end-to-end CAREER training + validation pass on the
# sample-data shipped with this repo.
#
# Usage:
#   bash verify_install.sh
#
# Runtime: ~1-2 min on a GPU node, ~3-5 min on CPU.
# Exits 0 on success. On failure, prints the failing phase and where to look
# for the full log.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SAMPLE_DATA="${SCRIPT_DIR}/sample-data"

# Prerequisites -----------------------------------------------------------
if [ ! -d "${SAMPLE_DATA}" ]; then
  echo "ERROR: sample-data/ not found at ${SAMPLE_DATA}" >&2
  exit 1
fi
if [ ! -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
  echo "ERROR: ${SCRIPT_DIR}/.venv not found. Run 'cd ${SCRIPT_DIR} && uv sync' first." >&2
  exit 1
fi

WORK_DIR=$(mktemp -d -t career_verify_XXXXXX)
trap "rm -rf '${WORK_DIR}'" EXIT
echo "Verification work dir (auto-cleanup on exit): ${WORK_DIR}"

phase() { printf "\n=== %s ===\n" "$*"; }
ok()    { printf "  [OK] %s\n" "$*"; }
fail()  { printf "  [FAIL] %s\n" "$*" >&2; echo "Full log: ${1:-(see above)}" >&2; exit 1; }

cd "${SCRIPT_DIR}"

# Phase 1: imports --------------------------------------------------------
phase "1/5  Python imports"
uv run python - <<'PY' || { echo "  Run manually: uv run python -c 'import fairseq'"; exit 1; }
import importlib, sys
modules = ['numpy', 'torch', 'omegaconf', 'sacrebleu', 'regex', 'tqdm',
           'bitarray', 'tensorboardX', 'fairseq', 'hydra']
for m in modules:
    importlib.import_module(m)
print(f'  imported {len(modules)} packages OK')

import torch, numpy, fairseq
assert torch.__version__.startswith('1.8.1'), f'unexpected torch: {torch.__version__}'
assert numpy.__version__ == '1.20.3', f'unexpected numpy: {numpy.__version__}'
print(f'  torch={torch.__version__}  numpy={numpy.__version__}  fairseq={fairseq.__version__}')

from fairseq.tasks.occupation_modeling import OccupationModelingTask  # noqa
from fairseq.models.transformer import TransformerModel              # noqa
from fairseq.models.bag_of_jobs import BagOfJobsModel                # noqa
print('  CAREER task + models load OK')
PY
ok "imports + version pins"

# Phase 2: compiled extensions --------------------------------------------
phase "2/5  fairseq C++/Cython extensions"
uv run python - <<'PY' || { echo "  Fix: uv sync --reinstall-package fairseq"; exit 1; }
from fairseq.data.data_utils_fast import batch_by_size_fn          # noqa
from fairseq.data.token_block_utils_fast import _get_slice_indices_fast  # noqa
PY
ok "data_utils_fast + token_block_utils_fast importable"

# Phase 3: CUDA -----------------------------------------------------------
phase "3/5  CUDA availability"
HAS_CUDA=$(uv run python -c "import torch; print('1' if torch.cuda.is_available() else '0')")
if [ "${HAS_CUDA}" = "1" ]; then
  DEVICE=$(uv run python -c "import torch; print(torch.cuda.get_device_name(0))")
  ok "CUDA available: ${DEVICE}"
  FP16_FLAG="--fp16"
else
  ok "no CUDA detected — running training in CPU mode (slower, still valid)"
  FP16_FLAG=""
fi

# Phase 4: preprocess sample-data ----------------------------------------
phase "4/5  Preprocess sample-data (job/year/gender/location)"
BIN="${WORK_DIR}/data-bin/sample"
mkdir -p "${BIN}"
for COV in job year gender location; do
  LOG="${WORK_DIR}/preprocess_${COV}.log"
  uv run fairseq-preprocess --only-source \
    --trainpref "${SAMPLE_DATA}/train.${COV}" \
    --validpref "${SAMPLE_DATA}/valid.${COV}" \
    --testpref  "${SAMPLE_DATA}/test.${COV}" \
    --destdir   "${BIN}/${COV}" \
    --workers 1 > "${LOG}" 2>&1 \
    || { tail -20 "${LOG}"; fail "${LOG}"; }
done
ok "binarized 4 covariates → ${BIN}"

# Phase 5: tiny CAREER training + validation ------------------------------
phase "5/5  CAREER training (10 updates) + validation"
SAVE="${WORK_DIR}/checkpoints/career"
LOG_DIR="${WORK_DIR}/logs/career"
TRAIN_LOG="${WORK_DIR}/train.log"
mkdir -p "${SAVE}" "${LOG_DIR}"

uv run python -m fairseq_cli.train --task occupation_modeling \
    "${BIN}" \
    --arch career \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 2 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 32 --sample-break-mode eos \
    --max-tokens 256 --update-freq 1 \
    --max-update 10 --validate-interval-updates 5 \
    --save-dir "${SAVE}" \
    --tensorboard-logdir "${LOG_DIR}" \
    ${FP16_FLAG} \
    --two-stage \
    --include-year --include-gender --include-location \
    --no-epoch-checkpoints \
    --log-interval 1 > "${TRAIN_LOG}" 2>&1 \
    || { tail -30 "${TRAIN_LOG}"; fail "${TRAIN_LOG}"; }

if [ ! -f "${SAVE}/checkpoint_last.pt" ]; then
    tail -30 "${TRAIN_LOG}"
    fail "${TRAIN_LOG} (training reported success but no checkpoint was saved)"
fi

# Pull the final loss out of the log as a sanity number.
FINAL_LOSS=$(grep -oE '"loss": "[0-9.]+"' "${TRAIN_LOG}" | tail -1 | grep -oE '[0-9.]+' || echo "n/a")
ok "trained 10 updates + validated, checkpoint at ${SAVE}/checkpoint_last.pt"
echo "       final reported loss: ${FINAL_LOSS}"

# Summary -----------------------------------------------------------------
echo
echo "============================================================"
echo " ALL CHECKS PASSED"
echo " Your CAREER environment is wired up correctly."
echo "============================================================"
