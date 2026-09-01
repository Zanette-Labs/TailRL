#!/usr/bin/env bash
# Check that everything the reward needs actually resolves, and print what it found.
#
# Run this before any training or evaluation job. The failure mode it exists to catch
# is the quiet one: with a missing toolchain or corpus the reward does not crash, it
# scores every rollout 0, and the run looks like a model that simply never improves
# any program. Several hours of GPU time can disappear that way.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

fail=0
bad()  { printf '  FAIL  %-22s %s\n' "$1" "$2"; fail=1; }

echo "== python =="
python3 - <<'PY' || fail=1
import importlib.util, sys
print(f"  ok    {'python':<22} {sys.version.split()[0]}")
need = ["torch", "transformers", "ray", "hydra", "omegaconf", "tensordict",
        "pandas", "pyarrow", "numpy", "datasets"]
opt = ["vllm", "matplotlib", "wandb"]
missing = []
for m in need:
    spec = importlib.util.find_spec(m)
    if spec is None:
        missing.append(m)
    else:
        mod = importlib.import_module(m)
        print(f"  ok    {m:<22} {getattr(mod, '__version__', '?')}")
for m in opt:
    present = importlib.util.find_spec(m) is not None
    print(f"  {'ok  ' if present else 'warn'}  {m:<22} "
          f"{'present' if present else 'absent (needed for rollouts / plots / logging)'}")
if missing:
    print(f"  FAIL  missing: {', '.join(missing)}")
    raise SystemExit(1)
PY

echo
echo "== package wiring =="
if python3 -c "import verl.trainer.main_ppo" 2>/dev/null; then
  printf '  ok    %-22s %s\n' "import verl" "the vendored fork"
else
  bad "import verl" "PYTHONPATH is wrong; source scripts/env.sh"
fi
est=$(python3 -c "
from verl.trainer.ppo.ray_trainer import AdvantageEstimator as A
print(' '.join(e.value for e in A))" 2>/dev/null || true)
case " ${est} " in
  *" tailrl "*) printf '  ok    %-22s %s\n' "estimators" "${est}" ;;
  *)            bad "estimators" "tailrl is not registered (got: ${est:-nothing})" ;;
esac

echo
echo "== data =="
if [ -d "${PIE_TEST_CASE_DIR}" ]; then
  n=$(find "${PIE_TEST_CASE_DIR}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
  if [ "${n}" -gt 3000 ]; then
    printf '  ok    %-22s %s\n' "test cases" "${n} problems in ${PIE_TEST_CASE_DIR}"
  else
    bad "test cases" "only ${n} problem dirs in ${PIE_TEST_CASE_DIR}; expected ~3907. Re-run scripts/download_data.sh"
  fi
else
  bad "test cases" "${PIE_TEST_CASE_DIR} does not exist; run scripts/download_data.sh"
fi
for f in pie_gem5_train.parquet pie_gem5_test.parquet; do
  if [ -f "${PIE_PARQUET_ROOT}/${f}" ]; then
    r=$(python3 -c "import pandas;print(len(pandas.read_parquet('${PIE_PARQUET_ROOT}/${f}')))" 2>/dev/null || echo '?')
    printf '  ok    %-22s %s rows\n' "${f}" "${r}"
  else
    bad "${f}" "missing; run scripts/prepare_dataset.sh"
  fi
done

echo
echo "== gem5 timing stack =="
python3 - <<'PY' || { fail=1; echo "  -> run scripts/setup_gem5.sh, or point the PIE_GEM5_* variables at an existing build"; }
import os
from code_opt.measurement import gem5_backend as g5
paths = [("gem5 binary", g5._GEM5), ("sysroot loader", g5._LD),
         ("target g++", g5._GXX), ("probe config", g5._PROBE),
         ("forkloop config", g5._SEQ), ("build pyenv", g5._ENVD)]
bad = False
for name, p in paths:
    hit = os.path.exists(p)
    print(f"  {'ok  ' if hit else 'FAIL'}  {name:<22} {p}")
    bad |= not hit
print(f"\n  gem5_available() -> {g5.gem5_available()}")
raise SystemExit(1 if bad else 0)
PY

echo
echo "== host compilers (native correctness gate) =="
python3 - <<'PY' || fail=1
try:
    from code_opt.measurement.core import GPP_PATH, GCC_PATH
    print(f"  ok    {'g++':<22} {GPP_PATH}")
    print(f"  ok    {'gcc':<22} {GCC_PATH}")
except Exception as e:
    print(f"  FAIL  {'host compiler':<22} {e}")
    raise SystemExit(1)
PY

echo
if [ "${fail}" -eq 0 ]; then
  echo "VERIFY OK"
  echo
  echo "One more check is worth the two minutes it costs, on a machine with the full"
  echo "stack -- it grades real dataset rows and asserts the source program scores ~1.0"
  echo "against itself and the reference fast version scores well above 1.0:"
  echo
  echo "    python3 code_opt/reward/gem5_reward.py --selftest --n 3"
else
  echo "VERIFY FAILED -- fix the items above before launching anything."
fi
exit "${fail}"
