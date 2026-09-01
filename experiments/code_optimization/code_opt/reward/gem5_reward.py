"""The reward: a correctness gate, then a gem5-measured speedup.

Contract:
- The reward FUNCTION is an argument (no single fixed transform). All transforms
  assume reward 0 unless the rollout passes correctness on EVERY case (and 0 if it
  exceeds the per-case tick budget); ln is the NATURAL log. With s = src_time/rollout_time:
    ratio           r = s
    log1p_ratio     r = 1 + ln(1 + s)
    gated_log_ratio r = ln(s) if rollout_time < src_time else 0
- Phase 1 (native): each rollout runs ALL the dataset's filtered cases for the
  problem (`usable_case_ids` from pie-gem5-{bysrc,pairs}); any wrong output /
  crash / timeout => reward 0. Comparator and compiler are byte-identical to the
  offline screen that produced `usable_case_ids` (measurement.core._compile at
  -O0 and _outputs_match: line-wise + 1e-3 float tolerance), so a case the
  screen accepted cannot be failed here for a toolchain reason.
- Phase 2 (gem5): only gate-passers are timed, on at most K usable cases
  (k_eff = min(K, n_usable); K=-1 = all). Per-case budget: a rollout exceeding
  budget_mult (3x) the SRC's ticks on a case is stopped and gets reward 0
  (forkloop budget at 3 x max(src over K) stops the sim early; an exact
  per-case 3x post-check enforces the rule case-by-case).
- src_time / rollout_time = sums of gem5 ticks over the SAME K cases; src ticks
  come from the dataset (`gem5_src_per_tc_ticks`), rollout ticks are measured
  live with the IDENTICAL pinned toolchain + gem5 build that produced them.
- K-case sampling default "group": every rollout of the same (problem, src) at
  a step gets the same K cases (advantages compare like-for-like); rotates per
  call/step. Also "rollout" (independent) and "fixed" (stable across steps).
- Robustness: every child runs in its own process group (timeout kills the
  whole tree), prlimit caps address space / file size (print bombs) / nproc
  (fork bombs) / core dumps; per-rollout cumulative native budget; a global
  batch deadline zero-fills stragglers and SIGKILLs their process groups so an
  RL step can never stall. Data/infra faults zero the rollout immediately and
  are never retried, so a step's wall time stays bounded; config errors, by
  contrast, raise loudly up front rather than silently zeroing a whole run.
- Memory: expected outputs are loaded once per problem per batch and freed
  after; program stdout goes to size-capped files (never unbounded pipes);
  per-rollout tmpdirs are removed in finally; duplicate (problem, code)
  rollouts are graded once and fanned out.
- Parallelism: ThreadPoolExecutor with nearest-power-of-2(available cores)
  workers, each thread driving one pinned subprocess at a time (compile /
  native run / gem5 are all separate processes, so all cores are busy without
  fork()ing the trainer driver).

verl wiring:
    custom_reward_function.path=code_opt/reward/gem5_reward.py
    custom_reward_function.name=compute_score
ground_truth JSON per sample must carry: problem_id, usable_case_ids,
gem5_src_per_tc_ticks (or src_per_tc_ticks), optionally src_id (group key).
Everything else (reward_kind, k_cases, sample_mode, the per-step `step` nonce,
etc.) is reward_kwargs / env, NOT per-sample ground_truth.

Self-test on a node with the real gem5 stack:
    python code_opt/reward/gem5_reward.py --selftest --n 3
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as cf_wait
from dataclasses import dataclass, field
from queue import Empty, Queue

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from code_opt.measurement.core import (  # noqa: E402
    _compile, _outputs_match, setup_pch,
)
from code_opt.measurement import gem5_backend as _g5  # noqa: E402


def _require_gem5_stack() -> None:
    """Fail loudly if the gem5 toolchain is missing.

    Without this the reward still "works": every rollout fails to time, scores 0,
    and the run looks like a model that never finds a speedup. Checked once per
    batch -- the check is four os.path.exists calls.
    """
    if _gem5_checked[0]:
        return
    if not _g5.gem5_available():
        raise RuntimeError(
            "the gem5 timing stack is not available -- every rollout would score 0. "
            "Expected a gem5 binary, an x86_64 sysroot, a g++-9 and the SE-mode "
            "configs; see gem5/README.md and scripts/setup_gem5.sh, and check the "
            "PIE_GEM5_* variables printed by scripts/verify_setup.sh."
        )
    _gem5_checked[0] = True


_gem5_checked = [False]


def _default_test_case_dir() -> str:
    """Root of the merged PIE test-case corpus (``<problem_id>/input.<i>.txt``).

    Required; there is no sensible default because the corpus is ~3.4 GB and is
    downloaded, not shipped. ``scripts/download_data.sh`` fetches it and prints the
    export line.
    """
    d = os.environ.get("PIE_TEST_CASE_DIR")
    if not d:
        raise RuntimeError(
            "PIE_TEST_CASE_DIR is not set. It must point at the merged PIE test-case "
            "corpus (one directory per problem_id, holding input.<i>.txt / "
            "output.<i>.txt). Run scripts/download_data.sh, or set it by hand."
        )
    return d


# No fallback parser here on purpose: a second extractor with slightly different
# rules would silently change which code gets graded.
from code_opt.reward.code_extraction import extract_code  # noqa: E402

_extract_code = extract_code

# Actor-checkpoint carve-out. The keep-all hf_model snapshotter
# patches RayPPOTrainer._save_checkpoint, which executes INSIDE the Ray
# TaskRunner actor — the same process that imports this reward module (verl
# exec_module's the custom reward fn here, and the threaded reward executor runs
# here too). train.py's DRIVER-side install never reaches that actor, so without
# this the carve-out directory stays empty. Install here as well:
# idempotent (_INSTALLED guard) and a no-op unless PIE_ACTOR_CARVEOUT_DIR is set.
try:
    from code_opt.guards.actor_carveout import (  # noqa: E402
        install_actor_carveout_hook as _install_actor_carveout_hook,
    )
    _install_actor_carveout_hook()
except Exception as _carveout_e:  # never let carve-out wiring break the reward
    print(f"[carveout] WARN: install from gem5_reward failed: {_carveout_e}", file=sys.stderr)

_PRLIMIT = "/usr/bin/prlimit"
_TASKSET = "/usr/bin/taskset"

# ----------------------------------------------------------------------------- config

def _env_f(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def _env_i(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


@dataclass
class Gem5RewardConfig:
    """All knobs. Defaults read the env at construction time (not import time)."""
    # reward selection
    reward_kind: str = field(default_factory=lambda: os.environ.get("PIE_GEM5_REWARD_KIND", "ratio"))
    k_cases: int = field(default_factory=lambda: _env_i("PIE_GEM5_REWARD_K", "5"))  # -1 = all usable
    sample_mode: str = field(default_factory=lambda: os.environ.get("PIE_GEM5_REWARD_SAMPLE", "group"))
    # WHICH usable cases get gem5-timed (correctness always runs on ALL usable):
    #   "sample" (default) -> random K via select_cases(sample_mode)
    #   "biggest"          -> the K largest-src-ticks usable cases (compute-heavy
    #                         cases that expose an algorithmic gap; deterministic).
    # Per-row ground_truth "gem5_case_ids" still overrides this when present.
    case_select: str = field(default_factory=lambda: os.environ.get("PIE_GEM5_CASE_SELECT", "sample"))
    seed: int = field(default_factory=lambda: _env_i("PIE_GEM5_REWARD_SEED", "0"))
    step: int | None = None          # RL step for group/rollout rotation; None -> per-call counter
    budget_mult: float = field(default_factory=lambda: _env_f("PIE_GEM5_BUDGET_MULT", "3.0"))
    # native gate
    native_timeout_s: float = field(default_factory=lambda: _env_f("PIE_NAT_TIMEOUT", "4"))
    native_rollout_budget_s: float = field(default_factory=lambda: _env_f("PIE_GEM5_REWARD_NATIVE_BUDGET", "120"))
    # Source-relative per-rollout WALL cap on the correctness gate.
    #   cap = clamp(k*src_sim_s + per_case*n_cases, floor=10s, ceil=native_rollout_budget_s)
    #   src_sim_s = sum(src ticks over usable cases) / gem5_ticks_per_sec.
    native_correct_floor_s: float = field(default_factory=lambda: _env_f("PIE_NAT_CORRECT_FLOOR", "10"))
    native_correct_mult: float = field(default_factory=lambda: _env_f("PIE_NAT_CORRECT_MULT", "3"))
    native_per_case_s: float = field(default_factory=lambda: _env_f("PIE_NAT_PER_CASE", "0.1"))
    gem5_ticks_per_sec: float = field(default_factory=lambda: _env_f("PIE_GEM5_TICKS_PER_SEC", "1e12"))
    # Wall cap on a single (possibly network-filesystem) file read.
    file_read_timeout_s: float = field(default_factory=lambda: _env_f("PIE_FILE_READ_TIMEOUT", "15"))
    mem_limit_mb: int = field(default_factory=lambda: _env_i("PIE_PROGRAM_MEM_LIMIT_GB", "1") * 1024)
    fsize_limit_mb: int = field(default_factory=lambda: _env_i("PIE_GEM5_REWARD_FSIZE_MB", "64"))
    nproc_limit: int = field(default_factory=lambda: _env_i("PIE_GEM5_REWARD_NPROC", "2048"))
    use_pch: bool = field(default_factory=lambda: os.environ.get("PIE_GEM5_REWARD_PCH", "1") != "0")
    # (native phase always stops at the first failing case — there is no
    #  run-all-cases-anyway mode, so it is not a configurable knob)
    # parallelism / deadline
    workers: int | None = None       # None -> nearest_pow2(available cores); env PIE_GEM5_REWARD_WORKERS
    no_cpu_pin: bool = False          # fine-shard mode: skip taskset core pinning so the OS
                                      # schedules freely. Prevents co-located shards' independent
                                      # core_qs from pinning many procs to the same cpu (which
                                      # would inflate wall-clock and spuriously trip native
                                      # timeouts -> false-zero a correct rollout). gem5 ticks are
                                      # simulated/deterministic, so pinning never affects the score.
    global_deadline_s: float = field(default_factory=lambda: _env_f("PIE_GEM5_REWARD_DEADLINE", "3600"))
    # data
    test_case_dir: str = field(default_factory=lambda: _default_test_case_dir())
    detail: bool = False             # include per-case native walls in results


# ----------------------------------------------------------------------- reward kinds

def _r_ratio(s: float) -> float:
    return s


def _r_log1p_ratio(s: float) -> float:
    return 1.0 + math.log1p(s)          # natural log; only ever applied on gate-pass


def _r_gated_log_ratio(s: float) -> float:
    return math.log(s) if s > 1.0 else 0.0   # natural log


def _r_log_ratio(s: float) -> float:
    # ln(s), NEGATIVE for s < 1: a correct-but-slower-than-src rollout scores
    # BELOW the 0 given to gate failures (deliberate ablation semantics).
    # budget_mult=3 truncates s < 1/3 upstream, so the range is [ln(1/3), ...).
    return math.log(max(s, 1e-12))          # natural log; guard exact-0 pathologies


def _r_one_plus_log_ratio(s: float) -> float:
    # 1 + ln(s): negative only below s < 1/e ~= 0.368; with budget_mult=3 the
    # sub-failure sliver is s in [1/3, 1/e) i.e. scores in [-0.099, 0).
    return 1.0 + math.log(max(s, 1e-12))    # natural log


def _r_ln1p_ratio(s: float) -> float:
    # ln(1+s): strictly positive on every pass (floor ln(4/3)=0.29 under
    # budget_mult=3), so correctness ALWAYS outranks the 0 given to failures.
    # Differs from the historical "log1p_ratio" kind by exactly -1.
    return math.log1p(s)                    # natural log


REWARD_KINDS = {
    # ALL gated: reward is 0 unless the rollout passes EVERY usable test case
    # (and 0 if it exceeds the per-case tick budget); these transforms are
    # applied only on the full-pass path. s = src_time / rollout_time.
    "ratio": _r_ratio,                    # s
    "log1p_ratio": _r_log1p_ratio,        # 1 + ln(1 + s)
    "gated_log_ratio": _r_gated_log_ratio,  # 1{rollout<src} ln(s)
    "log_ratio": _r_log_ratio,            # ln(s); slow-correct < failure(0)
    "one_plus_log_ratio": _r_one_plus_log_ratio,  # 1 + ln(s); failure ~ s=1/e
    "ln1p_ratio": _r_ln1p_ratio,          # ln(1+s); = log1p_ratio - 1
}


def nearest_pow2(n: int) -> int:
    """Nearest power of two (ties resolve DOWN: 96 -> 64, 72 -> 64, 288 -> 256)."""
    if n <= 1:
        return 1
    lo = 1 << (n.bit_length() - 1)
    hi = lo << 1
    return lo if (n - lo) <= (hi - n) else hi


def _resolve_workers(cfg: Gem5RewardConfig, ncores: int) -> int:
    env = os.environ.get("PIE_GEM5_REWARD_WORKERS")
    if cfg.workers:
        return max(1, cfg.workers)
    if env:
        return max(1, int(env))
    return nearest_pow2(ncores)


def _affinity_cores() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return list(range(os.cpu_count() or 1))


# ------------------------------------------------------------------------ case sampling

_CALL_COUNTER = itertools.count()


def _reset_call_counter() -> None:  # test hook
    global _CALL_COUNTER
    _CALL_COUNTER = itertools.count()


def select_cases(usable: list[int], k: int, mode: str, seed: int, nonce: int,
                 group_key: str, uid_idx: int) -> list[int]:
    """Pick the gem5-timed case ids. Pure + deterministic given the arguments.

    k=-1 or k>=n -> all cases (clamp k_eff = min(K, n); never pad/skip/zero a
    short problem). mode: group = same cases for every rollout sharing
    group_key this call; rollout = independent per rollout; fixed = stable
    across calls (no nonce). Returned sorted for stable input ordering.
    """
    n = len(usable)
    if k < 0 or k >= n:
        return sorted(usable)
    if mode == "group":
        key = f"{seed}|{nonce}|{group_key}"
    elif mode == "rollout":
        key = f"{seed}|{nonce}|{group_key}|{uid_idx}"
    elif mode == "fixed":
        key = f"{seed}|{group_key}"
    else:
        raise ValueError(f"unknown sample_mode {mode!r} (group|rollout|fixed)")
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    import random as _random
    return sorted(_random.Random(h).sample(sorted(usable), k))


# --------------------------------------------------------------------- process registry

class _ProcRegistry:
    """Live child process-group ids, so the global deadline can SIGKILL
    everything a straggler task has in flight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pg: dict[int, int] = {}

    def register(self, key: int, pgid: int) -> None:
        with self._lock:
            self._pg[key] = pgid

    def unregister(self, key: int) -> None:
        with self._lock:
            self._pg.pop(key, None)

    def kill_all(self) -> None:
        with self._lock:
            pgids = list(self._pg.values())
            self._pg.clear()
        for pg in pgids:
            try:
                os.killpg(pg, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _killpg_and_reap(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        p.wait(timeout=10)
    except Exception:
        pass


def _read_text_bounded(path: str, timeout_s: float, cap_bytes: int,
                       ) -> tuple[str | None, str | None]:
    """Read a (possibly network-filesystem) text file with a wall-clock cap that
    survives an uninterruptible D-state hang on the filer. The read runs in a
    DAEMON thread; open()/read() release the GIL during the syscall, so a wedged
    read never blocks the grader or other threads. The join has a timeout and, on
    miss, returns (None, ...) and abandons the thread, which exits on its own once
    the filer recovers. Decode semantics are utf-8 with errors='replace', matching
    a plain open(..., 'r', errors='replace').read()."""
    box: dict = {}

    def _do() -> None:
        try:
            with open(path, "rb") as f:
                box["data"] = f.read(cap_bytes)
        except OSError as e:
            box["err"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=_do, name="g5rw-read", daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        return None, f"read_timeout>{timeout_s}s"
    if "err" in box:
        return None, box["err"]
    return box["data"].decode("utf-8", "replace"), None


# -------------------------------------------------------------------------- native gate

def _run_case_native(exe: str, in_path: str, expected: str, cfg: Gem5RewardConfig,
                     cpu: int | None, reg: _ProcRegistry, reg_key: int,
                     wall_budget_s: float | None = None,
                     ) -> tuple[bool, str, float]:
    """Run one test case. Returns (passed, why, wall_s).

    Verdict-identical to the offline screen's core._run_and_compare (same comparator,
    same AS cap, rc!=0/timeout => fail) with three robustness upgrades that
    cannot flip a legitimate verdict: stdout goes to a size-capped FILE (a
    print bomb gets SIGXFSZ instead of OOMing the trainer), the child runs in
    its own process group (timeout kills the whole tree, incl. fork bombs,
    which prlimit --nproc additionally caps), and core dumps are off.
    """
    argv = [_PRLIMIT,
            f"--as={cfg.mem_limit_mb << 20}",
            f"--fsize={cfg.fsize_limit_mb << 20}",
            f"--nproc={cfg.nproc_limit}",
            "--core=0", "--"]
    if cpu is not None:
        argv += [_TASKSET, "-c", str(cpu)]
    argv += [exe]
    out_path = exe + ".out"
    t0 = time.monotonic()
    try:
        with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
            p = subprocess.Popen(argv, stdin=fin, stdout=fout,
                                 stderr=subprocess.DEVNULL,
                                 cwd=os.path.dirname(exe), start_new_session=True)
            reg.register(reg_key, p.pid)
            try:
                _to = (cfg.native_timeout_s if wall_budget_s is None
                       else max(0.1, min(cfg.native_timeout_s, wall_budget_s)))
                rc = p.wait(timeout=_to)
                # the leader exited; reap any descendants it left spinning
                # (e.g. a fork bomb whose parent returns 0) — the process
                # group outlives the leader while members are alive
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            except subprocess.TimeoutExpired:
                _killpg_and_reap(p)
                return False, "timeout", time.monotonic() - t0
            finally:
                reg.unregister(reg_key)
    except OSError as e:
        return False, f"os_error:{e}", time.monotonic() - t0
    wall = time.monotonic() - t0
    if rc != 0:
        return False, f"exit_{rc}", wall
    try:
        cap = (cfg.fsize_limit_mb << 20) + 1
        with open(out_path, "rb") as f:
            out = f.read(cap).decode("utf-8", "replace")
    except OSError as e:
        return False, f"read_error:{e}", wall
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    return (_outputs_match(out, expected), "mismatch", wall)


# gem5 timing indirection (monkeypatch point for tests)
_gem5_time_one_fn = _g5.gem5_time_one


# ------------------------------------------------------------------------- one rollout

def _zero(phase: str, reason: str, **extra) -> dict:
    return {"score": 0.0, "acc": 0.0, "speedup": 0.0, "phase": phase,
            "reason": reason, **extra}


def _grade_one(rep: dict, cfg: Gem5RewardConfig, transform, prob: dict,
               core_q: Queue, reg: _ProcRegistry, scratch: str,
               pch: str | None, deadline_t: float) -> dict:
    """Grade ONE unique (problem, code) rollout: native gate on all usable
    cases, then gem5 on its selected K cases. Never raises."""
    cases_all: list[int] = rep["usable"]
    src_ticks: dict[int, int] = rep["src_ticks"]
    sel: list[int] = rep["sel_cases"]
    reg_key = threading.get_ident()

    try:
        cpu = core_q.get(timeout=max(1.0, deadline_t - time.monotonic()))
    except Empty:
        return _zero("deadline", "no core before deadline")
    # core_q still throttles in-flight grades + bounds the deadline wait, but in
    # no_cpu_pin (fine-shard) mode we do NOT taskset-pin: let the OS schedule so
    # co-located shards don't pile multiple procs on the same core. Score is
    # unaffected either way (gem5 ticks are simulated; native gate is output match).
    pin = None if cfg.no_cpu_pin else cpu
    tmpdir = tempfile.mkdtemp(prefix="g5rw_", dir=scratch)
    t_nat0 = time.monotonic()
    try:
        exe, cerr = _compile(rep["code"], tmpdir, pch_include_path=pch)
        if exe is None and pch is not None:
            # a forced -include of bits/stdc++.h can (rarely) clash with code
            # that redefines stdlib symbols; the offline screen compiled without
            # a PCH, so retry bare before calling it a compile failure
            exe, cerr = _compile(rep["code"], tmpdir, pch_include_path=None)
        if exe is None:
            return _zero("compile", f"native compile: {cerr}",
                         wall_native_s=time.monotonic() - t_nat0)

        # phase 1: ALL usable cases, cheapest (by src ticks) first for fast fails.
        # Source-relative per-rollout WALL cap: a correct candidate should not run
        # natively much longer than the source's own simulated runtime. Without it,
        # a wedged native run or a D-state filer read holds the shard for the full
        # flat budget. cap = clamp(k*src_sim_s + per_case*n_cases, floor, native_budget).
        order = sorted(cases_all, key=lambda c: src_ticks.get(c, 1 << 62))
        src_sim_s = ((sum(v for v in (src_ticks.get(c) for c in order) if v)
                      / cfg.gem5_ticks_per_sec) if cfg.gem5_ticks_per_sec > 0 else 0.0)
        nat_wall_cap = min(cfg.native_rollout_budget_s,
                           max(cfg.native_correct_floor_s,
                               cfg.native_correct_mult * src_sim_s
                               + cfg.native_per_case_s * len(order)))
        walls: dict[int, float] = {}
        n_pass = 0
        for c in order:
            io = prob["cases"].get(c)
            if io is None:
                return _zero("infra", f"missing test files for case {c}",
                             n_native_pass=n_pass, n_native_total=len(order))
            _remaining = nat_wall_cap - (time.monotonic() - t_nat0)
            ok, why, w = _run_case_native(
                exe, io[0], io[1], cfg, pin, reg, reg_key,
                wall_budget_s=_remaining)
            walls[c] = w
            if not ok:
                # A timeout produced by the SHRINKING per-rollout WALL cap (the
                # remaining-budget clamp, NOT the flat per-case native_timeout)
                # is a budget overrun, not a correctness failure -- relabel it
                # native_budget so the phase stays diagnostic and keeps the same
                # meaning as the older cumulative-budget accounting. The
                # discriminator _remaining < native_timeout_s means the rollout
                # cap, not the per-case timeout, is what bounded this case.
                if (why == "timeout" and n_pass > 0
                        and _remaining < cfg.native_timeout_s):
                    return _zero("native_budget",
                                 f"native phase over wall cap {nat_wall_cap:.1f}s "
                                 f"(floor={cfg.native_correct_floor_s}s, "
                                 f"{cfg.native_correct_mult}x src_sim={src_sim_s:.2f}s "
                                 f"+ {cfg.native_per_case_s}x{len(order)}cases) "
                                 f"after {n_pass}/{len(order)} cases",
                                 n_native_pass=n_pass, n_native_total=len(order),
                                 wall_native_s=time.monotonic() - t_nat0)
                return _zero("native", f"case {c}: {why}",
                             n_native_pass=n_pass, n_native_total=len(order),
                             wall_native_s=time.monotonic() - t_nat0)
            n_pass += 1
            if (time.monotonic() - t_nat0 > nat_wall_cap
                    and n_pass < len(order)):
                return _zero("native_budget",
                             f"native phase over wall cap {nat_wall_cap:.1f}s "
                             f"(floor={cfg.native_correct_floor_s}s, "
                             f"{cfg.native_correct_mult}x src_sim={src_sim_s:.2f}s "
                             f"+ {cfg.native_per_case_s}x{len(order)}cases) "
                             f"after {n_pass}/{len(order)} cases",
                             n_native_pass=n_pass, n_native_total=len(order),
                             wall_native_s=time.monotonic() - t_nat0)
        wall_native = time.monotonic() - t_nat0

        # phase 2: gem5 on the selected cases (gate fully passed => acc=1)
        t_g50 = time.monotonic()
        srcs = [src_ticks.get(c) for c in sel]
        if not sel or any(s is None or s <= 0 for s in srcs):
            return _zero("infra", "missing/invalid src ticks for selected cases",
                         n_native_pass=n_pass, n_native_total=len(order)) | {"acc": 1.0}
        budget_abs = int(cfg.budget_mult * max(srcs))
        inputs = []
        for c in sel:
            txt, rerr = _read_text_bounded(prob["cases"][c][0],
                                           cfg.file_read_timeout_s,
                                           cfg.fsize_limit_mb << 20)
            if rerr is not None:
                return _zero("infra", f"read input {c}: {rerr}") | {"acc": 1.0}
            inputs.append(txt)
        res = _gem5_time_one_fn(rep["code"], inputs, cpu=pin,
                                max_tc_ticks=budget_abs)
        wall_g5 = time.monotonic() - t_g50
        base = {"acc": 1.0, "n_native_pass": n_pass, "n_native_total": len(order),
                "gem5_case_ids": sel, "wall_native_s": round(wall_native, 3),
                "wall_gem5_s": round(wall_g5, 3)}
        if cfg.detail:
            base["native_case_wall_s"] = {c: round(w, 4) for c, w in walls.items()}
        if res.get("error") or not res.get("per_tc"):
            return {**_zero("gem5", f"gem5: {res.get('error') or 'no per-tc ticks'}"),
                    **base}
        per_tc = res["per_tc"]
        if len(per_tc) != len(sel):
            return {**_zero("gem5", f"gem5 per-tc count {len(per_tc)} != {len(sel)}"),
                    **base}
        over = [c for c, s, t in zip(sel, srcs, per_tc)
                if t > cfg.budget_mult * s]
        if over:
            return {**_zero("gem5_over_budget",
                            f"rollout > {cfg.budget_mult}x src on case(s) {over}"),
                    **base}
        src_sum, cand_sum = sum(srcs), sum(per_tc)
        speedup = src_sum / cand_sum
        return {**base, "score": float(transform(speedup)), "speedup": float(speedup),
                "phase": "ok", "reason": None,
                "src_sum_ticks": int(src_sum), "cand_sum_ticks": int(cand_sum)}
    except Exception as e:  # never poison the batch
        return _zero("internal", f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        core_q.put(cpu)


# ------------------------------------------------------------------------ batch driver

_PCH_CACHE: dict[str, str | None] = {}
_PCH_LOCK = threading.Lock()


def _ensure_pch(scratch: str) -> str | None:
    with _PCH_LOCK:
        if scratch in _PCH_CACHE:
            return _PCH_CACHE[scratch]
        try:
            p = setup_pch(scratch)
        except Exception:
            p = None
        _PCH_CACHE[scratch] = p
        return p


def _load_problem(test_case_dir: str, pid: str, case_ids: set[int],
                  cache: dict[str, dict], read_timeout_s: float = 15.0,
                  cap_bytes: int = 64 << 20) -> dict:
    """Per-problem case data: {case_id: (input_path, expected_str)}.
    Loaded once per batch; missing files OR stalled filer reads map to None
    (-> infra zero). The expected-output read is wall-clock bounded."""
    cur = cache.get(pid)
    if cur is None:
        cur = {"cases": {}}
        cache[pid] = cur
    d = os.path.join(test_case_dir, pid)
    for c in case_ids:
        if c in cur["cases"]:
            continue
        ip = os.path.join(d, f"input.{c}.txt")
        op = os.path.join(d, f"output.{c}.txt")
        exp, rerr = _read_text_bounded(op, read_timeout_s, cap_bytes)
        if rerr is not None:                 # missing / stalled filer read
            cur["cases"][c] = None
            continue
        cur["cases"][c] = (ip, exp) if os.path.exists(ip) else None
    return cur


def compute_rewards_batch(jobs: list[dict], cfg: Gem5RewardConfig | None = None,
                          ) -> list[dict]:
    """Grade a batch of rollouts. Returns one result dict per job, in order.

    Each job: {uid?, problem_id, code | solution_str, usable_case_ids,
               src_per_tc_ticks, src_id? | group_key?, gem5_case_ids?}.
    Config errors raise immediately (a typo must not silently zero a run);
    per-rollout data/infra/runtime errors NEVER raise — they zero that rollout
    with a structured phase/reason.
    """
    cfg = cfg or Gem5RewardConfig()
    if cfg.reward_kind not in REWARD_KINDS:
        raise ValueError(f"unknown reward_kind {cfg.reward_kind!r}; "
                         f"choose from {sorted(REWARD_KINDS)}")
    if cfg.k_cases == 0 or cfg.k_cases < -1:
        raise ValueError("k_cases must be -1 (all) or >= 1")
    if cfg.budget_mult <= 0:
        raise ValueError("budget_mult must be > 0")
    if cfg.sample_mode not in ("group", "rollout", "fixed"):
        raise ValueError(f"unknown sample_mode {cfg.sample_mode!r}")
    if cfg.case_select not in ("sample", "biggest"):
        raise ValueError(f"unknown case_select {cfg.case_select!r} (sample|biggest)")
    _require_gem5_stack()
    transform = REWARD_KINDS[cfg.reward_kind]
    nonce = cfg.step if cfg.step is not None else next(_CALL_COUNTER)

    n = len(jobs)
    results: list[dict | None] = [None] * n
    if n == 0:
        return []

    # ---- normalize jobs; extract code; bad data -> immediate zero ----
    norm: list[dict | None] = [None] * n
    for i, j in enumerate(jobs):
        try:
            if j.get("_bad_gt"):
                results[i] = _zero("infra", f"bad ground_truth: {j['_bad_gt']}")
                continue
            code = j.get("code")
            if not code:
                code = _extract_code(j.get("solution_str") or "")
            if not code or not code.strip():
                results[i] = _zero("extract", "no C++ code in completion")
                continue
            uc = j["usable_case_ids"]
            if isinstance(uc, (str, bytes)):
                uc = json.loads(uc)
            usable = sorted({int(c) for c in uc})
            if not usable:
                results[i] = _zero("infra", "empty usable_case_ids")
                continue
            tk = j["src_per_tc_ticks"]
            if isinstance(tk, (str, bytes)):
                tk = json.loads(tk)
            ticks = {int(k): int(v) for k, v in dict(tk).items()}
            gk = j.get("group_key") or f"{j['problem_id']}|{j.get('src_id') or ''}"
            norm[i] = {"uid": j.get("uid", i), "problem_id": str(j["problem_id"]),
                       "code": code, "usable": usable, "src_ticks": ticks,
                       "group_key": gk, "sel_override": j.get("gem5_case_ids")}
        except Exception as e:
            results[i] = _zero("infra", f"bad job fields: {type(e).__name__}: {e}")

    live = [i for i in range(n) if norm[i] is not None]
    if not live:
        return [r | {"uid": jobs[i].get("uid", i)} for i, r in enumerate(results)]

    # ---- per-problem case data (loaded once) + per-job case selection ----
    prob_cache: dict[str, dict] = {}
    still = []
    for i in live:
        nj = norm[i]
        try:
            if nj["sel_override"]:                       # explicit per-row pin (highest priority)
                sel = sorted(int(c) for c in nj["sel_override"])
            elif cfg.case_select == "biggest":           # K largest-src-ticks usable cases
                k = cfg.k_cases if cfg.k_cases > 0 else len(nj["usable"])
                ranked = sorted(nj["usable"], key=lambda c: nj["src_ticks"].get(c, 0), reverse=True)
                sel = sorted(ranked[:k])
            else:                                        # random K (default)
                sel = select_cases(nj["usable"], cfg.k_cases, cfg.sample_mode,
                                   cfg.seed, nonce, nj["group_key"], i)
            nj["sel_cases"] = sel
            _load_problem(cfg.test_case_dir, nj["problem_id"],
                          set(nj["usable"]) | set(sel), prob_cache,
                          read_timeout_s=cfg.file_read_timeout_s,
                          cap_bytes=cfg.fsize_limit_mb << 20)
            still.append(i)
        except Exception as e:
            results[i] = _zero("internal", f"case selection/load: {type(e).__name__}: {e}")
            norm[i] = None
    live = still

    # ---- dedup identical rollouts. The key carries everything the reward
    # depends on: the row identity (group_key + usable + src ticks — the same
    # code against a DIFFERENT src has a different reward), the code, and the
    # selected cases (under "rollout" sampling two copies may differ).
    reps: dict[tuple, dict] = {}
    members: dict[tuple, list[int]] = {}
    for i in live:
        nj = norm[i]
        row_fp = hash((nj["group_key"], tuple(nj["usable"]),
                       tuple(sorted(nj["src_ticks"].items()))))
        key = (nj["problem_id"], row_fp,
               hashlib.sha1(nj["code"].encode("utf-8", "replace")).hexdigest(),
               tuple(nj["sel_cases"]))
        reps.setdefault(key, nj)
        members.setdefault(key, []).append(i)

    # ---- thread pool over unique rollouts ----
    cores = _affinity_cores()
    workers = min(_resolve_workers(cfg, len(cores)), max(1, len(reps)))
    core_q: Queue = Queue()
    for c in cores[::-1]:           # tail cores first (away from Ray/vLLM head cores)
        core_q.put(c)
    reg = _ProcRegistry()
    _g5.set_proc_registry(reg)          # in-flight gem5 sims become killable by reg.kill_all()
    scratch = (os.environ.get("PIE_COMPILE_SCRATCH")
               or os.environ.get("PIE_SCRATCH_DIR")
               or os.environ.get("TMPDIR") or tempfile.gettempdir())
    pch = _ensure_pch(scratch) if cfg.use_pch else None
    deadline_t = time.monotonic() + cfg.global_deadline_s

    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="g5rw")
    futs = {}
    try:
        for key, rep in reps.items():
            prob = prob_cache[rep["problem_id"]]
            futs[ex.submit(_grade_one, rep, cfg, transform, prob,
                           core_q, reg, scratch, pch, deadline_t)] = key
        done, not_done = cf_wait(futs, timeout=cfg.global_deadline_s)
        key_res: dict[tuple, dict] = {}
        for f in done:
            try:
                key_res[futs[f]] = f.result()
            except Exception as e:  # _grade_one never raises, but be paranoid
                key_res[futs[f]] = _zero("internal", f"{type(e).__name__}: {e}")
        if not_done:
            reg.kill_all()          # unblock straggler subprocesses immediately
            for f in not_done:
                f.cancel()
                key_res[futs[f]] = _zero(
                    "deadline", f"global deadline {cfg.global_deadline_s}s")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        _g5.set_proc_registry(None)     # detach the per-batch registry

    for key, idxs in members.items():
        r = key_res.get(key) or _zero("internal", "result missing")
        for i in idxs:
            results[i] = dict(r)

    out = []
    for i in range(n):
        r = results[i] or _zero("internal", "no result")
        r["uid"] = (norm[i] or {}).get("uid", jobs[i].get("uid", i))
        r["reward_kind"] = cfg.reward_kind
        out.append(r)
    return out


# ------------------------------------------------------------------------ verl adapters

_CFG_KEYS = ("reward_kind", "k_cases", "sample_mode", "case_select", "seed", "step", "budget_mult",
             "native_timeout_s", "native_rollout_budget_s", "native_correct_floor_s",
             "native_correct_mult", "native_per_case_s", "gem5_ticks_per_sec",
             "file_read_timeout_s", "mem_limit_mb",
             "fsize_limit_mb", "nproc_limit", "use_pch", "workers", "no_cpu_pin",
             "global_deadline_s", "test_case_dir", "detail")


def _cfg_from_kwargs(kw: dict) -> Gem5RewardConfig:
    return Gem5RewardConfig(**{k: kw[k] for k in _CFG_KEYS if k in kw and kw[k] is not None})


def _job_from_ground_truth(i: int, solution_str: str, ground_truth) -> dict:
    job = {"uid": i, "solution_str": solution_str or ""}
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, (str, bytes)) else dict(ground_truth)
        ticks = gt.get("gem5_src_per_tc_ticks", gt.get("src_per_tc_ticks"))
        if isinstance(ticks, (str, bytes)):
            ticks = json.loads(ticks)
        usable = gt.get("usable_case_ids")
        if isinstance(usable, (str, bytes)):
            usable = json.loads(usable)
        # Optional: pin the gem5-TIMED cases (correctness still runs on ALL
        # usable cases in _grade_one; only the speedup is measured on these).
        # Used by the algorithmic-headroom overfit: time only the large case
        # that exposes the asymptotic gap. Accepts gem5_case_ids|timed_case_ids.
        gem5_cases = gt.get("gem5_case_ids", gt.get("timed_case_ids"))
        if isinstance(gem5_cases, (str, bytes)):
            gem5_cases = json.loads(gem5_cases)
        job.update(problem_id=gt["problem_id"], usable_case_ids=usable,
                   src_per_tc_ticks=ticks, src_id=gt.get("src_id"),
                   gem5_case_ids=gem5_cases)
    except Exception as e:
        job["_bad_gt"] = f"{type(e).__name__}: {e}"
    return job


# --------------------------------------------------------------- distributed reward
# The reward normally runs as ONE Ray task on ONE node (verl's @ray.remote(num_cpus=1)
# compute_reward_async), so on a multi-node run the other nodes' cores + RAM sit idle.
# With PIE_GEM5_REWARD_DISTRIBUTED=1, compute_scores_batch shards the batch across the
# live Ray nodes (one child task per node, NodeAffinity-pinned), grades each shard with
# the SAME local path, and gathers IN ORDER. The whole gem5 stack + test cases live on
# the shared FS, so any node can grade any rollout. Default OFF (single-node behavior).
# Shards are GROUP-ALIGNED (split at multiples of G) so each prompt's G rollouts stay on
# one node -> within-group dedup preserved and (for case_select=sample) the per-group
# case choice stays self-consistent. Falls back to local on any error.


def _gkey(item: dict):
    """The (problem_id|src_id) group key from a reward_input's ground_truth, or None."""
    try:
        gt = item.get("ground_truth")
        gt = json.loads(gt) if isinstance(gt, (str, bytes)) else dict(gt)
        return f"{gt.get('problem_id')}|{gt.get('src_id')}"
    except Exception:
        return None


def _group_aligned_shards(reward_inputs: list, n_shards: int) -> list[list]:
    """Split into <= n_shards contiguous chunks that NEVER split a prompt's group,
    using the ACTUAL group_key from ground_truth (robust to any group size G, and to
    the rollouts being group-contiguous as verl lays them out). Order-preserving.
    Falls back to a fixed-G split if group keys can't be parsed."""
    if n_shards <= 1 or len(reward_inputs) <= 1:
        return [reward_inputs]
    keys = [_gkey(it) for it in reward_inputs]
    if any(k is None for k in keys):     # malformed -> safe fixed-size fallback
        return _split_group_aligned(reward_inputs, n_shards, _env_i("PIE_GEM5_GROUP_SIZE", "16"))
    bounds = [0]                          # start index of each contiguous group
    for i in range(1, len(keys)):
        if keys[i] != keys[i - 1]:
            bounds.append(i)
    bounds.append(len(reward_inputs))
    n_groups = len(bounds) - 1
    per = max(1, (n_groups + n_shards - 1) // n_shards)
    shards = []
    for g0 in range(0, n_groups, per):
        s = bounds[g0]; e = bounds[min(g0 + per, n_groups)]
        shards.append(reward_inputs[s:e])
    return shards


def _split_group_aligned(items: list, n_shards: int, group_size: int) -> list[list]:
    """Fixed-G fallback splitter: contiguous chunks whose boundaries land on multiples
    of group_size. Order-preserving. (Primary path is _group_aligned_shards.)"""
    if n_shards <= 1 or len(items) <= group_size:
        return [items]
    n_groups = (len(items) + group_size - 1) // group_size
    groups_per_shard = (n_groups + n_shards - 1) // n_shards
    chunk = max(1, groups_per_shard) * group_size
    return [items[i:i + chunk] for i in range(0, len(items), chunk)]


def _cancel_all(futs) -> None:
    """Best-effort cancel of submitted shard futures (force=True kills
    in-flight tasks). Called before any distributed->local fallback so the
    controller-node regrade doesn't run ON TOP of still-executing shard tasks,
    which would double the load and can OOM the node."""
    if not futs:
        return
    try:
        import ray
    except Exception:
        return
    for f in futs:
        try:
            ray.cancel(f, force=True, recursive=True)
        except Exception:
            pass


def _distribute_scores_batch(reward_inputs: list[dict], kwargs: dict):
    """Fan the batch out across live Ray nodes; return index-aligned results, or
    None to signal 'fall back to local'."""
    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        if not ray.is_initialized():
            return None
        nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU")]
        if len(nodes) < 2:
            return None
        # Resolve the reward config on THIS (head) node and pass it explicitly to
        # every shard, so a worker node with a different/missing PIE_GEM5_* env still
        # grades with identical settings. workers/step stay None so that each node
        # resolves its own worker count from its own environment, which is what
        # allows heterogeneous nodes to size their pools independently.
        cfg = _cfg_from_kwargs(kwargs)
        child_kwargs = dict(kwargs)
        child_kwargs.update({k: getattr(cfg, k) for k in _CFG_KEYS
                             if getattr(cfg, k, None) is not None})
        # FINE-SHARD WORK-STEALING (gated): many small group-aligned shards submitted
        # UNPINNED so Ray load-balances/steals across nodes (the coarse 4-node split
        # left fast nodes idle while one ground the heavy tail). A per-node custom
        # 'g5sim' resource caps concurrent gem5 sims (RAM-bound ~128/node @1.5GB), NOT
        # cores. Auto-falls back to the coarse path if the launcher didn't register
        # g5sim. Reward is byte-identical: per-rollout grading is independent and the
        # gather stays in shard order; whole groups stay intact (dedup atom).
        fine = (os.environ.get("PIE_GEM5_FINE_SHARDS") == "1"
                and float(ray.cluster_resources().get("g5sim", 0)) > 0)
        req = _env_i("PIE_GEM5_SHARD_SLOTS", "16")
        if fine:
            gps = max(1, _env_i("PIE_GEM5_GROUPS_PER_SHARD", "1"))
            keys = [_gkey(it) for it in reward_inputs]
            if any(k is None for k in keys):
                n_groups = max(1, len(reward_inputs) // _env_i("PIE_GEM5_GROUP_SIZE", "16"))
            else:
                n_groups = 1 + sum(1 for i in range(1, len(keys)) if keys[i] != keys[i - 1])
            n_shards = max(1, (n_groups + gps - 1) // gps)
            shards = _group_aligned_shards(reward_inputs, n_shards)
            child_kwargs["workers"] = req        # cap each shard's pool to its reservation
            child_kwargs["no_cpu_pin"] = True     # OS schedules; no cross-shard pin collision
        else:
            shards = _group_aligned_shards(reward_inputs, len(nodes))
        # The shard task MUST be a NESTED function so cloudpickle serializes it BY
        # VALUE. verl exec's this reward file as module 'custom_module', which is not
        # importable on remote Ray workers, so a module-level function fails with
        # ModuleNotFoundError: custom_module. The body re-imports the reward by its
        # REAL importable path (on PYTHONPATH on every node) and grades the shard.
        def _shard_task(shard, kw):
            import ray as _ray
            from code_opt.reward import gem5_reward as _g
            res = _g.compute_scores_batch(shard, _local_only=True, **kw)
            return {"node": _ray.get_runtime_context().get_node_id(), "res": res}
        futs, sizes, targets = [], [], []
        if fine:
            remote_fn = ray.remote(_shard_task)
            for shard in shards:
                if not shard:
                    continue
                slots = float(min(req, len(shard)))   # g5sim held while running (>= pool size)
                futs.append(remote_fn.options(num_cpus=0, resources={"g5sim": slots})
                            .remote(shard, child_kwargs))
                sizes.append(len(shard)); targets.append("steal")
        else:
            remote_fn = ray.remote(num_cpus=1)(_shard_task)
            for shard, node in zip(shards, nodes):
                if not shard:
                    continue
                strat = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=True)
                futs.append(remote_fn.options(scheduling_strategy=strat).remote(shard, child_kwargs))
                sizes.append(len(shard)); targets.append(node["NodeID"][:8])
        # Observability: record TRUE per-shard completion time (ray.wait, in finish order)
        # so a gem5 straggler shard is visible, plus actual-vs-target node placement so
        # head-node packing under soft affinity is visible. This is logged on the SUCCESS
        # path as well, because a silent success gives no evidence the reward actually
        # fanned out rather than quietly collapsing onto one node.
        # BOUNDED gather. Each shard internally honours global_deadline_s (its
        # own cf_wait), so wait a bounded slack PAST that. A wedged shard (hung gem5,
        # D-state filer) is ray.cancel(force=True)'d and its rollouts zero-filled
        # INDEX-ALIGNED, so one bad node can never freeze the RL step. ray.wait /
        # ray.get without a timeout would hang the step forever.
        gather_to = cfg.global_deadline_s + _env_f("PIE_GEM5_DIST_GATHER_SLACK", "300")
        get_to = _env_f("PIE_GEM5_DIST_GET_TIMEOUT", "120")
        t0 = time.time(); idx = {f: i for i, f in enumerate(futs)}; comp = {}; pending = list(futs)
        while pending:
            remaining = gather_to - (time.time() - t0)
            if remaining <= 0:
                break
            done, pending = ray.wait(pending, num_returns=1, timeout=remaining)
            if not done:                     # ray.wait itself timed out -> laggards remain
                break
            for d in done:
                comp[idx[d]] = round(time.time() - t0, 1)
        laggards = set(pending)
        if laggards:
            print(f"[dist-reward] gather timeout {gather_to:.0f}s: cancelling "
                  f"{len(laggards)} laggard shard(s) + zero-filling",
                  file=sys.stderr, flush=True)
            for f in laggards:
                try:
                    ray.cancel(f, force=True, recursive=True)
                except Exception:
                    pass

        def _zero_shard(m: int) -> list:
            return [{"score": 0.0, "acc": 0.0, "speedup": 0.0,
                     "reason": "dist shard timeout"} for _ in range(m)]

        out: list = []; actual = []
        for i, f in enumerate(futs):         # gather in shard order -> index-aligned
            if f in laggards:
                out.extend(_zero_shard(sizes[i])); actual.append("TIMEOUT"); continue
            try:
                r = ray.get(f, timeout=get_to)
            except Exception as e:           # finished-but-unfetchable / late straggler
                try:
                    ray.cancel(f, force=True, recursive=True)
                except Exception:
                    pass
                print(f"[dist-reward] shard {i} ray.get failed ({type(e).__name__}); zero-filling",
                      file=sys.stderr, flush=True)
                out.extend(_zero_shard(sizes[i])); actual.append("ERR"); continue
            out.extend(r["res"]); actual.append(r["node"][:8])
        per_shard = [comp.get(i) for i in range(len(futs)) if comp.get(i) is not None]
        wall = max(per_shard) if per_shard else 0
        if len(futs) > 8:                    # fine-shard mode: compact summary, not 256 numbers
            from collections import Counter as _Counter
            spread = dict(sorted(_Counter(actual).items(), key=lambda kv: -kv[1]))
            ds = (f"done_s[min/mean/max]={min(per_shard):.0f}/"
                  f"{sum(per_shard)/len(per_shard):.0f}/{max(per_shard):.0f}"
                  if per_shard else "done_s=?")
            print(f"[dist-reward] {len(reward_inputs)} rollouts -> {len(futs)} shard(s) over "
                  f"{len(nodes)} node(s) [FINE/steal]; {ds} wall={wall}s "
                  f"shards_per_node={spread} distinct_nodes={len(set(actual))}",
                  file=sys.stderr, flush=True)
        else:
            n_on_target = sum(1 for a, t in zip(actual, targets) if a == t)
            print(f"[dist-reward] {len(reward_inputs)} rollouts -> {len(futs)} shard(s) over "
                  f"{len(nodes)} node(s); sizes={sizes} done_s={per_shard} "
                  f"wall={wall}s on_target={n_on_target}/{len(futs)} "
                  f"distinct_nodes={len(set(actual))}", file=sys.stderr, flush=True)
        if len(out) != len(reward_inputs):    # shape sanity; never return a mismatched batch
            print(f"[dist-reward] shard count {len(out)} != {len(reward_inputs)}; local fallback",
                  file=sys.stderr)
            _cancel_all(futs)                 # don't leave shards running into the local regrade
            return None
        return out
    except Exception as e:
        print(f"[dist-reward] fell back to local: {type(e).__name__}: {e}", file=sys.stderr)
        _cancel_all(locals().get("futs"))     # cancel any submitted shards before local fallback
        return None


def compute_scores_batch(reward_inputs: list[dict], **kwargs) -> list[dict]:
    """verl batch entry. reward_inputs: [{solution_str, ground_truth}, ...].
    Returns [{score, acc, speedup, reason}, ...] (extra keys are pivoted into
    non_tensor_batch by verl's reward manager)."""
    fell_back = False
    if (not kwargs.pop("_local_only", False)
            and os.environ.get("PIE_GEM5_REWARD_DISTRIBUTED") == "1"
            and len(reward_inputs) > _env_i("PIE_GEM5_GROUP_SIZE", "16")):
        dist = _distribute_scores_batch(reward_inputs, dict(kwargs))
        if dist is not None:
            return dist
        fell_back = True                 # dist errored (and already cancelled its shard futures)
    cfg = _cfg_from_kwargs(kwargs)
    if fell_back and cfg.workers is None:
        # The distributed path failed, so the ENTIRE B*G batch is about to be
        # graded on this controller node, which also hosts the Ray head. An
        # uncapped regrade (nearest_pow2(cores) ~= 128 graders x ~1.5 GB per gem5
        # sim) would OOM the node, so hard-cap the local pool to a RAM-safe
        # constant.
        cap = _env_i("PIE_GEM5_FALLBACK_WORKERS", "32")
        cfg.workers = max(1, min(cap, _resolve_workers(cfg, len(_affinity_cores()))))
        print(f"[dist-reward] LOCAL FALLBACK: grading {len(reward_inputs)} rollouts "
              f"on one node, pool capped to {cfg.workers}", file=sys.stderr, flush=True)
    jobs = [_job_from_ground_truth(i, it.get("solution_str", ""), it.get("ground_truth"))
            for i, it in enumerate(reward_inputs)]
    res = compute_rewards_batch(jobs, cfg)
    # reason is purely diagnostic (logging/dump_generations); cap it so no single
    # rollout's compiler/gem5 stderr can balloon verl's controller-side array build.
    return [{"score": r["score"], "acc": r["acc"], "speedup": r["speedup"],
             "reason": (r.get("reason") or "")[:512]} for r in res]


def compute_score(solution_str: str, ground_truth, data_source=None,
                  extra_info=None, **kwargs) -> dict:
    """verl per-sample entry (thin wrapper over the batch path)."""
    return compute_scores_batch(
        [{"solution_str": solution_str, "ground_truth": ground_truth}], **kwargs)[0]


# ---------------------------------------------------------------------------- selftest

def _selftest(args) -> int:  # pragma: no cover - needs the real gem5 stack
    """Golden check on real dataset rows: src code as the rollout must score
    ratio ~= 1.0; the oracle tgt must score >> 1. Run on a compute node."""
    import pandas as pd
    if not _g5.gem5_available():
        print("gem5 stack not available on this node", file=sys.stderr)
        return 2
    # Read the dataset rows straight from the Hub, so the self-test needs nothing
    # built beyond the gem5 stack it is checking. PIE_RL_DATASET overrides with a
    # local directory of bysrc_<split>.parquet if the machine has no network.
    ds = os.environ.get("PIE_RL_DATASET")
    if ds:
        df = pd.read_parquet(f"{ds}/bysrc_{args.split}.parquet")
    else:
        from datasets import load_dataset
        hf_split = {"val": "validation"}.get(args.split, args.split)
        df = load_dataset("stablegradients/pie-gem5-bysrc", split=hf_split).to_pandas()
    df = df[df.n_usable >= max(args.k, 2)].head(args.n)
    jobs, expect = [], []
    for _, row in df.iterrows():
        gt = {"problem_id": row.problem_id, "src_id": row.src_id,
              "usable_case_ids": json.loads(row.usable_case_ids),
              "gem5_src_per_tc_ticks": row.gem5_src_per_tc_ticks}
        for tag, code in (("src", row.src_code), ("tgt", row.oracle_tgt_code)):
            jobs.append(_job_from_ground_truth(len(jobs), f"```cpp\n{code}\n```", json.dumps(gt)))
            expect.append((row.problem_id, tag,
                           1.0 if tag == "src" else float(row.our_speedup_usable)))
    cfg = Gem5RewardConfig(reward_kind="ratio", k_cases=args.k, sample_mode="fixed")
    t0 = time.time()
    res = compute_rewards_batch(jobs, cfg)
    ok = True
    for (pid, tag, exp), r in zip(expect, res):
        line = (f"{pid} {tag:3s} score={r['score']:.3f} acc={r['acc']:.0f} "
                f"phase={r['phase']} reason={r['reason']} (dataset ref ~{exp:.3f})")
        if tag == "src" and not (r["acc"] == 1.0 and 0.85 <= r["score"] <= 1.15):
            line += "  <-- FAIL (src-as-rollout should be ~1.0)"
            ok = False
        print(line)
    print(f"wall {time.time() - t0:.1f}s   {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest(a))
    ap.print_help()
