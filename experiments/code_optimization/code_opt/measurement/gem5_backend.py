"""gem5 timing backend for the runtime-optimization reward.

Times candidate programs with the x86 gem5 + Skylake-config pipeline:
  1. compile to a DYNAMICALLY linked x86_64 binary with the pinned g++-9 toolchain,
     against the sysroot, with an explicit --dynamic-linker. Dynamic, not static,
     because that is what the dataset's reference times were produced with; it is
     also why the sysroot has to exist, since the simulated loader resolves its
     libraries through gem5's guest-to-host path redirects at run time.
  2. probe the pre-input-read tick (program startup) via run_se_probe.py and
     GEM5_PREINPUT_FORK
  3. fork-amortized sequential loop over the K test inputs (run_se_forkloop.py);
     per-case tick = absolute run-to-completion tick, startup included, which is
     the semantics the PIE dataset's agg_runtime uses
  4. gem5_ticks = sum of per-case ticks over the SAME deterministic K-subset the
     correctness gate and the dataset's reference ticks use

Correctness is NOT checked here -- callers gate with the native runner first and
only send gate-passing unique programs (gem5 stdout is not compared).

Every toolchain path is an environment variable. They all default under
``PIE_GEM5_HOME`` (itself defaulting to ``<experiment>/gem5/build``), which is the
layout ``scripts/setup_gem5.sh`` produces; override any of them individually to
point at an existing build. ``gem5_available()`` reports whether the stack resolves.
Failure modes return ``ticks=None`` with an error string; callers decide the reward
consequence (score 0).
"""
from __future__ import annotations

import atexit
import glob
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, wait as cf_wait

# <experiment>/code_opt/measurement/gem5_backend.py -> <experiment>
_EXP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Everything scripts/setup_gem5.sh builds lands under one root.
_HOME = os.environ.get("PIE_GEM5_HOME", os.path.join(_EXP_ROOT, "gem5", "build"))
# Target sysroot: an x86_64 glibc-2.31 tree (Ubuntu 20.04) supplying the loader and
# the libstdc++/libgcc the simulated binary links against.
_RF = os.environ.get("PIE_GEM5_ROOTFS", os.path.join(_HOME, "rootfs"))
# Compiler root. On an x86_64 host this is a plain g++-9 prefix; on a non-x86 host
# setup_gem5.sh builds a cross-compiler here (see gem5/README.md).
_XROOT = os.environ.get("PIE_GEM5_XCROSS_ROOT", os.path.join(_HOME, "x86cross", "root"))
_GEM5 = os.environ.get("PIE_GEM5_BIN", os.path.join(_HOME, "gem5", "build", "X86", "gem5.fast"))
# Python env whose libpython the gem5 binary was linked against.
_ENVD = os.environ.get("PIE_GEM5_PYENV", os.path.join(_HOME, "pyenv"))
# The Skylake SE-mode configs shipped with this experiment. gem5 runs them with its
# own embedded interpreter on the host, so they live in the repo, not in the sysroot.
_CFG = os.environ.get("PIE_GEM5_CONFIG_DIR", os.path.join(_EXP_ROOT, "gem5", "skylake_config"))
_PROBE = f"{_CFG}/run_se_probe.py"
_SEQ = f"{_CFG}/run_se_forkloop.py"
_GXX = os.environ.get("PIE_GEM5_GXX", f"{_XROOT}/usr/bin/x86_64-linux-gnu-g++-9")
_LD = os.environ.get("PIE_GEM5_LOADER", f"{_RF}/lib/x86_64-linux-gnu/ld-2.31.so")

_COMPILE_TIMEOUT = float(os.environ.get("PIE_GEM5_COMPILE_TIMEOUT", "120"))
_PROBE_TIMEOUT = float(os.environ.get("PIE_GEM5_PROBE_TIMEOUT", "300"))
_FORKLOOP_TIMEOUT = float(os.environ.get("PIE_GEM5_FORKLOOP_TIMEOUT", "900"))
# Wall-clock cap on a whole gem5_time_batch call, the gem5-phase counterpart of
# grade_dedup_batch's global_timeout: stragglers past the cap are reported as
# {"ticks": None, "error": "gem5 batch timeout"} so one pathological program
# cannot stall an entire RL step.
_BATCH_TIMEOUT = float(os.environ.get("PIE_GEM5_BATCH_TIMEOUT", "1200"))
# How far before the measured first-stdin-read tick to place the fork. The fork
# must land after startup is complete but strictly before the program consumes
# any input, so that a single simulated startup can be reused for every test
# case; this margin is the empirically safe gap.
_FORK_MARGIN = 3000000

# Environment for the compiler itself. The compiler is a HOST binary, so its shared
# libraries live under the host's own multiarch triple -- x86_64-linux-gnu for a
# native build, aarch64-linux-gnu for the cross-compiler setup_gem5.sh builds on ARM.
_HOST_TRIPLE = f"{platform.machine()}-linux-gnu"
_GENV = {
    **os.environ,
    "PATH": f"{_XROOT}/usr/bin:" + os.environ.get("PATH", ""),
    "LD_LIBRARY_PATH": os.pathsep.join(p for p in (
        f"{_XROOT}/usr/lib/{_HOST_TRIPLE}",
        f"{_XROOT}/lib/{_HOST_TRIPLE}",
        f"{_XROOT}/usr/lib",
        os.environ.get("LD_LIBRARY_PATH", ""),
    ) if p),
}
# Link flags for the TARGET (always x86_64, since gem5 simulates a Skylake core).
_LFLAGS = [
    "--sysroot=" + _RF,
    f"-L{_RF}/usr/lib/gcc/x86_64-linux-gnu/9",
    f"-L{_RF}/usr/lib/x86_64-linux-gnu",
    f"-L{_RF}/lib/x86_64-linux-gnu",
]
# Environment for the gem5 binary. PIE_GEM5_ROOTFS is exported explicitly (not just
# inherited) because run_se_probe.py / run_se_forkloop.py read it to build the
# guest->host path redirects for the simulated dynamic loader, and it may only exist
# here as a resolved default rather than in the caller's environment.
_NATENV = {
    **os.environ,
    "LD_LIBRARY_PATH": f"{_ENVD}/lib",
    "PYTHONPATH": f"{_CFG}:{_CFG}/system",
    "PIE_GEM5_ROOTFS": _RF,
}

_PRAGMA_RE = re.compile(r"#pragma GCC target.*\n")

# Optional process-group registry installed by the reward's compute_rewards_batch
# (duck-typed .register(key, pgid) / .unregister(key)) so its global-deadline
# kill_all() can SIGKILL in-flight gem5 sims (probe/forkloop), not just the native
# gate. Keyed by thread id: each grader thread drives one gem5 sim at a time, and
# the native gate has already unregistered that key before the gem5 phase starts.
# Default None -> _run_pg behaves exactly as before. Set via a module global (not a
# gem5_time_one kwarg) so the fixed-signature test stubs of _gem5_time_one_fn are
# unaffected.
_PROC_REG = None


def set_proc_registry(reg) -> None:
    global _PROC_REG
    _PROC_REG = reg


def gem5_available() -> bool:
    """True if the cross-compiler + gem5 + config paths all exist."""
    return all(os.path.exists(p) for p in (_GXX, _GEM5, _PROBE, _SEQ, _LD))


def _scratch_base() -> str:
    return os.environ.get("PIE_GEM5_SCRATCH") or (
        "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    )


def _scratch_root() -> str:
    d = os.path.join(_scratch_base(), f"pie_gem5_{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    return d


def _sweep_stale_roots() -> None:
    """Remove pie_gem5_<pid> roots left by dead processes (crash/SIGKILL)."""
    for d in glob.glob(os.path.join(_scratch_base(), "pie_gem5_*")):
        try:
            pid = int(d.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)            # alive -> leave its scratch alone
        except ProcessLookupError:
            shutil.rmtree(d, ignore_errors=True)
        except PermissionError:
            pass                        # someone else's live process


_sweep_stale_roots()
atexit.register(lambda: shutil.rmtree(
    os.path.join(_scratch_base(), f"pie_gem5_{os.getpid()}"), ignore_errors=True))


def _run_pg(argv: list[str], env: dict, timeout: float) -> bytes | None:
    """Run argv in its own process group; on timeout SIGKILL the whole group
    (gem5's forkloop m5.fork()s real OS children that a plain subprocess.run
    kill would orphan mid-simulation). Returns stdout, or None on timeout."""
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    reg = _PROC_REG                          # snapshot (set once per reward batch)
    rkey = threading.get_ident() if reg is not None else None
    if reg is not None:
        try:
            reg.register(rkey, p.pid)        # pgid == leader pid (start_new_session)
        except Exception:
            pass
    try:
        out, _ = p.communicate(timeout=timeout)
        return out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.communicate(timeout=10)
        except Exception:
            pass
        return None
    finally:
        if reg is not None:
            try:
                reg.unregister(rkey)
            except Exception:
                pass


def _expand_cpuset(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _core_pool_cores() -> list[int]:
    """Cores for gem5 sims: PIE_GEM5_CPUSET > PIE_REWARD_CPUSET > affinity.
    Guards against being called from a single-core-pinned thread (which would
    silently serialize the whole batch)."""
    affinity = sorted(os.sched_getaffinity(0))
    spec = os.environ.get("PIE_GEM5_CPUSET") or os.environ.get("PIE_REWARD_CPUSET")
    if spec:
        try:
            cores = _expand_cpuset(spec)
            return cores or affinity
        except ValueError:
            pass
    if len(affinity) <= 1 and (os.cpu_count() or 1) > 1:
        return list(range(os.cpu_count()))
    return affinity


_XPCH_CACHE: dict[str, str | None] = {}
_XPCH_LOCK = threading.Lock()


def _ensure_xpch() -> str | None:
    """Build (once, cached) a precompiled <bits/stdc++.h> for the x86 cross-toolchain,
    with the SAME flags _cross_compile uses, and return the header path. g++ resolves
    an `-include foo.hpp` against `foo.hpp.gch` if present. Gated by PIE_GEM5_XCOMPILE_PCH;
    must be tick-identical to a bare compile (the gem5-timed binary is what this produces),
    so callers fall back to a bare compile if a PCH compile fails (header/symbol clash)."""
    root = _scratch_root()
    with _XPCH_LOCK:
        if root in _XPCH_CACHE:
            return _XPCH_CACHE[root]
        p = None
        try:
            d = os.path.join(root, "pie_xpch")
            os.makedirs(d, exist_ok=True)
            hpp = os.path.join(d, "xcommon.hpp")
            with open(hpp, "w") as f:
                f.write("#include <bits/stdc++.h>\n")
            r = subprocess.run([_GXX] + _LFLAGS + ["-O3", "-std=c++17",
                                "-x", "c++-header", hpp, "-o", hpp + ".gch"],
                               env=_GENV, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=120.0)
            if r.returncode == 0 and os.path.exists(hpp + ".gch"):
                p = hpp
        except Exception:
            p = None
        _XPCH_CACHE[root] = p
        return p


def _cross_compile(code: str, workdir: str, cpu: int | None,
                   _use_pch: bool | None = None) -> tuple[str | None, str | None]:
    """Compile to an x86 binary (the binary gem5 then times). Returns (binary_path, error)."""
    src = os.path.join(workdir, "cand.cpp")
    binp = os.path.join(workdir, "cand")
    with open(src, "w") as f:
        f.write(_PRAGMA_RE.sub("", code))
    if _use_pch is None:
        _use_pch = os.environ.get("PIE_GEM5_XCOMPILE_PCH") == "1"
    xpch = _ensure_xpch() if _use_pch else None
    argv = ["timeout", "-s", "KILL", str(max(1, int(_COMPILE_TIMEOUT) - 5))]
    if cpu is not None:
        argv += ["taskset", "-c", str(cpu)]
    argv += [_GXX] + _LFLAGS + ["-O3", "-std=c++17", "-Wl,--dynamic-linker=" + _LD]
    if xpch:
        argv += ["-include", xpch]
    argv += [src, "-o", binp]
    try:
        r = subprocess.run(argv, env=_GENV, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=_COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "gem5 cross-compile timeout"
    if r.returncode != 0 or not os.path.exists(binp):
        if xpch:   # a forced -include can clash with code redefining stdlib symbols;
            return _cross_compile(code, workdir, cpu, _use_pch=False)   # retry bare
        return None, "gem5 cross-compile failed: " + r.stderr.decode("utf-8", "ignore")[-300:]
    return binp, None


def _probe_startup(binp: str, inp_path: str, workdir: str, cpu: int | None) -> tuple[int | None, str | None]:
    """READ_TICK of the first stdin read (input-invariant). (tick, error)."""
    out = os.path.join(workdir, "probe_m5out")
    os.makedirs(out, exist_ok=True)
    argv = (["taskset", "-c", str(cpu)] if cpu is not None else []) + [
        _GEM5, "-d", out, "--stats-file=s.txt", _PROBE, "Verbatim", binp,
    ]
    env = {**_NATENV, "GEM5_STDIN": inp_path, "GEM5_PREINPUT_FORK": "1"}
    try:
        so = _run_pg(argv, env, _PROBE_TIMEOUT)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    if so is None:
        return None, "gem5 startup probe timeout"
    m = re.search(r"READ_TICK (\d+) cause (.+)", so.decode("utf-8", "ignore"))
    if not m:
        return None, "gem5 startup probe parse failure"
    if m.group(2).strip() != "preinput_read":
        # program never read stdin before exiting: the probe already simulated
        # the FULL run, and this tick is the exit tick (cross-checked against
        # full run-to-completion sims). The caller reuses it — no second sim.
        return int(m.group(1)), "no_stdin"
    return int(m.group(1)), None


def _forkloop_ticks(binp: str, inp_paths: list[str], fork_tick: int,
                    workdir: str, cpu: int | None,
                    max_tc_ticks: int | None = None) -> tuple[list[int] | None, str | None]:
    """Per-tc absolute completion ticks via the sequential forkloop.

    max_tc_ticks: optional ABSOLUTE per-tc tick budget (e.g. 3x the src's
    mean per-tc ticks). A child exceeding it reports cause=budget_exceeded
    and the program is censored as too-slow instead of simulating to
    completion or the wall timeout.
    """
    fix = os.path.join(workdir, "fixed_stdin")
    tcf = os.path.join(workdir, "tc_list")
    shutil.copyfile(inp_paths[0], fix)
    with open(tcf, "w") as f:
        f.write("\n".join(inp_paths))
    out = os.path.join(workdir, "fl_m5out")
    os.makedirs(out, exist_ok=True)
    argv = (["taskset", "-c", str(cpu)] if cpu is not None else []) + [
        _GEM5, "-d", out, "--stats-file=s.txt", _SEQ, "Verbatim", binp,
    ]
    env = {**_NATENV, "GEM5_FIXED_STDIN": fix, "GEM5_TC_LIST": tcf,
           "GEM5_FORK_TICK": str(fork_tick)}
    if max_tc_ticks:
        # forkloop budget is RELATIVE to the fork point; keep a sane minimum
        env["GEM5_TC_MAX_TICKS"] = str(max(int(max_tc_ticks) - fork_tick, 100_000_000))
    try:
        raw = _run_pg(argv, env, _FORKLOOP_TIMEOUT)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    if raw is None:
        return None, "gem5 forkloop timeout"
    so = raw.decode("utf-8", "ignore")
    ms = re.search(r"STARTUP cause=(.+) tick=(\d+)", so)
    if not ms or "limit reached" not in ms.group(1) or int(ms.group(2)) != fork_tick:
        return None, "gem5 forkloop bad startup phase"
    pairs = re.findall(r"TC \d+ tick (\d+) cause (.+)", so)
    over = [c for _, c in pairs if c.strip() == "budget_exceeded"]
    if over:
        return None, "gem5 tc budget exceeded (rollout slower than the budget factor x src)"
    bad = [c for _, c in pairs if not c.strip().startswith("exiting")]
    if bad:
        return None, f"gem5 forkloop abnormal tc exit ({len(bad)}/{len(pairs)})"
    if len(pairs) != len(inp_paths):
        return None, f"gem5 forkloop partial ({len(pairs)}/{len(inp_paths)} tcs)"
    return [int(t) for t, _ in pairs], None


def gem5_time_one(code: str, inputs: list[str], *, cpu: int | None = None,
                  scratch: str | None = None,
                  max_tc_ticks: int | None = None) -> dict:
    """Time one program on a list of test-case input STRINGS.

    Returns {"ticks": int | None, "per_tc": list[int] | None, "error": str | None}.
    ticks = sum of absolute per-tc completion ticks (startup included), the
    same quantity as the dataset's agg_runtime.
    max_tc_ticks: optional absolute per-tc tick budget (censors slow programs
    early; see _forkloop_ticks).
    """
    if not inputs:
        return {"ticks": None, "per_tc": None, "error": "no test inputs"}
    root = scratch or _scratch_root()
    workdir = tempfile.mkdtemp(dir=root, prefix="g5_")
    try:
        binp, err = _cross_compile(code, workdir, cpu)
        if binp is None:
            return {"ticks": None, "per_tc": None, "error": err}
        inp_paths = []
        for i, s in enumerate(inputs):
            p = os.path.join(workdir, f"input.{i}.txt")
            with open(p, "w") as f:
                f.write(s)
            inp_paths.append(p)
        startup, err = _probe_startup(binp, inp_paths[0], workdir, cpu)
        if startup is None:
            return {"ticks": None, "per_tc": None, "error": err}
        if err == "no_stdin":
            # input-invariant program: the probe's tick IS the full-run exit
            # tick (the hook never fired), and every tc costs the same
            return {"ticks": startup * len(inputs),
                    "per_tc": [startup] * len(inputs), "error": None}
        if startup <= _FORK_MARGIN + 1000:
            return {"ticks": None, "per_tc": None,
                    "error": "startup tick below fork margin"}
        per_tc, err = _forkloop_ticks(binp, inp_paths, startup - _FORK_MARGIN,
                                      workdir, cpu, max_tc_ticks=max_tc_ticks)
        if per_tc is None:
            return {"ticks": None, "per_tc": None, "error": err}
        return {"ticks": sum(per_tc), "per_tc": per_tc, "error": None}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def gem5_time_batch(tasks: dict[str, dict], *, num_workers: int | None = None,
                    batch_timeout: float | None = None) -> dict[str, dict]:
    """Time a batch of unique programs in parallel.

    tasks: {key: {"code": str, "inputs": [input_str, ...]}}
    Returns {key: result-of-gem5_time_one}; every input key is present.

    One core per in-flight program (compile + 2 sequential gem5 processes are
    all single-threaded). Cores come from PIE_GEM5_CPUSET / PIE_REWARD_CPUSET
    (tail-first, away from where Ray/vLLM workers land) or the affinity mask.
    Worker default is conservative (half the cores, max 64) — each gem5 is
    ~0.5-1.5 GB RSS and this runs on the training node.

    batch_timeout (default PIE_GEM5_BATCH_TIMEOUT=1200s) is a wall cap on the
    whole call: unfinished programs return {"ticks": None, "error": "gem5
    batch timeout"} instead of stalling the RL step. Their in-flight gem5
    subprocesses still die via their own per-phase process-group timeouts.
    """
    if not tasks:
        return {}
    cores = _core_pool_cores()
    env_workers = os.environ.get("PIE_GEM5_WORKERS")
    default_n = max(1, min(len(cores) // 2 or 1, 64))
    n = max(1, min(num_workers or (int(env_workers) if env_workers else default_n),
                   len(cores), len(tasks)))
    pool: queue.Queue = queue.Queue()
    for c in cores[-n:]:                 # tail cores: first cores host Ray/vLLM
        pool.put(c)
    root = _scratch_root()
    if batch_timeout is None:
        batch_timeout = _BATCH_TIMEOUT

    def _one(item: tuple[str, dict]) -> tuple[str, dict]:
        key, t = item
        cpu = pool.get()
        try:
            return key, gem5_time_one(t["code"], t["inputs"], cpu=cpu, scratch=root,
                                      max_tc_ticks=t.get("max_tc_ticks"))
        except Exception as e:  # never poison the batch
            return key, {"ticks": None, "per_tc": None,
                         "error": f"gem5 backend error: {type(e).__name__}: {e}"}
        finally:
            pool.put(cpu)

    out: dict[str, dict] = {}
    ex = ThreadPoolExecutor(max_workers=n)
    try:
        futs = {ex.submit(_one, item): item[0] for item in tasks.items()}
        done, not_done = cf_wait(futs, timeout=batch_timeout)
        for f in done:
            key, res = f.result()
            out[key] = res
        for f in not_done:
            f.cancel()
            out[futs[f]] = {"ticks": None, "per_tc": None,
                            "error": "gem5 batch timeout"}
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return out
