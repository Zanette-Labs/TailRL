"""
C++ measurement core.

Compiles C++ programs with g++ (resolved to an absolute path at import time),
gates them against expected outputs (strict all-pass), and on all-pass
records a "cost" — currently wall-clock nanoseconds summed across the gate
runs (MEASUREMENT_BACKEND="walltime"). Retired-instruction counting via
`perf stat -e instructions` was the original design, but hardware perf-event
counters serialise at the kernel level under concurrency, capping measurement
throughput at roughly 3x regardless of worker count. A percentile-rank style
cost is robust to wall-clock noise as long as baseline and candidate are
measured identically, so the throughput cost is what decided the trade-off in
favour of walltime.

The dict field is named `elapsed_ns` and carries total wall-clock ns
summed across the gate runs.

Public entrypoints:

    measure(code_str, test_cases) -> dict
    measure_from_disk(code_str, test_case_dir, problem_id) -> dict
    setup_pch(scratch_dir) -> str       # one-shot, run once at job start

The implementation here is deliberately minimal: it covers what the test
suite exercises, and extensions land only when a new failing test demands
them.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import struct
import subprocess
import sys
import tempfile
import time

# The host C++ compiler used for the native CORRECTNESS gate (output comparison)
# ONLY. Program *speed* is measured by gem5 with its own pinned g++-9 toolchain
# (see measurement/gem5_backend.py), so this compiler's version does not affect any
# reward value -- only whether a rollout compiles and produces right answers.
# -O0 because the gate is compile-bound and the pass/fail verdict is
# optimization-level-invariant for correct programs.
#
# Resolved at import so a misconfigured host fails loudly. Without this, every
# rollout would fail to compile and the run would report a plausible-looking wall
# of zero rewards with no error anywhere.
def _resolve(env_var: str, exe: str) -> str:
    import shutil
    p = os.environ.get(env_var) or shutil.which(exe)
    if not p:
        raise RuntimeError(
            f"no {exe} on PATH. Set {env_var} to the compiler used for the native "
            f"correctness gate, or install one. (This is NOT the gem5 timing "
            f"toolchain -- that is PIE_GEM5_GXX.)"
        )
    return p


GPP_PATH = _resolve("PIE_GXX", "g++")
GCC_PATH = _resolve("PIE_GCC", "gcc")
COMPILE_FLAGS = ["-O0", "-std=c++17"]

# Per-record compile scratch. Leaving `tempfile.TemporaryDirectory(dir=None)` to
# follow $TMPDIR puts compile scratch on node-local disk under most schedulers,
# which matters: compiling thousands of rollouts per step against a shared network
# filesystem serialises on metadata locks.
# In an interactive shell without a scheduler, $TMPDIR can fall back to a `/tmp`
# that is itself on a network filesystem, and compile cost rises noticeably.
#
# `PIE_COMPILE_SCRATCH` is the explicit override (preferred name; takes
# precedence over $TMPDIR). `PIE_SCRATCH_DIR` is kept as a backward-compatible
# alias of the same thing.
def _scratch_dir() -> str | None:
    return (
        os.environ.get("PIE_COMPILE_SCRATCH")
        or os.environ.get("PIE_SCRATCH_DIR")
        or None
    )

# Backend selection at module init. Reads `PIE_BACKEND` at module load:
# "walltime" (default) or "perf" (hardware retired-instruction counter via
# perf_event_open). The perf backend needs perf_event_paranoid <= 2 so an
# unprivileged process can open hardware counters on its own children; when
# that is unavailable the first failed use falls back to walltime, once per
# process (see _run_and_compare).
#
# History, because it explains why two backends exist: the original perf
# implementation shelled out to `perf stat`, which capped measurement
# throughput at roughly 3x under concurrency because the kernel serialises
# event-fd setup, so the code moved to wall-clock. Wall-clock noise then proved
# irreducible under concurrency on ARM server parts. The current perf backend
# calls `perf_event_open` directly (raw ctypes syscall) rather than going
# through the perf-stat wrapper: per-PID counters opened this way do not share
# the kernel path responsible for the 3x cap.
#
# The dict field `elapsed_ns` keeps its name across backends to avoid a
# repo-wide rename; under "perf" it carries instruction counts in the same
# slot. Consumers must read `measurement_backend` to interpret the unit.
MEASUREMENT_BACKEND = os.environ.get("PIE_BACKEND", "walltime")

# perf stat writes to stderr; the instruction-count line looks like
#     "     281,447      instructions:u  ..."
_INSTR_RE = re.compile(r"([\d,]+)\s+instructions:u")


def setup_pch(scratch_dir: str | None = None,
              header_text: str = "#include <bits/stdc++.h>\n") -> str:
    """Build a precompiled header for the standard C++ stdlib once, and return
    the path to the header source. Passing the returned path back as
    `pch_include_path` to `measure()` cuts per-compile time substantially
    (~2.65x on typical competitive-programming sources, empirically) and —
    more importantly — raises the parallel-compile speedup ceiling from ~4x to
    ~15x on a many-core node, because each compile no longer re-parses the
    full stdlib.

    g++ resolves an `-include foo.hpp` flag against `foo.hpp.gch` if present.

    The default header pulls in <bits/stdc++.h>, the GNU "everything stdlib"
    header used in competitive-programming code (which PIE programs are).
    Callers wanting tighter coverage can pass their own `header_text`.

    Caller must invoke this once before training/probe loops. The build is
    O(~2s) and is wasted if `measure()` is then called without
    `pch_include_path`.
    """
    scratch_dir = scratch_dir or _scratch_dir() or tempfile.gettempdir()
    pch_dir = os.path.join(scratch_dir, "pie_pch")
    os.makedirs(pch_dir, exist_ok=True)
    hpp_path = os.path.join(pch_dir, "common.hpp")
    with open(hpp_path, "w") as f:
        f.write(header_text)
    proc = subprocess.run(
        [GPP_PATH, *COMPILE_FLAGS, "-x", "c++-header", hpp_path, "-o", hpp_path + ".gch"],
        capture_output=True, text=True, errors="replace", timeout=60.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"PCH build failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return hpp_path


def _compile(code_str: str, tmpdir: str, pch_include_path: str | None = None
             ) -> tuple[str | None, str | None]:
    """Compile C++ source. Returns (executable_path, error_str). On failure exe is None.

    Failure modes — all return (None, error_str) instead of raising, so the
    caller (grade_dedup_batch.pool.map) doesn't surface an exception that
    would tear down the entire reward step:
      - rc != 0          → syntax/link error from g++
      - TimeoutExpired   → g++ hung past 30 s (some model-generated code with
                           huge template instantiations or pathological
                           constexpr does this; an uncaught 30 s g++ timeout
                           previously took down a whole training run)
      - OSError          → /tmp full, exe_path missing, etc.
    """
    src_path = os.path.join(tmpdir, "prog.cpp")
    exe_path = os.path.join(tmpdir, "prog")
    with open(src_path, "w") as f:
        f.write(code_str)
    args = [GPP_PATH, *COMPILE_FLAGS]
    if pch_include_path:
        args += ["-include", pch_include_path]
    args += ["-o", exe_path, src_path]
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            errors="replace",  # g++ echoes user-source bytes back in error messages
            # (e.g. "stray '\xXX' in program"); without errors='replace' a
            # non-UTF-8 byte in a model-generated C++ string literal would
            # crash the entire reward step with UnicodeDecodeError.
            timeout=30.0,
        )
    except subprocess.TimeoutExpired:
        return None, "compile timed out (30s)"
    except OSError as e:
        return None, f"compile OS error: {e}"
    if proc.returncode != 0:
        # Cap stderr: g++ on pathological model-generated C++ (huge template/constexpr
        # instantiation, "stray byte" errors that echo source per line) can emit tens of
        # MB of diagnostics. An uncapped string here becomes 'reason' downstream, which
        # numpy then broadcasts into a fixed-width unicode array on the controller and
        # OOMs the driver (see ray_trainer.py reward_extra_infos fix). Keep the first 2000
        # chars (the root error; later lines are cascading noise) plus a truncation marker.
        err = proc.stderr.strip()
        if len(err) > 2000:
            err = err[:2000] + f"  [+{len(err) - 2000} chars truncated]"
        return None, f"compile failed (rc={proc.returncode}): {err}"
    return exe_path, None


# Per-program address-space cap. Default 1 GiB; override via PIE_PROGRAM_MEM_LIMIT_GB.
# Model-generated programs occasionally allocate tens of GB, and at 64-worker
# concurrency that runs the node out of memory and kills the Ray worker.
#
# Originally implemented via `preexec_fn=setrlimit` on subprocess.run, but
# preexec_fn is documented UNSAFE with threads: it can deadlock between fork
# and exec when the calling Python is multi-threaded, and the reward worker
# runs at 64 threads. In practice this showed up as 100% gate failures, because
# the subprocess.run paths deadlocked silently.
#
# Use `prlimit -v <bytes> --` as an exec-side wrapper instead. prlimit is part
# of util-linux (always available on Linux servers), sets the limit on the
# spawned child only, and crucially does NOT require preexec_fn — so it's safe
# with threading.
_PRLIMIT_PATH = "/usr/bin/prlimit"
_MEM_LIMIT_BYTES = int(os.environ.get("PIE_PROGRAM_MEM_LIMIT_GB", "1")) * (1 << 30)


def _limited_argv(argv: list[str]) -> list[str]:
    """Prepend prlimit to argv so the spawned child has RLIMIT_AS = _MEM_LIMIT_BYTES."""
    return [_PRLIMIT_PATH, f"--as={_MEM_LIMIT_BYTES}", "--"] + argv


# ---------------------------------------------------------------------------
# perf_event_open backend (raw ctypes, no Cirron dependency).
#
# Why raw ctypes and not Cirron / py-spy / pyperf: this path adds no new
# dependencies and no process spawns beyond the candidate itself. The
# `perf stat` subprocess wrapper used by the earlier backend hit a 3x
# throughput cap under concurrency (kernel-level event-fd serialisation).
# perf_event_open invoked per-grade directly avoids that path entirely because
# each fd is independent and per-PID.
#
# Race window: between Popen returning (child PID exists, kernel has started
# the child) and the parent calling perf_event_open(pid=child_pid), the
# child may execute ld.so resolution + initial libc setup. Those instructions
# are NOT counted. That is a small constant offset; the determinism check
# (target rel_spread <= 1e-4 across repeated runs of the same program) is what
# confirms the offset is small enough to ignore. The common-case program also
# blocks on stdin within the first few hundred instructions (e.g. `cin >> n;`),
# so the window of uncounted work is small.
#
# perf_event_attr fields:
#   type   = PERF_TYPE_HARDWARE
#   config = PERF_COUNT_HW_INSTRUCTIONS
#   exclude_kernel = 1   (only user-space instructions)
#   inherit        = 0   (do NOT propagate to fork()'d children; matches
#                          EvalPerf convention. PIE candidates don't fork.)
#   disabled       = 0   (start counting as soon as fd opens)
# ---------------------------------------------------------------------------

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)

# perf_event_open syscall number: 241 on aarch64
# (arch/arm64/include/asm/unistd.h), 298 on x86_64. Selected from
# os.uname().machine because there is no libc wrapper for this syscall.
_SYS_perf_event_open = 241 if (os.uname().machine == "aarch64") else 298

# SIGSTOP-then-exec wrapper. Eliminates the perf-attach race window:
# the wrapper raises SIGSTOP before exec'ing the candidate program, so
# the parent has time to call perf_event_open on a known-stopped child
# before any candidate-program instructions run. After parent SIGCONTs,
# the wrapper execvp's the real binary; perf survives execve and counts
# from instruction 0.
_PERF_WRAPPER_PATH: str | None = None  # set lazily by _ensure_perf_wrapper()


def _ensure_perf_wrapper() -> str | None:
    """Compile the SIGSTOP wrapper once and return the path. None on failure
    (caller should fall back to walltime)."""
    global _PERF_WRAPPER_PATH
    if _PERF_WRAPPER_PATH is not None and os.path.isfile(_PERF_WRAPPER_PATH):
        return _PERF_WRAPPER_PATH
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "pie_perf_wrapper.c")
    if not os.path.isfile(src):
        return None
    # Cache the wrapper next to the scratch dir so it survives across worker
    # processes on the same node and across job restarts.
    scratch = _scratch_dir() or tempfile.gettempdir()
    out_dir = os.path.join(scratch, "pie_perf_wrapper")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "pie_perf_wrapper")
    try:
        proc = subprocess.run(
            [GCC_PATH, "-O2", "-o", out, src],
            capture_output=True, text=True, timeout=30.0,
        )
        if proc.returncode != 0:
            return None
    except (OSError, subprocess.TimeoutExpired):
        return None
    _PERF_WRAPPER_PATH = out
    return out

_PERF_TYPE_HARDWARE = 0
_PERF_COUNT_HW_INSTRUCTIONS = 1
_PERF_ATTR_SIZE_VER0 = 64  # baseline ABI — sufficient for type/config/flags

# read_format bits: ask the kernel to include enabled/running times in the
# read() output. Under heavy concurrency the kernel multiplexes events
# across CPUs and only counts during `running` ticks within a longer
# `enabled` window; without scaling the counts come back artificially small,
# and in practice nearly identical across workers because every one of them
# reports the same truncated prefix. Scaled count = raw * enabled / running.
_PERF_FORMAT_TOTAL_TIME_ENABLED = 1
_PERF_FORMAT_TOTAL_TIME_RUNNING = 2
_PERF_READ_FORMAT_BITS = (
    _PERF_FORMAT_TOTAL_TIME_ENABLED | _PERF_FORMAT_TOTAL_TIME_RUNNING
)
_PERF_READ_BYTES = 3 * 8  # count, time_enabled, time_running

# Bit positions in the flags u64 at offset 40 of perf_event_attr.
_FLAG_BIT_DISABLED = 0
_FLAG_BIT_INHERIT = 1
_FLAG_BIT_EXCLUDE_KERNEL = 5

# perf_event_open(2) on aarch64 takes args as long, but `syscall()` is
# variadic; we pass via c_long.
_syscall = _libc.syscall
_syscall.restype = ctypes.c_int


def _build_perf_attr_instructions() -> bytes:
    """Pack a minimal perf_event_attr for retired user-instructions counting."""
    attr = bytearray(_PERF_ATTR_SIZE_VER0)
    struct.pack_into("<I", attr, 0, _PERF_TYPE_HARDWARE)             # type
    struct.pack_into("<I", attr, 4, _PERF_ATTR_SIZE_VER0)            # size
    struct.pack_into("<Q", attr, 8, _PERF_COUNT_HW_INSTRUCTIONS)     # config
    # offset 16 (sample_period) stays 0
    # offset 24 (sample_type) stays 0
    struct.pack_into("<Q", attr, 32, _PERF_READ_FORMAT_BITS)         # read_format
    flags = (
        (0 << _FLAG_BIT_DISABLED)
        | (0 << _FLAG_BIT_INHERIT)
        | (1 << _FLAG_BIT_EXCLUDE_KERNEL)
    )
    struct.pack_into("<Q", attr, 40, flags)
    return bytes(attr)


_PERF_ATTR_BLOB = _build_perf_attr_instructions()


def _perf_event_open_pid(pid: int) -> int:
    """Open a perf event fd that counts retired user-space instructions on
    the given PID across all CPUs. Returns the fd; caller must os.close()."""
    attr_buf = ctypes.create_string_buffer(_PERF_ATTR_BLOB, len(_PERF_ATTR_BLOB))
    fd = _syscall(
        ctypes.c_long(_SYS_perf_event_open),
        ctypes.cast(attr_buf, ctypes.c_void_p),
        ctypes.c_long(pid),
        ctypes.c_long(-1),     # cpu = -1 (any)
        ctypes.c_long(-1),     # group_fd = -1
        ctypes.c_ulong(0),     # flags
    )
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"perf_event_open(pid={pid}): {os.strerror(err)}")
    return fd


def _perf_read_count(fd: int) -> int:
    """Read the multiplex-scaled instruction count from a perf event fd.

    With read_format = TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING the kernel
    returns 3 little-endian u64s: (raw_count, time_enabled, time_running).
    Scale: count = raw_count * (time_enabled / time_running). If running == 0
    the event never ran (likely failed silently); return 0.
    """
    buf = os.read(fd, _PERF_READ_BYTES)
    if len(buf) != _PERF_READ_BYTES:
        raise OSError(f"perf read returned {len(buf)} bytes "
                      f"(expected {_PERF_READ_BYTES})")
    raw_count, time_enabled, time_running = struct.unpack("<QQQ", buf)
    if time_running == 0:
        return 0
    if time_enabled == time_running:
        return raw_count
    # Multiplexed — apply the standard kernel-side rescaling factor.
    return int(raw_count * time_enabled / time_running)


def _run_and_compare_perf(
    exe_path: str, input_str: str, expected: str, timeout_s: float
) -> tuple[bool, int]:
    """Run program with stdin=input_str, return (passed, instruction_count).

    Same failure semantics as the walltime path: any error → (False, 0)
    so the gate loop can keep going.

    Uses the SIGSTOP wrapper to eliminate the perf-attach race window —
    the wrapper raises SIGSTOP before exec'ing the target, parent waits
    for the stop via waitpid(WUNTRACED), opens perf, then SIGCONTs. After
    SIGCONT the wrapper execvp's the real binary; perf survives execve
    and counts every instruction from main() onward.
    """
    import signal

    wrapper = _ensure_perf_wrapper()
    if wrapper is None:
        raise RuntimeError("perf wrapper not built; falling back to walltime")

    proc = subprocess.Popen(
        _limited_argv([wrapper, exe_path]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(exe_path),
    )
    fd = -1
    instr_count = 0
    try:
        # Wait for the wrapper to reach raise(SIGSTOP). Use os.waitpid with
        # WUNTRACED so we return when the child is stopped (not exited).
        try:
            wpid, status = os.waitpid(proc.pid, os.WUNTRACED)
        except OSError as e:
            return False, 0
        if not os.WIFSTOPPED(status):
            # Wrapper exited before stopping (probably argv error / exec
            # failed before raise()). Treat as gate fail.
            proc.returncode = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
            return False, 0

        try:
            fd = _perf_event_open_pid(proc.pid)
        except OSError as e:
            # Resume the child so it can exit, then return failure.
            try:
                os.kill(proc.pid, signal.SIGCONT)
            except OSError:
                pass
            try:
                proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            raise RuntimeError(f"perf_event_open failed: {e}") from e

        # Release the child to run the candidate program.
        try:
            os.kill(proc.pid, signal.SIGCONT)
        except OSError:
            return False, 0

        try:
            stdout_bytes, _stderr = proc.communicate(
                input=input_str.encode(errors="replace"),
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
            return False, 0
        except OSError:
            return False, 0

        try:
            instr_count = _perf_read_count(fd)
        except OSError:
            instr_count = 0

        if proc.returncode != 0:
            return False, instr_count

        actual = stdout_bytes.decode(errors="replace")
        return _outputs_match(actual, expected), instr_count
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if proc.poll() is None:
            try:
                # Make sure the child isn't still stopped (would leak).
                os.kill(proc.pid, signal.SIGCONT)
            except OSError:
                pass
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass


_PERF_FALLBACK_WARNED = False  # rate-limit fallback noise to once per process


def _run_and_compare(
    exe_path: str, input_str: str, expected: str, timeout_s: float
) -> tuple[bool, int]:
    """Run program with stdin=input_str, return (passed, cost).

    The `cost` slot carries either wall-clock nanoseconds (default backend)
    or retired user-space instructions (PIE_BACKEND=perf). Field name
    `elapsed_ns` is retained at the dict level for backward compatibility;
    consumers MUST consult `measurement_backend` to interpret the unit.

    elapsed_ns is the wall-clock time spent in `subprocess.run`. It is returned
    even when `passed` is False so callers can compute a fail-safe metric;
    `measure()` only uses it on the all-pass path.

    Failures (nonzero exit, timeout, OS error, undecodable output) all return
    (False, _) — never raise — so the strict-gate loop can keep going. A
    program that exceeds the RLIMIT_AS cap (default 1 GiB; tune via
    PIE_PROGRAM_MEM_LIMIT_GB) gets SIGKILL'd by the kernel and shows up here
    as a non-zero returncode — the gate rejects it normally, no special path.

    `errors='replace'` is load-bearing: some training programs print
    uninitialized memory (e.g., a stray 0xff byte) to stdout. Without it,
    subprocess.run's text decoder raises UnicodeDecodeError and aborts the
    whole batch. With `replace`, the bad bytes become U+FFFD and the gate
    rejects the program for output mismatch — the correct outcome.

    `cwd=os.path.dirname(exe_path)` is load-bearing: model-generated C++
    sometimes uses competitive-programming file I/O patterns
    (`freopen("foo.out", "w", stdout)`, hardcoded `ofstream("bar.txt")`).
    Without an explicit cwd, the binary inherits the parent's cwd (the
    launcher's working directory, usually the repository root) and litters
    it with files like `Beacon.out`. Worse, two parallel rollouts can race
    on the same filename and corrupt each other's output. Pinning cwd
    to the binary's own tmpdir (which `measure()` cleans up after) keeps
    those side-effects sandboxed.
    """
    if MEASUREMENT_BACKEND == "perf":
        try:
            return _run_and_compare_perf(exe_path, input_str, expected, timeout_s)
        except (OSError, RuntimeError) as e:
            global _PERF_FALLBACK_WARNED
            if not _PERF_FALLBACK_WARNED:
                print(f"[perf-fallback] {e}; reverting to walltime",
                      file=sys.stderr, flush=True)
                _PERF_FALLBACK_WARNED = True
            # fall through to walltime
    t0 = time.monotonic_ns()
    try:
        proc = subprocess.run(
            _limited_argv([exe_path]),
            input=input_str,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            cwd=os.path.dirname(exe_path),
        )
    except (subprocess.TimeoutExpired, OSError):
        return False, time.monotonic_ns() - t0
    elapsed_ns = time.monotonic_ns() - t0
    if proc.returncode != 0:
        return False, elapsed_ns
    return _outputs_match(proc.stdout, expected), elapsed_ns


def _outputs_match(output: str, expected: str) -> bool:
    """PIE's official output comparator, mirrored verbatim from the paper
    repo (LearningOpt/pie, gem5/benchmarking.py::get_accuracy): line-wise
    exact match with an ABSOLUTE 1e-3 float-tolerance fallback per line;
    a program passes iff every expected line matches. Two fidelity quirks
    kept on purpose: (a) extra OUTPUT lines beyond the expected count are
    ignored (their zip()), (b) the float fallback fires only when the whole
    line parses as one float. Without the tolerance, float-output problems
    (e.g. p02882: '75.9637565321' vs '75.963756532074') fail every case on
    formatting alone — the paper's human-reference %Correct=100% requires it."""
    exp_lines = expected.strip().splitlines()
    out_lines = output.strip().splitlines()
    if not exp_lines:
        return not out_lines
    n_correct = 0
    for gen, gt in zip(out_lines, exp_lines):
        ok = gen == gt
        if not ok:
            try:
                ok = abs(float(gen) - float(gt)) < 1e-3
            except (ValueError, OverflowError):
                pass
        n_correct += int(ok)
    return n_correct == len(exp_lines)


def _perf_count_instructions(exe_path: str, input_str: str) -> int | None:
    """Run exe under `perf stat -e instructions:u` and parse the retired-instruction count.

    Returns None if perf fails or its output can't be parsed.
    """
    proc = subprocess.run(
        ["perf", "stat", "-e", "instructions:u", "--", exe_path],
        input=input_str,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10.0,
    )
    if proc.returncode != 0:
        return None
    match = _INSTR_RE.search(proc.stderr)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _list_test_indices(test_case_dir: str, problem_id: str) -> list[int]:
    """Return the sorted list of test indices i for which both input.<i>.txt
    and output.<i>.txt exist in <test_case_dir>/<problem_id>/."""
    pdir = os.path.join(test_case_dir, problem_id)
    if not os.path.isdir(pdir):
        return []
    indices: list[int] = []
    for fn in os.listdir(pdir):
        if not (fn.startswith("input.") and fn.endswith(".txt")):
            continue
        try:
            i = int(fn[len("input."):-len(".txt")])
        except ValueError:
            continue
        if os.path.exists(os.path.join(pdir, f"output.{i}.txt")):
            indices.append(i)
    indices.sort()
    return indices


def measure_from_disk(
    code_str: str,
    *,
    test_case_dir: str,
    problem_id: str,
    per_test_timeout_s: float = 30.0,
    n_perf_samples: int = 3,
    skip_measurement: bool = False,
    pch_include_path: str | None = None,
) -> dict:
    """Compile + gate + (optional) timing, reading test cases from disk.

    Difference from `measure()`: test cases are discovered from the filesystem
    at `<test_case_dir>/<problem_id>/input.<i>.txt` + `output.<i>.txt` rather
    than passed in memory. The gate runs the in-Python `_run_and_compare` loop
    rather than batching test cases through a shell helper: under the walltime
    backend the gate IS the measurement, so per-call Python timing is needed.

    Returns the same dict shape as `measure()`. Same args otherwise.
    """
    if pch_include_path is None:
        pch_include_path = os.environ.get("PIE_PCH_INCLUDE_PATH") or None

    result: dict = {
        "compiled": False,
        "all_pass": False,
        "n_tests_run": 0,
        "n_tests_passed": 0,
        "elapsed_ns": None,
        "measurement_backend": MEASUREMENT_BACKEND,
        "error": None,
    }

    indices = _list_test_indices(test_case_dir, problem_id)
    result["n_tests_run"] = len(indices)
    if not indices:
        result["error"] = f"no test cases at {test_case_dir}/{problem_id}/"
        return result

    with tempfile.TemporaryDirectory(dir=_scratch_dir(), prefix="pie_meas_") as tmpdir:
        exe_path, err = _compile(code_str, tmpdir, pch_include_path=pch_include_path)
        if exe_path is None:
            result["error"] = err
            return result
        result["compiled"] = True

        # In-Python gate + timing (one subprocess per case). Under the walltime
        # backend the gate IS the measurement, so per-call timing is required.
        pdir = os.path.join(test_case_dir, problem_id)
        n_pass = 0
        total_elapsed_ns = 0
        for i in indices:
            with open(os.path.join(pdir, f"input.{i}.txt")) as f_in:
                input_str = f_in.read()
            with open(os.path.join(pdir, f"output.{i}.txt")) as f_out:
                expected = f_out.read()
            passed, elapsed_ns = _run_and_compare(exe_path, input_str, expected, per_test_timeout_s)
            if passed:
                n_pass += 1
                total_elapsed_ns += elapsed_ns
        result["n_tests_passed"] = n_pass

        all_pass = (len(indices) > 0) and (n_pass == len(indices))
        if not all_pass:
            return result
        result["all_pass"] = True
        if skip_measurement:
            return result
        result["elapsed_ns"] = total_elapsed_ns

    return result


def measure(
    code_str: str,
    test_cases: list[tuple[str, str]],
    *,
    per_test_timeout_s: float = 30.0,
    n_perf_samples: int = 3,
    skip_measurement: bool = False,
    pch_include_path: str | None = None,
) -> dict:
    """Compile, gate (strict all-pass), then measure retired instructions.

    Args:
        code_str: full C++ source.
        test_cases: list of (stdin_input, expected_stdout) pairs. Each pair is
            graded independently; the gate is satisfied only if EVERY pair passes.
        per_test_timeout_s: per-test execution wall-clock cap. Default 30s
            after the -O0 switch (-O0 makes programs 10-50× slower; the prior
            10s default was a 100% timeout on long-running tgt programs).
        n_perf_samples: how many perf runs per test case for the median. The
            default is median-of-3; 1 is a valid 3x saving on the perf phase,
            because repeated runs of the same program show rel_spread ~ 0 and a
            median brings no statistical benefit when the underlying variance is
            essentially zero.
        skip_measurement: when True, skip the perf phase entirely (compile + gate
            only). Useful for measuring the parallel speedup of the gate-only
            workload — the perf phase has its own kernel-level serialisation
            ceiling (~3.8x) while compile+gate scales much further. Returns the
            same dict shape; elapsed_ns is None.

    Returns:
        {
            "compiled": bool,
            "all_pass": bool,
            "n_tests_run": int,
            "n_tests_passed": int,
            "elapsed_ns": int | None,    # cost; None unless all_pass and not skip_measurement.
                                          # Under "walltime" backend (current), this is total
                                          # wall-clock ns summed across the gate runs.
            "measurement_backend": str,    # "perf" | "walltime" (default)
            "error": str | None,
        }
    """
    result: dict = {
        "compiled": False,
        "all_pass": False,
        "n_tests_run": 0,
        "n_tests_passed": 0,
        "elapsed_ns": None,
        "measurement_backend": MEASUREMENT_BACKEND,
        "error": None,
    }

    # Allow env-var fallback so SLURM scripts can supply PCH without threading
    # the param through every caller (notably insitu_eval, which calls measure()
    # via the redirect shim).
    if pch_include_path is None:
        pch_include_path = os.environ.get("PIE_PCH_INCLUDE_PATH") or None

    with tempfile.TemporaryDirectory(dir=_scratch_dir(), prefix="pie_meas_") as tmpdir:
        exe_path, err = _compile(code_str, tmpdir, pch_include_path=pch_include_path)
        if exe_path is None:
            result["error"] = err
            return result
        result["compiled"] = True

        # Walltime gate: time each subprocess.run, accumulate on all-pass.
        # There is no separate perf phase, because perf-event kernel contention
        # is prohibitive under concurrency. n_perf_samples is kept for API
        # compatibility but is a no-op under the walltime backend, since the
        # gate already runs each test exactly once.
        n_pass = 0
        total_elapsed_ns = 0
        for input_str, expected in test_cases:
            passed, elapsed_ns = _run_and_compare(
                exe_path, input_str, expected, per_test_timeout_s
            )
            if passed:
                n_pass += 1
                total_elapsed_ns += elapsed_ns
        result["n_tests_run"] = len(test_cases)
        result["n_tests_passed"] = n_pass

        all_pass = (len(test_cases) > 0) and (n_pass == len(test_cases))
        if not all_pass:
            return result
        result["all_pass"] = True
        if skip_measurement:
            return result
        result["elapsed_ns"] = total_elapsed_ns

    return result
