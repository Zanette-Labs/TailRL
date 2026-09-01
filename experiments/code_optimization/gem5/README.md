# The gem5 timing stack

The reward in this experiment is a *measured* speedup, not a proxy. Every candidate
program is compiled, checked for correctness against real test cases, and then
simulated cycle-by-cycle on a modelled Skylake core. The reward is the ratio of
simulated cycles.

`scripts/setup_gem5.sh` builds all of this. This file explains what it builds and
why each piece is pinned, which you need if the script fails or if you are adapting
it to an unusual host.

---

## Why a simulator at all

Wall-clock timing on a shared machine is not a usable reward signal. Two runs of the
same binary differ by more than the effect being optimized, and the noise is
correlated with whatever else is on the node — so a policy can be rewarded for other
people's jobs finishing. Retired-instruction counts are stable but measure the wrong
thing: they are close to uncorrelated with actual runtime on this task, because the
dominant costs are allocation, memory layout and I/O rather than instruction count.

gem5 in syscall-emulation mode gives a *deterministic* cycle count. The same program
on the same input always yields the same number of ticks, on any host, so the reward
is reproducible and comparable across machines and across arms. That determinism is
what makes group-relative advantages meaningful here: the differences between a
group's rewards are signal, not measurement noise.

The cost is speed — simulation is thousands of times slower than native execution —
and most of the engineering below exists to make it affordable.

## What gets built

Everything lands under `$PIE_GEM5_HOME`, by default `<experiment>/gem5/build`:

| path | what | size |
| --- | --- | --- |
| `gem5/build/X86/gem5.fast` | the simulator | ~20 MB binary, ~7 GB build tree |
| `pyenv/` | python 3.8 + scons 3.x | 132 MB |
| `rootfs/` | x86_64 Ubuntu 20.04 sysroot: glibc 2.31, libstdc++, g++-9 target libs | ~1.5 GB |
| `x86cross/root/` | the compiler that builds candidate programs | ~120 MB |

Each of these is a separate environment variable (`PIE_GEM5_BIN`, `PIE_GEM5_PYENV`,
`PIE_GEM5_ROOTFS`, `PIE_GEM5_XCROSS_ROOT`), so if you already have any of them you
can point at it instead of rebuilding. `code_opt/measurement/gem5_backend.py` is
where the defaults are resolved, and `scripts/verify_setup.sh` prints what they
resolved to.

## The pins, and why they are not negotiable

**gem5 v20.1.0.2** (commit `0d703041fcd5d119012b62287695723a2955b408`). The PIE
dataset's reference runtimes were produced with this version. The reward is a ratio
of *your* measured ticks to the dataset's *stored* ticks for the source program, so a
different microarchitectural model puts numerator and denominator on different
scales and every speedup in the dataset becomes wrong. Newer gem5 also does not build
here — the v25 X86 target fails under scons 4.

**Python 3.8 and SCons < 4**, for the build only. gem5 v20.1 bundles pybind11 2.4.1,
which calls `PyThreadState_DeleteCurrent()`, a private CPython symbol removed in 3.9;
and its `SConstruct` reaches into `SCons.Defaults.LinkAction` and `SCons.Node.FS`,
which SCons 4 rearranged. This environment is fully isolated from the trainer's: no
code here ever does `import m5` in-process, gem5 is only ever spawned as a
subprocess, so a 3.8 build environment coexists with a 3.10+ training environment
without either knowing about the other. The environment must survive the build,
though — the binary links `libpython3.8`.

**g++-9, `-O3`, dynamically linked.** This is what PIE compiled with. A different gcc
emits different valid x86 for the same source, which moves measured times off the
dataset's by more than the effects being measured. Dynamic linking (rather than
static) is why the sysroot exists at all: the simulated program resolves `ld-2.31.so`
and `libstdc++` through gem5's guest-to-host path redirects at run time.

**Ubuntu 20.04 sysroot.** That is the distribution that ships glibc 2.31 and g++-9
together. Any tree with `lib/x86_64-linux-gnu/ld-2.31.so`,
`usr/lib/x86_64-linux-gnu`, `usr/lib/gcc/x86_64-linux-gnu/9` and `usr/include` will
do; the setup script exports one from a container image because that is the reliable
way to get a coherent set.

## The six source patches

`gem5_v20.1.0.2_pie_timing.patch` touches six files. Three make a dynamically linked
binary runnable under SE mode, one is load-bearing for the reward's speed, and two
are build fixes for toolchains newer than 2020.

| file | change | why |
| --- | --- | --- |
| `src/base/loader/elf_object.cc` | guard the `PT_INTERP` open with `access(R_OK)` | with a sysroot and path redirects, the interpreter path in the ELF header is a *guest* path that does not exist on the host; unguarded, gem5 panics before the program starts |
| `src/sim/syscall_emul.hh` (mmap debug block) | same `access(R_OK)` guard around `createObjectFile` | same reason, for the debug symbol table. Symbols only; no effect on timing |
| `src/sim/vma.cc` | `mmap(fd, offset)` → `calloc` + `pread` | x86 ELF segments are 4 KB-aligned, so mapping them fails with `EINVAL` on a host with larger pages. Harmless on 4 KB hosts; keep it for portability |
| `src/sim/syscall_emul.hh` (`readFunc`) | **the read(0) hook** — see below | this is what makes the reward affordable |
| `src/sim/init_signals.cc` | `fatalSigStack[2 * SIGSTKSZ]` → a fixed 131072 | glibc 2.34 made `SIGSTKSZ` a runtime `sysconf()` value, so it can no longer size a static array |
| `src/systemc/ext/tlm_utils/instance_specific_extensions_int.h` | add `#include <typeinfo>` | gcc 11+ no longer includes it transitively |

### The read(0) hook, and why the reward is affordable

Simulating a competitive-programming program is dominated by *startup*: process
setup, dynamic linking, C++ static initialisers, iostream construction. For most of
these programs that is far more simulated time than the actual computation, and it is
identical every single time.

The hook exploits that. On the program's first `read()` from stdin — the moment
startup has finished and before any input has been consumed — gem5 stops and takes a
checkpoint of the whole simulated machine. Timing a test case then means forking from
that checkpoint and running only the part that depends on the input. One simulated
startup is shared by every case.

Two details make it correct. The flag is *static*, so children forked from the
checkpoint inherit "already triggered" and read their input normally instead of
stopping again. And the fork is placed a small margin *before* the measured
first-read tick rather than exactly at it, so it can never land after the program has
begun consuming input.

The reported per-case tick is still the absolute run-to-completion tick with startup
included, which is the same quantity the PIE dataset's `agg_runtime` measures — the
amortisation changes how long the measurement takes, not what it measures.

## The Skylake configuration

`skylake_config/` is [darchr/gem5-skylake-config](https://github.com/darchr/gem5-skylake-config)
at commit `d92b219`, BSD-3-Clause, by Jason Lowe-Power and Trivikram Reddy — see
[`skylake_config/NOTICE`](skylake_config/NOTICE). It models a Skylake core in enough
detail to be worth simulating: `VerbatimCPU` is an out-of-order core with fetch width
4, issue/dispatch/rename/writeback/commit width 8, 180 physical integer registers, a
3.5 GHz clock and a modelled cache hierarchy.

Two files in that directory are ours: `run_se_probe.py` measures the first-stdin-read
tick, and `run_se_forkloop.py` runs the fork-amortized loop over test inputs. Both
read `$PIE_GEM5_ROOTFS` to build the guest-to-host redirects; the backend exports it
for them. `run-se.py` is the upstream single-run entry point, kept as a reference.

## On a host that is not x86_64

The compiler must emit x86_64 regardless of what the host is, because that is what
gem5 simulates. On an x86_64 host that is just the local `g++-9` and
`scripts/setup_gem5.sh` handles it.

On any other host — an ARM login node, say — you need a cross-compiler: a
*host-native* `x86_64-linux-gnu-g++-9`. Build one from the gcc 9.3 sources against
the same sysroot, install it under a prefix, and set `PIE_GEM5_XCROSS_ROOT` to that
prefix. The setup script will not do this for you, and it is worth verifying rather
than assuming: compile a few hundred programs with both the cross-compiler and a
native `g++-9` and check the object code is byte-identical. If it is not, the measured
times are not comparable with the dataset's, and nothing downstream will tell you.

## Verifying

```bash
bash scripts/verify_setup.sh                          # every path resolves
python3 code_opt/reward/gem5_reward.py --selftest --n 3
```

The self-test is the one that matters. It grades real dataset rows: the source
program, submitted as if it were a rollout, must score ≈ 1.0 against its own stored
reference ticks, and the dataset's known-faster version must score well above 1.0. If
the first number is not ≈ 1.0, your toolchain does not match the one that produced
the dataset, and every reward will be systematically off.
