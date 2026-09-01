#!/usr/bin/env bash
# Build the gem5 timing stack this experiment's reward measures programs with.
#
#   ~9 GB of disk and 30-60 minutes of wall clock, almost all of it the gem5 build.
#   Re-running is cheap: each stage checks for its own output and skips.
#
# Four things get built under $PIE_GEM5_HOME (default <experiment>/gem5/build):
#
#   gem5/build/X86/gem5.fast   the simulator, gem5 v20.1.0.2 with six source patches
#   pyenv/                     python 3.8 + scons 3.x, needed to BUILD gem5 and, at
#                              run time, to supply the libpython3.8 it links against
#   rootfs/                    an x86_64 Ubuntu 20.04 sysroot: glibc 2.31, libstdc++,
#                              and the g++-9 target libraries
#   x86cross/root/             the compiler that builds candidate programs
#
# Why these versions are pinned, and why you should not "just use a newer gem5":
#
#   * gem5 v20.1.0.2 is the version the PIE paper's published runtimes were produced
#     with. A different microarchitectural model gives different tick counts, which
#     changes every speedup in the dataset and makes the reward incomparable with the
#     reference ticks shipped alongside it.
#   * gem5 v20.1 reaches into semi-internal SCons APIs that SCons 4 removed, and its
#     bundled pybind11 2.4.1 calls a CPython symbol deleted in Python 3.9. Hence
#     python 3.8 and scons<4. This build environment is fully isolated: nothing here
#     ever does `import m5` in-process, gem5 is only ever run as a subprocess, so the
#     3.8 environment is invisible to the 3.10+ environment the trainer runs in.
#   * g++-9 -O3, dynamically linked, is what PIE compiled with. A different compiler
#     emits different valid x86 and moves the measured times off the dataset's.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

GEM5_TAG="v20.1.0.2"
GEM5_COMMIT="0d703041fcd5d119012b62287695723a2955b408"
GEM5_REPO="${PIE_GEM5_REPO:-https://github.com/gem5/gem5.git}"
JOBS="${PIE_GEM5_BUILD_JOBS:-$(nproc)}"

HOME_DIR="${PIE_GEM5_HOME}"
SRC="${HOME_DIR}/gem5"
PYENV="${HOME_DIR}/pyenv"
ROOTFS="${HOME_DIR}/rootfs"
XROOT="${HOME_DIR}/x86cross/root"
PATCH="${EXP_ROOT}/gem5/gem5_v20.1.0.2_pie_timing.patch"

mkdir -p "${HOME_DIR}"
echo "[gem5] target root: ${HOME_DIR}"

# --------------------------------------------------------------------------- 1/4
# Build environment: python 3.8 + scons 3.x, isolated from everything else.
# ---------------------------------------------------------------------------
if [ ! -x "${PYENV}/bin/scons" ]; then
  command -v conda >/dev/null 2>&1 || {
    echo "FATAL: conda not found. It is the least painful way to get python 3.8 +" >&2
    echo "       scons<4 on a modern distro. If you have them another way, point" >&2
    echo "       PIE_GEM5_PYENV at that prefix and re-run." >&2
    exit 2
  }
  echo "[gem5] 1/4 creating the python 3.8 build environment (~132 MB)"
  conda create -y -p "${PYENV}" python=3.8 'scons<4'
else
  echo "[gem5] 1/4 build environment present, skipping"
fi

# --------------------------------------------------------------------------- 2/4
# Sysroot. The simulated program is a dynamically linked x86_64 binary, so gem5
# needs a target filesystem to resolve its loader and libraries against, and the
# compiler needs the same tree as its --sysroot. glibc 2.31 / g++-9 is Ubuntu 20.04.
# ---------------------------------------------------------------------------
if [ ! -e "${ROOTFS}/lib/x86_64-linux-gnu/ld-2.31.so" ]; then
  echo "[gem5] 2/4 building the x86_64 Ubuntu 20.04 sysroot"
  if command -v docker >/dev/null 2>&1; then
    # Exporting a container filesystem is the most reliable way to get a coherent
    # multiarch tree with matching glibc and libstdc++.
    printf 'FROM --platform=linux/amd64 ubuntu:20.04\nRUN apt-get update -qq && apt-get install -y -qq g++-9 libstdc++-9-dev\n' \
      | docker build --platform linux/amd64 -t codeopt-rootfs:20.04 -f - . >/dev/null
    tmpc="$(docker create --platform linux/amd64 codeopt-rootfs:20.04 /bin/true)"
    mkdir -p "${ROOTFS}"
    docker export "${tmpc}" | tar x -C "${ROOTFS}"
    docker rm -f "${tmpc}" >/dev/null
  else
    echo "FATAL: no docker, and there is no good distro-agnostic way to assemble a" >&2
    echo "       coherent Ubuntu 20.04 x86_64 sysroot without it." >&2
    echo "       Options: install docker/podman and re-run; or point PIE_GEM5_ROOTFS" >&2
    echo "       at an existing 20.04 tree (it needs at minimum" >&2
    echo "       lib/x86_64-linux-gnu/ld-2.31.so, usr/lib/x86_64-linux-gnu," >&2
    echo "       usr/lib/gcc/x86_64-linux-gnu/9 and usr/include)." >&2
    exit 2
  fi
else
  echo "[gem5] 2/4 sysroot present, skipping"
fi

# --------------------------------------------------------------------------- 3/4
# The compiler that builds candidate programs. On x86_64 this is just the sysroot's
# own g++-9, exposed under the target-triple name the backend looks for. On any
# other host architecture you need a real cross-compiler; see gem5/README.md.
# ---------------------------------------------------------------------------
if [ ! -x "${XROOT}/usr/bin/x86_64-linux-gnu-g++-9" ]; then
  echo "[gem5] 3/4 wiring up the target compiler"
  arch="$(uname -m)"
  if [ "${arch}" = "x86_64" ]; then
    mkdir -p "${XROOT}/usr/bin"
    gxx="$(command -v g++-9 || true)"
    if [ -z "${gxx}" ] && [ -x "${ROOTFS}/usr/bin/g++-9" ]; then
      gxx="${ROOTFS}/usr/bin/g++-9"
    fi
    [ -n "${gxx}" ] || {
      echo "FATAL: no g++-9 on this host. Install it (apt install g++-9), or set" >&2
      echo "       PIE_GEM5_GXX to a g++-9 you already have. The version matters:" >&2
      echo "       the dataset's reference ticks were produced with g++-9 -O3." >&2
      exit 2
    }
    ln -sf "${gxx}" "${XROOT}/usr/bin/x86_64-linux-gnu-g++-9"
    echo "[gem5]     linked ${gxx} -> ${XROOT}/usr/bin/x86_64-linux-gnu-g++-9"
  else
    echo "FATAL: host is ${arch}, not x86_64, so a native g++-9 cannot emit target" >&2
    echo "       binaries. You need an ${arch}-hosted x86_64-linux-gnu-g++-9 cross" >&2
    echo "       compiler; gem5/README.md describes how to build one and what to" >&2
    echo "       verify (codegen must be byte-identical to a native g++-9)." >&2
    echo "       Set PIE_GEM5_XCROSS_ROOT to its prefix and re-run." >&2
    exit 2
  fi
else
  echo "[gem5] 3/4 target compiler present, skipping"
fi

# --------------------------------------------------------------------------- 4/4
# gem5 itself.
# ---------------------------------------------------------------------------
if [ ! -x "${SRC}/build/X86/gem5.fast" ]; then
  if [ ! -d "${SRC}/.git" ]; then
    echo "[gem5] 4/4 cloning gem5 ${GEM5_TAG}"
    git clone --branch "${GEM5_TAG}" --depth 1 "${GEM5_REPO}" "${SRC}"
  fi
  head="$(git -C "${SRC}" rev-parse HEAD)"
  if [ "${head}" != "${GEM5_COMMIT}" ]; then
    echo "WARNING: gem5 HEAD is ${head}, expected ${GEM5_COMMIT}." >&2
    echo "         Tick counts are only comparable with the dataset at the pinned commit." >&2
  fi

  # Six patches. Three make a dynamically linked binary runnable under SE mode with a
  # sysroot; one is the read(0) hook the startup-amortization trick depends on; two are
  # build fixes for modern toolchains. gem5/README.md explains each.
  if ! git -C "${SRC}" diff --quiet; then
    echo "[gem5]     patch already applied, skipping"
  else
    echo "[gem5]     applying ${PATCH##*/}"
    git -C "${SRC}" apply "${PATCH}"
  fi

  echo "[gem5]     building gem5.fast with -j${JOBS} (25-45 min)"
  # gem5.fast rather than gem5.opt: NDEBUG only, so asserts and tracing are compiled
  # out. It produces identical sim_ticks and runs ~1.25x faster, which matters when
  # the reward simulates thousands of programs per training step.
  ( cd "${SRC}"
    PATH="${PYENV}/bin:${PATH}" \
    LD_LIBRARY_PATH="${PYENV}/lib:${LD_LIBRARY_PATH:-}" \
    scons build/X86/gem5.fast "-j${JOBS}" \
      CCFLAGS_EXTRA='-Wno-error' \
      PYTHON_CONFIG="${PYENV}/bin/python3.8-config" )
else
  echo "[gem5] 4/4 gem5.fast present, skipping"
fi

echo
echo "[gem5] done. Verify with:"
echo "    bash scripts/verify_setup.sh"
echo
echo "  export PIE_GEM5_HOME=${HOME_DIR}"
