#!/usr/bin/env python3
"""Enforce the supported `transformers` range. Run as a preflight by every launcher.

Why this is a hard gate and not a warning: outside the range the SFT checkpoints
load *silently wrong* -- the model still emits well-formed maze paths, the
training curves still move, and it reaches the goal 0% of the time. There is no
error to notice. Failing here is the only cheap signal.

Below MIN the checkpoint configs are misread (RoPE base falls back to a 100x-too-
small default, and the tokenizer class is unknown). At or above MAX the model and
generation APIs this verl fork targets have changed.

Usage:
    python scripts/check_transformers.py          # check the installed version
    python scripts/check_transformers.py 4.56.1   # check a specific version
"""
import re
import sys

MIN = (4, 51)          # inclusive
MAX = (5, 0)           # exclusive
SUPPORTED = ">=%d.%d,<%d.%d" % (MIN + MAX)


def parse(version):
    """('4.56.1',) -> (4, 56). Tolerates suffixes like '4.56.1.dev0' and '4.56'."""
    m = re.match(r"\s*v?(\d+)\.(\d+)", str(version))
    if not m:
        raise ValueError("cannot parse a transformers version from %r" % (version,))
    return int(m.group(1)), int(m.group(2))


def supported(version):
    """True iff `version` is inside the supported range."""
    return MIN <= parse(version) < MAX


def check(version):
    """Raise SystemExit with an actionable message if `version` is unsupported."""
    if supported(version):
        return
    where = "too old" if parse(version) < MIN else "too new"
    raise SystemExit(
        "[FATAL] transformers %s is %s. This experiment requires %s.\n"
        "        Outside that range the SFT checkpoints load silently wrong: the\n"
        "        model emits valid-looking maze paths and reaches the goal 0%% of\n"
        "        the time, with training curves that still look alive.\n"
        "        Fix:  pip install 'transformers%s'"
        % (version, where, SUPPORTED, SUPPORTED)
    )


def installed_version():
    import transformers  # imported lazily so the range logic stays testable without it

    return transformers.__version__


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else installed_version()
    check(version)
    print("[preflight] transformers %s OK (supported: %s)" % (version, SUPPORTED))


if __name__ == "__main__":
    main()
