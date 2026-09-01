# Probe: run startup with the read(0) hook, print the pre-input read tick, exit. Used to derive the
# per-binary fork_tick for fresh (RL rollout) candidates. Env: GEM5_STDIN=<file> GEM5_PREINPUT_FORK=1
import m5, os, sys
from m5.objects import *
import argparse
from system.se import MySystem
from system.core import *
valid = {c.__name__[:-3]: c for c in [VerbatimCPU, TunedCPU, UnconstrainedCPU]}
ap = argparse.ArgumentParser(); ap.add_argument('config', choices=valid.keys()); ap.add_argument('binary')
args = ap.parse_args()
class TestSystem(MySystem):
    _CPUModel = valid[args.config]
system = TestSystem(); system.setTestBinary(args.binary)
_stdin = os.environ.get("GEM5_STDIN")
if _stdin: system.cpu.workload[0].input = _stdin
_RF = os.environ.get("PIE_GEM5_ROOTFS")
if not _RF:
    raise SystemExit("PIE_GEM5_ROOTFS must point at the x86_64 sysroot; "
                     "code_opt/measurement/gem5_backend.py exports it automatically.")
system.redirect_paths = [RedirectPath(app_path=p, host_paths=[_RF + p])
                         for p in ["/lib64", "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib", "/lib"]]
root = Root(full_system=False, system=system)
m5.disableAllListeners()
m5.instantiate()
ev = m5.simulate()
sys.stdout.write("READ_TICK %d cause %s\n" % (m5.curTick(), ev.getCause())); sys.stdout.flush()
os._exit(0)
