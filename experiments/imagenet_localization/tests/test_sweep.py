"""Tests for experiments.imagenet_localization.sweep.

No real SLURM environment needed — all tests write to tmp_path.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from experiments.imagenet_localization.sweep import (
    generate_script_content,
    stage_groups,
    write_script,
    write_submit_all,
    write_submit_stage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WRITE_SCRIPT_DEFAULTS = dict(
    results_dir="/tmp/results",
    data_dir="/tmp/imagenet",
    partition="general",
    conda_env="base",
    train_subsample=100000,
    epochs=30,
    time="24:00:00",
    repo_dir="/tmp/repo",
)


def _write_one(tmp_path, method="tailrl", K=50, N=64, seed=42, **overrides) -> str:
    """Write a single script to tmp_path with sensible defaults."""
    kw = {**_WRITE_SCRIPT_DEFAULTS, **overrides}
    return write_script(
        out_dir=str(tmp_path),
        method=method,
        K=K,
        N=N,
        seed=seed,
        log_dir=str(tmp_path / "logs"),
        **kw,
    )


# ---------------------------------------------------------------------------
# Stage-group counts
# ---------------------------------------------------------------------------


def test_primary_total_is_84():
    """Stages 1-3 together must contain exactly 84 Primary configs."""
    stages = stage_groups(include_mse=False)
    primary_total = sum(len(stages[s]) for s in (1, 2, 3))
    assert primary_total == 84, f"Expected 84 primary configs, got {primary_total}"


def test_stage4_total_is_20():
    """Stage 4 must contain exactly 20 configs: N-sweep (tailrl, binary_maxrl) ×
    N ∈ {4, 16, 256} × 3 seeds = 18 (N=64 is deliberately skipped because
    Stage 1 already covers K=50, N=64 for both methods) + 2 K=2 sanity runs.
    """
    stages = stage_groups(include_mse=False)
    assert len(stages[4]) == 20, f"Expected 20 stage-4 configs, got {len(stages[4])}"


def test_total_count_default_is_104():
    """Grand total without --include_mse must be exactly 104 after Stage 4
    is deduplicated against Stage 1 (no K=50, N=64 collision)."""
    stages = stage_groups(include_mse=False)
    total = sum(len(v) for v in stages.values())
    assert total == 104, f"Expected 104 total configs, got {total}"


def test_total_count_with_mse_is_116():
    """Grand total with --include_mse must be exactly 116 (104 + 12 MSE)."""
    stages = stage_groups(include_mse=True)
    total = sum(len(v) for v in stages.values())
    assert total == 116, f"Expected 116 total configs with MSE, got {total}"


# ---------------------------------------------------------------------------
# Stage 1 content checks
# ---------------------------------------------------------------------------


def test_stage_1_contains_only_tailrl_and_binary_maxrl():
    """Stage 1 must only contain 'tailrl' and 'binary_maxrl' methods."""
    stages = stage_groups(include_mse=False)
    methods_in_stage1 = {method for method, K, N, seed in stages[1]}
    assert methods_in_stage1 == {"tailrl", "binary_maxrl"}, (
        f"Stage 1 methods: {methods_in_stage1}"
    )


def test_stage_1_contains_only_K_10_25_50_100():
    """Stage 1 must only use K ∈ {10, 25, 50, 100}."""
    stages = stage_groups(include_mse=False)
    k_values_in_stage1 = {K for method, K, N, seed in stages[1]}
    assert k_values_in_stage1 == {10, 25, 50, 100}, (
        f"Stage 1 K values: {k_values_in_stage1}"
    )


def test_stage_1_has_24_configs():
    """Stage 1: 2 methods × 4 K × 3 seeds = 24."""
    stages = stage_groups(include_mse=False)
    assert len(stages[1]) == 24, f"Stage 1 count: {len(stages[1])}"


def test_stage_2_has_24_configs_without_mse():
    """Stage 2 without MSE: 2 methods × 4 K × 3 seeds = 24."""
    stages = stage_groups(include_mse=False)
    assert len(stages[2]) == 24, f"Stage 2 count (no MSE): {len(stages[2])}"


def test_stage_2_has_36_configs_with_mse():
    """Stage 2 with MSE: (2 + 1) methods × 4 K × 3 seeds = 36."""
    stages = stage_groups(include_mse=True)
    assert len(stages[2]) == 36, f"Stage 2 count (with MSE): {len(stages[2])}"


def test_stage_3_has_36_configs():
    """Stage 3: 3 methods (grpo/rloo/reinforce) × 4 K × 3 seeds = 36."""
    stages = stage_groups(include_mse=False)
    assert len(stages[3]) == 36, f"Stage 3 count: {len(stages[3])}"


# ---------------------------------------------------------------------------
# No duplicate configs
# ---------------------------------------------------------------------------


def test_no_duplicate_configs_in_default_sweep():
    """After dedup fix, every (method, K, N, seed) tuple must be unique across
    all stages. Earlier versions had 6 intentional K=50/N=64 overlaps between
    Stage 1 and Stage 4 which caused submit_all.sh to re-submit the same
    scripts — that's been fixed by skipping N=N_PRIMARY in Stage 4's N-sweep.
    """
    from collections import Counter
    stages = stage_groups(include_mse=False)
    all_configs: list[tuple[str, int, int, int]] = []
    for configs in stages.values():
        all_configs.extend(configs)

    counts = Counter(all_configs)
    dupes = {cfg for cfg, cnt in counts.items() if cnt > 1}
    assert not dupes, f"Duplicate configs found: {dupes}"
    assert len(all_configs) == 104


def test_stage4_has_no_K50_N64_configs():
    """Stage 4 must not contain any K=50, N=64 configs — those collide with
    Stage 1 and were the source of the duplicate submissions before the fix.
    """
    stages = stage_groups(include_mse=False)
    colliding = [c for c in stages[4] if c[1] == 50 and c[2] == 64]
    assert not colliding, f"Stage 4 still has K=50/N=64 collisions: {colliding}"


# ---------------------------------------------------------------------------
# Generated script: bash syntax validity
# ---------------------------------------------------------------------------


def test_generated_script_is_valid_bash(tmp_path):
    """bash -n must exit 0 (syntax check only, no execution)."""
    (tmp_path / "logs").mkdir(exist_ok=True)
    path = _write_one(tmp_path)
    result = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"bash -n failed on generated script:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Generated script: executable bit
# ---------------------------------------------------------------------------


def test_generated_script_has_executable_bit(tmp_path):
    """Generated scripts must be chmod +x (owner execute bit set)."""
    (tmp_path / "logs").mkdir(exist_ok=True)
    path = _write_one(tmp_path)
    assert os.access(path, os.X_OK), (
        f"Script at {path!r} is not executable"
    )


# ---------------------------------------------------------------------------
# Generated script: content checks
# ---------------------------------------------------------------------------


def test_generated_script_has_shebang(tmp_path):
    """Generated scripts must start with #!/bin/bash."""
    (tmp_path / "logs").mkdir(exist_ok=True)
    path = _write_one(tmp_path)
    with open(path) as fh:
        first_line = fh.readline().strip()
    assert first_line == "#!/bin/bash", (
        f"Expected shebang, got: {first_line!r}"
    )


def test_generated_script_has_pipefail(tmp_path):
    """Generated job scripts must fail fast: 'set -eo pipefail'.

    Deliberately *not* 'set -euo': some conda env-activation scripts reference
    unbound variables (e.g. ADDR2LINE in activate-binutils_linux-64.sh) and
    hard-fail under `set -u`, killing the job before training starts. We keep
    -e and -o pipefail and drop -u. The launcher scripts, which never activate
    conda, do use the full 'set -euo pipefail'.
    """
    (tmp_path / "logs").mkdir(exist_ok=True)
    path = _write_one(tmp_path)
    with open(path) as fh:
        content = fh.read()
    assert "set -eo pipefail" in content
    assert "set -euo pipefail" not in content


def test_generated_script_per_run_output_dir(tmp_path):
    """Each script's --output_dir must be a run-specific subdirectory."""
    (tmp_path / "logs").mkdir(exist_ok=True)
    results_dir = "/tmp/results"
    path = _write_one(tmp_path, method="tailrl", K=50, N=64, seed=42,
                      results_dir=results_dir)
    with open(path) as fh:
        content = fh.read()
    expected_subdir = "tailrl_K50_N64_seed42"
    assert expected_subdir in content, (
        f"Expected run-specific subdir '{expected_subdir}' in script output_dir arg"
    )


def test_generated_script_job_name_is_opaque(tmp_path):
    """SLURM --job-name must be an opaque 'j_XXXXXXXX' identifier so that
    `squeue` on the shared cluster doesn't reveal which experiment or method
    is running. Deterministic so it's reproducible (same config → same name).
    """
    import re
    (tmp_path / "logs").mkdir(exist_ok=True)
    path = _write_one(tmp_path, method="tailrl", K=50, N=64, seed=42)
    with open(path) as fh:
        content = fh.read()
    m = re.search(r"#SBATCH --job-name=(j_[0-9a-f]{8})\b", content)
    assert m is not None, "Expected opaque job name 'j_XXXXXXXX' in generated script"
    # The name must NOT leak the experiment keyword.
    assert "imnet" not in m.group(1)
    assert "tailrl"   not in m.group(1)


def test_job_name_length_all_methods():
    """All method/K/N/seed combos must produce job names ≤ 64 characters."""
    from experiments.imagenet_localization.sweep import _job_name, METHODS, K_VALUES, SEEDS, N_PRIMARY
    for method in METHODS:
        for K in K_VALUES:
            for seed in SEEDS:
                name = _job_name(method, K, N_PRIMARY, seed)
                assert len(name) <= 64, (
                    f"Job name too long ({len(name)} chars): {name!r}"
                )


# ---------------------------------------------------------------------------
# submit_all.sh checks
# ---------------------------------------------------------------------------


def test_submit_all_sh_has_all_sbatch_lines(tmp_path):
    """submit_all.sh must contain one 'sbatch <path>' line per script."""
    logs = tmp_path / "logs"
    logs.mkdir()

    # Generate a small set of scripts and write launchers.
    stages = stage_groups(include_mse=False)
    per_stage_scripts: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}

    for stage_num, configs in stages.items():
        # Only write a representative subset to keep the test fast.
        for (method, K, N, seed) in configs[:2]:
            path = write_script(
                out_dir=str(tmp_path),
                method=method,
                K=K,
                N=N,
                seed=seed,
                log_dir=str(logs),
                **_WRITE_SCRIPT_DEFAULTS,
            )
            per_stage_scripts[stage_num].append(path)

    submit_all_path = write_submit_all(str(tmp_path), per_stage_scripts)

    with open(submit_all_path) as fh:
        content = fh.read()

    all_scripts = [p for scripts in per_stage_scripts.values() for p in scripts]
    for script_path in all_scripts:
        assert f'sbatch "{script_path}"' in content, (
            f"submit_all.sh missing sbatch line for {script_path!r}"
        )


def test_submit_all_sh_has_shebang_and_pipefail(tmp_path):
    """submit_all.sh must have a shebang and set -euo pipefail."""
    per_stage: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    submit_all_path = write_submit_all(str(tmp_path), per_stage)
    with open(submit_all_path) as fh:
        content = fh.read()
    assert content.startswith("#!/bin/bash"), "submit_all.sh missing shebang"
    assert "set -euo pipefail" in content, "submit_all.sh missing set -euo pipefail"


def test_submit_stage_scripts_have_only_their_stage(tmp_path):
    """submit_stage1.sh must not reference scripts from other stages."""
    logs = tmp_path / "logs"
    logs.mkdir()

    # Build two scripts for stage 1 and two for stage 2.
    per_stage: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}

    stage1_configs = [("tailrl", 10, 64, 42), ("tailrl", 25, 64, 42)]
    stage2_configs = [("ordinal_ce", 10, 64, 42), ("cross_entropy", 10, 64, 42)]

    for method, K, N, seed in stage1_configs:
        path = write_script(
            out_dir=str(tmp_path), method=method, K=K, N=N, seed=seed,
            log_dir=str(logs), **_WRITE_SCRIPT_DEFAULTS,
        )
        per_stage[1].append(path)

    for method, K, N, seed in stage2_configs:
        path = write_script(
            out_dir=str(tmp_path), method=method, K=K, N=N, seed=seed,
            log_dir=str(logs), **_WRITE_SCRIPT_DEFAULTS,
        )
        per_stage[2].append(path)

    stage1_launcher = write_submit_stage(str(tmp_path), 1, per_stage[1])
    with open(stage1_launcher) as fh:
        content = fh.read()

    # Stage 1 launcher must reference stage-1 scripts.
    for p in per_stage[1]:
        assert f'sbatch "{p}"' in content

    # Stage 1 launcher must NOT reference stage-2 scripts.
    for p in per_stage[2]:
        assert p not in content, (
            f"submit_stage1.sh unexpectedly references stage-2 script: {p!r}"
        )
