"""Allow running `python -m experiments.imagenet_localization` for a usage hint."""

import sys


def main():
    print(
        "experiments.imagenet_localization\n"
        "\n"
        "ImageNet localization RL experiment comparing TailRL\n"
        "against binary MaxRL, GRPO, and other baselines.\n"
        "\n"
        "Entry points:\n"
        "  python -m experiments.imagenet_localization.run      "
        "  -- train a single run with a given config\n"
        "  python -m experiments.imagenet_localization.pilot    "
        "  -- quick smoke-test / pilot run on a small subset\n"
        "  python -m experiments.imagenet_localization.sweep    "
        "  -- launch a hyperparameter sweep\n"
        "\n"
        "Tests:\n"
        "  python -m pytest experiments/imagenet_localization/tests/ -v\n"
        "\n"
        "Environment:\n"
        "  IMAGENET_DIR   -- required: root of the ImageNet localization release\n"
        "  TAILRL_ROOT / TAILRL_RESULTS_DIR / TAILRL_FIGURES_DIR / WANDB_PROJECT\n"
        "                 -- optional overrides; see paths.py for the defaults\n"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
