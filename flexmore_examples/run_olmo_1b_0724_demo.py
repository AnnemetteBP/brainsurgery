from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(cwd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full OLMo-1B dense-to-expert-MoE paper demo: "
            "BrainSurgery conversion, reference conversion, and diff validation."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for both BrainSurgery and the reference script.",
    )
    parser.add_argument(
        "--model-a",
        type=Path,
        default=Path("models/olmo_1b_0724_hf_dense"),
        help="Path to dense source checkpoint A.",
    )
    parser.add_argument(
        "--model-b",
        type=Path,
        default=Path("models/olmo_1b_0724_hf_dense"),
        help="Path to dense source checkpoint B.",
    )
    parser.add_argument(
        "--yaml-output",
        type=Path,
        default=Path("models/olmo_1b_0724_hf_dense_moe_demo"),
        help="Output directory written by the BrainSurgery YAML plan.",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=Path("models/olmo_1b_0724_hf_dense_moe_reference"),
        help="Output directory written by the reference converter.",
    )
    parser.add_argument(
        "--skip-yaml",
        action="store_true",
        help="Skip the BrainSurgery YAML conversion step.",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip the reference conversion step.",
    )
    parser.add_argument(
        "--skip-diff",
        action="store_true",
        help="Skip the final BrainSurgery diff validation step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if not args.skip_yaml:
        _run(
            [
                args.python,
                "-m",
                "brainsurgery.cli",
                "flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe.yaml",
                f"inputs[0]=m0::{args.model_a}",
                f"inputs[1]=m1::{args.model_b}",
                f"output.path={args.yaml_output}",
            ],
            cwd=repo_root,
        )

    if not args.skip_reference:
        _run(
            [
                args.python,
                "flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe_reference.py",
                "--model-a",
                str(args.model_a),
                "--model-b",
                str(args.model_b),
                "--target",
                str(args.reference_output),
                "--copy-metadata",
                "--write-example-config",
            ],
            cwd=repo_root,
        )

    if not args.skip_diff:
        _run(
            [
                args.python,
                "-m",
                "brainsurgery.cli",
                "flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe_validate.yaml",
                f"inputs[0]=yaml::{args.yaml_output}",
                f"inputs[1]=ref::{args.reference_output}",
            ],
            cwd=repo_root,
        )

    print("\nDemo flow completed.", flush=True)
    if not args.skip_diff:
        print(
            "If validation succeeded, the diff output should end with 'No differences found.'.",
            flush=True,
        )


if __name__ == "__main__":
    main()
