#!/usr/bin/env python3
"""Export captured Linux environments as anonymous paper evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = {
    "scaling": REPO / "log/revision_tests/eacl2027_scaling_linux_2dbcd50/scaling/environment.json",
    "competing_tools": REPO / "log/revision_tests/eacl2027_competing_linux_2dbcd50/competing_tools/environment.json",
    "robustness": REPO / "log/revision_tests/eacl2027_robustness_linux_2dbcd50/robustness/environment.json",
}
DEFAULT_OUTPUTS = {
    "scaling": REPO / "revision_tests/scaling/results/linux_2dbcd50/environment.json",
    "competing_tools": REPO / "revision_tests/competing_tools/results/linux_2dbcd50/environment.json",
    "robustness": REPO / "revision_tests/robustness/results/linux_2dbcd50/environment.json",
}
PAPER_OUTPUT = REPO / "revision_tests/plans/paper_environment.tex"
PACKAGE_ALLOWLIST = {
    "brainsurgery",
    "mergekit",
    "numpy",
    "psutil",
    "pyyaml",
    "safetensors",
    "torch",
    "torch-state-bridge",
    "transformers",
}
FORBIDDEN = ("/home/", "/workspace/", "github.com", "git@", "gpu_uuid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--paper-output", type=Path, default=PAPER_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"captured environment record is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"environment record is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_subset(snapshot: dict[str, Any]) -> dict[str, str]:
    packages = snapshot.get("packages", {})
    if isinstance(packages, dict):
        items = packages.items()
    else:
        items = packages
    result = {}
    for name, version in items:
        normalized = str(name).lower().replace("_", "-")
        if normalized in PACKAGE_ALLOWLIST:
            result[normalized] = str(version)
    return dict(sorted(result.items()))


def disk_total(value: Any) -> int | None:
    if isinstance(value, dict):
        total = value.get("total")
    elif isinstance(value, list) and value:
        total = value[0]
    else:
        total = None
    return int(total) if isinstance(total, (int, float)) else None


def gpu_inventory(value: Any) -> list[dict[str, str]]:
    result = []
    for line in str(value or "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or not fields[0]:
            continue
        result.append(
            {
                "name": fields[0],
                "memory": fields[2] if len(fields) > 2 else "unavailable",
                "driver": fields[3] if len(fields) > 3 else "unavailable",
            }
        )
    return result


def filesystem_type(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    fields = str(value.get("findmnt", "")).split()
    return fields[1] if len(fields) >= 2 else None


def common(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "platform": raw.get("platform") or raw.get("platform_release"),
        "system": raw.get("system") or raw.get("platform_system"),
        "machine": raw.get("machine") or raw.get("platform_machine"),
        "processor": raw.get("processor") or raw.get("cpu") or None,
        "cpu_count_logical": raw.get("cpu_count_logical") or raw.get("logical_cpu_count"),
        "cpu_count_physical": raw.get("cpu_count_physical"),
        "cpu_affinity": raw.get("cpu_affinity"),
        "memory_bytes": raw.get("memory_bytes"),
        "disk_total_bytes": disk_total(raw.get("disk")),
        "filesystem_type": filesystem_type(raw.get("filesystem")),
        "git_commit": raw.get("git_commit") or raw.get("repository", {}).get("commit"),
    }


def sanitize(name: str, raw: dict[str, Any], source: Path) -> dict[str, Any]:
    result = common(raw)
    result.update({"evaluation": name, "raw_record_sha256": sha256(source)})
    if name == "scaling":
        result.update(
            {
                "python": raw.get("python"),
                "packages": package_subset(raw),
                "gpu_inventory": gpu_inventory(raw.get("gpu_inventory_only_not_used")),
                "execution_device": raw.get("execution_device"),
                "cache_policy": raw.get("cache_policy"),
                "num_workers": raw.get("num_workers"),
                "sample_interval_ms": raw.get("sample_interval_ms"),
                "workload_note": raw.get("operator_workload_note"),
            }
        )
    elif name == "competing_tools":
        result.update(
            {
                "python_environments": {
                    tool: {
                        "python": snapshot.get("python"),
                        "packages": package_subset(snapshot),
                    }
                    for tool, snapshot in sorted(raw.get("python_environments", {}).items())
                },
                "cache_policy": raw.get("cache_policy"),
                "num_threads": raw.get("num_threads"),
                "sample_interval_ms": raw.get("sample_interval_ms"),
                "workload_note": raw.get("operator_workload_note"),
            }
        )
    else:
        result.update(
            {
                "python": raw.get("python"),
                "packages": {
                    key: raw.get(key)
                    for key in ("brainsurgery", "torch", "safetensors", "pyyaml")
                },
                "subprocess_environment": raw.get("subprocess_environment"),
            }
        )
    return result


def validate_anonymous(value: dict[str, Any], raw: dict[str, Any]) -> None:
    text = json.dumps(value, sort_keys=True).lower()
    forbidden = list(FORBIDDEN)
    hostname = str(raw.get("hostname", "")).strip().lower()
    if hostname:
        forbidden.append(hostname)
    matches = [token for token in forbidden if token and token in text]
    if matches or re.search(r"gpu-[0-9a-f-]{20,}", text):
        raise SystemExit(f"anonymous environment export contains identifiers: {matches}")


def latex_text(records: dict[str, dict[str, Any]]) -> str:
    scaling = records["scaling"]
    gpu = scaling.get("gpu_inventory") or []
    gpu_name = gpu[0]["name"] if gpu else "an NVIDIA B200 GPU"
    logical = scaling.get("cpu_count_logical") or "unreported"
    memory = scaling.get("memory_bytes")
    memory_gib = f"{memory / 2**30:.1f}" if isinstance(memory, int) else "unreported"
    platform_value = str(scaling.get("platform") or "Linux").replace("_", r"\_")
    return (
        "% Generated from sanitized captured run environments.\n"
        "\\paragraph{Linux experimental environment.}\n"
        f"The Linux evaluations ran on the recorded UCloud Ubuntu environment "
        f"({platform_value}) with {logical} logical CPU slots, {memory_gib}~GiB RAM, "
        f"and {gpu_name}. Scaling, robustness, and competing-tool measurements "
        "executed on CPU; the GPU was inventory only for those experiments. "
        "Each performance result used the workload, cache, worker, and package-version "
        "controls preserved in its accompanying sanitized environment record.\n"
    )


def main() -> int:
    args = parse_args()
    inputs = {
        "scaling": args.scaling,
        "competing_tools": args.competing_tools,
        "robustness": args.robustness,
    }
    records = {}
    for name, source in inputs.items():
        raw = read_json(source)
        record = sanitize(name, raw, source)
        validate_anonymous(record, raw)
        output = DEFAULT_OUTPUTS[name]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        records[name] = record
    args.paper_output.parent.mkdir(parents=True, exist_ok=True)
    args.paper_output.write_text(latex_text(records), encoding="utf-8")
    print("PASS: wrote three anonymous environment records and paper_environment.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
