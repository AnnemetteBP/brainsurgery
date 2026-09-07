from __future__ import annotations

from pathlib import Path

from revision_tests.export_linux_environment import sanitize, validate_anonymous


def test_scaling_export_removes_identity_and_gpu_uuid(tmp_path: Path):
    source = tmp_path / "environment.json"
    source.write_text("{}", encoding="utf-8")
    raw = {
        "platform": "Linux-6.8-x86_64",
        "system": "Linux",
        "machine": "x86_64",
        "hostname": "private-host",
        "python": "3.13.7",
        "packages": {"torch": "2.14.0", "unrelated": "1"},
        "cpu_count_logical": 16,
        "cpu_count_physical": 8,
        "memory_bytes": 64 * 2**30,
        "disk": {"total": 100 * 2**30},
        "gpu_inventory_only_not_used": "NVIDIA B200, GPU-deadbeef-dead-beef-dead-beefdeadbeef, 180000 MiB, 580.0",
        "execution_device": "cpu",
    }
    result = sanitize("scaling", raw, source)
    validate_anonymous(result, raw)
    assert result["packages"] == {"torch": "2.14.0"}
    assert result["gpu_inventory"][0] == {
        "name": "NVIDIA B200",
        "memory": "180000 MiB",
        "driver": "580.0",
    }
