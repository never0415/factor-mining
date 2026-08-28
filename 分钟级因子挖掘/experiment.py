"""Run isolation, atomic checkpoints and structured failure logging."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid

from min_gp.config import OUTPUT_DIR


def configuration_fingerprint(payload):
    """Stable SHA-256 identity for a JSON-compatible search configuration."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_directory(name, run_id=None, root=None):
    root = Path(root or OUTPUT_DIR) / "experiments"
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    path = root / f"{name}_{run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_failure(path, stage, genome, exc):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "time": datetime.now().isoformat(timespec="seconds"),
            "stage": stage, "genome": str(genome),
            "error_type": type(exc).__name__, "error": str(exc),
        }, ensure_ascii=False) + "\n")
