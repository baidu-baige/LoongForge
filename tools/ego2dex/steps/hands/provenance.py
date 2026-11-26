# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import zipfile
import tempfile
from pathlib import Path


def capture(output, hand, parameters):
    app = Path(__file__).resolve().parents[2]
    files = {str(p.relative_to(app)): p.read_bytes() for p in app.rglob("*.py") if "__pycache__" not in p.parts}
    hashes = {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    archives = Path(os.environ.get("EGO2ROBOT_SOURCE_ARCHIVES", str(Path(output) / "source_archives")))
    archives.mkdir(exist_ok=True, parents=True)
    archive = archives / (digest + ".zip")
    if not archive.exists():
        with tempfile.NamedTemporaryFile(dir=archives, suffix=".zip.tmp", delete=False) as temp:
            temporary = Path(temp.name)
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
            for n, v in files.items():
                z.writestr(n, v)
        os.replace(temporary, archive)
    from .registry import profile

    models = {
        s: dict(
            profile=profile(s),
            normalized_sha256=hashlib.sha256(Path(profile(s)["normalized"]).read_bytes()).hexdigest(),
        )
        for s in ["left", "right"]
    }
    record = dict(
        hand=hand,
        source_digest=digest,
        source_archive=str(archive),
        source_hashes=hashes,
        models=models,
        parameters=parameters,
        seed_policy=os.environ.get("EGO2ROBOT_HAND_SEED", "source"),
        objective_policy=os.environ.get("EGO2ROBOT_HAND_SOLVE", "balanced"),
        regression_parameters={k: v for k, v in os.environ.items() if k.startswith("EGO2ROBOT_")},
    )
    (Path(output) / "run_provenance.json").write_text(json.dumps(record, indent=2, default=str))
    return record
