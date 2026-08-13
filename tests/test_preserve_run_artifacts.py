import importlib.util
import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preserve_run_artifacts.py"
SPEC = importlib.util.spec_from_file_location("preserve_run_artifacts", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
main = MODULE.main
preserve_artifacts = MODULE.preserve_artifacts


def test_preserve_artifacts_copies_and_hashes_sources(tmp_path):
    source = tmp_path / "selected.pt"
    source.write_bytes(b"weights")

    hashes = preserve_artifacts(
        {"regnav/selected.pt": source},
        tmp_path / "preserved",
    )

    copied = tmp_path / "preserved/regnav/selected.pt"
    manifest = tmp_path / "preserved/SHA256SUMS.json"

    assert copied.read_bytes() == b"weights"
    assert source.read_bytes() == b"weights"
    assert hashes == {"regnav/selected.pt": sha256(b"weights").hexdigest()}
    assert json.loads(manifest.read_text()) == hashes


def test_preserve_artifacts_rejects_missing_source_and_existing_destination(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        preserve_artifacts({"missing.pt": tmp_path / "missing.pt"}, tmp_path / "new")

    source = tmp_path / "x.pt"
    source.write_bytes(b"x")
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="existing"):
        preserve_artifacts({"x.pt": source}, destination)


def test_preserve_artifacts_uses_atomic_rename(tmp_path, monkeypatch):
    source = tmp_path / "selected.pt"
    source.write_bytes(b"weights")
    calls = {}
    original_replace = os.replace

    def fake_replace(source_path, target):
        calls["source"] = Path(source_path)
        calls["target"] = Path(target)
        assert not Path(target).exists()
        return original_replace(source_path, target)

    monkeypatch.setattr(os, "replace", fake_replace)

    preserve_artifacts({"regnav/selected.pt": source}, tmp_path / "preserved")

    assert calls["target"] == tmp_path / "preserved"
    assert calls["source"].name.startswith(".preserved.")


def test_main_requires_five_inputs_and_writes_expected_layout(tmp_path, monkeypatch):
    selected = tmp_path / "selected.pt"
    validation = tmp_path / "validation-selected.json"
    test_json = tmp_path / "test.json"
    lite_best = tmp_path / "best.pt"
    lite_test = tmp_path / "lite-test.json"
    selected.write_bytes(b"selected")
    validation.write_text("{}\n")
    test_json.write_text('{"split":"test"}\n')
    lite_best.write_bytes(b"lite")
    lite_test.write_text('{"model":"lite"}\n')

    monkeypatch.setattr(
        "sys.argv",
        [
            "preserve_run_artifacts.py",
            "--destination",
            str(tmp_path / "preserved"),
            "--regnav-selected",
            str(selected),
            "--regnav-validation",
            str(validation),
            "--regnav-test",
            str(test_json),
            "--lite-best",
            str(lite_best),
            "--lite-test",
            str(lite_test),
        ],
    )

    main()

    assert (tmp_path / "preserved/regnav/selected.pt").read_bytes() == b"selected"
    assert (tmp_path / "preserved/regnav/validation-selected.json").read_text() == "{}\n"
    assert (tmp_path / "preserved/regnav/test.json").read_text() == '{"split":"test"}\n'
    assert (tmp_path / "preserved/lite/best.pt").read_bytes() == b"lite"
    assert (tmp_path / "preserved/lite/test.json").read_text() == '{"model":"lite"}\n'
