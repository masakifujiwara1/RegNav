import argparse
import ctypes
import errno
import json
import os
from hashlib import sha256
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_LIBC = None
try:
    _LIBC = ctypes.CDLL(None, use_errno=True)
    _RENAMEAT2 = _LIBC.renameat2
    _RENAMEAT2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    _RENAMEAT2.restype = ctypes.c_int
except (AttributeError, OSError):
    _RENAMEAT2 = None


def _validate_relative_name(relative_name: str) -> Path:
    relative_path = Path(relative_name)
    if relative_path.is_absolute():
        raise ValueError(f"artifact path must be relative, got absolute path: {relative_name}")
    if any(part == '..' for part in relative_path.parts):
        raise ValueError(f"artifact path must not contain .. segments: {relative_name}")
    return relative_path


def _rename_noreplace(source: Path, destination: Path) -> bool:
    if _RENAMEAT2 is None:
        return False
    result = _RENAMEAT2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return True
    err = ctypes.get_errno()
    if err in (errno.ENOSYS, errno.EINVAL):
        return False
    raise OSError(err, os.strerror(err), destination)


def _publish_path_no_clobber(staged: Path, destination: Path) -> None:
    if staged.is_dir():
        destination.mkdir()
        for child in staged.iterdir():
            _publish_path_no_clobber(child, destination / child.name)
        return
    if staged.is_file():
        os.link(staged, destination)
        return
    raise OSError(f"unsupported staged artifact type: {staged}")


def _publish_directory_no_clobber(staged: Path, destination: Path) -> None:
    if _rename_noreplace(staged, destination):
        return

    _publish_path_no_clobber(staged, destination)


def preserve_artifacts(sources: dict[str, Path], destination: Path) -> dict[str, str]:
    relative_paths = {name: _validate_relative_name(name) for name in sources}
    missing = [name for name, source in sources.items() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {', '.join(sorted(missing))}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temp_dir:
        staged = Path(temp_dir)
        hashes: dict[str, str] = {}
        for relative_name, source in sorted(sources.items()):
            target = staged / relative_paths[relative_name]
            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, target)
            hashes[relative_name] = sha256(target.read_bytes()).hexdigest()
        (staged / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
        _publish_directory_no_clobber(staged, destination)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description='Preserve verified run artifacts')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--regnav-selected', type=Path, required=True)
    parser.add_argument('--regnav-validation', type=Path, required=True)
    parser.add_argument('--regnav-test', type=Path, required=True)
    parser.add_argument('--lite-best', type=Path, required=True)
    parser.add_argument('--lite-test', type=Path, required=True)
    args = parser.parse_args()
    preserve_artifacts(
        {
            'regnav/selected.pt': args.regnav_selected,
            'regnav/validation-selected.json': args.regnav_validation,
            'regnav/test.json': args.regnav_test,
            'lite/best.pt': args.lite_best,
            'lite/test.json': args.lite_test,
        },
        args.destination,
    )


if __name__ == '__main__':
    main()
