import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory


def preserve_artifacts(sources: dict[str, Path], destination: Path) -> dict[str, str]:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    missing = [name for name, source in sources.items() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {', '.join(sorted(missing))}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temp_dir:
        staged = Path(temp_dir)
        hashes: dict[str, str] = {}
        for relative_name, source in sorted(sources.items()):
            target = staged / relative_name
            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, target)
            hashes[relative_name] = sha256(target.read_bytes()).hexdigest()
        (staged / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
        os.replace(staged, destination)
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
