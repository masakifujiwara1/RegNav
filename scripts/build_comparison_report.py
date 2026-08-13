import argparse
import json
import math
from pathlib import Path

_LATENCY_FIELDS = ("p50", "p95", "p99")


def _load_accepted_test(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"evaluation JSON must be an object: {path}")
    if data.get("split") != "test":
        raise ValueError(f"evaluation JSON must use the test split: {path}")
    checkpoint_hash = data.get("checkpoint_hash")
    if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
        raise ValueError(f"evaluation JSON requires a checkpoint_hash: {path}")
    aggregate = data.get("aggregate")
    if not isinstance(aggregate, dict) or aggregate.get("meets_baseline_acceptance") is not True:
        raise ValueError(f"evaluation JSON does not meet baseline acceptance: {path}")
    return data


def _validate_latencies(latencies: dict[str, dict[str, float]]) -> None:
    for model in ("regnav", "lite"):
        values = latencies.get(model)
        if not isinstance(values, dict) or any(field not in values for field in _LATENCY_FIELDS):
            raise ValueError(f"latencies must include {model} p50, p95, and p99")
        for field in _LATENCY_FIELDS:
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"latency {model}.{field} must be numeric")


def _metric(data: dict, name: str) -> object:
    try:
        return data["aggregate"]["model"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"evaluation JSON is missing aggregate.model.{name}") from exc


def build_report(
    regnav_test: Path,
    lite_test: Path,
    output: Path,
    latencies: dict[str, dict[str, float]],
) -> None:
    regnav = _load_accepted_test(regnav_test)
    lite = _load_accepted_test(lite_test)
    _validate_latencies(latencies)

    rows = []
    for label, key, data in (("RegNav", "regnav", regnav), ("RegNav-Lite", "lite", lite)):
        latency = latencies[key]
        threshold = "PASS" if latency["p95"] < 100 else "FAIL"
        rows.append(
            "| {label} | {ade} | {fde} | {heading} | {p50} ms | {p95} ms | {p99} ms | {threshold} | {checkpoint} |".format(
                label=label,
                ade=_metric(data, "ade"),
                fde=_metric(data, "fde"),
                heading=_metric(data, "heading_error"),
                p50=latency["p50"],
                p95=latency["p95"],
                p99=latency["p99"],
                threshold=threshold,
                checkpoint=data.get("checkpoint_hash", "unknown"),
            )
        )

    report = """# RegNav vs RegNav-Lite Comparison

Both evaluation files are held-out `test` split results and pass baseline acceptance.

| Model | ADE | FDE | Heading error | p50 | p95 | p99 | p95 < 100 ms | Checkpoint hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- |
{rows}

## Acceptance

- RegNav baseline acceptance: PASS
- RegNav-Lite baseline acceptance: PASS
- Batch-1 CUDA latency threshold: `p95 < 100 ms` (reported per model above).
- RECON collision metrics are **unavailable, not passed** because the dataset has no obstacle labels; `null` values are not treated as acceptance.

## Provenance

- Preserved artifacts are the verified selected RegNav checkpoint, validation/test JSON, and RegNav-Lite checkpoint/test JSON.
- RegNav's original `best.pt`, `last.pt`, and `metrics.jsonl` are excluded because they were contaminated by an unexplained second writer.
""".format(rows="\n".join(rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(report)


def _parse_latency(value: str) -> dict[str, float]:
    try:
        values = [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("latency must be comma-separated p50,p95,p99 floats") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("latency must contain exactly p50,p95,p99")
    return dict(zip(_LATENCY_FIELDS, values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RegNav comparison report")
    parser.add_argument("--regnav-test", type=Path, required=True)
    parser.add_argument("--lite-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regnav-latency", type=_parse_latency, required=True)
    parser.add_argument("--lite-latency", type=_parse_latency, required=True)
    args = parser.parse_args()
    build_report(
        args.regnav_test,
        args.lite_test,
        args.output,
        {"regnav": args.regnav_latency, "lite": args.lite_latency},
    )


if __name__ == "__main__":
    main()
