# Public data preparation

RegNav Phase 1 uses forward JPEG images and 2D odometry from these official sources:

- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html)
- [HuRoN](https://sites.google.com/view/sacson-review/huron-dataset)
- [TartanDrive 2.0](https://theairlab.org/TartanDrive2/)
- [ViNT/GNM processing reference](https://github.com/robodhruv/visualnav-transformer)

Check each source license before downloading, converting, or redistributing it. Keep raw archives, converted data, and caches outside this repository. Record each downloaded archive's URL, version/date, and SHA-256 checksum in a separate provenance file beside the dataset.

## Processed trajectory

Each trajectory follows the ViNT-style layout:

```text
trajectory-id/
├── 0.jpg
├── 1.jpg
├── ...
└── traj_data.pkl
```

`traj_data.pkl` is a mapping containing `position` as an `[N,2]` array and `yaw` as an `[N]` array. JPEG stems are zero-based frame numbers. Positions use metres, yaw uses radians, and `sample_period` in the manifest gives seconds per frame.

## Manifest

The manifest is JSONL. Each row contains exactly `trajectory_id`, `dataset`, `robot`, `environment`, `date`, `path`, `sample_period`, and optional `obstacle_dir`. `path` points to one processed trajectory. Optional privileged obstacle files use `obstacle_dir/<frame>.npy` with `[P,2]` XY points and are never model inputs.

Generate rows for a directory whose immediate children are trajectories:

```bash
uv run python scripts/build_manifest.py /data/regnav/recon /data/regnav/manifest.jsonl \
  --dataset recon --robot jackal --environment park \
  --date 2026-01-01 --sample-period 0.5
```

For RECON, derive each recording date from its trajectory ID:

```bash
uv run python scripts/build_manifest.py /home/ubuntu/data/regnav/processed/recon-smoke \
  /home/ubuntu/data/regnav/manifest-smoke.jsonl \
  --dataset recon --robot jackal --environment recon \
  --date-from-trajectory --sample-period 0.5
```

## SCAND conversion

SCAND provides wheeled Jackal and legged Spot demonstrations with camera
images and measured odometry. Convert each bag separately so `robot` remains
explicit in the manifest and in training/evaluation domain labels:

```bash
uv run --group data python scripts/convert_scand.py \
  /home/ubuntu/data/regnav/raw/scand-jackal-gdc-2021-10-29-11.bag \
  /home/ubuntu/data/regnav/processed/scand \
  /home/ubuntu/data/regnav/manifest-scand-jackal.jsonl \
  --robot jackal --date 2021-10-29 --environment ut-campus \
  --source-url https://doi.org/10.18738/T8/0PRYRH

uv run --group data python scripts/convert_scand.py \
  /home/ubuntu/data/regnav/raw/scand-spot-ballstructure-2021-11-10-60.bag \
  /home/ubuntu/data/regnav/processed/scand \
  /home/ubuntu/data/regnav/manifest-scand-spot.jsonl \
  --robot spot --date 2021-11-10 --environment ut-campus \
  --source-url https://doi.org/10.18738/T8/0PRYRH
```

The converter uses measured `nav_msgs/Odometry` poses interpolated at image
timestamps. It records the official SCAND DOI, source SHA-256, selected image
and odometry topics, and robot in `*-provenance.json`. Manifest rows use
`dataset=scand` with `robot=jackal` or `robot=spot`; training, evaluation, and
visualization expose these as `scand/jackal` and `scand/spot`.

The ROS bag dependency is optional and installed only for conversion:

```bash
uv run --group data python scripts/convert_scand.py --help
```

Combine manifests outside the repository when desired:

```bash
cat /home/ubuntu/data/regnav/manifest-recon.jsonl \
    /home/ubuntu/data/regnav/manifest-scand-jackal.jsonl \
    /home/ubuntu/data/regnav/manifest-scand-spot.jsonl \
  > /home/ubuntu/data/regnav/manifest-recon-scand.jsonl
```
