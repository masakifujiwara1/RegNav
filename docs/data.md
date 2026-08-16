# Public data preparation

RegNav Phase 1 uses forward JPEG images and 2D odometry from these official sources:

- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [BeoNav](https://sites.google.com/usc.edu/beonav/)
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

## BeoNav conversion

BeoNav supplies wheeled-robot camera/control logs from the USC campus. The
converter keeps the raw ROS bag outside this repository and writes one
ViNT-style trajectory plus a JSONL row:

```bash
uv run --group data python scripts/convert_beonav.py \
  /home/ubuntu/data/regnav/raw/2023-03-27-11-12-41.bag \
  /home/ubuntu/data/regnav/processed/beonav \
  /home/ubuntu/data/regnav/manifest-beonav.jsonl \
  --source-url https://ilab.usc.edu/kiran/beonav/2023-03-27-11-12-41.bag
```

`rosbags` is optional and only needed by this converter. This bag has image,
`/cmd_vel`, wheel-velocity, and IMU topics but no odometry topic, so the
converter records a provisional target by integrating the latest
`/cmd_vel` (`linear.x`, `angular.z`) with a unicycle model. The generated
`*-provenance.json` records that assumption and the source SHA-256; do not
treat these positions as ground-truth odometry when comparing against RECON.

To train or evaluate on both sources, concatenate the JSONL manifests outside
the repository:

```bash
cat /home/ubuntu/data/regnav/manifest-recon.jsonl \
    /home/ubuntu/data/regnav/manifest-beonav.jsonl \
  > /home/ubuntu/data/regnav/manifest-recon-beonav.jsonl
```
