# RegNav Training Acceleration Design

## Goal

Reduce RTX 3060 training time without changing the RegNav architecture, dataset, loss, or acceptance thresholds. Train for 15 epochs first and continue to 30 only when validation results require it.

## Approach

Benchmark the current FP16 path against BF16 and `torch.compile` on real RECON batches. Keep the fastest stable precision; do not assume BF16 is faster. Add only proven, semantics-preserving optimizations: optional model compilation plus pinned-memory, persistent-worker, and non-blocking CUDA transfer settings when benchmarks show a gain.

Use batch 128 and 12 workers as the starting point. Reject configurations that run out of memory after LoRA activates or provide negligible speedup at the cost of unsafe VRAM headroom.

## Staged Training

1. Train epochs 1–15 and retain `best.pt` and `last.pt`.
2. Evaluate `best.pt` on the validation split.
3. If validation meets baseline acceptance, proceed to the test split without further training.
4. Otherwise preserve the 15-epoch best checkpoint and resume optimizer state from `last.pt` through epoch 30.
5. Evaluate the final selected checkpoint on the test split exactly once, then benchmark CUDA batch-1 latency.

This avoids using the held-out test split to decide training duration.

## Verification

- Run focused and full tests for any training-loop changes.
- Compare steady-state samples/second and peak VRAM after warm-up.
- Require finite losses and successful checkpoint reload before the long run.
- Report unavailable privileged collision metrics as unavailable because RECON has no obstacle labels; do not treat missing labels as a pass.

## Alternatives Considered

- Always train 30 epochs: simplest but may waste roughly half the runtime.
- Reduce input resolution: likely faster, but changes the model input and may reduce accuracy, so it is excluded.
- Force BF16: supported by the RTX 3060, but the current path is FP16 and dtype will be selected by measurement.
