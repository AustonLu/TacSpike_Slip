# Stage 1 Data Sanity Result

Stage 1 implemented and validated HDF5 data reading, dynamic voxelization, and sample visualization.

## Code Layout

- `src/tacspike/`: reusable Python package.
- `src/tacspike/data/h5_dataset.py`: HDF5 manifest indexing, event selection, voxelization, and spatial pooling.
- `scripts/validate/`: validation and sanity-check entry points.
- `scripts/train/`: reserved for stage 2 training entry points.

## Remote Validation

Validated on `miller` with:

```text
/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0
```

Environment:

```text
/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python
```

Result files on remote:

```text
/lamport/makkapakka/jiajunlu/logs/tacspike_sanity/check_dataset_stage1.json
/lamport/makkapakka/jiajunlu/logs/tacspike_sanity/plots/train_sphere_batch_2_102_w22357_label1.png
/lamport/makkapakka/jiajunlu/logs/tacspike_sanity/plots/train_sphere_batch_2_25_w4865_label0.png
```

## Summary

- train: 1091 sequences, 20293980 windows.
- val: 234 sequences, 4173170 windows.
- test: 234 sequences, 4485259 windows.
- Random sample checks reported `mismatched_event_count=0`.
- Full voxel shape: `[20, 2, 128, 128]`.
- Default pooled shape with 4x4 spatial sum pooling: `[20, 2, 32, 32]`.

