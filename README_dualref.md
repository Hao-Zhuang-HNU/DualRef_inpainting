# TSR Dual-Reference Patch

This patch keeps the original official TSR baseline untouched and adds a **dual-reference training path**:

- **global reference**: the first frame in the same sequence
- **local reference**: the previous frame in the same sequence
- **reference encoding**: shared encoder + reference type embeddings
- **fusion**: axial self-attention on the current frame, then global cross-attention over concatenated reference tokens

## Added files

- `TSR_train_dualref.py`
- `datasets/dataset_TSR_dualref.py`
- `src/TSR_trainer_dualref.py`
- `src/models/transformer_dualref.py`
- `src/models/TSR_model_dualref.py`

## Dataset assumption

The image list in `--data_path` / `--validation_path` should contain frame paths whose **parent directory identifies the sequence**.
Frames are sorted lexicographically inside each sequence.

## Reference streams

- Global reference: first frame of each sequence
- Local reference: previous frame of each sequence
- For the first frame, the local reference is zero-padded and masked out

## Training entry

Run the new entry instead of the original one:

```bash
python TSR_train_dualref.py \
  --GPU_ids 0 \
  --data_path ./filter_results/DL3DV10K_filtered_line40/train_imgs_clean.txt \
  --validation_path ./filter_results/DL3DV10K_filtered_val/train_imgs_clean.txt \
  --train_line_path ./filter_results/DL3DV10K_filtered_line40/train_pkls_clean.txt \
  --val_line_path ./filter_results/DL3DV10K_filtered_val/train_pkls_clean.txt \
  --mask_path ./data_list/irregular_mask_40_50.txt \
  --valid_mask_path ./data_list/irregular_mask_30_40.txt \
  --batch_size 8 --train_epoch 20 --AMP
```

## Optional ablations

- `--no_global_ref`: disable the first-frame reference stream
- `--no_local_ref`: disable the previous-frame reference stream

