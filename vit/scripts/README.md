# Scripts

This folder contains scripts to reproduce our experimental results.

---

## Stage 1: Calibration Set Generation

Run the calibration generation scripts from the `./stage_one` directory:

- Use `s.sh`, `m.sh`, and `l.sh` for different model sizes.
- For faster ViT processing (which can be time-consuming), use the 8-GPU versions: `s_8.sh`, `m_8.sh`, and `l_8.sh`.

After generation, you may need to use `rescale_image.py` to convert the images.

---

## Stage 2: Task-Specific Distillation

Choose the appropriate folder under `./stage_two` based on your dataset:

- `exact_2k`: Scripts for distillation using exactly 2,000 images.
- `exact_full`: Scripts for distillation using the full dataset.
- `ours`: Scripts for our proposed method.

Remember to fill in your generated image path.

> **Note**: All scripts in Stage 2 are configured for 8-GPU training.
