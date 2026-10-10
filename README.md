# CIFAR-10 Image Classification with Transfer Learning (ResNet-18)

Fine-tunes a pre-trained ResNet-18 on the CIFAR-10 dataset, reaching roughly 94%
test accuracy. Built in PyTorch with data augmentation and learning-rate
scheduling.

## Overview

- **Model:** ResNet-18 pre-trained on ImageNet, fine-tuned for CIFAR-10's 10
  classes.
- **Techniques:** data augmentation, learning-rate scheduling, a fixed
  train/validation split (saved as `train_idx.npy` / `val_idx.npy` for
  reproducibility).
- **Result:** ~94% test accuracy.
- **Framework:** PyTorch + torchvision. Trained on GPU (CUDA).

## Repo contents

- `development_DL.ipynb` — the full notebook: data loading, model setup,
  training loop, evaluation, and results.
- `train_idx.npy` / `val_idx.npy` — the exact train/validation split indices.
- `requirements.txt` — Python dependencies.

The dataset and trained weights are not committed (see Setup).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the CIFAR-10 Python dataset from
   https://www.cs.toronto.edu/~kriz/cifar.html and extract it so the
   `cifar-10-batches-py/` folder sits in the project root.
3. Open and run the notebook:
   ```bash
   jupyter notebook development_DL.ipynb
   ```

## Notes

- The dataset (~160MB) and model weights (`.pth`) are gitignored. Download the
  dataset with the link above; re-run the notebook to regenerate the weights.
