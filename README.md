# CENG428 Neural Networks Take-Home Practice Exam

## Synthetic Shape Segmentation on MS COCO Images

This project implements a CNN-based semantic segmentation pipeline for detecting synthetic shapes added to natural MS COCO images.

COCO images are used only as natural image backgrounds. The original COCO object labels are not used. Synthetic shapes are added automatically and binary segmentation masks are generated from these shapes.

The main task is binary semantic segmentation:

- `0`: background pixel
- `1`: synthetic shape pixel

---

## Project Structure

```
01_data_creation.ipynb
02_model_training.ipynb
03_model_testing.ipynb
links.txt
README.md
```

The `links.txt` file contains public links for:

1. The generated dataset zip file
2. The trained model file used by the testing notebook

---

## Notebook Descriptions

### `01_data_creation.ipynb`

Prepares the dataset.

- Loads MS COCO 2017 using `torchvision.datasets.CocoDetection`
- Checks COCO folder structure and annotation files
- Creates fixed, reproducible train/validation/test splits (sorted image IDs)
- Adds synthetic shapes to COCO images
- Generates binary segmentation masks automatically
- Saves generated images and masks under `generated_data/`

Split sizes:

| Split      | Size   |
|------------|--------|
| Train      | 5,000  |
| Validation | 1,000  |
| Test       | 1,000  |

Train images: first 5,000 from `train2017` (sorted by image ID).  
Validation images: first 1,000 from `val2017` (sorted by image ID).  
Test images: next 1,000 from `val2017`.

Generated dataset structure:

```
generated_data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Images and masks are saved as `.png` files.

A fixed global seed (`GLOBAL_SEED = 2025`) is used to ensure validation and test samples are reproducible. Training samples are generated randomly to increase diversity. For val/test splits, a per-image deterministic seed is derived using SHA-256 hashing of `split_name + image_id + global_seed`.

### `02_model_training.ipynb`

Trains the CNN-based segmentation model.

- Loads generated train and validation image/mask pairs
- Creates PyTorch Dataset and DataLoader objects
- Defines a small U-Net style encoder-decoder CNN (`SmallUNet`)
- Trains using a combination of BCE with logits loss and Dice loss
- Saves the best model based on validation Dice score

Training hyperparameters:

| Parameter     | Value  |
|---------------|--------|
| Image size    | 256×256 |
| Batch size    | 8      |
| Optimizer     | Adam   |
| Learning rate | 1e-3   |
| Epochs        | 15     |

Saved model files:

```
models/best_unet_synthetic_shapes.pth   ← best validation Dice
models/last_unet_synthetic_shapes.pth   ← last epoch
```

Training metrics tracked per epoch:

- Loss (BCE + Dice)
- IoU / Jaccard Index
- Dice Score
- Foreground Precision
- Foreground Recall

Training history is saved as `results/metrics/training_history.csv`.

### `03_model_testing.ipynb`

Evaluates the trained model on the held-out test set.

- Loads the generated test image/mask pairs
- Loads the best trained U-Net model
- Computes final test metrics
- Generates prediction visualizations (12 samples: image / ground truth / prediction)
- Evaluates a simple grayscale thresholding baseline
- Compares the trained model against the baseline

Final evaluation metrics:

- Test IoU
- Test Dice
- Test Precision
- Test Recall

Comparison results are saved as `results/test_results/model_vs_baseline.csv`.

---

## Model Architecture

The model is a small U-Net style encoder-decoder CNN (`SmallUNet`, `base_channels=32`).

```
Input: [batch_size, 3, 256, 256]

Encoder:
  input_conv  → DoubleConv(3, 32)
  down1       → MaxPool2d + DoubleConv(32, 64)
  down2       → MaxPool2d + DoubleConv(64, 128)
  down3       → MaxPool2d + DoubleConv(128, 256)
  bottleneck  → MaxPool2d + DoubleConv(256, 512)

Decoder (with skip connections):
  up3 → ConvTranspose2d + DoubleConv(512→256, skip=256, out=256)
  up2 → ConvTranspose2d + DoubleConv(256→128, skip=128, out=128)
  up1 → ConvTranspose2d + DoubleConv(128→64,  skip=64,  out=64)
  up0 → ConvTranspose2d + DoubleConv(64→32,   skip=32,  out=32)

output_conv → Conv2d(32, 1, kernel_size=1)

Output: [batch_size, 1, 256, 256]  (raw logits)
```

Each `DoubleConv` block: Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU.

---

## Synthetic Data Generation

Synthetic shapes are added to COCO images to create a segmentation task. Shape types include:

- circles
- rectangles
- triangles
- ellipses
- polygons
- line segments

To make the task nontrivial, the following difficulty mechanisms are applied:

- random opacity
- low-contrast colors sampled from image statistics
- additive noise
- random blur
- overlapping shapes

**Positive samples** contain 1–3 synthetic shapes with non-zero binary masks.  
**Negative samples** contain no synthetic shapes; their masks are all zeros.

---

## Baseline

A simple grayscale thresholding baseline is implemented in the testing notebook for comparison. It requires no training and uses per-image intensity statistics to flag foreground regions:

```python
gray = image.mean(dim=1, keepdim=True)
prediction = (|gray - mean| > 1.2 * std)
```

The U-Net model is compared against this baseline on the same test metrics.

---

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
pillow
tqdm
pycocotools
```

Python 3.x, tested on Google Colab (GPU recommended).

---

## How to Run

The notebooks must be run in order:

```
1. 01_data_creation.ipynb   → generates images, masks, splits
2. 02_model_training.ipynb  → trains the U-Net model
3. 03_model_testing.ipynb   → evaluates model, compares with baseline
```

Each notebook requires Google Drive to be mounted at `/content/drive`.  
The COCO dataset must be downloaded separately and placed at the expected path.

Before submission: clear all outputs, restart kernel, run from top to bottom, save.

---

## External Files

The COCO dataset is not included in the submission.

The generated dataset and trained model are shared via public links in `links.txt`:

```
Generated Dataset (Train/Val/Test):
https://drive.google.com/drive/folders/1lYDS_98PThXEGJMxjxjAGlm7Xn9tWhlg?usp=sharing

Trained Model File:
https://drive.google.com/file/d/1ZnHwHsjT3kkNGOOgxinzWKFxQMqtUaAM/view?usp=sharing
```

---

## Submission Contents

```
01_data_creation.ipynb
02_model_training.ipynb
03_model_testing.ipynb
links.txt
README.md
```
