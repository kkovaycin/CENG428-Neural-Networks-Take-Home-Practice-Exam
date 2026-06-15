# CENG428 Neural Networks Take-Home Practice Exam

## Synthetic Shape Segmentation on MS COCO Images

This project implements a CNN-based semantic segmentation pipeline for detecting synthetic shapes added to natural MS COCO images.

The project uses MS COCO 2017 images as natural image backgrounds. The original COCO object labels are not used as target labels. Instead, synthetic shapes are added automatically to the images, and binary segmentation masks are generated from these shapes.

The main task is binary semantic segmentation:

* `0`: background
* `1`: synthetic shape pixel

## Project Structure

The project consists of three Jupyter notebooks:

```text
01_data_creation.ipynb
02_model_training.ipynb
03_model_testing.ipynb
```

A separate text file is also provided:

```text
links.txt
```

The `links.txt` file contains public links for:

1. the generated dataset zip file
2. the trained model file used by the testing notebook

## Notebook Descriptions

### 1. `01_data_creation.ipynb`

This notebook prepares the dataset.

Main steps:

* Loads MS COCO 2017 using `torchvision.datasets.CocoDetection`
* Checks COCO folder structure and annotation files
* Creates fixed train, validation, and test splits
* Adds synthetic shapes to COCO images
* Generates binary segmentation masks automatically
* Saves generated images and masks under `generated_data/`

The split sizes are:

```text
Train:      5000 images
Validation: 1000 images
Test:       1000 images
```

The generated dataset structure is:

```text
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

### 2. `02_model_training.ipynb`

This notebook trains a CNN-based segmentation model.

Main steps:

* Loads generated train and validation image/mask pairs
* Creates PyTorch Dataset and DataLoader objects
* Defines a small U-Net style encoder-decoder CNN
* Trains the model using a combination of:

  * Binary Cross Entropy Loss with logits
  * Dice Loss
* Monitors validation metrics
* Saves the best model according to validation Dice score

The saved model file is:

```text
models/best_unet_synthetic_shapes.pth
```

The main metrics used during training are:

* IoU / Jaccard Index
* Dice Score
* Foreground Precision
* Foreground Recall

### 3. `03_model_testing.ipynb`

This notebook evaluates the trained model on the test set.

Main steps:

* Loads the generated test image/mask pairs
* Loads the trained U-Net model
* Computes final test metrics
* Generates test prediction visualizations
* Evaluates a simple image-processing baseline
* Compares the trained model with the baseline

The final evaluation metrics are:

* Test IoU
* Test Dice
* Test Precision
* Test Recall

## Model Architecture

The project uses a small U-Net style CNN architecture.

The model includes:

* convolution blocks
* batch normalization
* ReLU activations
* max pooling
* transposed convolution upsampling
* skip connections

The input shape is:

```text
[batch_size, 3, 256, 256]
```

The output shape is:

```text
[batch_size, 1, 256, 256]
```

## Synthetic Data Generation

Synthetic shapes are added to COCO images to create a segmentation task.

The generator includes different shape types such as:

* circles
* rectangles
* triangles
* ellipses
* polygons
* line segments

To make the task nontrivial, several difficulty mechanisms are used:

* random opacity
* low-contrast colors
* colors sampled from image statistics
* additive noise
* random blur
* overlapping shapes
* positive and negative samples

Positive images contain 1 to 3 synthetic shapes. Negative images contain no target synthetic shape, and their masks are all zeros.

## Baseline

A simple grayscale thresholding baseline is implemented in the testing notebook.

This baseline does not learn from data. It uses image intensity differences to estimate foreground regions.

The trained U-Net model is compared against this baseline using the same test metrics.

## Requirements

The main libraries used in this project are:

```text
torch
torchvision
numpy
pandas
matplotlib
pillow
tqdm
pycocotools
```

## How to Run

The notebooks should be run in the following order:

```text
1. 01_data_creation.ipynb
2. 02_model_training.ipynb
3. 03_model_testing.ipynb
```

Before submission, each notebook should be cleared, restarted, run from beginning to end, and saved.

## External Files

The COCO dataset itself is not included in the submission zip.

The generated dataset zip and trained model file are provided through public links in:

```text
links.txt
```

## Submission Contents

The final submission zip contains:

```text
01_data_creation.ipynb
02_model_training.ipynb
03_model_testing.ipynb
links.txt
README.md
```

The `links.txt` file contains:

```text
Generated Train/Validation/Test Dataset Zip:
https://drive.google.com/drive/folders/1lYDS_98PThXEGJMxjxjAGlm7Xn9tWhlg?usp=sharing

Trained Model File:
https://drive.google.com/file/d/1ZnHwHsjT3kkNGOOgxinzWKFxQMqtUaAM/view?usp=sharing
```
