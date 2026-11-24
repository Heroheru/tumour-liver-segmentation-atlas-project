

# ATLAS Project: Liver and Liver-Tumour Segmentation in T1 CE-MRI

A 3D segmentation pipeline written from scratch in PyTorch, from preprocessing to evaluation, for the ATLAS challenge (liver and liver-tumour segmentation in contrast-enhanced MRI of patients with unresectable hepatocellular carcinoma).

The pipeline uses the configuration that nnU-Net derived for this dataset (target spacing, patch size, network settings) as a **reference**, but all code was written and trained independently, so that every step is understood. A second network (Res-SE blocks + multi-scale attention) was implemented and compared with the baseline 3D U-Net.

The trained model is intended to be integrated into a cloud-based GUI (joint project with the Cloud Computing course).


---

## Dataset

- From Atlas Challeng Dataset (Quinton et al, 2023) https://doi.org/10.3390/data8050079  with liver and tumour masks.
- In this project the amount of training set is 52 data and for validation set is 8 volumetric nifti data
- File naming: images `imXX.nii.gz`, masks `lbXX.nii.gz`. Labels: `0` background, `1` liver, `2` tumour.
- Our split of the 60 training cases: **52 for training, 8 for validation** (fixed seed, stored in `split.json` of each run).

The data is not included in this repository.

## Repository structure

```
Data_Processing/
    preprocess.py                  crop, resample, z-score, save .npy
    dataset.py                     patch sampling, foreground oversampling, augmentation
    visualisasi_pengembangan.py    TODO: describe (visualisation of development)
    check_augment.png              sanity check: image and mask stay aligned after augmentation
modelling/
    model.py                       baseline 3D U-Net
    model_rmau.py                  Res-SE-MAB 3D U-Net (Res-SE blocks + multi-scale attention)
    train.py                       training and validation loop, checkpoints, resume
    inference.py                   sliding-window inference back to the native image space
    check_pred.py                  diagnostics on patch-level predictions
    test_data/                     example images for testing inference
pred/
    baseline/  resmab/             predicted masks (NIfTI) on validation cases
runs/
    percobaan_ccub_rmau/           training outputs (log.csv, split.json, checkpoints) _> THIS IS .pth Model location link
requirements.txt
```

## Installation

```bash
conda create -n atlas_project python=3.11 -y
conda activate atlas_project
pip install -r requirements.txt
```

Training was done on a GPU node of the CCUB cluster (CUDA, mixed precision). 

## Usage

All commands are run from the folder of the script. Run `python <script>.py --help` for all options.

**1. Preprocess** (offline, once):

```bash
cd Data_Processing
python preprocess.py --images /path/to/imagesTr --labels /path/to/labelsTr --out ../preproc
```

**2. Smoke test, then train:**

```bash
cd ../modelling
python train.py --data ../preproc --smoke
python train.py --data ../preproc --out ../runs/baseline --epochs 50 --iters 250 --n_val 8
python train.py --data ../preproc --out ../runs/resmab --arch resmab --epochs 50 --iters 250 --n_val 8
# resume an interrupted run with the same arguments plus --resume
```

**3. Inference on one image** (raw NIfTI in, NIfTI mask out, same geometry as the input):

```bash
python inference.py --inp /path/to/im4.nii.gz --out ../pred/im4.nii.gz --ckpt ../runs/baseline/best.pt
```

**4. Evaluation** with the instructors' script (`calculate_metrics.py`, not included here). Prediction files must be named `imXX.nii.gz` and label files `lbXX.nii.gz`:

```bash
python calculate_metrics.py --segmentation_folder pred/resmab --label_folder eval/labels \
  --csv_output_file eval/resmab.csv --json_output_file eval/resmab.json
```

## Method

**Preprocessing.** Crop to the non-zero region, resample to 3.0 × 1.04 × 1.04 mm (z, y, x; image: cubic spline, mask: nearest neighbour), per-image z-score normalisation.

**Training.** Random patches of 48 × 192 × 224, batch size 2, about one third of the patches forced to contain foreground. Augmentation: rotation and scaling, Gaussian noise and blur, brightness, contrast, gamma, mirroring. Loss: soft Dice (per sample and class) + cross-entropy. SGD with Nesterov momentum 0.99, initial learning rate 0.01 with polynomial decay, gradient clipping at 12, mixed precision on CUDA. One epoch = 250 iterations.

**Networks.**

| | Baseline | Res-SE-MAB |
|---|---|---|
| Levels | 4 (3 max-pooling steps) | 5 (4 max-pooling steps) |
| Block | 2 × conv 3×3×3 + InstanceNorm + LeakyReLU | Residual block with squeeze-excitation |
| Extra | none | Multi-scale attention block (dilated convolutions 1, 2, 4, 8, channel then spatial attention) at the bottleneck and before the output |



## Results

Official metrics on **one validation case (im4)**, computed with the instructors' script. "Liver" is liver and tumour together (labels 1 and 2), as defined by the script. Distances are in mm.

| Metric | Liver, baseline | Liver, Res-SE-MAB | Tumour, baseline | Tumour, Res-SE-MAB |
|---|---|---|---|---|
| Dice (%) | 93.4 | **94.9** | 43.9 | **58.8** |
| Surface Dice 5 mm (%) | 90.2 | **93.4** | 45.2 | **60.8** |
| ASSD (mm) | 2.4 | **2.0** | 9.1 | **6.4** |
| Hausdorff (mm) | 147.3 | **56.7** | 75.2 | **68.8** |

For Res-SE-MAB on this case, tumour precision is 0.86 and recall 0.45: the tumour is under-segmented and small lesions are missed.

During training (baseline, epoch 50) the pooled patch-level validation Dice was about 0.85 for the liver and 0.78 for the tumour. These numbers are computed on patches and are not comparable to per-patient metrics.

TODO: replace the table with the mean and standard deviation over all 8 validation cases once they are evaluated.

