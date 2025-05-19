# VLM-enhanced Deformable DETR

This repository presents a multimodal tumor detection framework that integrates clinical text with imaging data using Vision-Language Models (VLMs). The model is optimized for breast cancer screening applications on the VinDr-Mammo [Link](https://physionet.org/content/vindr-mammo/) dataset.

## Proposed Architecture:

![arch](https://github.com/user-attachments/assets/72fc0c43-10ba-4d8d-8d8b-9257d58aa80b)


## 🌿 Project Structure: Two Development Branches

This project is organized into two core branches, the multimodal feature learning and the detection architecture.

### 🔹 `vlm` Branch
Focuses on building the vision-language model:
- Trains a contrastive model using **ClinicalBERT** and **EfficientNet-B5**.
- Outputs aligned visual and textual embeddings.

👉 Use this branch if you're interested in the **multimodal representation learning pipeline**.

### 🔹 `main` Branch
Implements object detection using the multimodal encoder:
- Integrates the trained EfficientNet-B5 encoder into **Deformable DETR** as a custom backbone.

👉 Use this branch if you're focused on **transformer-based localization**.

## ⚙️ Environment Setup
For the main branch, the setup is the same as the deformable DETR [Link](https://github.com/fundamentalvision/Deformable-DETR) repository. I would recommend going through it for setup and training.

For the VLM branch:
```
conda env create --name mammo -f environment.yml
conda activate mammo
```
Specifications:

- Python version: 3.8.18
- PyTorch version: 2.2.2
- CUDA version: 11.8

Use `python train.py --config-name pre_train_b5_clip.yaml` for training.

