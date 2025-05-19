# VLM-enhanced Deformable DETR

This repository presents a multimodal tumor detection framework that integrates clinical text with imaging data using Vision-Language Models (VLMs). The model is optimized for breast cancer screening applications on the VinDr-Mammo [Link](https://physionet.org/content/vindr-mammo/) dataset.

## Proposed Architecture:

![arch](https://github.com/user-attachments/assets/2d242a1e-95f2-4f48-b0f2-02b2e1431ca8)

## 🌿 Project Structure: Two Development Branches

This project is organized into two core branches, the multimodal feature learning and the detection architecture.

### 🔹 `vlm` Branch
Focuses on building the vision-language model:
- Processes imaging dataset and clinical metadata.
- Trains a contrastive model using **ClinicalBERT** and **EfficientNet-B5**.
- Outputs aligned visual and textual embeddings.

👉 Use this branch if you're interested in the **multimodal representation learning pipeline**.

### 🔹 `main` Branch
Implements object detection using the multimodal encoder:
- Integrates the trained EfficientNet-B5 encoder into **Deformable DETR** as a custom backbone.

👉 Use this branch if you're focused on **transformer-based localization**.

## ⚙️ Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
