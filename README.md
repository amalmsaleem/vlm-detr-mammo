# VLM-enhanced Deformable DETR

This repository presents a multimodal tumor detection framework that integrates clinical text with imaging data using Vision-Language Models (VLMs). The model is optimized for breast cancer screening applications on the [VinDr-Mammo](https://physionet.org/content/vindr-mammo/) dataset.

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

## ⚙️ Environment Setup & Training
For the main branch, the setup is the same as the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) repository. I would recommend going through it for setup and training.

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

## Results

![calc](https://github.com/user-attachments/assets/8fcba737-44e4-4475-b6e9-cc4bb0ab9ca2)
![mass](https://github.com/user-attachments/assets/5d143999-f54d-420a-a62a-03bb1de54d81)

## FAQ
**Q:** What is the point of this project? **A:** This was supposed to be my Master's thesis, maybe a good conference paper. It was like my child. But alas, like many children, it turned out to be a disappointment. But damn, I learned a lot raising it.
**Q:** Why isn't it published? **A:** I found a paper which was doing something VERY similar (and better).
**Q:** Why not change/improve the methodology? **A:** We started with breast ultrasounds. During the initial experimentation, I tried EVERYTHING on this planet from SAM models to adapters to transformers to LLAVA and reached to this method. We soon realised that the ultrasound dataset was too small to train a foundation model. So, we moved towards mammograms, by which time I found the other paper. Then, we tried to apply  this to chest x-rays. After the initial literature review for x-rays, and preprocessing, I had exactly three weeks to my final dissertation presentation. No results, just deadlines.
**Q:** Why is there no benchmark or ablation study to compare to other works? **A:** Girl.
**Q:** Why not work on this project after graduation? Go for a PhD even? **A:** I would rather drink acid.
**Q:** What did you learn from this? **A:** EVERYTHING related to VLMs, LLMs, transformers, ViT-based encoders, prompt encoders, text encoders, I was going through codes from Meta to random GitHub repos like a half-feral researcher, desperately trying to stitch together the sacred texts of the multimodal gods. I can confidently say I have implemented EVERYTHING that came out in 2024. And that this is a VERY fast paced research area, if you don't start with a proper baseline, you will end up wasting a lot of time.
