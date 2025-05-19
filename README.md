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

![calc](https://github.com/user-attachments/assets/4c8ee87e-4135-4209-8f80-b758235ba5d1)
![mass](https://github.com/user-attachments/assets/d57d1c62-6837-414a-af5f-bb1977c403c2)

## FAQ

**Q:** What is the point of this project?  
**A:** This was supposed to be my Master's thesis, maybe even a decent conference paper. It was like my child. But alas, like many children, it turned out to be a disappointment. But damn, I learned a lot raising it.  

**Q:** Why isn't it published? 
**A:** I found a paper which was doing something VERY similar (and better).  

**Q:** Why not change/improve the methodology?  
**A:** We started with breast ultrasounds. During the initial experimentation, I tried EVERYTHING on this planet from SAM models to adapters to transformers to LLAVA and reached to this method. We soon realised that the ultrasound dataset was too small to train a foundation model. So, we pivoted to mammograms, and found the other paper. Then we pivoted again to chest x-rays. After the initial literature review for x-rays, and preprocessing, I had exactly three weeks to my final dissertation presentation. No results, just deadlines.  

**Q:** Why is there no benchmark or ablation study to compare to other works? 
**A:** Girl.  

**Q:** Why not work on this after graduation? You could go for a PhD to further this project, right?  
**A:** It’s dead. I gave it a proper funeral. I’m not resurrecting it just to watch it die again in higher resolution.

**Q:** What did you learn from this? 
**A:** EVERYTHING related to VLMs, LLMs, transformers, ViT-based encoders, prompt encoders, text encoders, I was going through codes from Meta to random GitHub repos like a half-feral researcher, desperately trying to stitch together the sacred texts of the multimodal gods. This was a crash course in state-of-the-art 2024 computer vision and VLM research. And that this is a VERY fast-paced field, if you don't start with a solid baseline, you will end up wasting a lot of time.

**Q:** Was it a waste of time then? 
**A:** Absolutely not. It was the most frustratingly productive failure of my academic life.

**Q:** What does this say about your research approach? 
**A:** I go all in until I know exactly what’s broken and what’s worth keeping. I don’t waste time sugarcoating failure. When it’s time to pivot, I do it clean and fast, not because I quit, but because I’m playing the long game. Every setback just sharpens my focus.
