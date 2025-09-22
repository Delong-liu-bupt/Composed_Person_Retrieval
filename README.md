# [NeurIPS 2025] Composed Person Retrieval and the ITCPR Benchmark

[![arXiv](https://img.shields.io/badge/Arxiv-2311.16515-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.16515)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-ITCPR-yellow?logo=huggingface)](https://huggingface.co/datasets/a1557811266/ITCPR)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-SynCPR-yellow?logo=huggingface)](https://huggingface.co/datasets/a1557811266/SynCPR)

## News

* **2023.11.16:** Repository created. Code and dataset coming soon.
* **2023.11.25:** ITCPR dataset publicly available for download.
* **2025.05.20:** The previous training and testing code has been updated and is now located in the 'old_project' folder. The new code and data will be open-sourced soon.
* **2025.05.26:** SynCPR dataset publicly available for download.
* **2025.09.18:** Our paper *“Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieval”* has been **accepted to NeurIPS 2025**! 🎉 The related training and testing code will be open-sourced within one week. 
* **2025.09.22:** All retrieval-related training and inference code has been **open-sourced**!. 
---

## Introduction: Composed Person Retrieval (CPR)

Composed Person Retrieval (CPR) is a new cross-modal retrieval task that aims to identify individuals in large-scale person image databases by combining **both a reference image and a textual description** as the query. This task is inspired by real-world needs, such as searching for a missing person using an old photograph and a new verbal description, and it bridges the gap left by traditional retrieval methods that rely on either image-only or text-only queries.

### Why CPR?

* **Limitations of Existing Methods:** Image-based and text-based retrieval each leverage only one modality, missing out on richer, complementary information available in real scenarios.
* **CPR Definition:** In CPR, each query consists of a *reference image* and a *relative caption* describing differences or changes (e.g., clothes, appearance). The goal is to retrieve the target image(s) of the same identity that best match the combined query.

## ITCPR Dataset

### Overview

The **ITCPR** dataset is a comprehensive collection specifically designed for the Zero-Shot Composed Person Retrieval (ZS-CPR) task. It consists of a total of **2,225 annotated triplets**, derived from three distinct datasets: Celeb-reID, PRCC, and LAST. To access the ITCPR dataset, please use the following download link: [ITCPR Dataset Download](https://drive.google.com/file/d/1CTKxtkrDZ1b17TF5W0Kctylu1qGJ2sd2/view?usp=sharing). 

#### Dataset Scale
- **Total Annotated Triplets**: 2,225
- **Unique Query Combinations**: 2,202
- **Total Images**: 1,151 from Celeb-reID, 146 from PRCC, 905 from LAST
- **Total Identities**: 512 from Celeb-reID, 146 from PRCC, 541 from LAST
- **Target Gallery**: 20,510 images with 2,225 corresponding ground truths

### Image Sources
The images in the ITCPR dataset are sourced from the following datasets:
- **Celeb-reID**
- **PRCC**
- **LAST**

These are utilized solely for testing purposes in the ZS-CPR task.

### Annotation Files
The dataset includes two annotation files: `query.json` and `gallery.json`.

#### `query.json` Format
Each entry in the `query.json` file follows this structure:
```json
{
    "file_path": "Celeb-reID/001/1_1_0.jpg",
    "datasets": "Celeb-reID",
    "person_id": 1,
    "instance_id": 1,
    "caption": "Wearing a brown plaid shirt, black leather shoes, another dark gray T-shirt, another blue jeans"
}
```
- `file_path`: Reference image path relative to the data root directory.
- `datasets`: Source dataset of the image.
- `person_id`: Person ID in the original dataset.
- `instance_id`: Unique identifier for gallery ground truth matching.
- `caption`: Relative caption of the reference image.

#### `gallery.json` Format
Each entry in the `gallery.json` file follows this structure:
```json
{
    "file_path": "Celeb-reID/001/1_2_0.jpg",
    "datasets": "Celeb-reID",
    "person_id": 1,
    "instance_id": 1
}
```
- `instance_id`: Matches with `query.json` for target images; -1 for non-matching query instances.
- Others: Correspond to target image path, original dataset, and person ID.

### Data Directory Structure
```
data
|-- Celeb-reID
|   |-- 001
|   |-- 002
|   |-- 003
|   ...
|-- PRCC
|   |-- train
|   |-- val
|   |-- test
|-- LAST
|   |-- 000000
|   |-- 000001
|   |-- 000002
|   ...
|-- query.json
|-- gallery.json

```

### Dataset Download and Preparation
Download and prepare the datasets as follows:

1. **Celeb-reID**: [GitHub Repository](https://github.com/Huang-3/Celeb-reID)
2. **PRCC**: [Google Drive Link](https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view?usp=sharing)
3. **LAST**: [GitHub Repository](https://github.com/shuxjweb/last)

After downloading, use the `img_process.py` script to process Celeb-reID and LAST datasets into the standard format. The PRCC (subfolder PRCC/rgb) dataset can be directly placed in the corresponding directory upon extraction.

### Acknowledgments
We are deeply thankful to the creators of the Celebrities-ReID, PRCC, and LAST datasets for their significant contributions to the field of person re-identification. Their commitment to open-sourcing these valuable resources has greatly facilitated advancements in academic and practical research.

---
- **Celebrities-ReID**: "Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification" - [View Paper](https://ieeexplore.ieee.org/document/8851957)
- **PRCC**: "Person Re-identification by Contour Sketch under Moderate Clothing Change" - [View Paper](https://arxiv.org/abs/2002.02295)
- **LAST**: "Large-Scale Spatio-Temporal Person Re-identification: Algorithms and Benchmark" - [View Paper](https://arxiv.org/abs/2105.15076)
---
## Codes

1. **Initial Solution: Inversion-Based Approach**
   Our initial attempt to address the composed person retrieval problem was based on an inversion-based solution. All related code and scripts implementing this method have now been migrated to the `old_project` folder for archival and reference (See `old_project/README.md`).

2. **Current Solution: Scalable Synthetic Data and FAFA Framework**
   Building on our research insights, we have developed a new, scalable pipeline for automatic generation of synthetic CPR data, as described in our latest work. This pipeline utilizes large language models to generate diverse textual quadruples, fine-tuned generative models for identity-consistent image synthesis, and multimodal filtering to ensure data quality. On top of this data, we propose the Fine-grained Adaptive Feature Alignment (FAFA) framework, which enhances retrieval performance through fine-grained dynamic alignment and masked feature reasoning.
   **All new code, scripts, and the latest synthetic dataset will be released under the `FAFA_SynCPR` directory.**

## Citation

If you use our code or dataset, please cite:

```bibtex
@misc{liu2025automaticsyntheticdatafinegrained,
      title={Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieval}, 
      author={Delong Liu and Haiwen Li and Zhaohui Hou and Zhicheng Zhao and Fei Su and Yuan Dong},
      year={2025},
      eprint={2311.16515},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.16515}, 
}
```
