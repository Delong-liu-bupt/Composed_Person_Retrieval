# \[NeurIPS 2025] FAFA & SynCPR: Fine-grained Adaptive Feature Alignment and a Synthetic Dataset for Composed Person Retrieval

Official PyTorch implementation of **FAFA** (Fine-grained Adaptive Feature Alignment) and the accompanying **SynCPR** synthetic dataset for Composed Person Retrieval (CPR).

---

## 📋 Table of Contents

* [Part I — FAFA Method 🔧](#part-i--fafa-method-)

  * [1. Overview 🎯](#1-overview-)
  * [2. Installation 🛠️](#2-installation-)

    * [2.1. Requirements 📎](#21-requirements-)
    * [2.2. Setup ⚙️](#22-setup-)
  * [3. Data Preparation (for FAFA) 🗂️](#3-data-preparation-for-fafa-)

    * [3.1. SynCPR (Train) 📦](#31-syncpr-train-)
    * [3.2. ITCPR (Eval) 🧪](#32-itcpr-eval-)
  * [4. Training 🚀](#4-training-)

    * [4.1. Quick Start ⚡](#41-quick-start-)
    * [4.2. Key Arguments 🔑](#42-key-arguments-)
  * [5. Inference 🔍](#5-inference-)

    * [5.1. Run ▶️](#51-run-)
    * [5.2. Outputs 📈](#52-outputs-)
    * [5.3. Pre-trained Model 💾](#53-pre-trained-model-)
* [Part II — SynCPR Dataset 🗃️](#part-ii--syncpr-dataset-)

  * [6. Overview 🧭](#6-overview-)
  * [7. Construction Pipeline 🏗️](#7-construction-pipeline-)
  * [8. Key Features ✨](#8-key-features-)
  * [9. Data Structure 📐](#9-data-structure-)
  * [10. Download ⬇️](#10-download-)
* [Acknowledgements 🙏](#acknowledgements-)
* [Citation 📝](#citation-)

---

## Part I — FAFA Method 🔧

### 1. Overview 🎯

**FAFA** addresses **Composed Person Retrieval (CPR)**, where a system retrieves a target person image conditioned on a *reference image* plus a *text description* of appearance changes. FAFA introduces:

* **FDA — Fine-grained Dynamic Alignment**
  Dynamically selects top-k fine-grained features for adaptive similarity computation.
* **FD — Feature Diversity**
  Encourages diversity among visual features to capture comprehensive person attributes.
* **MFR — Masked Feature Reasoning**
  Enhances cross-modal understanding via masked feature prediction.

---

### 2. Installation 🛠️

#### 2.1. Requirements 📎

* Python ≥ 3.8
* CUDA ≥ 11.3
* PyTorch ≥ 1.13

#### 2.2. Setup ⚙️

```bash
# Clone the repository
git clone https://github.com/Delong-liu-bupt/Composed_Person_Retrieval.git
cd Composed_Person_Retrieval/FAFA_SynCPR

# Create and activate environment
conda create -n fafa python=3.10 -y
conda activate fafa

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Data Preparation (for FAFA) 🗂️

#### 3.1. SynCPR (Train) 📦

```
/path/to/SynCPR/
├── test1/
├── test2/
├── test3/
├── test4/
│   ...
└── SynCPR.json
```

#### 3.2. ITCPR (Eval) 🧪

```
/path/to/ITCPR/
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

---

### 4. Training 🚀

#### 4.1. Quick Start ⚡

```bash
python src/blip_fine_tune_new.py \
  --dataset cpr \
  --syncpr-data-path /your/custom/syncpr/root \
  --itcpr-root /your/custom/itcpr/root \
  --json-path SynCPR.json \
  --exp-name FAFA_SynCPR_FDA_FD_MFR \
  --blip-model-name blip2_fafa_cpr \
  --setting annotations \
  --num-epochs 10 \
  --num-workers 4 \
  --learning-rate 2e-6 \
  --batch-size 256 \
  --transform squarepad \
  --save-training \
  --save-best \
  --validation-frequency 1 \
  --validation-step 500 \
  --loss-fda 1.0 \
  --loss-fd 1.0 \
  --loss-mfr 0.5 \
  --fda-k 6 \
  --fda-alpha 0.5 \
  --fd-margin 0.5
```

#### 4.2. Key Arguments 🔑

**Essential**

* `--dataset` — dataset type (use `cpr`)
* `--syncpr-data-path` — SynCPR training root
* `--itcpr-root` — ITCPR validation root
* `--exp-name` — experiment name for logging/saving

**Model**

* `--blip-model-name` — architecture (`blip2_fafa_cpr`)
* `--backbone` — vision backbone (`pretrain` for ViT-G, `pretrain_vitL` for ViT-L)
* `--num-query-token` — query tokens (default: 32)

**Optimization**

* `--batch-size` — default: 256
* `--learning-rate` — default: 2e-6
* `--num-epochs` — default: 10
* `--num-workers` — default: 2

**FDA (Fine-grained Dynamic Alignment)**

* `--fda-k` — top-k features (default: 6)
* `--fda-alpha` — soft-label strength (default: 0.5)

**FD (Feature Diversity)**

* `--loss-fd` — FD loss weight λ₁ (default: 1.0)
* `--fd-margin` — margin *m* (default: 0.5)

**MFR (Masked Feature Reasoning)**

* `--loss-mfr` — MFR loss weight λ₂ (default: 0.5)

**Data Preprocessing**

* `--transform` — `squarepad` | `targetpad` | `resize`
* `--target-ratio` — for `targetpad` (default: 1.25)

**Validation & Checkpoints**

* `--validation-frequency` — by epochs (default: 1)
* `--validation-step` — by steps (default: 1\_000\_000; set smaller to validate more often)
* `--save-training` — save checkpoints during training
* `--save-best` — keep best model
* `--save-last` — keep last model

---

### 5. Inference 🔍

#### 5.1. Run ▶️

```bash
python inference_fafa.py \
  --exp-dir output/cpr/FAFA_experiment \
  --model-name tuned_recall_at1_step.pt \
  --itcpr-root /path/to/ITCPR \
  --batch-size 256
```

Optional:

* `--device` — `cuda` or `cpu` (defaults to CUDA if available)

#### 5.2. Outputs 📈

* Retrieval metrics: **Recall\@1/5/10**, **mAP**
* Results saved to `inference_results_{model_name}.json`

#### 5.3. Pre-trained Model 💾

We also provide the **official released model weights** corresponding to the paper version.
You can directly download and use it without additional training.
Notably, its performance is **slightly higher than the numbers reported in the paper**.

👉 [Download Pre-trained Model](https://drive.google.com/file/d/10sYj3bxddFhku_OKZVcvurUPQQQxXw3Q/view?usp=sharing)

---

## Part II — SynCPR Dataset 🗃️

### 6. Overview 🧭

**SynCPR** is a large-scale, fully synthetic dataset purpose-built for **Composed Person Retrieval**. Constructed via an automated pipeline, SynCPR provides high diversity, realism, and scale for person-centric retrieval research.

![SynCPR Teaser](https://github.com/user-attachments/assets/0fc2cd5c-896c-4edb-a82b-665feca5b6e5)

---

### 7. Construction Pipeline 🏗️

1. **Textual Quadruple Generation**
   Using [Qwen2.5-70B](https://github.com/QwenLM/Qwen2.5-VL), we generate **140,500 textual quadruples**.

2. **Image Generation**
   With a fine-tuned **LoRA** ([LoRA](https://arxiv.org/abs/2106.09685)) on **Flux.1** ([Flux.1](https://github.com/black-forest-labs/flux)) we create **2.81M triplets**.

3. **Rigorous Filtering**
   After filtering, **1.15M high-quality triplets** remain, covering **177K unique GIDs**.

---

### 8. Key Features ✨

* **Diversity** — scenes, ages, attire, clarity, and appearance
* **Realism** — realism-oriented fine-tuning with advanced generative backbones
* **Scale** — over **1.15M** curated triplets with varied captions
* **Comprehensiveness** — synthetic construction enables wider attribute coverage

---

### 9. Data Structure 📐

Each sample is defined in `SynCPR.json` (see [Hugging Face dataset page](https://huggingface.co/datasets/a1557811266/SynCPR)).

**Fields:**

* `reference_caption` / `target_caption`
* `reference_image_path` / `target_image_path`
* `edit_caption`
* `cpr_id`

**Example**

```json
[
  {
    "reference_caption": "The young woman ... silver clutch.",
    "target_caption": "The young woman ... black leather handbag.",
    "reference_image_path": "test2/sub_img/img_left/10732-1_left.png",
    "target_image_path": "test2/sub_img/img_right/10732-1_right.png",
    "edit_caption": "Wearing light gray skirt, carrying a large black leather handbag.",
    "cpr_id": 0
  }
]
```

---

### 10. Download ⬇️

The SynCPR dataset is publicly available for research:
**➡️ [Hugging Face: a1557811266/SynCPR](https://huggingface.co/datasets/a1557811266/SynCPR)**

---

## Acknowledgements 🙏

This work builds upon **[LAVIS](https://github.com/salesforce/LAVIS)** and **[SPRC](https://github.com/chunmeifeng/SPRC)**. We thank the authors for their excellent contributions.

---

## Citation 📝

```bibtex
@misc{liu2025automaticsyntheticdatafinegrained,
  title         = {Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieval},
  author        = {Delong Liu and Haiwen Li and Zhaohui Hou and Zhicheng Zhao and Fei Su and Yuan Dong},
  year          = {2025},
  eprint        = {2311.16515},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2311.16515}
}
```

