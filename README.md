# Face Generation with Generative Adversarial Networks  
Udacity Deep Learning Nanodegree – Project 4  

![GAN pipeline](assets/gan_pipeline.png)

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository Layout](#repository-layout)  
3. [Quick Start](#quick-start)  
4. [Data](#data)  
5. [Model](#model)  
6. [Training & Re-producing Results](#training--re-producing-results)  
7. [Unit Tests](#unit-tests)  
8. [Results](#results)  
9. [Improvements & Next Steps](#improvements--next-steps)  
10. [References](#references)  
11. [License](#license)  

---

## Project Overview
The goal of this project is to build and train a **Generative Adversarial Network (GAN)** that can synthesize 64 × 64 RGB images of human faces that look realistic.  
You will:

* Implement a clean data pipeline for the (cropped & resized) **CelebA** dataset  
* Code a **DCGAN-style** *Generator* and *Discriminator* from scratch in PyTorch  
* Train, evaluate and visualise results inside a single Jupyter notebook (`dlnd_face_generation.ipynb`)  
* Pass the automated shape/value sanity-checks in `tests.py`  

**Why this matters** — GANs underpin state-of-the-art image and video synthesis (StyleGAN, Stable Diffusion, etc.).  Building one from first principles deepens your intuition for:
* adversarial game-theoretic training
* convolutional up-/down-sampling design
* practical tricks that stabilise notoriously brittle GAN objectives

The completed notebook is the deliverable submitted to Udacity for grading; this repository merely keeps the surrounding assets, lecture notes and environment files tidy.

---

## Repository Layout
```text
4_face_generation/
├─ dlnd_face_generation.ipynb     # ← main project notebook (cleaned – ≤40 KB)
├─ tests.py                       # helper tests used by the notebook
├─ environment.yml                # full conda environment (GPU or CPU)
├─ requirements.txt               # minimal pip requirements
├─ processed_celeba_small/        # ← place the dataset here (see below)
├─ assets/                        # figures used in README / notebook
├─ images/                        # lecture illustration material
├─ C_*.md, S_ALL_*.md             # course notes – optional reading
└─ ...
```
*The notebook is self-contained: opening it and running **Run All** will train the model provided the dataset is present and PyTorch can see a GPU.*

---

## Quick Start
### 1. Clone the repo
```bash
git clone https://github.com/<your-user>/4_face_generation.git
cd 4_face_generation
```

### 2. Create the Python environment  
Choose **conda** (recommended) *or* **pip**:

<details>
<summary>Conda (all dependencies, CPU/GPU)</summary>

```bash
conda env create -f environment.yml
conda activate ml        # "ml" is the env name inside the file
```
</details>

<details>
<summary>pip (only the essentials)</summary>

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
</details>

### 3. Download the pre-processed CelebA subset
Udacity hosts a 200 MB zip containing 32 600 cropped faces at 64 × 64 × 3.

```bash
wget https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip
unzip processed-celeba-small.zip
# this creates processed_celeba_small/celeba/*.jpg
```

*(Skip if the folder is already present.)*

### 4. Launch Jupyter and run the notebook
```bash
jupyter notebook dlnd_face_generation.ipynb
```
or
```bash
jupyter lab
```
Select **Kernel ▸ Restart & Run All**.  
Training progress, losses and sample outputs will display inline.

---

## Data
* **Dataset** [CelebFaces Attributes (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
* **Subset** 32 600 images, pre-cropped around faces and down-sampled to 64 × 64  
* **Normalisation** Each image is scaled to `[–1, 1]` via
  `Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))`

---

## Model
### Discriminator
DCGAN-inspired CNN  
`3×64×64 → [64,128,256,512] Conv + BatchNorm + LeakyReLU → 1×1 score → Sigmoid`

### Generator
Transpose-conv "mirror" of the discriminator  
`z ∈ ℝ^256×1×1 → ConvT blocks [512,256,128,64] + BatchNorm + ReLU → 3×64×64 → Tanh`

### Losses & Optimisation
* **Binary Cross-Entropy (BCE)** for both Generator & Discriminator  
* **Adam** (β₁ = 0.5, β₂ = 0.999)  
  *G*: lr = 2e-4  *D*: lr = 1e-4 (slower to stabilise GAN training)

Additional options (implemented but disabled by default):

* Gradient Penalty (for WGAN-GP style training)  
* Learning-rate schedulers (`StepLR`)  
* Training G twice per D step

---

## Training & Re-producing Results
Main hyper-parameters (all configurable at the top of **Training** section):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `latent_dim` | 256 | Dimension of noise vector *z* |
| `batch_size` | 128 | Increase if GPU memory allows |
| `n_epochs` | 50 | 100 + recommended for crisper faces |
| `device` | `cuda` ¦ `mps` ¦ `cpu` | Auto-detected |

Expected total run-time on a modern GPU: **≈30 min** for 50 epochs.  
Sample outputs are displayed every epoch using a fixed latent vector to show convergence.

---

## Unit Tests
`tests.py` provides three simple integrity checks:

```python
import tests, torch
from dlnd_face_generation import Generator, Discriminator   # if you copy the classes out
latent_dim = 256

# dataset comes from the notebook
tests.check_dataset_outputs(dataset)            # length, shape, range

disc = Discriminator(); gen = Generator(latent_dim)
tests.check_discriminator(disc)

# generator image shape
tests.check_generator(gen, latent_dim)
```
The notebook runs these automatically; you can invoke them separately when refactoring.

---

## Results
After ≈50 epochs you should see samples similar to:

| Epoch | Samples |
|-------|---------|
| 0 | ![epoch0](assets/Deep_Learning_ND_P4_C_1_01.png) |
| 25 | ![epoch25](assets/Deep_Learning_ND_P4_C_3_01.png) |
| 50 | ![epoch50](assets/Deep_Learning_ND_P4_C_4_03.png) |

Faces become progressively sharper; slight blurriness and mode bias remain – see *Improvements*.

---

## Improvements & Next Steps
* **Longer Training** – 150-200 epochs improves fidelity  
* **WGAN-GP loss** – uncomment the gradient penalty helpers  
* **Progressive Growing** – start at 4 × 4 and double resolution  
* **StyleGAN-style Mapping Network** – disentangle latent space  
* **Dataset Diversity** – include non-celebrity faces to mitigate bias  

---

## References
* *Generative Adversarial Nets* – Goodfellow et al., 2014  
* *DCGAN* – Radford et al., 2016  
* Udacity Deep Learning Nanodegree, Project 4 curriculum materials  

---

## License
This project is released under the MIT License – see [`LICENSE`](LICENSE) for details.  
CelebA is licensed separately by the Chinese University of Hong Kong for non-commercial research.

---
