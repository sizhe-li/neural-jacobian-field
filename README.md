# Neural Jacobian Fields

**Sizhe Lester Li, Annan Zhang, Boyuan Chen, Hanna Matusik, Chao Liu, Daniela Rus, Vincent Sitzmann**  
📄 [**Paper** (Nature, 2025)](https://www.nature.com/articles/s41586-025-09170-0) | 🌐 [**Project Website**](https://sizhe-li.github.io/publication/neural_jacobian_field/) | 📖 [**Tutorial**](https://sizhe-li.github.io/blog/2025/jacobian-fields-tutorial/) | 🎥 [**Explainer**](https://youtu.be/dFZ1RvJMN7A) | 📦[**Dataset**](https://huggingface.co/datasets/sizhe-lester-li/neural-jacobian-field)

> [TL;DR] Neural Jacobian Fields are a general-purpose representation of robotic systems that can be learned from perception.

<img width="960" alt="explainer" src="https://github.com/user-attachments/assets/32a8bec9-fee7-4338-ab74-8ffe08fef75a" />

---

## 📢  Announcements

- **[2025-09-23]** Added [FAQ](#faq) about training time and supervision types.
- **[2025-08-29]** Released the [Allegro-Hand-Only Dataset](https://huggingface.co/datasets/sizhe-lester-li/neural-jacobian-field-allegro-only) — a lighter version containing only the Allegro Hand, making it much faster to download.
- **[2025-06-25]** Our paper is now published in [**Nature**](https://www.nature.com/articles/s41586-025-09170-0).
- **[2025-04-20]** Dataset now live on HuggingFace: [Link](https://huggingface.co/datasets/sizhe-lester-li/neural-jacobian-field).
- **[2025-03-23]** Major tutorial updates for training in 2D simulations.

---

## 🚀 Quickstart

We provide the software implementations of:
- 🧠 3D Jacobian Field: `project/neural_jacobian_field`  
- ✋ 2D Jacobian Field: `project/jacobian`  
- 🧪 Custom simulator: [`mujoco-phys-sim`](https://github.com/sizhe-li/mujoco-phys-sim.git)

### 📦 Installation

#### 1. Create Conda Environment

```bash
conda create --name neural-jacobian-field python=3.10.8
conda activate neural-jacobian-field
```

#### 2. Install Dependencies (CUDA 11.8)

```
bash install.sh
```

## ▶️ Running the Code
### 📥 Download Pretrained Checkpoints

Download from [Google Drive](https://drive.google.com/drive/folders/1fq0nngkeRWhCJ_CAyzQopYda20Zu-Zu8?usp=drive_link) and place them under:

```
notebooks/inference_demo_data/real_world_pretrained_ckpts
notebooks/tutorial/tutorial_pretrained_ckpts
```

## 🧪 Simulated Experiments
![FingerExample](https://github.com/user-attachments/assets/3cd3014c-a755-47e8-9375-f84e2a4bc542)

Tutorial Notebooks (2D, ~30 mins each)

- 🧩 [Tutorial 1 – 2D Pusher](https://github.com/sizhe-li/neural-jacobian-field/blob/6badf88418a4f39378dc4e708a8d0f1b3ba1b6eb/notebooks/tutorial/1_training_pusher_jacobian_in_2D.ipynb)
- ✋ [Tutorial 2 – 2D Finger](https://github.com/sizhe-li/neural-jacobian-field/blob/6badf88418a4f39378dc4e708a8d0f1b3ba1b6eb/notebooks/tutorial/2_training_finger_jacobian_in_2D.ipynb)
- 🤖 [Tutorial 3 – Finger Control](https://github.com/sizhe-li/neural-jacobian-field/blob/6badf88418a4f39378dc4e708a8d0f1b3ba1b6eb/notebooks/tutorial/3_control_demo_block_pushing.ipynb)

## 🦾 Real-World Experiments

✔️ Ready-to-Run Demos
- 📊 [Visualize Jacobian Fields](https://github.com/sizhe-li/neural-jacobian-field/blob/main/notebooks/real_world/1_visualize_jacobian_fields.ipynb)
- 🎯 [Inverse Dynamics Optimization](https://github.com/sizhe-li/neural-jacobian-field/blob/main/notebooks/real_world/2_inverse_dynamics.ipynb)

## 📦 Dataset (HuggingFace)

We provide two datasets depending on your needs:

### 1. [Allegro-Only Dataset](https://huggingface.co/datasets/sizhe-lester-li/neural-jacobian-field-allegro-only)  
✨ **Recommended** — lightweight, faster to download and work with.

Command to download: 
```
huggingface-cli download --resume-download --repo-type dataset sizhe-lester-li/neural-jacobian-field-allegro-only
```

---

### 2. [Full Neural Jacobian Field Dataset](https://huggingface.co/datasets/sizhe-lester-li/neural-jacobian-field)  
A comprehensive **multiview video-action dataset** with camera poses, containing:  
- 🤖 Pneumatic robot hand (mounted on robot arm)  
- ✋ Allegro robot hand  
- 🧩 Handed Shearing Auxetics platform  
- 🦾 Poppy robot arm  

Command to download: 
```
huggingface-cli download --resume-download --repo-type dataset sizhe-lester-li/neural-jacobian-field
```


## 🏋️‍♀️ Training

**On a 4 x A8000s server, perception training takes 1 day, and Jacobian training takes 12 hours to 1 day.**

### A. Train Perception Module (PixelNeRF)

```
python3 -m neural_jacobian_field.train dataset=dataset_allegro model=model_allegro dataset.mode=perception
```
### B. Train Jacobian Fields

Replace the `checkpoint` flag with what you have on wandb :) and start training

```
python3 -m neural_jacobian_field.train dataset=dataset_allegro model=model_allegro dataset.mode=action checkpoint.load=wandb://entity/project/usoftylr:v5
```

## 🎥 Camera Conventions
- Extrinsics: OpenCV-style camera-to-world matrices
(+Z = look vector, +X = right, –Y = up)
- Intrinsics: Normalized (row 1 ÷ width, row 2 ÷ height)

## FAQ

### Q: Training seems extremely slow (e.g., 1300 hours estimated on an NVIDIA A40). Is this normal?
Yes, everything is fine! The number of training steps in the default config (**50 million**) is somewhat arbitrary. In practice, you can stop once you see good 3D reconstruction results during stage 1 (PixelNeRF), and then move on to fitting Jacobian fields. You usually don’t need to run the full 50M steps.

---

### Q: What hardware did you use for training?
We tested training on:
- **4 × A8000s**
- **4 × A100s**

For testing on a local robot-ready PC after training, we used a **single RTX 4090**.

---

### Q: Can I train with multiple GPUs?
Yes. The training script supports multi-GPU setups. By default, the script will use all available GPUs. You can set `CUDA_VISIBLE_DEVICES` to select specific GPUs. We recommend multi-GPU for large-scale training, especially with the full dataset.

---

### Q: My run goes out-of-memory (OOM) even with small `rays_per_batch`. Why?
This usually happens if `action_supervision_type` is set to `tracks`.

- In **track supervision**, `rays_per_batch` is **ignored**.  
- Instead, the number of rays is determined by: `num_positive_samples` + `num_negative_samples`. If both values are `null` (default), the dataloader uses **all tracks** (often ~10,000 rays), which easily causes OOM.

---

### Q: What supervision type should I use for the Allegro hand?
For the Allegro hand dataset, we by default use **optical flow supervision** (via RAFT), *not* track supervision. Both types of supervision have been tested and work well.

## 📚 Citation
If you find our work useful, please consider citing us:

```
@Article{Li2025,
  author={Li, Sizhe Lester
  and Zhang, Annan
  and Chen, Boyuan
  and Matusik, Hanna
  and Liu, Chao
  and Rus, Daniela
  and Sitzmann, Vincent},
  title={Controlling diverse robots by inferring Jacobian fields with deep networks},
  journal={Nature},
  year={2025},
  month={Jun},
  day={25},
  issn={1476-4687},
  doi={10.1038/s41586-025-09170-0},
  url={https://doi.org/10.1038/s41586-025-09170-0}
}
```

## 🙏 Acknowledgements

The authors thank Hyung Ju Terry Suh for his writing suggestions (system dynamics) and Tao Chen and Pulkit Agrawal for their hardware support on the Allegro hand.
V.S. acknowledges support from the Solomon Buchsbaum Research Fund through MIT’s Research Suppport Committee. 
S.L.L. was supported through an MIT Presidential Fellowship. 
A.Z., H.M., C.L., and D.R. acknowledge support from the National Science Foundation EFRI grant 1830901 and the Gwangju Institute of Science and Technology.
