# Neural Jacobian Fields

**Sizhe Lester Li, Annan Zhang, Boyuan Chen, Hanna Matusik, Chao Liu, Daniela Rus, Vincent Sitzmann**  
📄 [**Paper** (Nature, 2025)](https://www.nature.com/articles/s41586-025-09170-0) | 🌐 [**Project Website**](https://sizhe-li.github.io/publication/neural_jacobian_field/) | 📖 [**Tutorial**](https://sizhe-li.github.io/blog/2025/jacobian-fields-tutorial/) | 🎥 [**Explainer**](https://youtu.be/dFZ1RvJMN7A)

> [TL;DR] Neural Jacobian Fields are a general-purpose representation of robotic systems that can be learned from perception.

<img width="960" alt="explainer" src="https://github.com/user-attachments/assets/32a8bec9-fee7-4338-ab74-8ffe08fef75a" />

---

## 📢  Announcements

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
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/sizhe-li/nerfstudio.git
```

#### 3. Install Simulator

```
git submodule update --init --recursive
cd mujoco-phys-sim/phys_sim
pip install -r requirements.txt
pip install -e .
```

#### 4. Install Jacobian Fields Codebase

```
cd project
pip install -r requirements_new.txt
pip install -e .
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

## ✔️ Ready-to-Run Demos
- 📊 [Visualize Jacobian Fields](https://github.com/sizhe-li/neural-jacobian-field/blob/main/notebooks/real_world/1_visualize_jacobian_fields.ipynb)
- 🎯 [Inverse Dynamics Optimization](https://github.com/sizhe-li/neural-jacobian-field/blob/main/notebooks/real_world/2_inverse_dynamics.ipynb)

## 📦 Dataset (HuggingFace)

### [Neural Jacobian Field Dataset]((https://huggingface.co/datasets/sizhe-lester-li/neural-jacobian-field))

A multiview video-action dataset with camera poses that includes
- Pneumatic robot hand (on robot arm)
- Allegro robot hand
- Handed Shearing Auxetics platform
- Poppy robot arm


## 🏋️‍♀️ Training

### A. Train Perception Module (PixelNeRF)

```
python3 -m neural_jacobian_field.train dataset=dataset_allegro model=model_allegro dataset.mode=perception
```
### B. Train Jacobian Fields

#### Visual motion extraction
- Install TAPIR [here](https://github.com/google-deepmind/tapnet).
- Use `scripts/dataset/extract_tapir_motion_tracks.py` to extract the motion tracks of the allegro hand. 
- We are working on a simplified implementation using CoTracker for the motion data extraction process, which will be released before the end of June.


Replace the `checkpoint` flag with what you have on wandb :) and start training

```
python3 -m neural_jacobian_field.train dataset=dataset_allegro model=model_allegro dataset.mode=action checkpoint.load=wandb://entity/project/usoftylr:v5
```

## 🎥 Camera Conventions
- Extrinsics: OpenCV-style camera-to-world matrices
(+Z = look vector, +X = right, –Y = up)
- Intrinsics: Normalized (row 1 ÷ width, row 2 ÷ height)


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
