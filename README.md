# Neural Jacobian Fields
*Sizhe Lester Li, Annan Zhang, Boyuan Chen, Hanna Matusik, Chao Liu, Daniela Rus, and Vincent Sitzmann*

[[Project Website]](https://sizhe-li.github.io/publication/neural_jacobian_field/)[[Paper]](https://arxiv.org/abs/2407.08722)[[Video]](https://youtu.be/dFZ1RvJMN7A)

[TL;DR] Neural Jacobian Fields are a general-purpose representation of robotic systems that can be learned from perception.

<!-- insert some visualization -->
https://github.com/user-attachments/assets/62786f4c-94e9-48b0-a924-d50d92aaa0a9

# Announcements
- **[03/23/25] Major updates**: we added tutorials on training 2D Jacobian Fields in simulations, and the codes are now highly modularized and clean.

  
# Quickstart
We provide python implementations of
- 3D Jacobian Fields: `project/neural_jacobian_field`
- 2D Jacobian Fields: `project/jacobian`
- A customized mujoco simulator [[github repo]](https://github.com/sizhe-li/mujoco-phys-sim.git) for simulated experiments in 2D and 3D: `mujoco-phys-sim`


## Installation 

### Prerequisites
You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. 

### Create environment
```bash
conda create --name neural-jacobian-field python=3.10.8
conda activate neural-jacobian-field
```

### 1. Install dependencies 
For CUDA 11.8:
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/sizhe-li/nerfstudio.git
```


### 2. Install customized mujoco wrapper for simulated experiments [[github repo]](https://github.com/sizhe-li/mujoco-phys-sim.git)

```bash
git submodule update --init --recursive
```

Please check out the installation guide inside `mujoco-phys-sim` repository. We provide a brief installation guide here.

```bash
cd mujoco-phys-sim/phys_sim
pip install -r requirements.txt
pip install -e .
```


### 3. Install Jacobian Fields

<!-- (TODO @ Lester) update the description -->
```bash
cd project
pip install -r requirements_new.txt
python3 -m pip install -e .
```

# How to run

## A. Reproducing simulated experiments (30 mins)
![FingerExample](https://github.com/user-attachments/assets/3cd3014c-a755-47e8-9375-f84e2a4bc542)

**1. Warm-up: Training 2D Jacobian Fields**: please follow the following notebooks
- `notebooks/tutorial/1_training_pusher_jacobian_in_2D.ipynb`
- `notebooks/tutorial/2_training_finger_jacobian_in_2D.ipynb`
- (Upcoming) `notebooks/tutorial/3_controlling_robots_with_2D_jacobian_fields.ipynb`
- (Upcoming) `notebooks/tutorial/4_jacobian_fields_in_3D.ipynb`


## B. Reproducing real-world experiments 

### Demos that are directly runnable (no need to train anything!)
We show how to visualize the learned Jacobian fields and solve for robot commands via inverse dynamics.
- **1. Visualize Jacobian Fields:** `notebooks/1_visualize_jacobian_fields.ipynb`
- **2. Inverse Dynamics Optimization:** `notebooks/2_inverse_dynamics_optimization.ipynb`
- **3. Deployment on real-robot** Upcoming by mid-April 2025!

### Dataset requirements

Our Jacobian Fields were trained with our multi-view robot dataset [[paper]](https://arxiv.org/abs/2407.08722). Our dataset includes a pneumatic robot hand (mounted on a robot arm), the Allegro robot hand, the Handed Shearing Auxetics platform, and the Poppy robot arm. We will release all our robot data to encourage future research endeavors. 

**We are actively working on uploading our dataset to the web. Updates on this coming very soon!** Please contact sizheli@mit.edu if you need it urgently.

### Pre-trained checkpoints

You can find pre-trained checkpoints for **Allegro Hand** and **Toy Arm** inside `notebooks/inference_demo_data/pretrained_ckpts`.

### Training

The main entry point is `project/neural_jacobian_field/train.py`. Call it via:

```bash
python3 -m neural_jacobian_field.train dataset.mode=perception 
```

- To reduce memory usage, you can change the batch size as follows: `training.data.batch_size=1`
- Our code supports multi-GPU training. The above batch size is the per-GPU batch size.

### Camera Conventions

Our extrinsics are OpenCV-style camera-to-world matrices. This means that +Z is the camera look vector, +X is the camera right vector, and -Y is the camera up vector. Our intrinsics are normalized, meaning that the first row is divided by image width, and the second row is divided by image height.


## BibTeX

Please consider citing our work if you find that our work is helpful for your research endeavors :D

```
@misc{li2024unifying3drepresentationcontrol,
      title={Unifying 3D Representation and Control of Diverse Robots with a Single Camera}, 
      author={Sizhe Lester Li and Annan Zhang and Boyuan Chen and Hanna Matusik and Chao Liu and Daniela Rus and Vincent Sitzmann},
      year={2024},
      eprint={2407.08722},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.08722}, 
}
```

## Acknowledgements

The authors thank Hyung Ju Terry Suh for his writing suggestions (system dynamics) and Tao Chen and Pulkit Agrawal for their hardware support on the Allegro hand.
V.S. acknowledges support from the Solomon Buchsbaum Research Fund through MIT’s Research Suppport Committee. 
S.L.L. was supported through an MIT Presidential Fellowship. 
A.Z., H.M., C.L., and D.R. acknowledge support from the National Science Foundation EFRI grant 1830901 and the Gwangju Institute of Science and Technology.
