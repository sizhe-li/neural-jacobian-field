from dataclasses import dataclass
from typing import List, Literal, TypedDict

import numpy as np
from jaxtyping import Float
from torch.utils.data import Dataset

Stage = Literal["training", "validation", "test"]


class Trajectory(TypedDict):
    seq_rgb_obs: Float[np.ndarray, "num_steps height width channels"]
    seq_qpos: Float[np.ndarray, "num_steps command_dim"]
    seq_flow_obs: Float[np.ndarray, "num_steps 2 height width"]
    # # privileged information, only for testing purposes
    # seq_robot_mask: Float[np.ndarray, "num_steps height width 1"]
