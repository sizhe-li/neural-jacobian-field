from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch
import numpy as np

from jaxtyping import Float


@dataclass
class ModelConfig:
    # model-related
    model_cfg_path: Path = Path(
        "../../../../neural-jacobian-field/project/configurations/config"
    )
    model_ckpt_path: Path = Path("../notebooks/artifacts/model-lhatlmi4:v38/model.ckpt")

    # inference-related
    num_points_per_dim: int = 40
    # num_points_per_dim: int = 20
    density_threshold: int = 20
    num_actions: int = 1

    # batch optimization
    batch_size: int = 20
    num_optim_iter: int = 50
    learning_rate: int = 1e-1
    flow_multiplier: float = 1.0
    use_ot_loss: bool = False


@dataclass
class CameraConfig:
    # camera-related
    downscale_factor: int = 1
    ctxt_camera_idx: int = 0
    trgt_camera_idx: int = 0
    recording_fps: int = 10
    recording_pix_fmt: str = "bgr24"
    video_crf: int = 21
    thread_per_video: int = 2


@dataclass
class ProgramConfig:
    model_config: ModelConfig = ModelConfig()
    camera_config: CameraConfig = CameraConfig()
    device: torch.device = torch.device("cuda:0")


@dataclass
class InferenceRecord:
    desired_motion_visualization: List[Float[np.ndarray, "H W 3"]] = field(
        default_factory=list
    )
    # motion planning
    current_keypoints: List[Float[np.ndarray, "num_points 2"]] = field(
        default_factory=list
    )
    target_keypoints: List[Float[np.ndarray, "num_points 2"]] = field(
        default_factory=list
    )
    # observation collection
    current_observation: List[Float[np.ndarray, "H W 3"]] = field(default_factory=list)
    next_observation: List[Float[np.ndarray, "H W 3"]] = field(default_factory=list)
    # robot state

    current_robot_state: List[Float[np.ndarray, "action_dim"]] = field(
        default_factory=list
    )
    desired_robot_state: List[Float[np.ndarray, "action_dim"]] = field(
        default_factory=list
    )
    # robot action
    robot_action: List[Float[np.ndarray, "action_dim"]] = field(default_factory=list)
    # reconstruction from perception
    pred_depth_images: List[Float[np.ndarray, "H W 3"]] = field(default_factory=list)
    pred_flow_images: List[Float[np.ndarray, "H W 3"]] = field(default_factory=list)
    # demonstration order
    demonstration_index: List[int] = field(default_factory=list)
    action_index: List[int] = field(default_factory=list)
    action_features: List[Float[np.ndarray, "H W Channel"]] = field(
        default_factory=list
    )
    model_type: List[str] = field(default_factory=list)

    def append_record(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k).append(v)
