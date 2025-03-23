import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, TypedDict

import numpy as np
import torch
from jacobian.dataset.types import Stage
from jacobian.utils.io import load_gzip_file, numpy_to_torch_image
from jacobian.utils.misc import cyan
from jaxtyping import Float
from torch.utils.data import Dataset

from .types import Trajectory


@dataclass
class DatasetPusherCfg:
    name: Literal["pusher"]
    root: Path


class DatasetPusher(Dataset):
    cfg: DatasetPusherCfg
    stage: Stage
    trajectory_paths: List[Path]
    trajectory_data: List[Trajectory]
    repeat: int = 100000

    # normalization params
    min_qpos = torch.tensor([-1.7, -1.09], dtype=torch.float32)[None]
    max_qpos = torch.tensor([0.4, 1.0], dtype=torch.float32)[None]

    # normalize the action by this scale, not important and can be set as 1.0
    action_scale: float = 4.0

    def __init__(self, cfg: DatasetPusherCfg, stage: Stage):
        assert stage in ["train", "val", "test"]

        self.cfg = cfg
        self.stage = stage

        self.trajectory_paths = list(Path((cfg.root), self.stage).glob("*.pkl"))
        print(
            cyan(
                f"Stage: {stage} | Num Files {len(self.trajectory_paths)} | Root {cfg.root}"
            )
        )

    @property
    def num_files(self) -> int:
        return len(self.trajectory_paths)

    def __len__(self) -> int:
        orig_len = self.num_files

        return orig_len * self.repeat if self.stage == "train" else orig_len

    def __getitem__(self, idx: int):
        if self.stage == "train":
            return self.get_train_item(idx)
        else:
            return self.get_val_item(idx)

    def get_train_item(self, idx: int):
        trajectory_idx = idx % self.num_files
        trajectory_path = self.trajectory_paths[trajectory_idx]

        trajectory: Trajectory = load_gzip_file(trajectory_path)

        image_sequence = trajectory["seq_rgb_obs"]
        num_frames = len(image_sequence)

        # randomly sample a frame
        rand_frame_idx = np.random.randint(num_frames - 1)

        joint_pos_sequence = torch.from_numpy(trajectory["seq_qpos"])
        joint_pos_sequence = joint_pos_sequence[..., :2]  # num_steps x command_dim

        traj_flow_sequence = torch.from_numpy(trajectory["seq_flow_obs"])
        trgt_flow_curr = traj_flow_sequence[rand_frame_idx]

        # normalize the joint_pos sequence
        joint_pos_sequence = (joint_pos_sequence - self.min_qpos) / (
            self.max_qpos - self.min_qpos
        )

        # grab the current and next frame
        frame_curr = numpy_to_torch_image(image_sequence[rand_frame_idx])
        frame_next = numpy_to_torch_image(image_sequence[rand_frame_idx + 1])

        # grab the current joint pos
        input_command = self.action_scale * (
            joint_pos_sequence[rand_frame_idx + 1] - joint_pos_sequence[rand_frame_idx]
        )

        return {
            "input_frame_curr": frame_curr.float(),
            "input_frame_next": frame_next.float(),
            "input_command": input_command.float(),
            "trgt_flow_curr": trgt_flow_curr.float(),
        }

    def get_val_item(self, idx: int):
        # trajectory_idx = random.randint(0, len(self.trajectory_paths) - 1)
        trajectory_path = self.trajectory_paths[idx]

        trajectory: Trajectory = load_gzip_file(trajectory_path)

        input_video_sequence = trajectory["seq_rgb_obs"]
        joint_pos_sequence = torch.from_numpy(trajectory["seq_qpos"])
        joint_pos_sequence = joint_pos_sequence[..., :2]  # num_steps x command_dim

        traj_flow_sequence = torch.from_numpy(trajectory["seq_flow_obs"])

        # normalize the joint_pos sequence
        joint_pos_sequence = (joint_pos_sequence - self.min_qpos) / (
            self.max_qpos - self.min_qpos
        )

        input_video_sequence = torch.stack(
            [numpy_to_torch_image(img) for img in input_video_sequence], dim=0
        )

        # grab the current joint pos
        input_command_sequence = self.action_scale * (
            joint_pos_sequence[1:] - joint_pos_sequence[:-1]
        )

        return {
            "input_video_sequence": input_video_sequence.float(),
            "input_command_sequence": input_command_sequence.float(),
            "trgt_flow_sequence": traj_flow_sequence.float(),
        }
