import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, TypedDict

import numpy as np
import torch
from jacobian.dataset.types import Stage, Trajectory
from jacobian.utils.io import load_gzip_file, numpy_to_torch_image
from jacobian.utils.misc import cyan
from jaxtyping import Float
from raft_wrapper.raft import resize_batched_flow
from torch.utils.data import Dataset


@dataclass
class DatasetPlanarHandCfg:
    name: Literal["planar_hand"]
    root: Path
    new_H: int | None = None
    new_W: int | None = None


class DatasetPlanarHand(Dataset):
    cfg: DatasetPlanarHandCfg
    stage: Stage
    trajectory_paths: List[Path]
    trajectory_data: List[Trajectory]
    repeat: int = 100000
    # normalization params
    min_qpos = torch.tensor([-1.4, -1.4], dtype=torch.float32)[None]
    max_qpos = torch.tensor([0.0, 0.0], dtype=torch.float32)[None]
    action_scale: float = 4.0

    def __init__(self, cfg: DatasetPlanarHandCfg, stage: Stage):
        self.cfg = cfg
        self.stage = stage

        self.trajectory_paths = sorted(list(Path((cfg.root), stage).glob("*.pkl")))
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

        # if self.stage != "train":
        #     print(cyan("Warning: validating using only one sample"))
        # return orig_len

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

        # randomly sample a frame that is not the last frame
        rand_frame_idx = np.random.randint(trajectory["seq_rgb_obs"].shape[0] - 1)

        image_sequence = trajectory["seq_rgb_obs"]
        image_sequence = torch.stack(
            [numpy_to_torch_image(img) for img in image_sequence], dim=0
        )

        joint_pos_sequence = torch.from_numpy(trajectory["seq_qpos"])
        joint_pos_sequence = joint_pos_sequence[..., :2]  # num_steps x command_dim

        traj_flow_sequence = torch.from_numpy(trajectory["seq_flow_obs"])

        # normalize the joint_pos sequence
        joint_pos_sequence = (joint_pos_sequence - self.min_qpos) / (
            self.max_qpos - self.min_qpos
        )

        if self.cfg.new_H is not None and self.cfg.new_W is not None:
            image_sequence = torch.nn.functional.interpolate(
                image_sequence, size=(self.cfg.new_H, self.cfg.new_W)
            )

            traj_flow_sequence = resize_batched_flow(
                traj_flow_sequence, self.cfg.new_H, self.cfg.new_W
            )

        # grab the current and next frame
        frame_curr = image_sequence[rand_frame_idx]
        frame_next = image_sequence[rand_frame_idx + 1]
        trgt_flow_curr = traj_flow_sequence[rand_frame_idx]

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
        if self.cfg.new_H is not None and self.cfg.new_W is not None:
            input_video_sequence = torch.nn.functional.interpolate(
                input_video_sequence, size=(self.cfg.new_H, self.cfg.new_W)
            )

            traj_flow_sequence = resize_batched_flow(
                traj_flow_sequence, self.cfg.new_H, self.cfg.new_W
            )

        # grab the current joint pos
        input_command_sequence = self.action_scale * (
            joint_pos_sequence[1:] - joint_pos_sequence[:-1]
        )

        return {
            "input_video_sequence": input_video_sequence.float()[:-1],
            "input_command_sequence": input_command_sequence.float(),
            "trgt_flow_sequence": traj_flow_sequence.float()[:-1],
        }
