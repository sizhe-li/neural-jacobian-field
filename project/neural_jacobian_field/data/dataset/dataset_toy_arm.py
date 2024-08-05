import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.utils import poses as pose_utils
from torch.utils.data import Dataset

from neural_jacobian_field.rendering.geometry import get_pixel_coordinates
from neural_jacobian_field.utils import convention, io

from .config_parser import (
    DNeRFDataParser,
    DNeRFDataParserConfig,
    DNeRFDataParserOutputs,
)
from .dataset import DatasetCfgCommon
from .image_augmentation import RandomBackground, ZeroMaskPatchedImage

NUM_POSITIVE_SAMPLES = 128
NUM_NEGATIVE_SAMPLES = 128
MAX_FRAME_DISPLACEMENT = 1
MAX_NUM_FRAMES = 10


def normalize_actions(
    curr_vals,
    old_min,
    old_max,
    new_min=0.0,
    new_max=1.0,
):
    values = (curr_vals - old_min) / (old_max - old_min)
    values = (new_max - new_min) * values + new_min
    return values


def get_pixel_selector(pos_y, pos_x, negative_yx, image_width=640):
    pixel_selector = torch.cat(
        [
            torch.cat([pos_y, pos_x], dim=-1),
            negative_yx,
        ],
        dim=0,
    )

    pixel_selector = torch.round(
        pixel_selector[:, 0] * image_width + pixel_selector[:, 1]
    ).long()

    return pixel_selector


def load_tapir_tracks(
    trgt_img_filename: str,
    traj_idx: int,
    curr_frame_idx: int,
    next_frame_idx: int,
    num_negative: int = 512,
    num_positive: int = 512,
):
    ### load positive mask
    pos_hand_mask = (
        str(trgt_img_filename).replace("rgb", "mask").replace(".png", ".npy")
    )
    pos_hand_mask = torch.from_numpy(np.load(pos_hand_mask)).float()

    ### get negative mask
    neg_hand_mask = (1 - pos_hand_mask).float()
    negative_yx = neg_hand_mask.nonzero()

    negative_yx = negative_yx[torch.randperm(len(negative_yx))[:num_negative]]

    point_track_filename = trgt_img_filename.replace("rgb", "tapir").replace(
        f"{traj_idx:05d}_{curr_frame_idx:05d}.png", f"{traj_idx:05d}_001.npz"
    )

    try:
        tapir_data = np.load(point_track_filename)
    except:
        return False

    ### load point tracks
    ### (x, y) order
    point_track_data = torch.from_numpy(tapir_data["tracks"]).float()
    pixel_visible_mask = torch.from_numpy(tapir_data["visibles"]).float()

    rand_indices = torch.randperm(len(point_track_data))
    point_track_data = point_track_data[rand_indices][:num_positive]
    pixel_visible_mask = pixel_visible_mask[rand_indices][:num_positive]

    # converting to long is super important!!!
    point_track_data[:, :, 0] = point_track_data[:, :, 0].clip(0, 639).long()
    point_track_data[:, :, 1] = point_track_data[:, :, 1].clip(0, 479).long()

    return {
        "point_track_data": point_track_data,
        "pixel_visible_mask": pixel_visible_mask,
        "negative_yx": negative_yx,
    }


@dataclass
class DatasetToyArmPointTrackCfg(DatasetCfgCommon):
    name: Literal["allegro"]
    root: Path
    num_total_joints: int
    disabled_joints: List[int]
    other_roots: List[Path]

    augment_ctxt_img: bool = False
    train_flow: bool = False
    mask_ratio: Optional[float] = 0.0


class DatasetToyArmPointTrack(Dataset):
    cfg: DatasetToyArmPointTrackCfg
    near: float = 0.5
    far: float = 3.5
    repeat = 1000
    scale_factor = 1.0

    def __init__(
        self,
        cfg: DatasetToyArmPointTrackCfg,
        stage: Literal["train", "val", "test"] = "train",
    ):
        super().__init__()

        self.cfg = cfg
        self.stage = stage
        self.root_dir = Path(cfg.root)

        downscale_factor = 1 if (stage in ["train", "test"]) else 5

        dataparser: DNeRFDataParser
        dataparser = DNeRFDataParserConfig(
            data=self.root_dir,
            center_method="focus",
            downscale_factor=downscale_factor,
        ).setup()

        dataparser_outputs: DNeRFDataParserOutputs
        dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
        self._dataparser_outputs = dataparser_outputs

        if cfg.other_roots is not None:
            print(f"Combining roots! {cfg.other_roots}")
            total_dataparser_outputs = io.combine_roots(
                cfg, dataparser_outputs, downscale_factor
            )
            # update dataparser_outputs
            dataparser_outputs = dataparser.merge_datparser_outputs(
                total_dataparser_outputs
            )
            self._dataparser_outputs = dataparser_outputs
            print(
                "Number of files of combining roots",
                len(dataparser_outputs.image_filenames),
            )

        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)

        # load camera-related
        self.sample_to_camera_idx = dataparser_outputs.sample_to_camera_idx
        self.multiview_cam2worlds = pose_utils.to4x4(self.cameras.camera_to_worlds)
        self.multiview_intrinsics = self.cameras.get_intrinsics_matrices()

        # load depth-related
        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]

        ### load image coordinates
        self.coordinates = self.cameras.get_image_coords()

        ### load action-related
        self.num_joints = self.cfg.num_total_joints
        self.disabled_joints = self.cfg.disabled_joints
        self.active_joints = [
            x for x in range(self.num_joints) if x not in self.disabled_joints
        ]
        self.times = self.metadata["times"]
        self.joint_positions: Dict[str, torch.tensor] = self.metadata["joint_positions"]
        joint_positions_np = torch.stack(list(self.joint_positions.values()), dim=0)
        self.servo_pos_min = joint_positions_np.min(0).values.float()
        self.servo_pos_max = joint_positions_np.max(0).values.float()
        # data augmentation flags
        self.random_background = RandomBackground() if cfg.augment_ctxt_img else None
        self.zero_background = (
            ZeroMaskPatchedImage(patch_size=20, mask_ratio=cfg.mask_ratio)
            if cfg.mask_ratio
            else None
        )

    def load_extrinsics(self, camera_idx: int):
        c2w = self.multiview_cam2worlds[camera_idx].clone()
        c2w = convention.post_process_camera_to_world(c2w)

        return c2w

    def load_intrinsics(self, camera_idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        intrinsics = self.multiview_intrinsics[camera_idx].clone()  # [3, 3]

        # normalize intrinsics
        image_width = self.cameras.width[camera_idx].clone()
        image_height = self.cameras.height[camera_idx].clone()

        intrinsics[:2] /= torch.tensor([image_width, image_height])[:, None].float()

        return intrinsics, (image_height.item(), image_width.item())

    def load_trgt_depth(self, trgt_img_filename, trgt_cam_idx: int):
        trgt_depth_filename = Path(trgt_img_filename.replace("rgb", "depth"))
        height = int(self.cameras.height[trgt_cam_idx])
        width = int(self.cameras.width[trgt_cam_idx])

        depth_scale_factor = (
            self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        )
        trgt_depth_img = get_depth_image_from_path(
            filepath=trgt_depth_filename,
            height=height,
            width=width,
            scale_factor=depth_scale_factor,
        ).permute(2, 0, 1)

        return trgt_depth_img

    @staticmethod
    def random_select_action_type(
        curr_frame_idx: int, frame_displacement: int, max_num_frames: int = 8
    ):
        if curr_frame_idx <= frame_displacement - 1:
            return "fwd"
        elif curr_frame_idx >= max_num_frames - frame_displacement:
            return "bwd"
        else:
            return random.choice(["fwd", "bwd"])

    def load_robot_action(
        self,
        sample_idx: int,
        curr_frame_idx: int,
        next_frame_idx: int,
    ):
        ### load robot action
        curr_frame_keyname = f"{sample_idx:05d}_{curr_frame_idx:05d}"
        next_frame_keyname = f"{sample_idx:05d}_{next_frame_idx:05d}"

        curr_joint_pos = self.joint_positions[curr_frame_keyname].clone()
        next_joint_pos = self.joint_positions[next_frame_keyname].clone()

        curr_joint_pos = normalize_actions(
            curr_joint_pos,
            old_min=self.servo_pos_min.clone(),
            old_max=self.servo_pos_max.clone(),
            new_min=-1.0,
            new_max=1.0,
        )
        next_joint_pos = normalize_actions(
            next_joint_pos,
            old_min=self.servo_pos_min.clone(),
            old_max=self.servo_pos_max.clone(),
            new_min=-1.0,
            new_max=1.0,
        )

        # normalize [-2, 2] to [-1, 1]
        robot_action = (next_joint_pos - curr_joint_pos) / 2.0
        # extract only active joints
        robot_action = robot_action[self.active_joints]

        return robot_action

    def load_action_and_flow(self, trgt_img_filename: str, image_width: int = 640):
        #### sample flow type
        traj_idx, curr_frame_idx = convention.extract_traj_and_frame_idx(
            trgt_img_filename
        )
        # random select action gap 1 - 3 frames apart
        frame_displacement = random.randint(1, MAX_FRAME_DISPLACEMENT)

        flow_type = self.random_select_action_type(
            curr_frame_idx,
            frame_displacement,
            max_num_frames=MAX_NUM_FRAMES,
        )

        ### load next frame
        next_frame_idx = (
            curr_frame_idx + frame_displacement
            if flow_type == "fwd"
            else curr_frame_idx - frame_displacement
        )

        ### load action
        robot_action = self.load_robot_action(traj_idx, curr_frame_idx, next_frame_idx)

        tapir_track_data = load_tapir_tracks(
            trgt_img_filename, traj_idx, curr_frame_idx, next_frame_idx
        )
        if tapir_track_data is False:
            return None

        point_track_data = tapir_track_data["point_track_data"]
        pixel_visible_mask = tapir_track_data["pixel_visible_mask"]
        negative_yx = tapir_track_data["negative_yx"]
        num_negative = len(negative_yx)

        curr_pos_x, curr_pos_y = point_track_data[:, curr_frame_idx, :].split(1, dim=-1)
        next_pos_x, next_pos_y = point_track_data[:, next_frame_idx, :].split(1, dim=-1)

        flow_x, flow_y = (next_pos_x - curr_pos_x), (next_pos_y - curr_pos_y)
        pixel_selector = torch.cat(
            [
                torch.cat([curr_pos_y, curr_pos_x], dim=-1),
                negative_yx,
            ],
            dim=0,
        )

        pixel_selector = torch.round(
            pixel_selector[:, 0] * image_width + pixel_selector[:, 1]
        ).long()

        pixel_motion = torch.cat(
            [
                torch.cat([flow_x, flow_y], dim=-1),
                torch.zeros_like(negative_yx),
            ],
            dim=0,
        )

        pixel_visible_mask = torch.cat(
            [
                pixel_visible_mask[:, next_frame_idx],
                torch.ones((num_negative,), dtype=torch.float32),
            ]
        )

        return {
            "robot_action": robot_action,
            "pixel_selector": pixel_selector,
            "pixel_motion": pixel_motion,
            "pixel_visible_mask": pixel_visible_mask,
        }

    def augment_image(self, image: torch.Tensor, image_filename: str):

        mask = str(image_filename).replace("rgb", "mask").replace(".png", ".npy")
        mask = torch.from_numpy(np.load(mask)).float()
        image = self.random_background(image, mask)
        return image

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ):
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return value

    @property
    def num_files(self) -> int:
        return len(self._dataparser_outputs.image_filenames)

    def __len__(self) -> int:
        return self.num_files * self.repeat if self.stage == "train" else 100

    def __getitem__(self, ctxt_file_idx: int) -> dict:
        ctxt_file_idx = ctxt_file_idx % self.num_files

        if self.cfg.overfit_to_scene is not None:
            ctxt_file_idx = int(self.cfg.overfit_to_scene)

        ctxt_cam_idx = int(self.sample_to_camera_idx[ctxt_file_idx].item())
        trgt_cam_idx = random.choice(range(len(self.cameras)))

        ctxt_img_filename = self._dataparser_outputs.image_filenames[ctxt_file_idx]
        trgt_img_filename = convention.get_trgt_view_filename(
            str(ctxt_img_filename), ctxt_cam_idx, trgt_cam_idx
        )
        ### process rgb
        ctxt_rgb = io.load_image_file_to_torch(ctxt_img_filename, self.scale_factor)
        trgt_rgb = io.load_image_file_to_torch(trgt_img_filename, self.scale_factor)

        ### process extrinsics
        ctxt_c2w = self.load_extrinsics(ctxt_cam_idx)
        trgt_c2w = self.load_extrinsics(trgt_cam_idx)

        inverse_ctxt_c2w = torch.inverse(ctxt_c2w)

        ctxt_c2w = torch.einsum("ij, jk -> ik", inverse_ctxt_c2w, ctxt_c2w)
        trgt_c2w = torch.einsum("ij, jk -> ik", inverse_ctxt_c2w, trgt_c2w)

        ### process intrinsics
        ctxt_intr, (image_height, image_width) = self.load_intrinsics(ctxt_cam_idx)
        trgt_intr, _ = self.load_intrinsics(trgt_cam_idx)

        ### process depth
        try:
            trgt_depth_img = self.load_trgt_depth(trgt_img_filename, trgt_cam_idx)
        except:
            print("Error loading depth image", trgt_img_filename)
            raise ValueError

        ### process coordinates
        coordinates, _ = get_pixel_coordinates(image_height, image_width)

        if self.cfg.augment_ctxt_img:
            ctxt_rgb = self.augment_image(ctxt_rgb, ctxt_img_filename)

        if self.stage == "test" and self.zero_background is not None:
            ctxt_rgb = self.zero_background(ctxt_rgb)

        ret_dict = {
            "context": {
                "rgb": ctxt_rgb,
                "extrinsics": ctxt_c2w,
                "intrinsics": ctxt_intr,
            },
            "target": {
                "rgb": trgt_rgb,
                "depth": trgt_depth_img,
                "extrinsics": trgt_c2w,
                "intrinsics": trgt_intr,
            },
            "scene": {
                "near": self.get_bound("near", num_views=1),
                "far": self.get_bound("far", num_views=1),
                "coordinates": coordinates,
            },
        }

        ret_dict["context"]["robot_action"] = torch.zeros(
            (self.num_joints,), dtype=torch.float32
        )

        if self.cfg.train_flow:
            ### process action and flow
            action_and_flow_data = self.load_action_and_flow(
                trgt_img_filename=trgt_img_filename,
                image_width=image_width,
            )
            if action_and_flow_data is None:
                random_index = random.randint(0, self.num_files)
                return self.__getitem__(random_index)

            ret_dict["context"]["robot_action"] = action_and_flow_data["robot_action"]

            # (num_rays)
            ret_dict["target"]["pixel_selector"] = action_and_flow_data[
                "pixel_selector"
            ]
            # (num_rays, 2) ; order: (y, x)
            ret_dict["target"]["pixel_motion"] = action_and_flow_data["pixel_motion"]
            ret_dict["target"]["pixel_visible_mask"] = action_and_flow_data[
                "pixel_visible_mask"
            ]

        return ret_dict
