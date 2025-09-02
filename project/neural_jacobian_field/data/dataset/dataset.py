import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generic, List, Literal, Tuple, TypeVar

import numpy as np
import torch
from jaxtyping import Float, Int
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.utils import poses as pose_utils
from torch import Tensor
from torch.utils.data import Dataset

from ...rendering.geometry import get_pixel_coordinates
from ...utils import convention, io_utils, misc
from .config_parser import (
    Cameras,
    DNeRFDataParser,
    DNeRFDataParserConfig,
    DNeRFDataParserOutputs,
)
from .image_augmentation import RandomBackground, ZeroMaskPatchedImage

Stage = Literal["train", "val", "test"]


@dataclass
class DatasetStandardItems:
    dataparser_scale: float
    image_filenames: List[Path]
    depth_filenames: List[str]
    depth_unit_scale_factor: float
    times: List[float]


@dataclass
class DatasetCameraItems:
    cameras: Cameras
    sample_to_camera_idx: Int[Tensor, "num_samples"]
    multiview_cam2worlds: Float[Tensor, "*camera 4 4"]
    multiview_intrinsics: Float[Tensor, "*camera 3 3"]
    coordinates: Float[Tensor, "height width 2"]


@dataclass
class DatasetActionItems:
    num_joints: int
    active_joints: List[int]
    keyname_to_qpos: Dict[str, Float[Tensor, "action_dim"]]
    qpos_minimum: Float[Tensor, "action_dim"]
    qpos_maximum: Float[Tensor, "action_dim"]


@dataclass
class DatasetItems:
    standard: DatasetStandardItems
    camera: DatasetCameraItems
    action: DatasetActionItems


@dataclass
class QposItems:
    curr_qpos: Float[Tensor, "action_dim"]
    next_qpos: Float[Tensor, "action_dim"]
    qpos_minimum: Float[Tensor, "action_dim"]
    qpos_maximum: Float[Tensor, "action_dim"]


@dataclass
class TrackSupervision:
    pixel_selector: Float[Tensor, "num_pixels"]
    pixel_motion: Float[Tensor, "num_pixels 2"]
    pixel_visible_mask: Float[Tensor, "num_pixels"]


@dataclass
class DatasetCfgCommon:
    name: str
    mode: Literal["perception", "action"]
    # file related
    overfit_to_scene: None | str
    root: Path
    other_roots: None | List[Path] | List[str]
    # action related
    num_total_joints: int
    disabled_joints: None | List[int]
    # creating supervision
    max_frame_displacement: int
    max_num_frames_per_traj: int
    action_supervision_type: Literal["optical_flow", "tracks"]
    num_positive_samples: int | None
    num_negative_samples: int | None
    # training
    augment_ctxt_image: bool
    # testing
    testing_mask_ratio: None | float = None


# T = TypeVar("T")


class DatasetCommon(Dataset):
    cfg: DatasetCfgCommon
    near: float
    far: float
    repeat: int
    scale_factor: float
    dataset_items: DatasetItems

    def __init__(self, cfg: DatasetCfgCommon, stage: Stage):
        self.cfg = cfg
        self.stage = stage

        downscale_factor = 1 if (stage in ["train", "test"]) else 5
        self.dataset_items = self.load_dataset(downscale_factor=downscale_factor)
        # data augmentation flags
        self.random_background = RandomBackground() if cfg.augment_ctxt_image else None
        if stage == "test":
            if cfg.testing_mask_ratio is not None:
                self.zero_background = ZeroMaskPatchedImage(
                    patch_size=20, mask_ratio=cfg.testing_mask_ratio
                )

    def load_dataset(self, downscale_factor: int = 1) -> DatasetItems:
        dataparser: DNeRFDataParser
        dataparser = DNeRFDataParserConfig(
            data=Path(self.cfg.root),
            center_method="focus",
            downscale_factor=downscale_factor,
        ).setup()

        dataparser_outputs: DNeRFDataParserOutputs
        dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
        self._dataparser_outputs = dataparser_outputs

        if self.cfg.other_roots is not None:
            print(f"Combining roots! {self.cfg.other_roots}")
            total_dataparser_outputs = io_utils.combine_roots(
                self.cfg, dataparser_outputs, downscale_factor
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

        metadata = deepcopy(dataparser_outputs.metadata)
        cameras = deepcopy(dataparser_outputs.cameras)

        # load camera-related
        sample_to_camera_idx = dataparser_outputs.sample_to_camera_idx
        multiview_cam2worlds = pose_utils.to4x4(cameras.camera_to_worlds)
        multiview_intrinsics = cameras.get_intrinsics_matrices()

        # load depth-related
        depth_filenames = metadata["depth_filenames"]
        depth_unit_scale_factor = metadata["depth_unit_scale_factor"]

        ### load image coordinates
        coordinates = cameras.get_image_coords()

        ### load action-related
        num_joints = self.cfg.num_total_joints
        active_joints = list(range(num_joints))
        if self.cfg.disabled_joints is not None:
            disabled_joints = self.cfg.disabled_joints
            # filter out disabled joints
            active_joints = [x for x in active_joints if x not in disabled_joints]

        times = metadata["times"]
        # unpack joint positions
        keyname_to_qpos: Dict[str, torch.tensor] = metadata["joint_positions"]
        qpos = torch.stack(list(keyname_to_qpos.values()), dim=0)
        qpos_minimum = qpos.min(0).values.float()
        qpos_maximum = qpos.max(0).values.float()

        return DatasetItems(
            standard=DatasetStandardItems(
                dataparser_scale=dataparser_outputs.dataparser_scale,
                image_filenames=dataparser_outputs.image_filenames,
                depth_filenames=depth_filenames,
                depth_unit_scale_factor=depth_unit_scale_factor,
                times=times,
            ),
            camera=DatasetCameraItems(
                cameras=cameras,
                sample_to_camera_idx=sample_to_camera_idx,
                multiview_cam2worlds=multiview_cam2worlds,
                multiview_intrinsics=multiview_intrinsics,
                coordinates=coordinates,
            ),
            action=DatasetActionItems(
                num_joints=num_joints,
                active_joints=active_joints,
                keyname_to_qpos=keyname_to_qpos,
                qpos_minimum=qpos_minimum,
                qpos_maximum=qpos_maximum,
            ),
        )

    @staticmethod
    def random_select_action_type(
        curr_frame_idx: int, frame_displacement: int, max_num_frames: int = 8
    ) -> Literal["fwd", "bwd"]:
        if curr_frame_idx <= frame_displacement - 1:
            return "fwd"
        elif curr_frame_idx >= max_num_frames - frame_displacement:
            return "bwd"
        else:
            return random.choice(["fwd", "bwd"])

    def preprocess_qpos(self, qpos_items: QposItems) -> QposItems:
        """Dataset dependent, default to identity"""

        return qpos_items

    def load_robot_action(
        self,
        sample_idx: int,
        curr_frame_idx: int,
        next_frame_idx: int,
    ):
        ### load robot action
        curr_frame_keyname = f"{sample_idx:05d}_{curr_frame_idx:05d}"
        next_frame_keyname = f"{sample_idx:05d}_{next_frame_idx:05d}"

        curr_qpos = self.dataset_items.action.keyname_to_qpos[
            curr_frame_keyname
        ].clone()
        next_qpos = self.dataset_items.action.keyname_to_qpos[
            next_frame_keyname
        ].clone()

        qpos_items = self.preprocess_qpos(
            QposItems(
                curr_qpos=curr_qpos,
                next_qpos=next_qpos,
                qpos_minimum=self.dataset_items.action.qpos_minimum.clone(),
                qpos_maximum=self.dataset_items.action.qpos_maximum.clone(),
            )
        )
        curr_qpos, next_qpos, qpos_minimum, qpos_maximum = (
            qpos_items.curr_qpos,
            qpos_items.next_qpos,
            qpos_items.qpos_minimum,
            qpos_items.qpos_maximum,
        )

        curr_qpos = convention.normalize(
            curr_qpos,
            old_min=qpos_minimum,
            old_max=qpos_maximum,
            new_min=-1.0,
            new_max=1.0,
        )
        next_qpos = convention.normalize(
            next_qpos,
            old_min=qpos_minimum,
            old_max=qpos_maximum,
            new_min=-1.0,
            new_max=1.0,
        )

        # normalize [-2, 2] to [-1, 1]
        robot_action = (next_qpos - curr_qpos) / 2.0
        # extract only active joints
        robot_action = robot_action[self.dataset_items.action.active_joints]

        return robot_action

    def load_extrinsics(self, camera_idx: int):
        c2w = self.dataset_items.camera.multiview_cam2worlds[camera_idx].clone()
        c2w = convention.post_process_camera_to_world(c2w)

        return c2w

    def load_intrinsics(self, camera_idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        intrinsics = self.dataset_items.camera.multiview_intrinsics[
            camera_idx
        ].clone()  # [3, 3]

        # normalize intrinsics
        image_width = self.dataset_items.camera.cameras.width[camera_idx].clone()
        image_height = self.dataset_items.camera.cameras.height[camera_idx].clone()

        intrinsics[:2] /= torch.tensor([image_width, image_height])[:, None].float()

        return intrinsics, (image_height.item(), image_width.item())

    def load_depth(self, trgt_img_filename, trgt_cam_idx: int):
        trgt_depth_filename = Path(trgt_img_filename.replace("rgb", "depth"))
        height = int(self.dataset_items.camera.cameras.height[trgt_cam_idx])
        width = int(self.dataset_items.camera.cameras.width[trgt_cam_idx])

        depth_scale_factor = (
            self.dataset_items.standard.depth_unit_scale_factor
            * self._dataparser_outputs.dataparser_scale
        )
        trgt_depth_img = get_depth_image_from_path(
            filepath=trgt_depth_filename,
            height=height,
            width=width,
            scale_factor=depth_scale_factor,
        ).permute(2, 0, 1)

        return trgt_depth_img

    @property
    def num_files(self) -> int:
        return len(self._dataparser_outputs.image_filenames)

    def __len__(self) -> int:
        return self.num_files * self.repeat

    @staticmethod
    def get_relative_transform(ctxt_c2w, trgt_c2w):
        inverse_ctxt_c2w = torch.inverse(ctxt_c2w)
        ctxt_c2w = torch.einsum("ij, jk -> ik", inverse_ctxt_c2w, ctxt_c2w)
        trgt_c2w = torch.einsum("ij, jk -> ik", inverse_ctxt_c2w, trgt_c2w)

        return ctxt_c2w, trgt_c2w

    def __getitem__(self, ctxt_file_idx: int):
        ctxt_file_idx = ctxt_file_idx % self.num_files

        if self.cfg.overfit_to_scene is not None:
            ctxt_file_idx = int(self.cfg.overfit_to_scene)

        ctxt_cam_idx = self.dataset_items.camera.sample_to_camera_idx[ctxt_file_idx]
        trgt_cam_idx = random.choice(range(len(self.dataset_items.camera.cameras)))

        ctxt_img_filename = self._dataparser_outputs.image_filenames[ctxt_file_idx]
        trgt_img_filename = convention.get_trgt_view_filename(
            str(ctxt_img_filename), ctxt_cam_idx, trgt_cam_idx
        )
        ###################
        ### process rgb ###
        ###################
        try:
            ctxt_rgb = io_utils.load_image_file_to_torch(
                ctxt_img_filename, self.scale_factor
            )
            trgt_rgb = io_utils.load_image_file_to_torch(
                trgt_img_filename, self.scale_factor
            )
        except OSError:
            print(misc.cyan(f"Error loading image: {ctxt_img_filename}"))
            return self.__getitem__(random.randint(0, self.num_files))

        image_height, image_width = ctxt_rgb.shape[-2:]

        ##########################
        ### process extrinsics ###
        ##########################
        ctxt_c2w = self.load_extrinsics(ctxt_cam_idx)
        trgt_c2w = self.load_extrinsics(trgt_cam_idx)
        inverse_ctxt_c2w = torch.inverse(ctxt_c2w)
        ctxt_c2w = torch.einsum("ij, jk -> ik", inverse_ctxt_c2w, ctxt_c2w)
        trgt_c2w = torch.einsum("ij, jk -> ik", inverse_ctxt_c2w, trgt_c2w)

        ##########################
        ### process intrinsics ###
        ##########################

        ctxt_intr, (render_height, render_width) = self.load_intrinsics(ctxt_cam_idx)
        trgt_intr, _ = self.load_intrinsics(trgt_cam_idx)

        #####################
        ### process depth ###
        #####################
        trgt_depth_img = self.load_depth(trgt_img_filename, trgt_cam_idx)

        ###########################
        ### process coordinates ###
        ###########################
        coordinates, _ = get_pixel_coordinates(render_height, render_width)

        ##### Augmentation
        if self.cfg.augment_ctxt_image:
            ctxt_rgb = self.augment_image(ctxt_rgb, ctxt_img_filename)

        if self.stage == "test" and self.zero_background is not None:
            ctxt_rgb = self.zero_background(ctxt_rgb)

        data_dict = {
            "context": {
                "rgb": ctxt_rgb,
                "extrinsics": ctxt_c2w,
                "intrinsics": ctxt_intr,
                "robot_action": torch.zeros(
                    (len(self.dataset_items.action.active_joints),), dtype=torch.float32
                ),
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

        if self.cfg.mode == "action":
            #######################
            ### process actions ###
            #######################

            traj_idx, curr_frame_idx = convention.get_traj_and_frame_idx(
                trgt_img_filename
            )
            max_num_frames_per_traj = self.get_max_num_frames_per_traj(traj_idx)

            rand_frame_displacement = random.randint(1, self.cfg.max_frame_displacement)
            flow_type = self.random_select_action_type(
                curr_frame_idx=curr_frame_idx,
                frame_displacement=rand_frame_displacement,
                max_num_frames=max_num_frames_per_traj,
            )
            ### load next frame
            next_frame_idx = (
                curr_frame_idx + rand_frame_displacement
                if flow_type == "fwd"
                else curr_frame_idx - rand_frame_displacement
            )
            ### load action
            robot_action = self.load_robot_action(
                traj_idx, curr_frame_idx, next_frame_idx
            )
            data_dict["context"]["robot_action"] = robot_action

            if self.cfg.action_supervision_type == "optical_flow":
                flow = self.load_optical_flow_supervision(
                    trgt_img_filename, traj_idx, curr_frame_idx, flow_type
                )
                data_dict["target"]["flow"] = flow
            else:
                track_supervision = self.load_tracks_supervision(
                    trgt_img_filename,
                    traj_idx,
                    curr_frame_idx,
                    next_frame_idx,
                    image_width=image_width,
                )
                data_dict["target"]["pixel_selector"] = track_supervision.pixel_selector
                # (num_rays, 2) ; order: (y, x)
                data_dict["target"]["pixel_motion"] = track_supervision.pixel_motion
                data_dict["target"][
                    "pixel_visible_mask"
                ] = track_supervision.pixel_visible_mask

        return data_dict

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ):
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return value

    def augment_image(self, image: torch.Tensor, image_filename: Path):
        mask = str(image_filename).replace("rgb", "mask").replace(".png", ".npy")
        mask = torch.from_numpy(np.load(mask)).float()
        image = self.random_background(image, mask)
        return image

    def load_optical_flow_supervision(
        self,
        trgt_img_filename: str,
        traj_idx: int,
        curr_frame_idx: int,
        flow_type: Literal["fwd", "bwd"],
    ) -> Float[Tensor, "2 H W"]:
        trgt_optical_flow_filename = convention.get_optical_flow_filename(
            trgt_img_filename, traj_idx, curr_frame_idx, flow_type
        )
        trgt_optical_flow = io_utils.load_optical_flow(trgt_optical_flow_filename)
        trgt_optical_flow = torch.from_numpy(trgt_optical_flow).float()

        return trgt_optical_flow

    def load_tracks_supervision(
        self,
        trgt_img_filename: str,
        traj_idx: int,
        curr_frame_idx: int,
        next_frame_idx: int,
        image_width: int = 640,
    ) -> TrackSupervision:
        tapir_track_data = io_utils.load_tapir_tracks(
            trgt_img_filename=trgt_img_filename,
            traj_idx=traj_idx,
            curr_frame_idx=curr_frame_idx,
            next_frame_idx=next_frame_idx,
            num_negative=self.cfg.num_negative_samples,
            num_positive=self.cfg.num_positive_samples,
        )

        # if tapir_track_data is False:
        #     return None

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

        return TrackSupervision(
            pixel_selector=pixel_selector,
            pixel_motion=pixel_motion,
            pixel_visible_mask=pixel_visible_mask,
        )

    def get_max_num_frames_per_traj(self, traj_idx: int) -> int:
        return self.cfg.max_num_frames_per_traj
