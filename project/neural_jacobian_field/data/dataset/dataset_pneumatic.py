import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Optional

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
from .image_augmentation import RandomBackground


def process_joints_new_hand(
    orig_servo_pos_min, orig_servo_pos_max, orig_curr_joint_pos, orig_next_joint_pos
):
    # sum every two joints | TODO: this is special for the new pneuamtic hand only.
    servo_pos_min = orig_servo_pos_min[::2]
    servo_pos_min[:-1] += orig_servo_pos_min[:-1][1::2]

    servo_pos_max = orig_servo_pos_max[::2]
    servo_pos_max[:-1] += orig_servo_pos_max[:-1][1::2]

    curr_joint_pos = orig_curr_joint_pos[::2]
    curr_joint_pos[:-1] += orig_curr_joint_pos[:-1][1::2]

    next_joint_pos = orig_next_joint_pos[::2]
    next_joint_pos[:-1] += orig_next_joint_pos[:-1][1::2]

    # print("servo_pos_min", servo_pos_min)
    # print("servo_pos_max", servo_pos_max)
    # print("curr_joint_pos", curr_joint_pos)
    # print("next_joint_pos", next_joint_pos)

    return servo_pos_min, servo_pos_max, curr_joint_pos, next_joint_pos


def process_joints_old_hand(
    orig_servo_pos_min, orig_servo_pos_max, orig_curr_joint_pos, orig_next_joint_pos
):
    # this is for old pneumatic hand
    servo_pos_min = orig_servo_pos_min[::2] + orig_servo_pos_min[1::2]
    servo_pos_max = orig_servo_pos_max[::2] + orig_servo_pos_max[1::2]
    curr_joint_pos = orig_curr_joint_pos[::2] + orig_curr_joint_pos[1::2]
    next_joint_pos = orig_next_joint_pos[::2] + orig_next_joint_pos[1::2]

    return servo_pos_min, servo_pos_max, curr_joint_pos, next_joint_pos


def process_joints_move_arm(
    orig_servo_pos_min, orig_servo_pos_max, orig_curr_joint_pos, orig_next_joint_pos
):
    return (
        orig_servo_pos_min[:2],
        orig_servo_pos_max[:2],
        orig_curr_joint_pos[:2],
        orig_next_joint_pos[:2],
    )


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
    trgt_img_filename,
    traj_idx,
    curr_frame_idx,
    next_frame_idx,
    num_negative=512,
    num_positive=512,
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

    # ## multiple files for tracks could exist
    # point_track_filename = random.choice(
    #     convention.get_tracking_data_filenames(
    #         trgt_img_filename, traj_idx, curr_frame_idx
    #     )
    # )

    point_track_filename = trgt_img_filename.replace("rgb", "tapir").replace(
        f"{traj_idx:05d}_{curr_frame_idx:05d}.png", f"{traj_idx:05d}_001.npz"
    )

    tapir_data = np.load(point_track_filename)

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
class DatasetPneumaticPointTrackCfg(DatasetCfgCommon):
    name: Literal["allegro"]
    root: Path
    num_total_joints: int
    disabled_joints: List[int]
    augment_ctxt_img: bool = False
    train_flow: bool = False
    use_correspondence_loss: bool = False
    other_roots: Optional[List[Path]] = None


class DatasetPneumaticPointTrack(Dataset):
    cfg: DatasetPneumaticPointTrackCfg
    near: float = 0.5
    far: float = 10.0
    repeat = 1000
    scale_factor = 1.0

    def __init__(
        self,
        cfg: DatasetPneumaticPointTrackCfg,
        stage: Literal["train", "val", "test"] = "train",
    ):
        super().__init__()

        self.cfg = cfg
        self.stage = stage
        self.root_dir = Path(cfg.root)

        downscale_factor = 1 if stage == "train" else 5

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
            print("Combining data with other roots", cfg.other_roots)

            total_dataparser_outputs = [dataparser_outputs]
            for other_root in cfg.other_roots:
                # print("other root.exists", os.path.exists(other_root), other_root)

                other_dataparser = DNeRFDataParserConfig(
                    data=Path(other_root),
                    center_method="focus",
                    downscale_factor=downscale_factor,
                ).setup()
                total_dataparser_outputs.append(
                    other_dataparser.get_dataparser_outputs(split="train")
                )

            # check joint positions and pad to the same length with zeros
            max_len = -1
            for i, other_dataparser_outputs in enumerate(total_dataparser_outputs):
                joint_positions = other_dataparser_outputs.metadata["joint_positions"]
                max_len = max(max_len, len(list(joint_positions.values())[0]))

            for i, other_dataparser_outputs in enumerate(total_dataparser_outputs):
                joint_positions = other_dataparser_outputs.metadata["joint_positions"]
                for key, value in joint_positions.items():
                    if len(value) < max_len:
                        pad_len = max_len - len(value)
                        pad_value = torch.zeros(
                            (pad_len,), dtype=value.dtype, device=value.device
                        )
                        joint_positions[key] = torch.cat([value, pad_value], dim=0)

                other_dataparser_outputs.metadata["joint_positions"] = joint_positions

            # update dataparser_outputs
            dataparser_outputs = dataparser.merge_datparser_outputs(
                total_dataparser_outputs
            )
            self._dataparser_outputs = dataparser_outputs

            print("num files after combining", len(dataparser_outputs.image_filenames))

        self.scene_box = deepcopy(dataparser_outputs.scene_box)
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

    def load_extrinsics(self, camera_idx: int):
        c2w = self.multiview_cam2worlds[camera_idx].clone()
        c2w = convention.post_process_camera_to_world(c2w)

        return c2w

    def load_intrinsics(self, camera_idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        intrinsics = self.multiview_intrinsics[camera_idx].clone()  # [3, 3]

        # normalize intrinsics
        img_width = self.cameras.width[camera_idx].clone()
        img_height = self.cameras.height[camera_idx].clone()

        intrinsics[:2] /= torch.tensor([img_width, img_height])[:, None].float()

        return intrinsics, (img_height.item(), img_width.item())

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
    def random_select_action_type(timestamp: float):
        if 0.0 < timestamp < 1.0:
            # can sample both backward and forward in time.
            flow_type = random.choice(["fwd", "bwd"])
        elif timestamp == 0.0:
            # can only sample forward in time.
            flow_type = "fwd"
        else:
            # can only sample backward in time.
            flow_type = "bwd"

        return flow_type

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

        if len(curr_joint_pos) == 11:
            (
                servo_pos_min,
                servo_pos_max,
                curr_joint_pos,
                next_joint_pos,
            ) = process_joints_new_hand(
                self.servo_pos_min.clone(),
                self.servo_pos_max.clone(),
                curr_joint_pos,
                next_joint_pos,
            )
        else:
            (
                servo_pos_min,
                servo_pos_max,
                curr_joint_pos,
                next_joint_pos,
            ) = process_joints_move_arm(
                self.servo_pos_min.clone(),
                self.servo_pos_max.clone(),
                curr_joint_pos,
                next_joint_pos,
            )

        curr_joint_pos = convention.normalize(
            curr_joint_pos,
            old_min=servo_pos_min,
            old_max=servo_pos_max,
            new_min=-1.0,
            new_max=1.0,
        )
        next_joint_pos = convention.normalize(
            next_joint_pos,
            old_min=servo_pos_min,
            old_max=servo_pos_max,
            new_min=-1.0,
            new_max=1.0,
        )

        # normalize [-2, 2] to [-1, 1]
        robot_action = (next_joint_pos - curr_joint_pos) / 2.0
        # extract only active joints
        robot_action = robot_action[self.active_joints]

        return robot_action

    def load_correspondence(
        self, ctxt_file_idx: int, trgt_img_filename: str, image_width: int = 640
    ):
        traj_idx, curr_frame_idx = convention.extract_traj_and_frame_idx(
            trgt_img_filename
        )

        #### sample flow type
        flow_type = self.random_select_action_type(self.times[ctxt_file_idx])
        ### load next frame
        next_frame_idx = (
            curr_frame_idx + 1 if flow_type == "fwd" else curr_frame_idx - 1
        )

        tapir_track_data = load_tapir_tracks(
            trgt_img_filename,
            traj_idx,
            curr_frame_idx,
            next_frame_idx,
            num_positive=128,
            num_negative=128,
        )
        point_track_data = tapir_track_data["point_track_data"]
        pixel_visible_mask = tapir_track_data["pixel_visible_mask"]

        curr_pos_x, curr_pos_y = point_track_data[:, curr_frame_idx, :].split(1, dim=-1)
        next_pos_x, next_pos_y = point_track_data[:, next_frame_idx, :].split(1, dim=-1)

        pixel_visible_mask = (
            pixel_visible_mask[:, curr_frame_idx]
            * pixel_visible_mask[:, next_frame_idx]
        )

        pixel_selector_curr = torch.cat([curr_pos_y, curr_pos_x], dim=-1)
        pixel_selector_curr = torch.round(
            pixel_selector_curr[:, 0] * image_width + pixel_selector_curr[:, 1]
        ).long()

        pixel_selector_next = torch.cat([next_pos_y, next_pos_x], dim=-1)
        pixel_selector_next = torch.round(
            pixel_selector_next[:, 0] * image_width + pixel_selector_next[:, 1]
        ).long()

        # load next frame trgt_img
        trgt_img_next_filename = str(trgt_img_filename).replace(
            f"{traj_idx:05d}_{curr_frame_idx:05d}.png",
            f"{traj_idx:05d}_{next_frame_idx:05d}.png",
        )
        trgt_img_next = io.load_image_file_to_torch(
            trgt_img_next_filename, self.scale_factor
        )

        return {
            "pixel_selector_curr": pixel_selector_curr,
            "pixel_selector_next": pixel_selector_next,
            "pixel_visible_mask": pixel_visible_mask,
            "trgt_img_next": trgt_img_next,
        }

    def load_action_and_flow(
        self,
        ctxt_file_idx: int,
        ctxt_img_filename: str,
        trgt_img_filename: str,
        trgt_cam_idx: int,
        return_other_views=True,
    ):
        #### sample flow type
        flow_type = self.random_select_action_type(self.times[ctxt_file_idx])
        traj_idx, curr_frame_idx = convention.extract_traj_and_frame_idx(
            trgt_img_filename
        )
        ### load next frame
        next_frame_idx = (
            curr_frame_idx + 1 if flow_type == "fwd" else curr_frame_idx - 1
        )
        ### load action
        robot_action = self.load_robot_action(traj_idx, curr_frame_idx, next_frame_idx)

        tapir_track_data = load_tapir_tracks(
            trgt_img_filename, traj_idx, curr_frame_idx, next_frame_idx
        )
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

        image_width = 640
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

        # ### load optical flow
        # trgt_optical_flow_filename = convention.get_optical_flow_filename(
        #     trgt_img_filename, traj_idx, curr_frame_idx, flow_type
        # )
        # trgt_optical_flow = io.load_optical_flow(trgt_optical_flow_filename)
        # trgt_optical_flow = torch.from_numpy(trgt_optical_flow).float()
        #
        # ctxt_optical_flow_filename = convention.get_optical_flow_filename(
        #     ctxt_img_filename, traj_idx, curr_frame_idx, flow_type
        # )
        # ctxt_optical_flow = io.load_optical_flow(ctxt_optical_flow_filename)
        # ctxt_optical_flow = torch.from_numpy(ctxt_optical_flow).float()

        # other_views = None
        # if return_other_views:
        #     image_height, image_width = self.load_intrinsics(trgt_cam_idx)[1]
        #     # resize and squeeze the first dimension
        #     trgt_optical_flow = convention.resize_optical_flow(
        #         trgt_optical_flow.unsqueeze(0),
        #         trgt_size=(image_height, image_width),
        #     ).squeeze(0)
        #
        #     other_views = list(range(len(self.cameras)))
        #     other_views.remove(trgt_cam_idx)
        #
        #     other_flows = []
        #     other_extrinsics = []
        #     other_intrinsics = []
        #     for view_idx in other_views:
        #         this_flow_filename = deepcopy(trgt_optical_flow_filename).replace(
        #             f"view_{trgt_cam_idx}", f"view_{view_idx}"
        #         )
        #         flow = io.load_optical_flow(this_flow_filename)
        #         flow = torch.from_numpy(flow).float()
        #         # resize and squeeze the first dimension
        #         flow = convention.resize_optical_flow(
        #             flow.unsqueeze(0), (image_height, image_width)
        #         ).squeeze(0)
        #
        #         other_flows.append(flow)
        #         other_extrinsics.append(self.load_extrinsics(view_idx))
        #         other_intrinsics.append(self.load_intrinsics(view_idx)[0])
        #
        #     other_views = {
        #         "flows": torch.stack(other_flows, 0),
        #         "extrinsics": torch.stack(other_extrinsics, 0),
        #         "intrinsics": torch.stack(other_intrinsics, 0),
        #     }

        return {
            # "trgt_optical_flow": trgt_optical_flow,
            # "ctxt_optical_flow": ctxt_optical_flow,
            "robot_action": robot_action,
            # "other_views": other_views,
            "pixel_selector": pixel_selector,
            "pixel_motion": pixel_motion,
            "pixel_visible_mask": pixel_visible_mask,
        }

    def augment_image(self, image: torch.Tensor, image_filename: str):
        # flow_mask = optical_flow.norm(p=2, dim=0) > 0.8  # (H, W)

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
        return self.num_files * self.repeat

    def __getitem__(self, ctxt_file_idx: int) -> dict:
        ctxt_file_idx = ctxt_file_idx % self.num_files

        if self.cfg.overfit_to_scene is not None:
            ctxt_file_idx = int(self.cfg.overfit_to_scene)

        ctxt_cam_idx = self.sample_to_camera_idx[ctxt_file_idx]
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
        ctxt_intr, (img_h, img_w) = self.load_intrinsics(ctxt_cam_idx)
        trgt_intr, _ = self.load_intrinsics(trgt_cam_idx)

        ### process depth
        trgt_depth_img = self.load_trgt_depth(trgt_img_filename, trgt_cam_idx)

        ### process coordinates
        coordinates, _ = get_pixel_coordinates(img_h, img_w)

        if self.cfg.augment_ctxt_img:
            ctxt_rgb = self.augment_image(ctxt_rgb, ctxt_img_filename)

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

        if self.cfg.use_correspondence_loss:
            assert not self.cfg.train_flow
            trgt_corr_data = self.load_correspondence(
                ctxt_file_idx, trgt_img_filename, img_w
            )

            ret_dict["target"]["rgb_next"] = trgt_corr_data["trgt_img_next"]
            ret_dict["target"]["pixel_selector_curr"] = trgt_corr_data[
                "pixel_selector_curr"
            ]
            ret_dict["target"]["pixel_selector_next"] = trgt_corr_data[
                "pixel_selector_next"
            ]
            ret_dict["target"]["pixel_visible_mask"] = trgt_corr_data[
                "pixel_visible_mask"
            ]

        if self.cfg.train_flow:
            ### process action and flow
            action_and_flow_data = self.load_action_and_flow(
                ctxt_file_idx=ctxt_file_idx,
                ctxt_img_filename=str(ctxt_img_filename),
                trgt_img_filename=trgt_img_filename,
                trgt_cam_idx=trgt_cam_idx,
                return_other_views=(False if self.stage == "train" else True),
            )

            ret_dict["context"]["robot_action"] = action_and_flow_data["robot_action"]
            # ret_dict["target"]["flow"] = action_and_flow_data["trgt_optical_flow"]

            # (num_rays)
            ret_dict["target"]["pixel_selector"] = action_and_flow_data[
                "pixel_selector"
            ]
            # (num_rays, 2) ; order: (y, x)
            ret_dict["target"]["pixel_motion"] = action_and_flow_data["pixel_motion"]
            ret_dict["target"]["pixel_visible_mask"] = action_and_flow_data[
                "pixel_visible_mask"
            ]

            # ### render optical flow at other views for comparison.
            # if self.stage != "train":
            #     other_views = action_and_flow_data["other_views"]
            #     other_views["extrinsics"] = torch.einsum(
            #         "ij, njk -> nik", inverse_ctxt_c2w, other_views["extrinsics"]
            #     )
            #     ret_dict["other_views"] = other_views

        return ret_dict
