# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for blender dataset"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type, List

import numpy as np
import torch
from PIL import Image
from jaxtyping import Int
from torch import Tensor

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600


@dataclass
class DNeRFDataParserOutputs(DataparserOutputs):
    sample_to_camera_idx: Int[Tensor, "num_samples"] = torch.tensor(
        [], dtype=torch.long
    )


@dataclass
class DNeRFDataParserConfig(DataParserConfig):
    """D-NeRF dataset parser config"""

    _target: Type = field(default_factory=lambda: DNeRFDataParser)
    """target class to instantiate"""
    data: Path = Path("data/dnerf/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background""" """Directory or explicit json file path specifying location of data."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""


@dataclass
class DNeRFDataParser(DataParser):
    """Nerfstudio DatasetParser"""

    config: DNeRFDataParserConfig
    downscale_factor: Optional[int] = None
    includes_time: bool = True
    sample_idx_enabled: bool = True

    def _generate_dataparser_outputs(
        self, split="train", timetable: Optional[List[str]] = None
    ):
        assert (
            self.config.data.exists()
        ), f"Data directory {self.config.data} does not exist."

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        # sample related
        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        dense_features_filenames = []
        joint_pos_filenames = []

        times = []
        sample_inds = []
        sample_to_camera_idx = []
        timestep_indices = []
        joint_positions = dict()

        # camera related
        poses = []
        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sample dependent. we are decoupling cameras and samples
        for sample_cnt, frame in enumerate(meta["frames"]):
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if timetable is not None:
                # TODO: filtering the file based on the time table
                keyname = str(fname).split("/")[-1][:-4] + ".npz"
                if keyname not in timetable:
                    continue
                else:
                    timestep_indices.append(timetable.index(keyname))

            times.append(frame["time"])

            sample_idx = (
                int(frame["time"] * 10)
                if frame.get("sample_idx") is None
                else frame["sample_idx"]
            )
            # TODO: should print warning that says we are assuming the first case to be hsa finger
            sample_inds.append(sample_idx)

            sample_to_camera_idx.append(frame["camera_idx"])

            image_filenames.append(fname)
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(
                    depth_filepath, data_dir, downsample_folder_prefix="depths_"
                )
                depth_filenames.append(depth_fname)

            if "joint_pos" in frame:
                joint_pos = frame["joint_pos"]
                keyname = str(fname).split("/")[-1][:-4]
                if keyname not in joint_positions:
                    joint_positions[keyname] = torch.tensor(joint_pos).float().squeeze()

            dense_features_filepath = data_dir / Path(
                str(filepath)
                .replace("images", "dense_features")
                .replace(".png", ".npz")
            )
            if dense_features_filepath.exists():
                dense_features_filenames.append(str(dense_features_filepath))

            joint_path_head = (
                str(fname.parent.parent.parent)
                .replace("nerfstudio_ready", "raw_captures")
                .replace("small_grid", "08-04-grid")
            )
            joint_path_base = str(fname.name)[:-4] + ".npz"
            joint_filename = Path(joint_path_head) / Path(joint_path_base)
            joint_pos_filenames.append(str(joint_filename))

        # camera dependent
        for cam_cfg in meta["cameras"]:
            if not fx_fixed:
                assert "fl_x" in cam_cfg, "fx not specified in frame"
                fx.append(float(cam_cfg["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in cam_cfg, "fy not specified in frame"
                fy.append(float(cam_cfg["fl_y"]))
            if not cx_fixed:
                assert "cx" in cam_cfg, "cx not specified in frame"
                cx.append(float(cam_cfg["cx"]))
            if not cy_fixed:
                assert "cy" in cam_cfg, "cy not specified in frame"
                cy.append(float(cam_cfg["cy"]))
            if not height_fixed:
                assert "h" in cam_cfg, "height not specified in frame"
                height.append(int(cam_cfg["h"]))
            if not width_fixed:
                assert "w" in cam_cfg, "width not specified in frame"
                width.append(int(cam_cfg["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(cam_cfg["k1"]) if "k1" in cam_cfg else 0.0,
                        k2=float(cam_cfg["k2"]) if "k2" in cam_cfg else 0.0,
                        k3=float(cam_cfg["k3"]) if "k3" in cam_cfg else 0.0,
                        k4=float(cam_cfg["k4"]) if "k4" in cam_cfg else 0.0,
                        p1=float(cam_cfg["p1"]) if "p1" in cam_cfg else 0.0,
                        p2=float(cam_cfg["p2"]) if "p2" in cam_cfg else 0.0,
                    )
                )
            poses.append(np.array(cam_cfg["transform_matrix"]))

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        has_split_files_spec = any(
            f"{split}_filenames" in meta for split in ("train", "val", "test")
        )
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(
                self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"]
            )
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(
                    f"Some filenames for split {split} were not found: {unmatched_filenames}."
                )

            # indices = [
            #     i for i, path in enumerate(image_filenames) if path in split_filenames
            # ]
            # CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            # indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(
                f"The dataset's list of filenames for split {split} is missing."
            )
        else:
            # filter image_filenames and poses based on train/eval split percentage
            num_images = len(image_filenames)
            num_train_images = math.ceil(num_images * self.config.train_split_fraction)
            num_eval_images = num_images - num_train_images
            i_all = np.arange(num_images)
            i_train = np.linspace(
                0, num_images - 1, num_train_images, dtype=int
            )  # equally spaced training images starting and ending at 0 and num_images-1
            i_eval = np.setdiff1d(
                i_all, i_train
            )  # eval images are the remaining images
            assert len(i_eval) == num_eval_images

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(
                f"[yellow] Dataset is overriding orientation method to {orientation_method}"
            )
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -0.0],
                    # [-aabb_scale, -aabb_scale, -0.0],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)
        height = (
            int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)
        )
        width = (
            int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)
        )
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(
                meta["applied_transform"], dtype=transform_matrix.dtype
            )
            transform_matrix = transform_matrix @ torch.cat(
                [
                    applied_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                ],
                0,
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        sample_to_camera_idx = torch.from_numpy(np.array(sample_to_camera_idx)).to(
            torch.long
        )
        times = torch.tensor(times, dtype=torch.float32)
        sample_inds = torch.tensor(sample_inds, dtype=torch.long)

        dataparser_outputs = DNeRFDataParserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": (
                    depth_filenames if len(depth_filenames) > 0 else None
                ),
                "dense_features_filenames": (
                    dense_features_filenames
                    if len(dense_features_filenames) > 0
                    else None
                ),
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "times": times,
                "sample_inds": sample_inds if self.sample_idx_enabled else None,
                "joint_pos_filenames": joint_pos_filenames,
                "timestep_indices": (
                    timestep_indices if len(timestep_indices) > 0 else None
                ),
                "joint_positions": (
                    joint_positions if len(joint_positions) > 0 else None
                ),
            },
            sample_to_camera_idx=sample_to_camera_idx,
        )
        return dataparser_outputs

    def get_dataparser_outputs(
        self, split: str = "train", timetable: Optional[List[str]] = None
    ) -> DNeRFDataParserOutputs:
        """Returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).
            kwargs: kwargs for generating dataparser outputs.

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
        dataparser_outputs = self._generate_dataparser_outputs(split)
        return dataparser_outputs

    def merge_datparser_outputs(
        self, outputs: List[DNeRFDataParserOutputs]
    ) -> DNeRFDataParserOutputs:
        """Merge the outputs of multiple data parsers."""
        if len(outputs) == 1:
            return outputs[0]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        dense_features_filenames = []
        joint_pos_filenames = []
        times = []
        sample_inds = []
        sample_to_camera_idx = []
        timestep_indices = []
        joint_positions = dict()
        for i, output in enumerate(outputs):
            image_filenames.extend(output.image_filenames)
            if output.mask_filenames is not None:
                mask_filenames.extend(output.mask_filenames)
            if output.metadata["depth_filenames"] is not None:
                depth_filenames.extend(output.metadata["depth_filenames"])
            if output.metadata["dense_features_filenames"] is not None:
                dense_features_filenames.extend(
                    output.metadata["dense_features_filenames"]
                )
            if output.metadata["joint_pos_filenames"] is not None:
                joint_pos_filenames.extend(output.metadata["joint_pos_filenames"])
            if output.metadata["timestep_indices"] is not None:
                timestep_indices.extend(output.metadata["timestep_indices"])
            if output.metadata["joint_positions"] is not None:
                joint_positions.update(output.metadata["joint_positions"])
            times.extend(output.metadata["times"])
            sample_inds.extend(output.metadata["sample_inds"])
            sample_to_camera_idx.extend(output.sample_to_camera_idx)

        return DNeRFDataParserOutputs(
            image_filenames=image_filenames,
            cameras=outputs[0].cameras,
            scene_box=outputs[0].scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=outputs[0].dataparser_scale,
            dataparser_transform=outputs[0].dataparser_transform,
            metadata={
                "depth_filenames": (
                    depth_filenames if len(depth_filenames) > 0 else None
                ),
                "dense_features_filenames": (
                    dense_features_filenames
                    if len(dense_features_filenames) > 0
                    else None
                ),
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "times": times,
                "sample_inds": sample_inds if self.sample_idx_enabled else None,
                "joint_pos_filenames": joint_pos_filenames,
                "timestep_indices": (
                    timestep_indices if len(timestep_indices) > 0 else None
                ),
                "joint_positions": (
                    joint_positions if len(joint_positions) > 0 else None
                ),
            },
            sample_to_camera_idx=torch.tensor(sample_to_camera_idx, dtype=torch.long),
        )

    def _get_fname(
        self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_"
    ) -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (
                        data_dir
                        / f"{downsample_folder_prefix}{2**(df+1)}"
                        / filepath.name
                    ).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        # if self.downscale_factor > 1:
        #     return (
        #         data_dir
        #         / f"{downsample_folder_prefix}{self.downscale_factor}"
        #         / filepath.name
        #     )
        return data_dir / filepath
