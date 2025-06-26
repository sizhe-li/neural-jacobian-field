import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import functools
import glob
from pathlib import Path

import haiku as hk
import imageio
import jax
import numpy as np
import torch
import tqdm
import tree
from jaxtyping import Float
from lang_sam import LangSAM
from natsort import natsorted
from numpy import ndarray
from PIL import Image
from tapnet import tapir_model
from tapnet.utils import model_utils as tapnet_model_utils
from tapnet.utils import viz_utils

# Constants
NUM_VIEWS = 12
BATCH_SIZE = 200
MODEL_TYPE = "bootstapir"
PROMPTS = ["robot hand", "black robot hand", "robot"]


def segment_robot(image: Float[ndarray, "H W 3"], seg_model: LangSAM):
    """Segment robot using a series of fallback prompts."""
    for prompt in PROMPTS:
        masks = segment_robot_all_masks(image, seg_model, prompt)
        if len(masks) > 0:
            return masks[0]
    raise ValueError("No mask found for any prompt.")


def segment_robot_all_masks(
    image: Float[ndarray, "H W 3"], seg_model: LangSAM, prompt: str = "white robot hand"
):
    pil_image = Image.fromarray(image.astype("uint8"))
    masks, _, _, _ = seg_model.predict(pil_image, prompt)
    return masks


def build_model(frames, query_points, model_type="tapir"):
    """Construct the appropriate TAPIR model based on type."""
    if model_type == "tapir":
        model = tapir_model.TAPIR(
            bilinear_interp_with_depthwise_conv=False, pyramid_level=0
        )
    elif model_type == "bootstapir":
        model = tapir_model.TAPIR(
            bilinear_interp_with_depthwise_conv=False,
            pyramid_level=1,
            extra_convs=True,
            softmax_temperature=10.0,
        )
    return model(video=frames, is_training=False, query_points=query_points, query_chunk_size=64)


def inference_given_query_points(frames, query_points):
    """Run inference on selected query points."""
    frames = tapnet_model_utils.preprocess_frames(frames)
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    rng = jax.random.PRNGKey(42)
    outputs, _ = model_apply(params, state, rng, frames, query_points)
    outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)

    tracks = outputs["tracks"]
    visibles = tapnet_model_utils.postprocess_occlusions(outputs["occlusion"], outputs["expected_dist"])
    return {"tracks": tracks, "visibles": visibles}


def inference_all_pixels(video, mask, frame_idx: int = 0, batch_size: int = 200):
    """Run inference for all masked pixels in a given frame."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    y, x = mask.nonzero()
    indices = np.stack([y, x], axis=1)
    total_points, num_frames = indices.shape[0], video.shape[0]

    total_tracks = np.zeros((total_points, num_frames, 2))
    total_visibles = np.zeros((total_points, num_frames))

    for i in range(0, total_points, batch_size):
        print(f"Processing batch [{i}:{i + batch_size}] out of {total_points}")
        start_idx = max(i, 0)
        end_idx = min(i + batch_size, total_points)

        batch_query_points = np.concatenate(
            (frame_idx * np.ones((end_idx - start_idx, 1)), indices[start_idx:end_idx]), axis=-1
        ).astype(np.int32)

        batch_outputs = inference_given_query_points(video, batch_query_points)
        total_tracks[start_idx:end_idx] = batch_outputs["tracks"]
        total_visibles[start_idx:end_idx] = batch_outputs["visibles"]

    return {"total_tracks": total_tracks, "total_visibles": total_visibles}


def load_numpy_image(image_filename, scale_factor: float = 1.0):
    """Load and optionally resize an image as a numpy array."""
    pil_image = Image.open(image_filename)
    if scale_factor != 1.0:
        newsize = (int(pil_image.width * scale_factor), int(pil_image.height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
    image = np.array(pil_image, dtype="uint8")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    assert image.ndim == 3 and image.shape[2] in [3, 4]
    return image


def preprocess_frames(frames):
    """Normalize frames to [-1, 1] float32."""
    return (frames.astype(np.float32) / 255.0) * 2 - 1


def load_video(root_path: Path, view_idx: int, traj_idx: int, num_frames: int = 10):
    """Load a video sequence from disk."""
    video = [
        load_numpy_image(root_path / f"view_{view_idx}" / "rgb" / f"{traj_idx:05d}_{i:05d}.png")
        for i in range(num_frames)
    ]
    return np.stack(video, axis=0)


def save_track_data(name: Path, tracks: ndarray, visibles: ndarray):
    """Save tracking results to compressed npz."""
    np.savez_compressed(name, tracks=tracks, visibles=visibles)


def main(
    capture_folder: Path,
    num_views: int = 12,
    traj_idx_low: int = 0,
    traj_idx_high: int = None,
    num_frames_per_traj: int = 10,
    process_view_idx: int = 0,
    version_stamp: int = 0,
):
    seg_model = LangSAM()

    for view_idx in range(num_views):
        (capture_folder / f"view_{view_idx}" / "tapir").mkdir(parents=True, exist_ok=True)
        (capture_folder / f"view_{view_idx}" / "tapir_vis").mkdir(parents=True, exist_ok=True)

    raw_filenames = natsorted(glob.glob(str(capture_folder / "view_*" / "rgb" / "*.png")))
    if traj_idx_high is None:
        traj_idx_low = max(int(Path(raw_filenames[0]).stem.split("_")[0]), traj_idx_low)
        traj_idx_high = int(Path(raw_filenames[-1]).stem.split("_")[0])

    print(f"Processing Trajs [{traj_idx_low} - {traj_idx_high}]")

    tapir_dir = capture_folder / f"view_{process_view_idx}" / "tapir"
    vis_dir = capture_folder / f"view_{process_view_idx}" / "tapir_vis"
    assert 0 <= version_stamp < num_frames_per_traj

    for traj_idx in tqdm.trange(traj_idx_low, traj_idx_high + 1, desc=f"view_idx: {process_view_idx}"):
        video = load_video(capture_folder, process_view_idx, traj_idx, num_frames=num_frames_per_traj)
        mask = segment_robot(video[version_stamp], seg_model)

        outputs = inference_all_pixels(video, mask, frame_idx=version_stamp, batch_size=BATCH_SIZE)
        save_track_data(tapir_dir / f"{traj_idx:05d}_{version_stamp:03d}.npz", outputs["total_tracks"], outputs["total_visibles"])

        # Visualization
        select_idx = np.random.choice(outputs["total_tracks"].shape[0], 200, replace=False)
        video_viz = viz_utils.paint_point_track(video, outputs["total_tracks"][select_idx], outputs["total_visibles"][select_idx])
        imageio.mimwrite(vis_dir / f"{traj_idx:05d}_{version_stamp:03d}.mp4", video_viz, fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture_folder", type=str, required=True)
    parser.add_argument("--traj_idx_low", type=int, default=0)
    parser.add_argument("--traj_idx_high", type=int, default=None)
    parser.add_argument("--num_frames_per_traj", type=int, default=10)
    parser.add_argument("--view_idx", type=int, default=0)
    parser.add_argument("--version_stamp", type=int, default=0)
    args = parser.parse_args()

    capture_folder = Path(args.capture_folder)

    # Load checkpoint
    checkpoint_path = {
        "tapir": "./tapnet/checkpoints/tapir_checkpoint_panning.npy",
        "bootstapir": "./tapnet/checkpoints/bootstapir_checkpoint.npy"
    }[MODEL_TYPE]
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state["params"], ckpt_state["state"]

    # JIT compile
    build_model_fn = functools.partial(build_model, model_type=MODEL_TYPE)
    model = hk.transform_with_state(build_model_fn)
    model_apply = jax.jit(model.apply)

    main(
        capture_folder=capture_folder,
        num_views=NUM_VIEWS,
        traj_idx_low=args.traj_idx_low,
        traj_idx_high=args.traj_idx_high,
        process_view_idx=args.view_idx,
        version_stamp=args.version_stamp,
        num_frames_per_traj=args.num_frames_per_traj,
    )
