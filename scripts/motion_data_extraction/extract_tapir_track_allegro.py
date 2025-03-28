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

NUM_VIEWS = 12
BATCH_SIZE = 200
MODEL_TYPE = "bootstapir"

PROMPTS = [
    "robot hand",
    "black robot hand",
    "robot",
]


def segment_robot(image: Float[ndarray, "H W 3"], seg_model: LangSAM):
    idx = 0
    while True:
        if len(PROMPTS) == idx:
            raise ValueError("No mask found for any prompt.")

        masks = segment_robot_all_masks(image, seg_model, PROMPTS[idx])
        idx += 1

        if len(masks) > 0:
            break

    best_mask = masks[0]
    return best_mask


def segment_robot_all_masks(
    image: Float[ndarray, "H W 3"],
    seg_model: LangSAM,
    prompt: str = "white robot hand",
):
    pil_image = Image.fromarray(image.astype("uint8"))
    masks, _, _, _ = seg_model.predict(pil_image, prompt)

    return masks


def build_model(frames, query_points, model_type="tapir"):
    """Compute point tracks and occlusions given frames and query points."""
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
    outputs = model(
        video=frames,
        is_training=False,
        query_points=query_points,
        query_chunk_size=64,
    )
    return outputs


def inference_given_query_points(frames, query_points):
    """Inference on one video.
    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8
      query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

    Returns:
      tracks: [num_points, 3], [-1, 1], [t, y, x]
      visibles: [num_points, num_frames], bool
    """

    # Preprocess video to match model inputs format
    frames = tapnet_model_utils.preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    # Model inference
    rng = jax.random.PRNGKey(42)
    outputs, _ = model_apply(params, state, rng, frames, query_points)
    outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
    tracks, occlusions, expected_dist = (
        outputs["tracks"],
        outputs["occlusion"],
        outputs["expected_dist"],
    )

    # Binarize occlusions
    visibles = tapnet_model_utils.postprocess_occlusions(occlusions, expected_dist)
    # return tracks, visibles

    return {
        "tracks": tracks,
        "visibles": visibles,
    }


# need a function that takes a video as input, mask as input, and run inference for all pixels in the mask
def inference_all_pixels(video, mask, frame_idx: int = 0, batch_size: int = 200):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    x, y = mask.nonzero()
    indices = np.stack([x, y], axis=1)

    # # randomly select 5000 points
    # np.random.shuffle(indices)
    # indices = indices[:5000]

    total_points = indices.shape[0]
    num_frames = video.shape[0]

    total_tracks = np.zeros((total_points, num_frames, 2))
    total_visibles = np.zeros((total_points, num_frames))

    # process points in batches. For last step compute the last batch elements
    for i in range(0, total_points, batch_size):
        print(f"Processing batch [{i}:{i+batch_size}] out of {total_points}")

        start_idx = i
        end_idx = i + batch_size
        if i + batch_size > total_points:
            start_idx = total_points - batch_size
            end_idx = total_points

        batch_query_points = np.concatenate(
            (frame_idx * np.ones((batch_size, 1)), indices[start_idx:end_idx]), axis=-1
        ).astype(np.int32)

        batch_outputs = inference_given_query_points(video, batch_query_points)
        total_tracks[start_idx:end_idx], total_visibles[start_idx:end_idx] = (
            batch_outputs["tracks"],
            batch_outputs["visibles"],
        )

    return {
        "total_tracks": total_tracks,
        "total_visibles": total_visibles,
    }


def load_numpy_image(image_filename, scale_factor: float = 1.0):
    pil_image = Image.open(image_filename)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)
    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
    return image


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames / 255 * 2 - 1
    return frames


def load_video(
    root_path: Path,
    view_idx: int,
    traj_idx: int,
    num_frames: int = 10,
):
    video = []
    for frame_idx in range(num_frames):
        curr_image_file = str(
            root_path
            / Path(f"view_{view_idx}")
            / Path("rgb")
            / Path(f"{traj_idx:05d}_{frame_idx:05d}.png")
        )
        video.append(load_numpy_image(curr_image_file))

    video = np.stack(video, axis=0)

    return video


def save_track_data(
    name: Path,
    tracks: ndarray,
    visibles: ndarray,
):
    np.savez_compressed(
        name,
        tracks=tracks,
        visibles=visibles,
    )


def main(
    capture_folder: Path,
    num_views: int = 12,
    traj_idx_low: int = 0,
    traj_idx_high: int = None,
    num_frames_per_traj: int = 10,
    process_view_idx: int = 0,
    version_stamp: int = 0,
):
    # set batch-size depending on gpu memory

    seg_model = LangSAM()

    ### create folder directory
    for view_idx in range(num_views):
        tapir_dir = capture_folder / f"view_{view_idx}" / "tapir"
        vis_dir = capture_folder / f"view_{view_idx}" / "tapir_vis"

        os.makedirs(tapir_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    raw_capture_filenames = natsorted(
        glob.glob(str(capture_folder / "view_*" / "rgb" / "*.png"))
    )

    if traj_idx_high is None:
        traj_idx_low = max(
            int(raw_capture_filenames[0].split("/")[-1].split("_")[0]), traj_idx_low
        )
        traj_idx_high = int(raw_capture_filenames[-1].split("/")[-1].split("_")[0])

    print(f"Processing Trajs [{traj_idx_low} - {traj_idx_high}]")

    # for view_idx in range(num_views):

    tapir_dir = capture_folder / f"view_{process_view_idx}" / "tapir"
    vis_dir = capture_folder / f"view_{process_view_idx}" / "tapir_vis"

    assert 0 <= version_stamp < num_frames_per_traj

    ran = tqdm.trange(traj_idx_low, traj_idx_high + 1)
    for traj_idx in ran:
        ran.set_description_str(f"view_idx: {process_view_idx}, traj_idx: {traj_idx}")

        video = load_video(
            capture_folder,
            process_view_idx,
            traj_idx,
            num_frames=num_frames_per_traj,
        )

        mask = segment_robot(video[version_stamp], seg_model)

        outputs = inference_all_pixels(
            video, mask, frame_idx=version_stamp, batch_size=BATCH_SIZE
        )
        tracks, visibles = outputs["total_tracks"], outputs["total_visibles"]

        save_track_data(
            tapir_dir / f"{traj_idx:05d}_{version_stamp:03d}.npz",
            tracks,
            visibles,
        )

        viz_top_n = 200
        select_idx = np.random.choice(tracks.shape[0], viz_top_n, replace=False)

        video_viz = viz_utils.paint_point_track(
            video, tracks[select_idx], visibles[select_idx]
        )

        imageio.mimwrite(
            vis_dir / f"{traj_idx:05d}_{version_stamp:03d}.mp4",
            video_viz,
            fps=10,
        )


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

    if MODEL_TYPE == "tapir":
        checkpoint_path = "./tapnet/checkpoints/tapir_checkpoint_panning.npy"
    else:
        checkpoint_path = "./tapnet/checkpoints/bootstapir_checkpoint.npy"

    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state["params"], ckpt_state["state"]

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
