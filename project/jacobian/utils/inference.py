import os
from pathlib import Path

import hydra
import mediapy as media
import numpy as np
import torch
import random
from einops import rearrange
from hydra import compose, initialize
from jacobian.dataset.pusher_dataset import DatasetPusher
from jacobian.models.scene_jacobian import Model
from jacobian.raft_wrapper import InputPadder, RaftWrapper
from jacobian.utils.io import numpy_to_torch_image
from mujoco_sim.env.push_env import PushEnv
from omegaconf import DictConfig
from torchvision.utils import flow_to_image


CKPT_FILEPATH = "/home/sizheli/project/scene-jacobian-discovery/outputs/2024-09-18/12-49-21/scene-jacobian-discovery/3xjy4qlx/checkpoints/epoch=0-step=20000.ckpt"
H, W = 256, 256
LETTERS = ["A", "E", "G", "M", "R", "T", "V"]
NUM_RUNS = 10


def load_jacobian(ckpt_fp: str, cfg: DictConfig, device: str):
    ckpt = torch.load(ckpt_fp, weights_only=True)

    # remove "model" from prefix
    state_dict = {}
    for key in ckpt["state_dict"]:
        if key.startswith("raft"):
            continue
        new_key = key
        if key.startswith("model"):
            new_key = key[6:]
        state_dict[new_key] = ckpt["state_dict"][key]

    model = Model(cfg.wrapper.model)

    model.load_state_dict(state_dict)
    model.eval()

    model = model.to(device)

    return model


def load_raft(device: str):
    raft = RaftWrapper()
    raft = raft.eval()

    raft = raft.to(device)

    return raft


def run_jacobian(model: Model, video: list, joint_pos: list, device: str):

    # normalize input images
    seq_input_img_th = torch.stack([numpy_to_torch_image(img) for img in video], dim=0)[
        :-1
    ]
    seq_input_img_th = seq_input_img_th.to(device)
    # normalize input robot commands
    seq_input_cmd_th = torch.FloatTensor(joint_pos)[..., :2]
    seq_input_cmd_th = (seq_input_cmd_th - DatasetPusher.min_qpos) / (
        DatasetPusher.max_qpos - DatasetPusher.min_qpos
    )
    seq_input_cmd_th = 4 * torch.diff(seq_input_cmd_th, dim=0)
    seq_input_cmd_th = seq_input_cmd_th.to(device)

    return model.forward(seq_input_img_th, seq_input_cmd_th)


def run_raft(raft: RaftWrapper, video: list, device: str, window_size: int = 12):
    # convert video to torch
    video_th = torch.from_numpy(np.stack(video, axis=0))
    video_th = rearrange(video_th, "t h w c -> t c h w").to(device)
    video_th = (video_th / 255.0).float()

    H, W = video_th.shape[-2:]

    # resize the long dimension to 768
    long_dim = 768
    if H > W:
        new_H, new_W = long_dim, int(long_dim / H * W)
    else:
        new_H, new_W = int(long_dim / W * H), long_dim

    video_th = torch.nn.functional.interpolate(video_th, (new_H, new_W))
    input_padder = InputPadder(video_th.shape)

    # processing in small batches to avoid GPU OOM, sliding window
    num_frames = video_th.shape[0]

    # Initialize the parameters
    start = 0
    end = num_frames

    total_pred_flows_raft = []
    # Loop to generate the sliding windows
    for i in range(start, end, window_size - 1):
        window_start = i
        window_end = i + window_size
        window_end = min(window_end, end)

        sbatch = video_th[window_start:window_end]

        # Convert the video to a torch tensor
        sbatch = input_padder.pad(sbatch)[0].unsqueeze(0).to(device)
        with torch.no_grad():
            sbatch_pred_flows = raft.forward_flow(sbatch, chunk=50)

        total_pred_flows_raft.append(sbatch_pred_flows)

    total_pred_flows_raft = torch.cat(total_pred_flows_raft, dim=0)  # T - 1, 2, H, W
    print(total_pred_flows_raft.shape)

    return total_pred_flows_raft


def produce_video(
    model: Model, raft: RaftWrapper, video: list, joint_pos: list, device: str
) -> np.array:
    jacobian_output = run_jacobian(model, video, joint_pos, device)
    raft_output = run_raft(raft, video, device)
    # plot flow
    total_pred_flows_rgb = flow_to_image(jacobian_output.flow)
    # reshape to original height and width
    total_pred_flows_rgb = torch.nn.functional.interpolate(total_pred_flows_rgb, (H, W))
    total_pred_flows_rgb = rearrange(total_pred_flows_rgb, "t c h w -> t h w c")
    total_pred_flows_rgb = total_pred_flows_rgb.cpu().numpy()

    # plot raft flow
    total_pred_flows_raft_rgb = flow_to_image(raft_output)
    # reshape to original height and width
    total_pred_flows_raft_rgb = torch.nn.functional.interpolate(
        total_pred_flows_raft_rgb, (H, W)
    )
    total_pred_flows_raft_rgb = rearrange(
        total_pred_flows_raft_rgb, "t c h w -> t h w c"
    )
    total_pred_flows_raft_rgb = total_pred_flows_raft_rgb.cpu().numpy()

    print(np.array(video).shape)
    print(total_pred_flows_raft_rgb.shape)
    print(total_pred_flows_rgb.shape)

    # concatenate the flow and the video side by side, trim the last frame of the video to match time length
    vis_video = np.concatenate(
        [np.array(video)[:-1, ...], total_pred_flows_rgb, total_pred_flows_raft_rgb],
        axis=2,
    )
    return vis_video


def main():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../../project/configurations")
    cfg = compose(config_name="wrapper/pusher.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_jacobian(CKPT_FILEPATH, cfg, device)
    raft = load_raft(device)

    for run in range(NUM_RUNS):
        video = []
        joint_pos = []

        letters = random.sample(LETTERS, 3)

        def get_obs(self):
            image = self.render("birdview")
            video.append(image)
            joint_pos.append(self.data.get_body_xpos(f"pusher_main").copy())

        write_path = str(
            Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
            / "results"
            / "letters_generalization"
            / f"letter_{letters}_run_{run}.mp4"
        )
        env = PushEnv(use_object=True, object_type=letters, num_blocks=3)
        env.sample_rand_traj(horizon=50, get_obs=get_obs)
        vis_video = produce_video(model, raft, video, joint_pos, device)
        media.write_video(write_path, vis_video)


if __name__ == "__main__":
    main()
