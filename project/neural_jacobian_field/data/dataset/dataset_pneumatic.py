from dataclasses import dataclass
from typing import Literal

from PIL import ImageFile

from .dataset import DatasetCfgCommon, DatasetCommon, Stage
from .dataset import QposItems

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


@dataclass
class DatasetPneumaticHandOnlyCfg(DatasetCfgCommon):
    name: Literal["pneumatic_hand_only"]


class DatasetPneumaticHandOnly(DatasetCommon):
    cfg: DatasetPneumaticHandOnlyCfg
    near: float = 0.5
    far: float = 10.0
    repeat = 1000
    scale_factor = 1.0

    def __init__(
        self,
        cfg: DatasetPneumaticHandOnlyCfg,
        stage: Stage = "train",
    ):
        super().__init__(cfg, stage)

    def preprocess_qpos(self, qpos_items: QposItems) -> QposItems:
        curr_qpos = qpos_items.curr_qpos.clone()
        next_qpos = qpos_items.next_qpos.clone()

        qpos_minimum = qpos_items.qpos_minimum.clone()
        qpos_maximum = qpos_items.qpos_maximum.clone()

        if len(curr_qpos) == 11:
            (
                new_qpos_minimum,
                new_qpos_maximum,
                curr_qpos,
                next_qpos,
            ) = process_joints_new_hand(
                qpos_minimum,
                qpos_maximum,
                curr_qpos,
                next_qpos,
            )
        else:
            (
                new_qpos_minimum,
                new_qpos_maximum,
                curr_qpos,
                next_qpos,
            ) = process_joints_move_arm(
                qpos_minimum,
                qpos_maximum,
                curr_qpos,
                next_qpos,
            )

        return QposItems(
            curr_qpos=curr_qpos,
            next_qpos=next_qpos,
            qpos_minimum=new_qpos_minimum,
            qpos_maximum=new_qpos_maximum,
        )

    # def get_max_num_frames_per_traj(self, traj_idx: int) -> int:
    #     return 7 if traj_idx == 511 else self.cfg.max_num_frames_per_traj
