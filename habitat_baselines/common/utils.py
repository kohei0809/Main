#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from habitat.utils.visualizations.utils import images_to_video
import quaternion

from habitat.core.logging import logger

import random


class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        #logger.info("################################")
        #logger.info(x)
        pre_x = x
        x = self.linear(x)
        #logger.info(x)

        # 0行目が全てnanであるかを判定
        is_nan_row = torch.isnan(x).all(dim=1)

        # NaN行がある場合のみ処理
        if is_nan_row.any():
            # NaNではない行の値
            non_nan_rows = x[~is_nan_row]

            # 非NaN行がない場合はランダム値で補完
            if len(non_nan_rows) == 0:
                random_values = torch.tensor([[random.random() for _ in range(x.size(1))]], dtype=x.dtype, device=x.device)
                x[is_nan_row] = random_values
            else:
                # 最初の非NaN行でNaN行を一括補完
                x[is_nan_row] = non_nan_rows[0]
        

        """
        # 0行目が全てnanである場合、他の行をコピーした値にする
        for i in range(x.size(0)):
            if is_nan_row[i]:
                logger.info(f"######## There is none i={i}#######")
                #logger.info(pre_x)
                #logger.info(x)
                y = x[~is_nan_row]
                if len(y) == 0:
                    logger.info("len(y) == 0")
                    y = torch.from_numpy(np.array([random.random() for _ in range(3)]))
                    x[i] = y
                else:
                    x[i] = y[0]
        """

        return CustomFixedCategorical(logits=x)


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)
    recon_sensor = ["delta", "pose_estimation_mask", "pose_refs"]

    for obs in observations:
        for sensor in obs:
            if sensor in recon_sensor:
                continue
            if sensor == "semantic":
                obs[sensor] = obs[sensor].astype(np.int64)     
            
            batch[sensor].append(_to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = (
            torch.stack(batch[sensor], dim=0)
            .to(device=device)
            .to(dtype=torch.float)
        )

    return batch


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:
    r""" Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    # models_paths.sort(key=os.path.getmtime)
    models_paths.sort(key = lambda x: int(x.split(".")[1]))
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: int,
    metrics: Dict[str, float],
    name_ci = None,
    fps: int = 10,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    in_metric = ['exp_area', 'ci']
    for k, v in metrics.items():
        if k in in_metric:
            metric_strs.append(f"{k}={v:.2f}")

    if name_ci is None:
        video_name = f"episode={episode_id}-".join(
            metric_strs
        )
    else:
        video_name = f"episode={episode_id}-" + str(name_ci)
    
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)

def generate_video2(
    video_dir: Optional[str],
    images: List[np.ndarray],
    fps: int = 100,
    video_name: str = "video",
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return
    
    images_to_video(images, video_dir, video_name, fps)

def quat_from_angle_axis(theta: float, axis: np.ndarray) -> np.quaternion:
    r"""Creates a quaternion from angle axis format

    :param theta: The angle to rotate about the axis by
    :param axis: The axis to rotate about
    :return: The quaternion
    """
    axis = axis.astype(float)
    axis /= np.linalg.norm(axis)
    return quaternion.from_rotation_vector(theta * axis)

class to_grid():
    def __init__(self, global_map_size, coordinate_min, coordinate_max):
        self.global_map_size = global_map_size
        self.coordinate_min = coordinate_min
        self.coordinate_max = coordinate_max
        self.grid_size = (coordinate_max - coordinate_min) / global_map_size

    def get_grid_coords(self, positions):
        grid_x = ((self.coordinate_max - positions[:, 0]) / self.grid_size).round()
        grid_y = ((positions[:, 1] - self.coordinate_min) / self.grid_size).round()
        return grid_x, grid_y
