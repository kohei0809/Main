#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import quaternion  # noqa # pylint: disable=unused-import


def quaternion_to_rotation(q_r, q_i, q_j, q_k):
    r"""
    ref: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    s = 1  # unit quaternion
    rotation_mat = np.array(
        [
            [
                1 - 2 * s * (q_j ** 2 + q_k ** 2),
                2 * s * (q_i * q_j - q_k * q_r),
                2 * s * (q_i * q_k + q_j * q_r),
            ],
            [
                2 * s * (q_i * q_j + q_k * q_r),
                1 - 2 * s * (q_i ** 2 + q_k ** 2),
                2 * s * (q_j * q_k - q_i * q_r),
            ],
            [
                2 * s * (q_i * q_k - q_j * q_r),
                2 * s * (q_j * q_k + q_i * q_r),
                1 - 2 * s * (q_i ** 2 + q_j ** 2),
            ],
        ],
        #dtype=np.float32,
        dtype=np.float64,
    )
    return rotation_mat


def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion

    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate

    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format
    """
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def compute_pixel_coverage(instance_seg, object_id):
    cand_mask = instance_seg == object_id
    score = cand_mask.sum().astype(np.float64) / cand_mask.size
    return score

def compute_heading_from_quaternion(rotation: np.quaternion) -> float:
    r"""Converts np.quaternion to a heading angle scalar.

    Args:
        rotation - represents a counter-clockwise rotation about Y-axis.
    Returns:
        Heading angle with clockwise rotation from -Z to X being +ve.
    """
    direction_vector = np.array([0, 0, -1])  # Forward vector
    heading_vector = quaternion_rotate_vector(rotation.inverse(), direction_vector)
    # Flip sign to compute clockwise rotation
    phi = -cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

    return phi
