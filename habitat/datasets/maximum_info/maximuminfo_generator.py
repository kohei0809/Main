#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np

import habitat_sim
from habitat.core.simulator import Simulator
from habitat.datasets.utils import get_action_shortest_path
from habitat.tasks.nav.nav import MaximumInformationEpisode
from habitat.core.logging import logger

r"""A minimum radius of a plane that a point should be part of to be
considered  as a target or source location. Used to filter isolated points
that aren't part of a floor.
"""
ISLAND_RADIUS_LIMIT = 1.5

def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.

    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(
    s, t, sim, near_dist, far_dist, geodesic_to_euclid_ratio
):
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)
    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
        #  t are from same floor
        return False, 0
    d_separation = sim.geodesic_distance(s, [t])
    if d_separation == np.inf:
        return False, 0
    if not near_dist <= d_separation <= far_dist:
        return False, 0
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0
    return True, d_separation


def _create_episode(
    episode_id, 
    scene_id, 
    start_position, 
    start_rotation
) -> Optional[MaximumInformationEpisode]:
    return MaximumInformationEpisode(
        episode_id=str(episode_id),
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        description=""
    )


def generate_maximuminfo_episode(
    sim: Simulator, 
    num_episodes: int = -1, 
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
) -> MaximumInformationEpisode:
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        if episode_count % 10000 == 0:
            logger.info(episode_count)
        target_position = sim.sample_navigable_point()

        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            continue

        for retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )
        if is_compatible:
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            scene_id = sim.config.SCENE
            episode = _create_episode(
                episode_id=episode_count,
                scene_id=scene_id,
                start_position=source_position,
                start_rotation=source_rotation,
            )

            episode_count += 1
            yield episode
            
        
def generate_maximuminfo_episode2(sim: Simulator, num_episodes: int = -1, z_list: list = []) -> MaximumInformationEpisode:
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        if episode_count % 1000 == 0:
            logger.info(episode_count)
        source_position = sim.sample_navigable_point()
        
        if (len(z_list) != 0) and ((source_position[1] in z_list) == False):
            continue
        
        #print(source_position)
        angle = np.random.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

        episode = _create_episode(
            episode_id=episode_count,
            scene_id=sim.config.SCENE,
            start_position=source_position,
            start_rotation=source_rotation,
        )

        episode_count += 1
        yield episode
