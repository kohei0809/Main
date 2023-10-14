#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np

from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import MaximumInformationEpisode
from habitat.core.logging import logger


def _create_episode(episode_id, scene_id, start_position, start_rotation,) -> Optional[MaximumInformationEpisode]:
    return MaximumInformationEpisode(
        episode_id=str(episode_id),
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
    )


def generate_maximuminfo_episode(sim: Simulator, num_episodes: int = -1,) -> MaximumInformationEpisode:
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        if episode_count % 100000 == 0:
            logger.info(episode_count)
        source_position = sim.sample_navigable_point()
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
        
        
def generate_maximuminfo_episode2(sim: Simulator, num_episodes: int = -1, z_list: list = []) -> MaximumInformationEpisode:
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        if episode_count % 10000 == 0:
            logger.info(episode_count)
        source_position = sim.sample_navigable_point()
        if (source_position[1] in z_list) == False:
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
