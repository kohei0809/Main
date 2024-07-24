#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@habitat.registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type

import numpy as np
import pandas as pd

from habitat.core.logging import logger
import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.utils.visualizations import fog_of_war, maps
from habitat.core.env import RLEnv
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_rotate_vector,
)


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return habitat.registry.get_env(env_name)


@habitat.registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._take_measure_name = self._rl_config.SUCCESS_MEASURE
        self._subsuccess_measure_name = self._rl_config.SUBSUCCESS_MEASURE
        self._previous_measure = None

        super().__init__(self._core_env_config, dataset)

    def reset(self):
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        #logger.info("######## step in NavRLEnv ##########")
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations, **kwargs):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        if self._episode_subsuccess():
            current_measure = self._env.task.foundDistance

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_subsuccess():
            self._previous_measure = self._env.get_metrics()[self._reward_measure_name]

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
        elif self._episode_subsuccess():
            reward += self._rl_config.SUBSUCCESS_REWARD
        elif self._env.task.is_found_called and self._rl_config.FALSE_FOUND_PENALTY:
            reward -= self._rl_config.FALSE_FOUND_PENALTY_VALUE

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="InfoRLEnv")
class InfoRLEnv(RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._take_picture_name = self._rl_config.TAKE_PICTURE_MEASURE
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._picture_measure_name = self._rl_config.PICTURE_MEASURE


        self._previous_area = None
        self._previous_distance = None
        
        self._map_resolution = (300, 300)
        self._coordinate_min = -120.3241-1e-6
        self._coordinate_max = 120.0399+1e-6
        
        super().__init__(self._core_env_config, dataset)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def reset(self):
        observations = super().reset()
        self._previous_area = 0.0
        
        self._scene_data = pd.DataFrame(columns=['object_name', 'id'])
        #semantic_scene = self._env._sim._sim.semantic_scene
        semantic_scene = self._env._sim.semantic_scene
        for level in semantic_scene.levels:
            for region in level.regions:
                for obj in region.objects:
                    new_row = pd.DataFrame({'object_name': [obj.category.name()], 'id': [int(obj.id.split('_')[-1])]})
                    self._scene_data = pd.concat([self._scene_data, new_row], ignore_index=True)

        return observations

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)
    
    def step2(self, *args, **kwargs):
        return super().step2(*args, **kwargs)
    

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )
        
    # 観測済みのマップを作成
    def _cal_explored_rate(self, top_down_map, fog_of_war_map):
        num = 0.0 # 探索可能範囲のグリッド数
        num_exp = 0.0 # 探索済みの範囲のグリッド数
        rate = 0.0
        
        for i in range(len(top_down_map)):
            for j in range(len(top_down_map[0])):
                # 探索可能範囲
                if top_down_map[i][j] != 0:
                    # 探索済み範囲
                    if fog_of_war_map[i][j] == 1:
                        num_exp += 1
                    
                    num += 1
                    
        if num == 0:
            rate = 0.0
        else:
            rate = num_exp / num
                
        return rate

    def get_reward(self, observations, **kwargs):
        #logger.info("###### get_reward 1 in InfoRLEnv #########")
        reward = self._rl_config.SLACK_REWARD
        ci = -1000
        picture_value = -1
        info = self.get_info(observations)
        #logger.info("###### get_reward 2 in NvRLEnv #########")

        if "smooth_map_value" in info:
            #logger.info("###### get_reward 3 in NvRLEnv #########")
            smooth_current_area = info["smooth_map_value"]
            area_reward = smooth_current_area / 50      
            output = 0.0
            reward += area_reward
            #logger.info(f"smooth_current_area={smooth_current_area}, smooth_value={smooth_value}")
        else:
            # area_rewardの計算
            _top_down_map = info["top_down_map"]["map"]
            _fog_of_war_map = info["top_down_map"]["fog_of_war_mask"]

            current_area = self._cal_explored_rate(_top_down_map, _fog_of_war_map)
            current_area *= 10
            # area_rewardを足す
            area_reward = current_area - self._previous_area
            reward += area_reward
            output = 0.0
            self._previous_area = current_area
        
        #logger.info("###### get_reward 4 in NvRLEnv #########")
        measure = self._env.get_metrics()[self._picture_measure_name]
        picture_value = measure

        agent_position = self._env._sim.get_agent_state().position

        #logger.info("###### get_reward 5 in NvRLEnv #########")
        return reward, picture_value, area_reward, output, self._take_picture(), self._scene_data, agent_position[0], agent_position[2]

    
    def get_reward2(self, observations, **kwargs):
        reward = self._rl_config.SLACK_REWARD
        #reward = 0
        ci = -1000
        matrics = None
        
        agent_position = self._env._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        agent_position = np.array([a_x, a_y])
    

        picture_value = self._env.get_metrics()["saliency"]

        return picture_value  
    
    def get_polar_angle(self):
        agent_state = self._env._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation
        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def _take_picture(self):
        return self._env.get_metrics()[self._take_picture_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
