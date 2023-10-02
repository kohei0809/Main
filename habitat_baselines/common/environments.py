#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
from utils.log_manager import LogManager
from utils.log_writer import LogWriter
import sys
import numpy as np

from habitat.core.logging import logger
import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.utils.visualizations import fog_of_war, maps
from habitat.core.env import RLEnv
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
)


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._take_measure_name = self._rl_config.SUCCESS_MEASURE
        self._subsuccess_measure_name = self._rl_config.SUBSUCCESS_MEASURE


        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
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


    def _episode_subsuccess(self):
        return self._env.get_metrics()[self._subsuccess_measure_name]

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


        self._previous_measure = None
        self._previous_action = None
        
        self._map_resolution = (300, 300)
        self._coordinate_min = -120.3241-1e-6
        self._coordinate_max = 120.0399+1e-6
        
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        self.fog_of_war_map_all = None
        observations = super().reset()
        self._previous_measure = 0.0
        
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

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
        reward = self._rl_config.SLACK_REWARD
        ci = -sys.float_info.max

        #観測領域のreward
        #exp_area = self._env.get_metrics()[self._reward_measure_name].sum() 
        #current_measure = exp_area * 100
        """
        logger.info("AREA: " + str(exp_area) + "," + str(current_measure))
        logger.info(current_measure - self._previous_measure)
        top_down_map = self._env.task.sceneMap
        count = 0
        for i in range(top_down_map.shape[0]):
            for j in range(top_down_map.shape[1]):
                if top_down_map[i][j] != maps.MAP_INVALID_POINT:
                    count += 1
                
        logger.info("TOP_DOWN_MAP: " + str(count))
        logger.info(str(current_measure/count*100) + "%")
        """
        
        matrics = None
        fog_of_war_map = None
        self._top_down_map = self._env.task.sceneMap
        
        agent_position = self._env._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        agent_position = np.array([a_x, a_y])
        
        # area_rewardの計算
        if self.fog_of_war_map_all is None:
            self.fog_of_war_map_all = np.zeros_like(self._top_down_map)
            
        self.fog_of_war_map_all = fog_of_war.reveal_fog_of_war(
            self._top_down_map,
            self.fog_of_war_map_all,
            agent_position,
            self.get_polar_angle(),
            fov=self._config.TASK.PICTURE_MAP.FOV,
            max_line_len=self._config.TASK.PICTURE_MAP.VISIBILITY_DIST
            * max(self._map_resolution)
            / (self._coordinate_max - self._coordinate_min)
        )
        
        current_measure = self._cal_explored_rate(self._top_down_map, self.fog_of_war_map_all)
        current_measure *= 10000
        #current_measure *= 100

        if self._take_picture():
            measure = self._env.get_metrics()[self._picture_measure_name]
            ci, matrics = measure[0], measure[1]
                
            fog_of_war_map = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                np.zeros_like(self._top_down_map),
                agent_position,
                self.get_polar_angle(),
                fov=self._config.TASK.PICTURE_MAP.FOV,
                max_line_len=self._config.TASK.PICTURE_MAP.VISIBILITY_DIST
                * max(self._map_resolution)
                / (self._coordinate_max - self._coordinate_min)
            )
            
            ##
            #壁などをマーク付け(1: 写真を撮った範囲, 2: 不可侵領域, 3: 壁, (4:現在位置))
            OUT_REGION = 1
            WALL_REGION = 2
            ##
            x, y = self._map_resolution
            for i in range(y):
                for j in range(x):
                    if self._top_down_map[i][j] == OUT_REGION:
                        fog_of_war_map[i][j] = 2
                    elif self._top_down_map[i][j] == WALL_REGION:
                        fog_of_war_map[i][j] = 3
            
            
            #####################

        elif self._env.task.is_found_called and self._rl_config.FALSE_FOUND_PENALTY:
            reward -= self._rl_config.FALSE_FOUND_PENALTY_VALUE
            
        # area_rewardを足す
        area_reward = current_measure - self._previous_measure
        reward += area_reward
        output = self._previous_measure
        self._previous_measure = current_measure

        return [reward, ci, current_measure, output], matrics, fog_of_war_map, self._top_down_map
    
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

