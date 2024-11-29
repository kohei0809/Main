#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type, Union

import attr
import numpy as np
from gym import spaces
import math
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torchvision import transforms
from scipy import stats
from scipy.ndimage import label
import copy

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
    compute_heading_from_quaternion,
)
from habitat.utils.visualizations import fog_of_war, maps

from reconstruction_model.common import quat_from_coeffs

from TranSalNet.utils.data_process import preprocess_img, postprocess_img
from TranSalNet.TranSalNet_Dense import TranSalNet

from log_manager import LogManager

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

cv2 = try_cv2_import()


MAP_THICKNESS_SCALAR: int = 1250


def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy.
    """

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius.
    """

    room_id: str = attr.ib(default=None, validator=not_none_validator)
    room_name: Optional[str] = None

@attr.s(auto_attribs=True, kw_only=True)
class MaximumInformationEpisode(Episode):
    """
    start_room: Optional[str] = None
    shortest_paths: Optional[List[ShortestPathPoint]] = None
    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    object_category: Optional[List[str]] = None
    object_index: Optional[int] = None
    currGoalIndex: Optional[int] = 0 
    """

@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.
    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    start_room: Optional[str] = None
    shortest_paths: Optional[List[ShortestPathPoint]] = None
    
    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    object_category: Optional[List[str]] = None
    object_index: Optional[int] = None
    currGoalIndex: Optional[int] = 0 


@registry.register_sensor
class PointGoalSensor(Sensor):
    r"""Sensor for PointGoal observations which are used in PointGoal Navigation.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.
            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]
    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim

        self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pointgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _compute_pointgoal(
        self, source_position, source_rotation, goal_position
    ):
        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._goal_format == "POLAR":
            if self._dimensionality == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            else:
                _, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                theta = np.arccos(
                    direction_vector_agent[1]
                    / np.linalg.norm(direction_vector_agent)
                )
                rho = np.linalg.norm(direction_vector_agent)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            if self._dimensionality == 2:
                return np.array(
                    [-direction_vector_agent[2], direction_vector_agent[0]],
                    dtype=np.float32,
                )
            else:
                return direction_vector_agent

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):
        source_position = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            source_position, rotation_world_start, goal_position
        )


@registry.register_sensor(name="PointGoalWithGPSCompassSensor")
class IntegratedPointGoalGPSAndCompassSensor(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.
            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]
    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pointgoal_with_gps_compass"

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )


@registry.register_sensor(name="PositionSensor")
class AgentPositionSensor(Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return self._sim.get_agent_state().position


@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "heading"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=float)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        return self._quat_to_xy_heading(rotation_world_agent.inverse())


@registry.register_sensor(name="CompassSensor")
class EpisodicCompassSensor(HeadingSensor):
    r"""The agents heading in the coordinate frame defined by the epiosde,
    theta=0 is defined by the agents state at t=0
    """

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "compass"

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        return self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )


@registry.register_sensor(name="GPSSensor")
class EpisodicGPSSensor(Sensor):
    r"""The agents current location in the coordinate frame defined by the episode,
    i.e. the axis it faces along and the origin is defined by its state at t=0
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position = agent_state.position

        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        if self._dimensionality == 2:
            return np.array(
                [-agent_position[2], agent_position[0]], dtype=np.float32
            )
        else:
            return agent_position.astype(np.float32)


@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "MAX_DETECTION_RADIUS", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "proximity"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=float,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        return self._sim.distance_to_closest_obstacle(
            current_position, self._max_detection_radius
        )

RGBSENSOR_DIMENSION = 3
@registry.register_sensor(name="PoseEstimationRGBSensor")
class PoseEstimationRGBSensor(Sensor):
    r"""Sensor for PoseEstimation observations.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PoseEstimation sensor.

    Attributes:
        _nRef: number of pose references
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
        ):
        self._sim = sim
        self._nRef = 100
        super().__init__(config=config)

        self.current_episode_id = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pose_refs"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(
                self._nRef,
                self.config.HEIGHT,
                self.config.WIDTH,
                RGBSENSOR_DIMENSION,
            ),
            dtype=np.uint8,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        episode_id = (episode.episode_id, episode.scene_id)
        # Render the pose references only at the start of each episode.
        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id
            #ref_positions = episode.pose_ref_positions
            #ref_rotations = episode.pose_ref_rotations

            ref_positions = []
            ref_rotations = []
            for _ in range(self._nRef):
                position = self._sim.sample_navigable_point()            
                angle = np.random.uniform(0, 2 * np.pi)
                rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
                ref_positions.append(position)
                ref_rotations.append(rotation)

            start_position = episode.start_position
            start_rotation = episode.start_rotation
            start_quat = quat_from_coeffs(start_rotation)
            # Measures the angle from forward to right directions.
            start_heading = compute_heading_from_quaternion(start_quat)
            xs, ys = -start_position[2], start_position[0]
            ref_rgb = []
            ref_reg = []
            for position, rotation in zip(ref_positions, ref_rotations):
                # Get data only from the RGB sensor
                obs = self._sim.get_observations_at(position, rotation)["rgb"]
                # remove alpha channel
                #obs = obs[:, :, :RGBSENSOR_DIMENSION]
                ref_rgb.append(obs)

                rotation = quat_from_coeffs(rotation)
                # Measures the angle from forward to right directions.
                ref_heading = compute_heading_from_quaternion(rotation)
                xr, yr = -position[2], position[0]
                # Compute vector from start to ref assuming start is
                # facing forward @ (0, 0)
                rad_sr = np.sqrt((xr - xs) ** 2 + (yr - ys) ** 2)
                phi_sr = np.arctan2(yr - ys, xr - xs) - start_heading
                theta_sr = ref_heading - start_heading
                # Normalize theta_sr
                theta_sr = np.arctan2(np.sin(theta_sr), np.cos(theta_sr))
                ref_reg.append((rad_sr, phi_sr, theta_sr, 0.0))

            # Add dummy images to compensate for fewer than nRef references.
            if len(ref_rgb) < self._nRef:
                dummy_image = np.zeros_like(ref_rgb[0])
                for i in range(len(ref_rgb), self._nRef):
                    ref_rgb.append(dummy_image)
                for i in range(len(ref_reg), self._nRef):
                    ref_reg.append((0.0, 0.0, 0.0, 0.0))
            self._pose_ref_rgb = np.stack(ref_rgb, axis=0)
            self._pose_ref_reg = np.array(ref_reg)

            return np.copy(self._pose_ref_rgb), np.copy(self._pose_ref_reg)

        else:
            return None, None


@registry.register_sensor(name="PoseEstiomationMaskSensor")
class PoseEstimationMaskSensor(Sensor):
    r"""Sensor for PoseEstimation observations. Returns the mask
    indicating which references are valid.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PoseEstimation sensor.

    Attributes:
        _nRef: number of pose references
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._nRef = 100
        super().__init__(config=config)

        self.current_episode_id = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pose_estimation_mask"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-1000000.0, high=1000000.0, shape=(self._nRef,), dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        episode_id = (episode.episode_id, episode.scene_id)
        if self.current_episode_id != episode_id:
            pose_ref_mask = np.ones((self._nRef,))
            #pose_ref_mask[len(episode.pose_ref_positions) :] = 0
            self._pose_ref_mask = pose_ref_mask
            return np.copy(self._pose_ref_mask)
        else:
            return None


@registry.register_sensor(name="DeltaSensor")
class DeltaSensor(Sensor):
    r"""Sensor that returns the odometer readings from the previous action."""
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

        self.current_episode_id = None
        self.prev_position = None
        self.prev_rotation = None
        self.start_position = None
        self.start_rotation = None
        
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "delta"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-1000000.0, high=1000000.0, shape=(4,), dtype=np.float32,)

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        episode_id = (episode.episode_id, episode.scene_id)
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        agent_rotation = agent_state.rotation

        if self.current_episode_id != episode_id:
            # A new episode has started
            self.current_episode_id = episode_id
            delta = np.array([0.0, 0.0, 0.0, 0.0])
            self.start_position = copy.deepcopy(agent_position)
            self.start_rotation = copy.deepcopy(agent_rotation)
        else:
            current_position = agent_position
            current_rotation = agent_rotation
            # For the purposes of this sensor, forward is X and rightward is Y.
            # The heading is measured positively from X to Y.
            curr_x, curr_y = -current_position[2], current_position[0]
            curr_heading = compute_heading_from_quaternion(current_rotation)
            prev_x, prev_y = -self.prev_position[2], self.prev_position[0]
            prev_heading = compute_heading_from_quaternion(self.prev_rotation)
            dr = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            dphi = math.atan2(curr_y - prev_y, curr_x - prev_x)
            dhead = curr_heading - prev_heading
            # Convert these to the starting point's coordinate system.
            start_heading = compute_heading_from_quaternion(self.start_rotation)
            dphi = dphi - start_heading
            delta = np.array([dr, dphi, dhead, 0.0])
        self.prev_position = copy.deepcopy(agent_position)
        self.prev_rotation = copy.deepcopy(agent_rotation)

        return delta

        
@registry.register_measure
class Picture(Measure):
    r"""Whether or not the agent take picture
    """

    cls_uuid: str = "picture"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        if (
            hasattr(task, "is_found_called")
            and task.is_found_called
        ):
            self._metric = 1
        else:
            #self._metric = 0
            # without TAKE-PICTURE
            self._metric = 1
            
@registry.register_measure
class Saliency(Measure):
    # saliencyのcountを出力
    # ただし、0を除いた最頻値が1ではないときは-1を返す

    cls_uuid: str = "saliency"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transalnet_model = TranSalNet()
        self.transalnet_model.load_state_dict(torch.load('TranSalNet/pretrained_models/TranSalNet_Dense.pth'))
        self.transalnet_model = self.transalnet_model.to(self.device) 
        self.transalnet_model.eval()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid
    
    def get_metric(self):
        return self._metric

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [Picture.cls_uuid])
        self.update_metric(*args, episode=episode, task=task, **kwargs)
    
    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        take_picture = task.measurements.measures[Picture.cls_uuid].get_metric()
        
        observation = self._sim.get_observations_at()
        
        if take_picture:
            self._metric = self._cal_picture_value(observation)
        else:
            self._metric = -1
            ####
            self._metric = self._cal_picture_value(observation)

    def _to_category_id(self, obs):
        scene = self._sim._sim.semantic_scene
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])

        if mapping.shape <= obs.max():
            logger.info(f"########### mapping={mapping.shape}, obs_max = {obs.max()} ##############")
            obs[obs >= mapping.shape] = 0
            logger.info(f"########### Modified !!! mapping={mapping.shape}, obs_max = {obs.max()} ##############")
        semantic_obs = np.take(mapping, obs)
        semantic_obs[semantic_obs>=40] = 0
        semantic_obs[semantic_obs<0] = 0
        return semantic_obs

    def _cal_picture_value(self, obs):
        rgb_obs = obs["rgb"]
        img = preprocess_img(image=rgb_obs) # padding and resizing input image into 384x288
        img = np.array(img)/255.
        img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)
        img = torch.from_numpy(img)
    
        if torch.cuda.is_available():
            img = img.type(torch.cuda.FloatTensor).to(self.device)
        else:
            img = img.type(torch.FloatTensor).to(self.device)
        
        raw_saliency, pred_saliency = self.transalnet_model(img)
        toPIL = transforms.ToPILImage()
        pic = toPIL(pred_saliency.squeeze())

        pred_saliency = postprocess_img(pic, org_image=rgb_obs)
        
        # 0を削除
        non_zero_pred_saliency = pred_saliency[pred_saliency != 0]
        #flag = (stats.mode(non_zero_pred_saliency).mode == 1)

        count_sal = raw_saliency[raw_saliency > 0].shape[0]
        sem_obs = self._to_category_id(obs["semantic"])
        H = sem_obs.shape[0]
        W = sem_obs.shape[1]
        obs_shape = H*W

        sem_obs = np.array(sem_obs)
        # 各カテゴリの数をカウント
        category_num = np.bincount(sem_obs.flatten(), minlength=40)
        num_category = np.count_nonzero(category_num)
    
        """
        #objectのcategoryリスト
        category_num = [0] * 40
        for i in range(H):
            for j in range(W):
                obs = sem_obs[i][j]
                category_num[obs] += 1

        num_category = 0
        for i in range(40):
            if category_num[i] > 0:
                num_category += 1
        """

        picture_value = count_sal * num_category
        
        return picture_value
        #return self._count_saliency_regions(pred_saliency)

@registry.register_measure
class CI(Measure):
    # observationの10%以上を占めるオブジェクトの数
    cls_uuid: str = "ci"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid
    
    def get_metric(self):
        return self._metric

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [Picture.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)

        #########
        #scene = self._sim._sim.semantic_scene
        #self.print_scene_recur(scene, 1000)
    
    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        take_picture = task.measurements.measures[
            Picture.cls_uuid
        ].get_metric()
        """
        observation = self._sim.get_observations_at()
        semantic_obs = self._to_category_id(observation["semantic"])
        H = semantic_obs.shape[0]
        W = semantic_obs.shape[1]
        self._matrics = np.zeros((H, W))
        """

        self._metric = self._calCI()


    def print_scene_recur(self, scene, limit_output=10):
        count = 0
        for level in scene.levels:
            logger.info(
                f"Level id:{level.id}, center:{level.aabb.center},"
                f" dims:{level.aabb.sizes}"
            )
            for region in level.regions:
                logger.info(
                    f"Region id:{region.id}, category:{region.category.name()},"
                    f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
                )
                for obj in region.objects:
                    logger.info(
                        f"Object id:{obj.id}, category:{obj.category.name()},"
                        f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                    )
                    count += 1
                    if count >= limit_output:
                        return None
            
    def _to_category_id(self, obs):
        #logger.info(obs)
        scene = self._sim._sim.semantic_scene
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])

        semantic_obs = np.take(mapping, obs)
        semantic_obs[semantic_obs>=40] = 0
        semantic_obs[semantic_obs<0] = 0
        return semantic_obs

        
    def _calCI(self, *args: Any, **kwargs: Any):
        observation = self._sim.get_observations_at()
        semantic_obs = self._to_category_id(observation["semantic"])
        #depth_obs = observation["depth"]
        H = semantic_obs.shape[0]
        W = semantic_obs.shape[1]
        obs_shape = H*W

        sem_obs = np.array(sem_obs)
        # 各カテゴリの数をカウント
        category_num = np.bincount(sem_obs.flatten(), minlength=40)
        num_category = np.count_nonzero(category_num)

        """
        #objectのcategoryリスト
        category_num = [0] * 40
        for i in range(H):
            for j in range(W):
                obs = semantic_obs[i][j]
                category_num[obs] += 1

        num_category = 0
        for i in range(40):
            if category_num[i] > obs_shape*0.1:
                num_category += 1
        """

        return num_category
        

@registry.register_measure
class Success(Measure):
    r"""Whether or not the agent succeeded at its task
    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [SubSuccess.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        subsuccess = task.measurements.measures[
            SubSuccess.cls_uuid
        ].get_metric()

        if subsuccess ==1 and task.currGoalIndex >= len(episode.goals):
            self._metric = 1
        else:
            self._metric = 0

@registry.register_measure
class PercentageSuccess(Measure):
    r"""Variant of SubSuccess. It tells how much of the episode 
        is successful
    """

    cls_uuid: str = "percentage_success"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any): ##Called only when episode begins
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToCurrGoal.cls_uuid]
        )
        self._metric=0
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_subgoal = task.measurements.measures[
            DistanceToCurrGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(task, "is_found_called")
            and task.is_found_called
            and distance_to_subgoal < self._config.SUCCESS_DISTANCE
        ):
            self._metric += 1/len(episode.goals)

@registry.register_measure
class RawMetrics(Measure):
    """All the raw metrics we might need
    """
    cls_uuid: str = "raw_metrics"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()

        self._start_end_episode_distance = 0

        self._agent_episode_distance = 0.0
        self._metric = None

        self.update_metric(*args, episode=episode, task=task, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position
        
        self._metric = {
            'agent_path_length': self._agent_episode_distance,
            'episode_lenth': task.measurements.measures[EpisodeLength.cls_uuid].get_metric()
        }



@registry.register_measure
class WPL(Measure):
    """
    MSPL but without the multiplicative factor of Success
    """

    cls_uuid: str = "wpl"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = 0
        for goal_number in range(len(episode.goals) ):  # Find distances between successive goals and keep adding them
            if goal_number == 0:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    self._previous_position, episode.goals[0][0].position
                )
            else:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.goals[goal_number - 1][0].position, episode.goals[goal_number][0].position
                )
        self._agent_episode_distance = 0.0
        self._metric = None
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid]
        )
        self.update_metric(*args, episode=episode, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position

        self._metric = 1 * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


#@registry.register_measure
class STEPS(Measure):
    r"""Count for steps taken
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "wpl"

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        # ep_success = 0
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        # if (
        #     hasattr(task, "is_stop_called")
        #     and task.is_stop_called
        #     and distance_to_target < self._config.SUCCESS_DISTANCE
        # ):
        #     ep_success = 1

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = 1 * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )   ##changed just this line



@registry.register_measure
class Collisions(Measure):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "collisions"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        if self._metric is None:
            self._metric = {"count": 0, "is_collision": False}
        self._metric["is_collision"] = False
        if self._sim.previous_step_collided:
            self._metric["count"] += 1
            self._metric["is_collision"] = True


@registry.register_measure
class TopDownMap(Measure):
    r"""Top Down Map measure
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self._shortest_path_points = None
        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        self.line_thickness = int(
            np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_map"

    def _check_valid_nav_point(self, point: List[float]):
        self._sim.is_navigable(point)

    def get_original_map(self):
        top_down_map = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            self._config.DRAW_BORDER,
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        self._ind_x_min = range_x[0]
        self._ind_x_max = range_x[-1]
        self._ind_y_min = range_y[0]
        self._ind_y_max = range_y[-1]

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[0],
            position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        if self._config.DRAW_SOURCE:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )
            
        self.update_metric(None, None)

    def _clip_map(self, _map):
        return _map[
            self._ind_x_min
            - self._grid_delta : self._ind_x_max
            + self._grid_delta,
            self._ind_y_min
            - self._grid_delta : self._ind_y_max
            + self._grid_delta,
        ]

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        clipped_house_map = self._clip_map(house_map)

        clipped_fog_of_war_map = None
        if self._config.FOG_OF_WAR.DRAW:
            clipped_fog_of_war_map = self._clip_map(self._fog_of_war_mask)

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_map,
            "agent_map_coord": (
                map_agent_x - (self._ind_x_min - self._grid_delta),
                map_agent_y - (self._ind_y_min - self._grid_delta),
            ),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            thickness = int(
                np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
            )
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                * max(self._map_resolution)
                / (self._coordinate_max - self._coordinate_min),
            )


@registry.register_measure
class NewTopDownMap(Measure):
    r"""New Top Down Map measure
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        self.z_dict = {}
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "new_top_down_map"

    def get_original_map(self, z_value):
        top_down_map = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            self._config.DRAW_BORDER,
        )
        #logger.info("##### in get_original_map ########")
        #logger.info(top_down_map)
        #logger.info(top_down_map.shape)

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        self._ind_x_min = range_x[0]
        self._ind_x_max = range_x[-1]
        self._ind_y_min = range_y[0]
        self._ind_y_max = range_y[-1]

        self.z_dict[z_value] = (
                                top_down_map,
                                self._ind_x_min,
                                self._ind_x_max,
                                self._ind_y_min,
                                self._ind_y_max
                                )

        return top_down_map

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        agent_position = self._sim.get_agent_state().position
        self._top_down_map = self.get_original_map(agent_position[1])
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
            
        self.update_metric(None, None)

    def _clip_map(self, _map):
        return _map[
            self._ind_x_min
            - self._grid_delta : self._ind_x_max
            + self._grid_delta,
            self._ind_y_min
            - self._grid_delta : self._ind_y_max
            + self._grid_delta,
        ]

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        clipped_house_map = self._clip_map(house_map)
        clipped_fog_of_war_map = np.ones(clipped_house_map.shape, dtype="int8")

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_map,
            "agent_map_coord": (
                map_agent_x - (self._ind_x_min - self._grid_delta),
                map_agent_y - (self._ind_y_min - self._grid_delta),
            ),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def update_map(self, agent_position):
        if agent_position[1] in self.z_dict:
            (
                self._top_down_map, 
                self._ind_x_min, 
                self._ind_x_max, 
                self._ind_y_min, 
                self._ind_y_max
            ) = self.z_dict[agent_position[1]]
        else:
            self._top_down_map = self.get_original_map(agent_position[1])
        #logger.info("####### in nav.py #########")
        #logger.info(self._top_down_map)
        #logger.info(self._top_down_map.shape)
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )

        return self._top_down_map, a_x, a_y
            

@registry.register_measure
class ExploredMap(Measure):
    r"""Explored Map measure"""

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "explored_map"


    def get_original_map(self):
        top_down_map = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            self._config.DRAW_BORDER,
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        self._ind_x_min = range_x[0]
        self._ind_x_max = range_x[-1]
        self._ind_y_min = range_y[0]
        self._ind_y_max = range_y[-1]

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)

        return top_down_map

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )

        self.start_position_x = agent_position[0]
        self.start_position_y = agent_position[2]
        self.start_grid_x = a_x
        self.start_grid_y = a_y

        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))
    
        self.update_metric(None, None)

    def _clip_map(self, _map):
        return _map[
            self._ind_x_min
            - self._grid_delta : self._ind_x_max
            + self._grid_delta,
            self._ind_y_min
            - self._grid_delta : self._ind_y_max
            + self._grid_delta,
        ]

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        clipped_house_map = self._clip_map(house_map)
        #clipped_house_map = house_map

        clipped_fog_of_war_map = None
        if self._config.FOG_OF_WAR.DRAW:
            clipped_fog_of_war_map = self._clip_map(self._fog_of_war_mask)
            #clipped_fog_of_war_map= self._fog_of_war_mask

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_map,
            "agent_map_coord": (
                map_agent_x - (self._ind_x_min - self._grid_delta),
                map_agent_y - (self._ind_y_min - self._grid_delta),
            ),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                * max(self._map_resolution)
                / (self._coordinate_max - self._coordinate_min),
            )


@registry.register_measure
class SmoothCoverage(Measure):
    r"""Smooth Coverage measure"""

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self.top_down_map = None
        self._coverage_map = None
        super().__init__()
        """
        self.log_manager = LogManager()
        self.log_manager.setLogDirectory("./smooth_value")
        self.log_index = 0
        """

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "smooth_coverage"


    def get_original_map(self):
        self.top_down_map = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            True,
        )

        self._coverage_map = np.zeros_like(self.top_down_map)
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(self.top_down_map)
        #logger.info(f"########## TopDown Map Shape={self._top_down_map.shape}")


    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = None
        self.get_original_map()
            
        self.update_metric(None, None)


    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        """
        self.log_writer = self.log_manager.createLogWriter(f"smooth_value_{self.log_index}")
        self.log_index += 1
        for i in range(self._top_down_map.shape[0]):
            for j in range(self._top_down_map.shape[1]):
                self.log_writer.write(str(self._top_down_map[i][j]))
            self.log_writer.writeLine()
        """

        result, seen_count = self.update_top_down_map()

        """
        seen_count = 1e-8
        result = 0.0
        for i in range(len(self._coverage_map)):
            for j in range(len(self._coverage_map[0])):
                if self._fog_of_war_mask[i,j] == 1:
                    seen_count += 1 
                    count = self._coverage_map[i,j]
                    result += 1 / np.sqrt(count)
        """

        #logger.info(f"######### result = {result} ###########")
        #logger.info(f"########## seen_count = {seen_count} ############")
        self._metric = result / seen_count
        #logger.info(f"####### smooth_reward={self._metric} #########")

    def update_top_down_map(self):
        # フィルタリングして条件を満たすインデックスを取得
        mask = self._fog_of_war_mask == 1
        self._coverage_map[mask] += 1

        seen_count = np.count_nonzero(mask) + 1e-8
        result = np.sum(1 / np.sqrt(self._coverage_map[mask]))

        """
        seen_count = 1e-8
        result = 0.0
        for i in range(len(self._coverage_map)):
            for j in range(len(self._coverage_map[0])):
                if self._fog_of_war_mask[i,j] == 1:
                    #logger.info(f"i={i}, j={j}")
                    self._coverage_map[i,j] += 1

                    seen_count += 1 
                    count = self._coverage_map[i,j]
                    result += 1 / np.sqrt(count)
        """

        return result, seen_count

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self.top_down_map,
                np.zeros_like(self.top_down_map),
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                * max(self._map_resolution)
                / (self._coordinate_max - self._coordinate_min),
            )

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

@registry.register_measure
class NoveltyValue(Measure):
    r"""Novelty Value measure"""

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        super().__init__()
        """
        self.log_manager = LogManager()
        self.log_manager.setLogDirectory("./novelty")
        self.log_index = 0
        """

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "novelty_value"

    def get_original_map(self):
        top_down_map = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            False,
        )

        self._top_down_map = np.zeros(top_down_map.shape)

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = None
        self.get_original_map()
            
        self.update_metric(None, None)


    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )

        self._top_down_map[a_x, a_y] += 1

        """
        self.log_writer = self.log_manager.createLogWriter(f"novelty_value_{self.log_index}")
        self.log_index += 1
        logger.info("Write Count Map")
        for i in range(self._top_down_map.shape[0]):
            for j in range(self._top_down_map.shape[1]):
                self.log_writer.write(str(self._top_down_map[i][j]))
            self.log_writer.writeLine()
        """

        visit_count = self._top_down_map[a_x, a_y]
        #logger.info("###### visit_count = {visit_count}")
        
        # 平方の逆数を計算
        novelty = 1 / np.sqrt(visit_count)
        #logger.info("###### sqrt = {np.sqrt(visit_count)}")
    
        self._metric = novelty


@registry.register_measure
class PictureRangeMap(Measure):
    r"""Picture Range Map measure
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self._shortest_path_points = None
        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        self.line_thickness = int(
            np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "picture_range_map"


    def get_original_map(self):
        top_down_map = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            self._config.DRAW_BORDER,
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        self._ind_x_min = range_x[0]
        self._ind_x_max = range_x[-1]
        self._ind_y_min = range_y[0]
        self._ind_y_max = range_y[-1]

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[0],
            position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        if self._config.DRAW_SOURCE:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )
            
        self.update_metric(None, None)

    def _clip_map(self, _map):
        return _map[
            self._ind_x_min
            - self._grid_delta : self._ind_x_max
            + self._grid_delta,
            self._ind_y_min
            - self._grid_delta : self._ind_y_max
            + self._grid_delta,
        ]

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        clipped_house_map = self._clip_map(house_map)

        clipped_fog_of_war_map = None
        if self._config.FOG_OF_WAR.DRAW:
            clipped_fog_of_war_map = self._clip_map(self._fog_of_war_mask)

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_map,
            "agent_map_coord": (
                map_agent_x - (self._ind_x_min - self._grid_delta),
                map_agent_y - (self._ind_y_min - self._grid_delta),
            ),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                np.zeros_like(self._top_down_map),
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                * max(self._map_resolution)
                / (self._coordinate_max - self._coordinate_min),
            )


@registry.register_measure
class FowMap(Measure):
    r"""FOW map measure
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._map_resolution = (300, 300)
        self._coordinate_min = -120.3241-1e-6
        self._coordinate_max = 120.0399+1e-6
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "fow_map"

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._metric = None
        self._top_down_map = task.sceneMap
        self._fog_of_war_mask = np.zeros_like(self._top_down_map)
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        agent_position = np.array([a_x, a_y])

        self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
            self._top_down_map,
            self._fog_of_war_mask,
            agent_position,
            self.get_polar_angle(),
            fov=self._config.FOV,
            max_line_len=self._config.VISIBILITY_DIST
            * max(self._map_resolution)
            / (self._coordinate_max - self._coordinate_min),
        )

        self._metric = self._fog_of_war_mask


    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation
        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

@registry.register_measure
class Ratio(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "ratio"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = None
        current_position = self._sim.get_agent_state().position.tolist()
        if self._config.DISTANCE_TO == "POINT":
            initial_geodesic_distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[0].position
            )
            for goal_number in range(0, len(episode.goals)-1):
                initial_geodesic_distance_to_target += self._sim.geodesic_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )

            initial_euclidean_distance_to_target = self._euclidean_distance(
                current_position, episode.goals[0].position
            )
            for goal_number in range(0, len(episode.goals)-1):
                initial_euclidean_distance_to_target += self._euclidean_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )
        else:
            logger.error(
                f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
            )
        self._metric = initial_geodesic_distance_to_target / initial_euclidean_distance_to_target

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        pass

@registry.register_measure
class EpisodeLength(Measure):
    r"""Calculates the episode length
    """
    cls_uuid: str = "episode_length"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._episode_length = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._episode_length = 0
        self._metric = self._episode_length

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        self._episode_length += 1
        self._metric = self._episode_length



@registry.register_task_action
class MoveForwardAction(SimulatorTaskAction):
    name: str = "MOVE_FORWARD"

    def step(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = False ##C
        return self._sim.step(HabitatSimActions.MOVE_FORWARD)


@registry.register_task_action
class TurnLeftAction(SimulatorTaskAction):
    def step(self, *args: Any,  task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = False ##C
        return self._sim.step(HabitatSimActions.TURN_LEFT)


@registry.register_task_action
class TurnRightAction(SimulatorTaskAction):
    def step(self, *args: Any,  task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = False ##C
        return self._sim.step(HabitatSimActions.TURN_RIGHT)


@registry.register_task_action
class MoveLeftAction(SimulatorTaskAction):
    def step(self, *args: Any,  task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = False ##C
        return self._sim.step(HabitatSimActions.MOVE_LEFT)


@registry.register_task_action
class MoveRightAction(SimulatorTaskAction):
    def step(self, *args: Any,  task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = False ##C
        return self._sim.step(HabitatSimActions.MOVE_RIGHT)


@registry.register_task_action
class StopAction(SimulatorTaskAction):
    name: str = "STOP"

    def reset(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.is_stop_called = False
        task.is_found_called = False ##C

    def step(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_stop_called = True
        task.is_found_called = False ##C
        return self._sim.get_observations_at()
    
@registry.register_task_action
class TakePicture(SimulatorTaskAction):
    name: str = "TAKE_PICTURE"

    def reset(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.is_stop_called = False
        task.is_found_called = False ##C

    def step(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = True
        return self._sim.get_observations_at()


@registry.register_task_action
class LookUpAction(SimulatorTaskAction):
    def step(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = False
        return self._sim.step(HabitatSimActions.LOOK_UP)


@registry.register_task_action
class LookDownAction(SimulatorTaskAction):
    def step(self, *args: Any,  task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = False
        return self._sim.step(HabitatSimActions.LOOK_DOWN)


@registry.register_task_action
class FoundObjectAction(SimulatorTaskAction):
    name: str = "FOUND"
    def reset(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.is_stop_called = False
        task.is_found_called = False ##C

    def step(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = True
        return self._sim.get_observations_at()



@registry.register_task_action
class TeleportAction(SimulatorTaskAction):
    # TODO @maksymets: Propagate through Simulator class
    COORDINATE_EPSILON = 1e-6
    COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
    COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "TELEPORT"

    def step(
        self,
        *args: Any,
        position: List[float],
        rotation: List[float],
        **kwargs: Any,
    ):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        logger.info(f"####### position={position}, rotation={rotation} ##########")

        if not isinstance(rotation, list):
            rotation = list(rotation)

        if not self._sim.is_navigable(position):
            return self._sim.get_observations_at()

        return self._sim.get_observations_at(
            position=position, rotation=rotation, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self):
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([self.COORDINATE_MIN] * 3),
                    high=np.array([self.COORDINATE_MAX] * 3),
                    dtype=np.float32,
                ),
                "rotation": spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
                ),
            }
        )


@registry.register_task(name="Nav-v0")
class NavigationTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)


@registry.register_task(name="Info-v0")
class InformationTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        
    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
