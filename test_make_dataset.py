import os
import random

import numpy as np
from gym import spaces
import gzip

from matplotlib import pyplot as plt

from PIL import Image

from habitat_sim.utils.common import d3_40_colors_rgb
from habitat_baselines.config.default import get_config  
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.multi_nav.multi_nav_dataset import MultiNavDatasetV1
from habitat.datasets.pointnav.pointnav_generator import is_compatible_episode, generate_pointnav_episode
from habitat.core.env_point import Env

   
if __name__ == '__main__':
    exp_config = "./habitat/config/test.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    dataset_path = "test_dataset.json.gz"
        
    config.defrost()
    config.DATASET.DATA_PATH = dataset_path
    config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/p5wJjkQkbXX/p5wJjkQkbXX.glb"
    #config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb"
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = ("./data/default.phys_scene_config.json")
    config.TASK_CONFIG.TASK.MEASUREMENTS = ['DISTANCE_TO_CURR_GOAL', 'DISTANCE_TO_MULTI_GOAL', 'SUB_SUCCESS', 'SUCCESS', 'EPISODE_LENGTH', 'MSPL', 'PERCENTAGE_SUCCESS', 'RATIO', 'PSPL', 'RAW_METRICS', 'TOP_DOWN_MAP']
    config.TASK_CONFIG.TRAINER_NAME = "oracle"
    config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    config.freeze()
    
    print(config)
    print("########################")
    
    sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
    dataset = PointNavDatasetV1()
    #dataset = MultiNavDatasetV1()
    dataset.episodes += generate_pointnav_episode(sim=sim, num_episodes=1)
    #    X                          Z
    #(10.45676  ,  3.5141459,  8.119155)
    
    #datasetを.gzに圧縮
    with gzip.open(dataset_path, "wt") as f:
        f.write(dataset.to_json())
    