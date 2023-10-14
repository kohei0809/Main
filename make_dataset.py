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
from habitat.datasets.maximum_info.maximuminfo_dataset import MaximumInfoDatasetV1
from habitat.datasets.maximum_info.maximuminfo_generator import generate_maximuminfo_episode
from habitat.core.env import Env

   
if __name__ == '__main__':
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    opts = None
    config = get_config(exp_config, opts)
    print(config)
    
        
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = ("./data/default.phys_scene_config.json")
    config.TASK_CONFIG.TRAINER_NAME = "oracle-ego"
    config.freeze()
    
    dir_path = "data/scene_datasets/mp3d"
    dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    train_scene_num = 61
    val_scene_num = 11
    test_scene_num = 18
    episode_num = 20000
    
    i = 0
    dataset_path = "data/datasets/v4/maximum/"
    dataset = MaximumInfoDatasetV1()
    split = ""
    while(True):
        config.defrost()
        if i < train_scene_num:
            split = "train"
            episode_num = 20000
        elif i < train_scene_num+val_scene_num:
            if i == train_scene_num:
                #datasetを.gzに圧縮
                with gzip.open(dataset_path + split + "/" + split +  ".json.gz", "wt") as f:
                    random.shuffle(dataset.episodes)
                    f.write(dataset.to_json())
                    
                dataset = MaximumInfoDatasetV1()
                    
            split = "val"
            episode_num = 7
        elif i < train_scene_num+val_scene_num+test_scene_num:
            if i == train_scene_num+val_scene_num:
                #datasetを.gzに圧縮
                with gzip.open(dataset_path + split + "/" + split +  ".json.gz", "wt") as f:
                    random.shuffle(dataset.episodes)
                    f.write(dataset.to_json())
                    
                dataset = MaximumInfoDatasetV1()
                
            split = "test"
            episode_num = 500
        else:
            break
            
        scene = dirs[i]
        config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/" + scene + "/" + scene + ".glb"
        config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path + split + "/" + split +  ".json.gz"
        config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path + split + "/" + split +  ".json.gz"
        config.freeze()
        
        sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
        dataset.episodes += generate_maximuminfo_episode(sim=sim, num_episodes=episode_num)
        print(str(i) + ": SPLIT:" + split + ", NUM:" + str(episode_num) + ", TOTAL_NUM:" + str(len(dataset.episodes)))
        print("SCENE:" + scene)
        sim.close()
        
        i += 1

    #datasetを.gzに圧縮
    with gzip.open(dataset_path + split + "/" + split +  ".json.gz", "wt") as f:
        f.write(dataset.to_json())
                    