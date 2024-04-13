import os
import random
import pathlib

import numpy as np
from gym import spaces
import gzip
import torch
import datetime
import multiprocessing

from matplotlib import pyplot as plt

from PIL import Image

from habitat_baselines.config.default import get_config  
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.datasets.maximum_info.maximuminfo_dataset import MaximumInfoDatasetV1
from habitat.datasets.maximum_info.maximuminfo_generator import generate_maximuminfo_episode, generate_maximuminfo_episode2
from habitat_baselines.common.environments import InfoRLEnv
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.logging import logger
        
        
def make_dataset_random(scene_idx):
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    dir_path = "data/scene_datasets/mp3d"
    dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    
    
    scene_name = dirs[scene_idx]
    logger.info("START FOR: " + scene_name)
        
    dataset_path = "PictureSelector/data/datasets/" + scene_name + ".json.gz"    
    
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/" + scene_name + "/" + scene_name + ".glb"
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = ("data/default.phys_scene_config.json")
    config.TASK_CONFIG.TRAINER_NAME = "oracle-ego"
    config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    config.freeze()
    
    num = 1000
    dataset = MaximumInfoDatasetV1()
    sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
    dataset.episodes += generate_maximuminfo_episode2(sim=sim, num_episodes=num)
    sim.close()
        
    #datasetを.gzに圧縮
    with gzip.open(dataset_path, "wt") as f:
        f.write(dataset.to_json())
        
        
def collect_pictures(scene_idx):
    dir_path = "data/scene_datasets/mp3d"
    dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    
    scene_name = dirs[scene_idx]
    logger.info("########### START FOR: " + scene_name + " ####################")
    
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    opts = None
    config = get_config(exp_config, opts)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    dataset_path = dataset_path = "PictureSelector/data/datasets/" + scene_name + ".json.gz" 

    config.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/" + scene_name + "/" + scene_name + ".glb"
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.5
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.TASK_CONFIG.TASK.MEASUREMENTS.append('FOW_MAP')
    config.NUM_PROCESSES = 1
    config.TASK_CONFIG.TRAINER_NAME = "oracle-ego"
    config.freeze()
    
    #フォルダがない場合は、作成
    p_dir = pathlib.Path("PictureSelector/data/pictures/" + scene_name)
    if not p_dir.exists():
        p_dir.mkdir(parents=True)    
    with InfoRLEnv(config=config) as env:
        for i in range(len(env.episodes)):
            if i % 100 == 0:
                logger.info("STEP: " + str(i))
                
            #エピソードの変更
            env._env.current_episode = env.episodes[i]
            
            observation = env.reset()
            outputs = env.step2()
            
            picture = observation["rgb"]
            image = Image.fromarray(picture)
            
            # 画像をJPEG形式で保存
            image.save("PictureSelector/data/pictures/" + scene_name + "/" + str(i) + ".jpg")
        env.close()
                
if __name__ == '__main__':
    for i in range(73, 90):
        try:
            logger.info("######### " + str(i) + " ############")
            scene_idx = i
            make_dataset_random(scene_idx)
            collect_pictures(scene_idx)
        except:
            continue
    
    logger.info("################# FINISH EXPERIMENT !!!!! ##########################")