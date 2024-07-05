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
from habitat.datasets.maximum_info.maximuminfo_generator import generate_maximuminfo_episode, generate_maximuminfo_episode3
from habitat_baselines.common.environments import InfoRLEnv
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.logging import logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.utils import generate_video2

translate_action = {
    "move_forward": "MOVE_FORWARD",
    "turn_left": "TURN_LEFT",
    "turn_right": "TURN_RIGHT",
}
        
        
def make_dataset(scene_name, initial_position, initial_rotation):
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    logger.info("START FOR: " + scene_name)
        
    dataset_path = "viewer_video/datasets/" + scene_name + ".json.gz"    
    
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/" + scene_name + "/" + scene_name + ".glb"
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.POSITION = [0, 1.5, 0]
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = ("data/default.phys_scene_config.json")
    config.TASK_CONFIG.TRAINER_NAME = "oracle-ego"
    config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    config.TASK_CONFIG.TASK.MEASUREMENTS = ['NEW_TOP_DOWN_MAP', 'FOW_MAP']
    config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.07
    config.TASK_CONFIG.SIMULATOR.TURN_ANGLE = 1.5
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING=True
    config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 1000000000
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'MOVE_BACKWARD']
    config.freeze()

    logger.info(config)
    
    num = 1
    dataset = MaximumInfoDatasetV1()
    #sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
    dataset.episodes += generate_maximuminfo_episode3(config, num_episodes=num, initial_position=initial_position, initial_rotation=initial_rotation)
    #sim.close()
        
    #datasetを.gzに圧縮
    with gzip.open(dataset_path, "wt") as f:
        f.write(dataset.to_json())


def read_file_into_list(file_path):
    actions = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                actions.append(line.strip())
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return actions
        
        
def create_video(scene_name, action_file):
    logger.info("########### Create FOR: " + scene_name + " ####################")
    
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    opts = None
    config = get_config(exp_config, opts)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    
    dataset_path = f"viewer_video/datasets/{scene_name}.json.gz"    
    
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.SCENE = f"data/scene_datasets/mp3d/{scene_name}/{scene_name}.glb"
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.POSITION = [0, 1.5, 0]
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = ("data/default.phys_scene_config.json")
    config.TASK_CONFIG.TRAINER_NAME = "oracle-ego"
    config.TASK_CONFIG.TASK.MEASUREMENTS = ['NEW_TOP_DOWN_MAP', 'FOW_MAP']
    config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.07
    config.TASK_CONFIG.SIMULATOR.TURN_ANGLE = 1.5
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING=True
    config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 1000000000
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'MOVE_BACKWARD']
    config.freeze()

    logger.info(config)

    action_list = read_file_into_list(action_file)
    
    #フォルダがない場合は、作成
    p_dir = pathlib.Path(f"viewer_video/videos/{scene_name}")
    if not p_dir.exists():
        p_dir.mkdir(parents=True)    
    rgb_frames = []
    with InfoRLEnv(config=config) as env:
        observation = env.reset()
        outputs = env.step2("TURN_RIGHT")
        outputs = env.step2("TURN_LEFT")
        observation, info = outputs
        frame, change_w = observations_to_image(observation, info)
        rgb_frames.append(frame)
        for _ in range(10):
            rgb_frames.append(frame)

        for i in range(len(action_list)):
            #logger.info(f"step={i}")
            action = action_list[i]
            if action == "stop":
                frame = rgb_frames[-1]
                rgb_frames.append(frame)
            else:
                outputs = env.step2(translate_action[action])
                observation, info = outputs
                frame, _ = observations_to_image(observation, info, change_w=change_w)
                #logger.info(f"frame_size={frame.shape}")
                rgb_frames.append(frame)

        last_frame = rgb_frames[-1] 
        for _ in range(20):
            rgb_frames.append(last_frame) 
        
        generate_video2(
            video_dir=f"viewer_video/videos/{scene_name}",
            images=rgb_frames,
        )

                
if __name__ == '__main__':
    logger.info("START PROGRAM")
    scene_name = "VFuaQ6m2Qom"
    initial_position = (-14.56984, 0.15195204, -1.1711477)
    # 1, 2, 3, 0
    initial_rotation = (0, 7.06150603946298e-05, 0, 1)
    file_name = "24-07-04_03-58-47"
    action_file = f"/gs/fs/tga-aklab/matsumoto/Main/viewer_video/action_log/{scene_name}/{file_name}.txt"
    make_dataset(scene_name, initial_position, initial_rotation)
    create_video(scene_name, action_file)
    
    logger.info("################# FINISH EXPERIMENT !!!!! ##########################")