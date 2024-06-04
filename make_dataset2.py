import os
import random

import numpy as np
from gym import spaces
import gzip

from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image

from habitat_sim.utils.common import d3_40_colors_rgb
from habitat_baselines.config.default import get_config  
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.datasets.maximum_info.maximuminfo_dataset import MaximumInfoDatasetV1
from habitat.datasets.maximum_info.maximuminfo_generator import generate_maximuminfo_episode
from habitat.core.env import Env

from habitat.core.logging import logger


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

    # ファイルを読み込んで行ごとにリストに格納する
    with open('data/scene_datasets/mp3d/description.txt', 'r') as file:
        lines = file.readlines()

    # scene id と文章を抽出してデータフレームに変換する
    scene_ids = []
    types = []
    descriptions = []
    for i in range(0, len(lines), 3):
        scene_ids.append(lines[i].strip())
        types.append(lines[i+1].strip())
        descriptions.append(lines[i+2].strip())

    df = pd.DataFrame({'scene_id': scene_ids, 'type': types, 'description': descriptions})   

    i = 0
    dataset_path = "data/datasets/maximuminfo/v3/"
    dataset = MaximumInfoDatasetV1()
    split = ""
    mode = 0
    while(True):
        logger.info("!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.info(f"i: {i}")
        if i >= 90:
            mode += 1
            i = 0
        if mode == 3:
            break

        config.defrost()
        scene = dirs[i]
        if (mode == 0) and (df[df["scene_id"]==scene]["type"].item()=="train"):
            split = "train"
            episode_num = 20000
        elif (mode == 1) and (df[df["scene_id"]==scene]["type"].item()=="val"):
            if i == 0:
                #datasetを.gzに圧縮
                with gzip.open(dataset_path + split + "/" + split +  ".json.gz", "wt") as f:
                    random.shuffle(dataset.episodes)
                    f.write(dataset.to_json())

                dataset = MaximumInfoDatasetV1()

            split = "val"
            episode_num = 50
        elif (mode == 2) and (df[df["scene_id"]==scene]["type"].item()=="test"):
            if i == 0:
                #datasetを.gzに圧縮
                with gzip.open(dataset_path + split + "/" + split +  ".json.gz", "wt") as f:
                    random.shuffle(dataset.episodes)
                    f.write(dataset.to_json())

                dataset = MaximumInfoDatasetV1()

            split = "test"
            episode_num = 50
        else:
            break

        #scene = dirs[i]
        config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/" + scene + "/" + scene + ".glb"
        config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path + split + "/" + split +  ".json.gz"
        config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path + split + "/" + split +  ".json.gz"
        config.freeze()

        sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
        dataset.episodes += generate_maximuminfo_episode(sim=sim, num_episodes=episode_num)
        logger.info("####################")
        logger.info(str(i) + ": SPLIT:" + split + ", NUM:" + str(episode_num) + ", TOTAL_NUM:" + str(len(dataset.episodes)))
        logger.info(f"mode: {mode}, i: {i}, SCENE: {scene}")
        sim.close()

        i += 1

    #datasetを.gzに圧縮
    with gzip.open(dataset_path + split + "/" + split +  ".json.gz", "wt") as f:
        f.write(dataset.to_json())