import os
import random

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
from utils.log_manager import LogManager
from utils.log_writer import LogWriter
from habitat.core.logging import logger

        
def research_ci(scene_idx):
    dir_path = "data/scene_datasets/mp3d"
    dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    
    scene_name = dirs[scene_idx]
    logger.info("START FOR: " + scene_name)
    
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    opts = None
    config = get_config(exp_config, opts)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    dataset_path = "map_dataset/" + scene_name + ".json.gz"

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
        
    #ログファイルの設定   
    log_manager = LogManager()
    log_manager.setLogDirectory("map_ci/")

    map_list = []
    
    with InfoRLEnv(config=config) as env:
        logger.info("EPISODE NUM: "+ str(len(env.episodes)))
        
        #for i in range(len(env.episodes)):
        for i in range(50000):
        
            if i % 1000 == 0:
                logger.info("STEP: " + str(i))
                
            #エピソードの変更
            env._env.current_episode = env.episodes[i]
            
            observation = env.reset()
            outputs = env.step2()
            rewards, done, info = outputs
        
            ci = rewards[0][1]
            top_down_map = info["top_down_map"]["map"]
            map_idx = 0
            if i == 0:
                map = top_down_map.copy()
                map = map.astype(np.float32)
                #mapの壁は-100, それ以外は-10にする
                for j in range(map.shape[0]):
                    for k in range(map.shape[1]):
                        if map[j][k] == 0:
                            map[j][k] = -100.0
                        else:
                            map[j][k] = -200.0
                    
                map_list.append(map)
            
            flag = False
            flag2 = False
            map = None
            map_length = len(map_list)
            for idx in range(map_length):
                map = map_list[idx]
                #map_listの中にshapeが同じマップがあったらそれをmapにする
                if (map.shape[0] == top_down_map.shape[0]) and (map.shape[1] == top_down_map.shape[1]):
                    map_idx = idx
                    flag2 = True
                    break
              
            #新しいマップの時は、map_listに追加する  
            if flag2 == False:
                map = top_down_map.copy()
                map = map.astype(np.float32)
                y, x = map.shape
                
                #log_writer = log_manager.createLogWriter("map_" + str(i) + "_pre")
                #mapの壁は-100, それ以外は-200にする
                for j in range(y):
                    for k in range(x):
                        if map[j][k] == 0:
                            map[j][k] = -100.0
                        else:
                            map[j][k] = -200.0
                            
                        #log_writer.write(str(map[j][k]))
                        
                    #log_writer.writeLine()
                          
                logger.info("New map appended: " + str(map.shape))  
                map_list.append(map)
                map_idx = len(map_list)-1 
                
            #現在位置を特定
            px = -1
            py = -1
            w = 0
              
            y, x = map.shape  
            for j in range(y):
                for k in range(x):
                    if top_down_map[j][k] == 4:
                        flag = True
                        px = k
                        py = j
                        for l in range(k, x):
                            if top_down_map[j][l] == 4:
                                w += 1
                            else:
                                break
                        
                    if flag == True:
                        break
                if flag == True:
                        break   
                       
            #現在位置の中心点を求める
            px += w//2
            py += w//2
            
            ci = rewards[0][1]
            
            # ciが大きかったら変更
            if map[py][px] < ci:
                map[py][px] = ci
                map_list[map_idx] = map
        
        env.close()
        
    #map_i.csvに書く(x, y)のciの最大値のマップを記録
    
    logger.info("MAP NUM: " + str(len(map_list)))
    for i in range(len(map_list)):
        log_writer = log_manager.createLogWriter(scene_name + "_" + str(i))
        map = map_list[i]
        y, x = map.shape
        logger.info("SIZE: " + str(map.shape))
        for j in range(y):
            for k in range(x):
                log_writer.write(str(map[j][k]))
            log_writer.writeLine()
            
def check_dir():
    dir_path = "data/scene_datasets/mp3d"
    dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    
    for i in range(len(dirs)):
        logger.info(dirs[i])
        
       
def research_valid_z(scene_idx):
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    dir_path = "data/scene_datasets/mp3d"
    dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    
    scene_name = dirs[scene_idx]
    logger.info("START FOR: " + scene_name)
        
    dataset_path = "map_dataset/" + scene_name + ".json.gz"    
    
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/" + scene_name + "/" + scene_name + ".glb"
    config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = ("./data/default.phys_scene_config.json")
    config.TASK_CONFIG.TRAINER_NAME = "oracle-ego"
    config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    config.freeze()
        
        
    #データセットに入れるz軸の候補を決める
    num = 1000000
    dataset = MaximumInfoDatasetV1()
    sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
    dataset.episodes += generate_maximuminfo_episode(sim=sim, num_episodes=num)         
        
    position_list = []
    num_list = []
    for i in range(len(dataset.episodes)):
        position = dataset.episodes[i].start_position[1]
            
        if position in position_list:
            idx = position_list.index(position)
            num_list[idx] += 1
        else:
            position_list.append(position)
            num_list.append(1)
                
    logger.info("LIST_SIZE: " + str(len(position_list)))
        
    #z軸が少数だったものは削除
    to_delete = []
    for i, n in enumerate(num_list):
        if n < (num/10):
            to_delete.append(i)
        
    for i in reversed(to_delete):
        num_list.pop(i)
        position_list.pop(i)
             
    logger.info("POSITION_LIST: " + str(len(position_list)))    
    for i in range(len(position_list)):
        logger.info(str(position_list[i])+ ", " + str(num_list[i]))
        
    z_list = position_list
        
        
def make_dataset(scene_idx, z_list):        
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    dir_path = "data/scene_datasets/mp3d"
    dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    
    scene_name = dirs[scene_idx]
    logger.info("START FOR: " + scene_name)
        
    dataset_path = "map_dataset/" + scene_name + ".json.gz"    
    
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/" + scene_name + "/" + scene_name + ".glb"
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = ("./data/default.phys_scene_config.json")
    config.TASK_CONFIG.TRAINER_NAME = "oracle-ego"
    config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    config.freeze()
    
    # position_listのz軸のみのデータセットを作成    
    num = 1000000    
    dataset = MaximumInfoDatasetV1()
    sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
    dataset.episodes += generate_maximuminfo_episode2(sim=sim, num_episodes=num, z_list=z_list)
        
    #datasetを.gzに圧縮
    with gzip.open(dataset_path, "wt") as f:
        f.write(dataset.to_json())
             
                
if __name__ == '__main__':
    scene_idx = 71
    z_list = [3.8914499282836914, -3.508549928665161, 0.09144997596740723] 
    navでのpictureのところを修正する
    #research_valid_z(scene_idx)
    #make_dataset(scene_idx, z_list)    
    research_ci(scene_idx)
    #check_dir()
    
    logger.info("################# FINISH EXPERIMENT !!!!! ##########################")