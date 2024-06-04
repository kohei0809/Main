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
from lavis.models import load_model_and_preprocess
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from torchvision import transforms
from scipy import stats


from habitat_baselines.config.default import get_config  
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.datasets.maximum_info.maximuminfo_dataset import MaximumInfoDatasetV1
from habitat.datasets.maximum_info.maximuminfo_generator import generate_maximuminfo_episode, generate_maximuminfo_episode2
from habitat_baselines.common.environments import InfoRLEnv
from habitat_baselines.common.baseline_registry import baseline_registry
from log_manager import LogManager
from log_writer import LogWriter
from habitat.core.logging import logger

from TranSalNet.utils.data_process import preprocess_img, postprocess_img
from TranSalNet.TranSalNet_Dense import TranSalNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
lavis_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
transalnet_model = TranSalNet()
transalnet_model.load_state_dict(torch.load('TranSalNet/pretrained_models/TranSalNet_Dense.pth'))
transalnet_model = transalnet_model.to(device) 
transalnet_model.eval()

    
def create_description(picture):
    # pictureのdescriptionを作成
    image = Image.fromarray(picture)
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    generated_text = lavis_model.generate({"image": image}, use_nucleus_sampling=True,num_captions=1)[0]
    return generated_text
    
def calculate_similarity(pred_description, origin_description):
    # 文をSentence Embeddingに変換
    embedding1 = bert_model.encode(pred_description, convert_to_tensor=True)
    embedding2 = bert_model.encode(origin_description, convert_to_tensor=True)
    
    # コサイン類似度を計算
    sentence_sim = util.pytorch_cos_sim(embedding1, embedding2).item()
    
    return sentence_sim
        
        
def make_dataset_random(scene_idx):
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    dir_path = "data/scene_datasets/mp3d"
    dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    
    
    scene_name = dirs[scene_idx]
    logger.info("START FOR: " + scene_name)
    
    # ファイルを読み込んで行ごとにリストに格納する
    with open('data/scene_datasets/mp3d/description.txt', 'r') as file:
        lines = file.readlines()

    # scene id と文章を抽出してデータフレームに変換する
    scene_ids = []
    descriptions = []
    for i in range(0, len(lines), 3):
        scene_ids.append(lines[i].strip())
        descriptions.append(lines[i+2].strip())

    description_df = pd.DataFrame({'scene_id': scene_ids, 'description': descriptions})
    description = description_df[description_df["scene_id"]==scene_name]["description"].item()
    if description == "wip":
        logger.info("############# wip ###############")
        return False

    dataset_path = "map_dataset/" + scene_name + ".json.gz" 
    if os.path.exists(dataset_path):
        logger.info("########### EXIST #############")
        return True   
    
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
    num = 100
    dataset = MaximumInfoDatasetV1()
    sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
    dataset.episodes += generate_maximuminfo_episode2(sim=sim, num_episodes=num)
    sim.close()
        
    #datasetを.gzに圧縮
    with gzip.open(dataset_path, "wt") as f:
        f.write(dataset.to_json())
    
    return True
        
def research_saliency_and_similarity(scene_idx):
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
    log_manager.setLogDirectory("research_picture_value_and_similarity/csv/")
    log_writer = log_manager.createLogWriter("similarity_" + scene_name)
    
    # ファイルを読み込んで行ごとにリストに格納する
    with open('data/scene_datasets/mp3d/description.txt', 'r') as file:
        lines = file.readlines()

    # scene id と文章を抽出してデータフレームに変換する
    scene_ids = []
    descriptions = []
    for i in range(0, len(lines), 3):
        scene_ids.append(lines[i].strip())
        descriptions.append(lines[i+2].strip())

    description_df = pd.DataFrame({'scene_id': scene_ids, 'description': descriptions})
        
    with InfoRLEnv(config=config) as env:
        for i in range(100):        
            #エピソードの変更
            env._env.current_episode = env.episodes[i]
            
            observation = env.reset()
            outputs = env.step2()
            rewards, done, info = outputs
        
            if rewards == -1:
                continue

            description = description_df[description_df["scene_id"]==env._env.current_episode.scene_id[-15:-4]]["description"].item()
            pred_descriotion = create_description(observation["rgb"])
            similarity = calculate_similarity(pred_descriotion, description)
            
            log_writer.writeLine(str(similarity) + "," + str(rewards))
    
        env.close()
                
if __name__ == '__main__':
    for i in range(90):
        scene_idx = i
        is_create = make_dataset_random(scene_idx)
        if is_create == True:
            research_saliency_and_similarity(scene_idx)
    
    logger.info("################# FINISH EXPERIMENT !!!!! ##########################")
