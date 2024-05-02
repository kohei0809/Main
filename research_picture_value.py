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
        
        
def make_dataset_random(scene_idx):
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    dir_path = "data/scene_datasets/mp3d"
    dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    
    
    scene_name = dirs[scene_idx]
    logger.info("START FOR: " + scene_name)

    dataset_path = "map_dataset/" + scene_name + ".json.gz" 
    
    """   
    if os.path.exists(dataset_path):
        return True
    """
    
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

def _to_category_id(obs):
    scene = self._sim._sim.semantic_scene
    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
    mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])

    semantic_obs = np.take(mapping, obs)
    semantic_obs[semantic_obs>=40] = 0
    semantic_obs[semantic_obs<0] = 0
    return semantic_obs
        
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
        
    with InfoRLEnv(config=config) as env:
        logger.info("EPISODE NUM: "+ str(len(env.episodes)))
        
        #for i in range(len(env.episodes)):
        for i in range(100):
            #エピソードの変更
            env._env.current_episode = env.episodes[i]
            
            observation = env.reset()
            outputs = env.step2()
            rewards, done, info = outputs
        
            #ci = rewards[0][1]
            obs = observation["rgb"]
            semantic_obs = _to_category_id(observation["semantic"])
            H, W = semantic_obs.shape

            img = preprocess_img(image=obs) # padding and resizing input image into 384x288
            img = np.array(img)/255.
            img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)
            img = torch.from_numpy(img)
            if torch.cuda.is_available():
                img = img.type(torch.cuda.FloatTensor).to(device)
            else:
                img = img.type(torch.FloatTensor).to(device)
            pred_saliency = transalnet_model(img)
            raw_saliency = pred_saliency
            max_sal = raw_saliency.max()
            mean_sal = raw_saliency.mean()
            
            #objectのcategoryリスト
            category_list = []
            for i in range(H):
                for j in range(W):
                    category = semantic_obs[i][j]
                    if category not in category_list:
                        category_list.append(category)

            #score = max_sal * len(category_list)
            score = mean_sal * len(category_list)

            picture = Image.fromarray(np.uint8(observation["rgb"]))
            os.makedirs(f"picture_value/mean/{scene_name}", exist_ok=True)
            file_path = f"picture_value/mean/{scene_name}/{score}_{mean_sal}_{len(category_list)}.png"
            picture.save(file_path)

        env.close()
                
if __name__ == '__main__':
    for i in range(90):
        scene_idx = i
        is_create = True
        is_create = make_dataset_random(scene_idx)
        if is_create == True:
            research_saliency_and_similarity(scene_idx)
    
    logger.info("################# FINISH EXPERIMENT !!!!! ##########################")
