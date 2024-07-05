import os
import random

import numpy as np
from gym import spaces

from typing import cast

from matplotlib import pyplot as plt

from PIL import Image

from habitat_sim.utils.common import d3_40_colors_rgb

from habitat_baselines.config.default import get_config  
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat.core.env_point import Env
#from habitat.core.env import Env
from habitat.utils.visualizations import maps
from log_manager import LogManager
from log_writer import LogWriter

def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), sem_writer=None, env=None, config=None, opt=-1
):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    
    arr = [rgb_obs]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )

        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")
        print("ID")
        scene = env.sim._sim.semantic_scene
        print(semantic_obs[0][0])
        print(scene.semantic_index_to_object_index(semantic_obs[0][0]))
        print(semantic_obs[255][0])
        print(scene.semantic_index_to_object_index(semantic_obs[255][0]))
        #print(d3_40_colors_rgb.shape)
        #print(d3_40_colors_rgb)
        
        for i in range(semantic_obs.shape[0]):
            for j in range(semantic_obs.shape[1]):
                sem_writer.write(str(semantic_obs[i][j]))
            sem_writer.writeLine()
        
        
    if depth_obs.size != 0:
        depth_obs = (depth_obs - config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH) / (
                    config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH - config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                )
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        arr.append(depth_img)
        titles.append("depth")
        
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    
    path = "./fig.png"
    if opt !=-1:
        path = "./fig" + str(opt) + ".png"
    plt.savefig(path)
    plt.show(block=False)
    
def print_scene_recur(scene, limit_output=1000):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
    #print(len(scene.semantic_index_map))

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None
                
def check_semantic(env, semantic_obs, sem_writer, sem2_writer):
    # obtain mapping from instance id to semantic label id
    scene = env.sim._sim.semantic_scene
    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
    mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])
    
    obs = np.take(mapping, semantic_obs)
    print(obs[255][0])
    print(obs[255][1])
    
    for i in range(semantic_obs.shape[0]):
        for j in range(semantic_obs.shape[1]):
            sem2_writer.write(str(semantic_obs[i][j]))
        sem2_writer.writeLine()

    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            sem_writer.write(str(obs[i][j]))
        sem_writer.writeLine()
        
    semantic_img = Image.new(
        "P", (semantic_obs.shape[1], semantic_obs.shape[0])
    )

    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(semantic_img)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)
    path = "./semantic.png"
    plt.savefig(path)
    plt.show(block=False)
    
    
if __name__ == '__main__':
    exp_config = "./habitat/config/test.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    dataset_path = "test_dataset.json.gz"
        
    config.defrost()
    config.DATASET.DATA_PATH = dataset_path
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.TASK_CONFIG.TASK.MEASUREMENTS = ['DISTANCE_TO_CURR_GOAL', 'DISTANCE_TO_MULTI_GOAL', 'SUB_SUCCESS', 'SUCCESS', 'EPISODE_LENGTH', 'MSPL', 'PERCENTAGE_SUCCESS', 'RATIO', 'PSPL', 'RAW_METRICS', 'TOP_DOWN_MAP']
    config.TASK_CONFIG.TRAINER_NAME = "oracle"
    config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    config.freeze()
    
    #print(config)
    #print("########################")
    
    log_manager = LogManager()
    log_manager.setLogDirectory("./")
    sem_writer = log_manager.createLogWriter("semantic")
    sem2_writer = log_manager.createLogWriter("semantic_each")
    
    with Env(config=config.TASK_CONFIG) as env:
        observation = env.reset()
        info = env.get_metrics()
        #display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), sem_writer, env, config.TASK_CONFIG, opt=-1)
        scene = env.sim._sim.semantic_scene
        #print_scene_recur(scene)
        check_semantic(env, observation['semantic'], sem_writer, sem2_writer)
    