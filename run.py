
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
#sys.path.insert(0, "/gs/fs/tga-aklab/matsumoto/Main")
#sys.path.append(".")
#sys.path.insert(0, '../')
import argparse
import pathlib
import random
import datetime
import numpy as np
import torch
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config    
from log_manager import LogManager
from habitat.core.logging import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "train2", "train3", "train4", "train5", "eval", "eval2", "eval3", "eval4", "eval5", "random", "random2", "random3", "random4", "random5", "collect"],
        default=None,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--area-reward-type",
        choices=["coverage", "smooth-coverage", "curiosity", "novelty", "reconstruction"],
        default=None,
        help="area reward type of the experiment",
    )

    """
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--agent-type",
        choices=["no-map", "oracle", "oracle-ego", "proj-neural", "obj-recog"],
        required=True,
        help="agent type: oracle, oracleego, projneural, objrecog",
    )
    """

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    #run_exp(**vars(args))
    test(**vars(args))
    

def test(run_type: str, area_reward_type: str, opts=None):    
    exp_config = "habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    agent_type = "oracle-ego"
    
    if run_type is None:
        run_type = "train"
        #run_type = "eval"
        #run_type = "random"
        #run_type = "random2"
        
    logger.info("RUN TYPE: " + run_type)
    start_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    
    config = get_config(exp_config)
    
    if run_type in ["train", "train2", "train3", "train4"]:
        datadate = "" 
        config.defrost()
        config.NUM_PROCESSES = 24
        config.RL.PPO.num_mini_batch = 4
        #config.NUM_PROCESSES = 4
        #config.RL.PPO.num_mini_batch = 1
        config.TORCH_GPU_ID = 0
        config.TASK_CONFIG.AREA_REWARD = area_reward_type
        config.freeze()

    elif run_type in ["train5"]:
        logger.info(f"######## area_reward_type = {area_reward_type} ############")
        datadate = "" 
        config.defrost()
        config.NUM_PROCESSES = 24
        config.RL.PPO.num_mini_batch = 4
        #config.NUM_PROCESSES = 1
        #config.RL.PPO.num_mini_batch = 1
        config.TORCH_GPU_ID = 0
        config.TASK_CONFIG.AREA_REWARD = area_reward_type
        config.freeze()
        if area_reward_type == "novelty":
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append('NOVELTY_VALUE')
            config.freeze()
        elif area_reward_type == "smooth-coverage":
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append('SMOOTH_COVERAGE')
            config.freeze()
        elif area_reward_type == "reconstruction":
            config.defrost()
            config.TASK_CONFIG.TASK.SENSORS.append('POSE_ESTIMATION_RGB_SENSOR')
            config.TASK_CONFIG.TASK.SENSORS.append('DELTA_SENSOR')
            config.TASK_CONFIG.TASK.SENSORS.append('POSE_ESTIMATION_MASK_SENSOR')
            config.RL.PPO.num_steps = 64
            config.NUM_PROCESSES = 1
            config.RL.PPO.num_mini_batch = 1
            config.CHECKPOINT_INTERVAL = 8
            config.LOG_INTERVAL = 8
            config.freeze()

    elif run_type in ["eval", "eval2", "eval3", "eval4", "eval5"]:
        logger.info(f"######## area_reward_type = {area_reward_type} ############")
        #datadate = "24-07-25 06-34-14"
        #datadate = "24-08-11 21-55-35"
        #datadate = "24-10-05 21-07-23"
        #datadate = "24-10-07 06-13-55"
        #datadate = "24-10-08 21-09-33"
        datadate = "24-10-19 21-11-05"
        #datadate = "24-10-23 19-23-28"
        #datadate = "24-10-23 00-52-02"
        #current_ckpt = 120
        #current_ckpt = 165
        #current_ckpt = 155
        #current_ckpt = 653
        #current_ckpt = 662
        current_ckpt = 206
        #current_ckpt = 20
        #current_ckpt = 2051

        config.defrost()
        config.RL.PPO.num_mini_batch = 4
        config.NUM_PROCESSES = 28
        #config.RL.PPO.num_mini_batch = 1
        #config.NUM_PROCESSES = 1
        #config.TEST_EPISODE_COUNT = 220
        config.TEST_EPISODE_COUNT = 110
        config.VIDEO_OPTION = ["disk"]
        #config.VIDEO_OPTION = []
        config.TORCH_GPU_ID = 0
        #config.TASK_CONFIG.DATASET.DATA_PATH: "data/datasets/maximuminfo/v4/{split}/{split}.json.gz"
        #config.TASK_CONFIG.DATASET.DATA_PATH: "data/datasets/maximuminfo/v4/test/test.json.gz"
        config.TASK_CONFIG.AREA_REWARD = area_reward_type
        config.freeze()

        if area_reward_type == "novelty":
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append('NOVELTY_VALUE')
            config.freeze()
        elif area_reward_type == "smooth-coverage":
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append('SMOOTH_COVERAGE')
            config.freeze()
        elif area_reward_type == "reconstruction":
            config.defrost()
            config.TASK_CONFIG.TASK.SENSORS.append('POSE_ESTIMATION_RGB_SENSOR')
            config.TASK_CONFIG.TASK.SENSORS.append('DELTA_SENSOR')
            config.TASK_CONFIG.TASK.SENSORS.append('POSE_ESTIMATION_MASK_SENSOR')
            config.RL.PPO.num_steps = 64
            config.NUM_PROCESSES = 1
            config.RL.PPO.num_mini_batch = 1
            config.CHECKPOINT_INTERVAL = 8
            config.LOG_INTERVAL = 8
            config.freeze()

    elif run_type in ["random", "random2", "random3", "random4"]:
        datadate = "" 
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = "val"
        config.RL.PPO.num_mini_batch = 4
        config.NUM_PROCESSES = 28
        #config.NUM_PROCESSES = 8
        config.RL.PPO.num_mini_batch = 1
        config.NUM_PROCESSES = 1
        config.TEST_EPISODE_COUNT = 220
        config.TEST_EPISODE_COUNT = 110
        config.TEST_EPISODE_COUNT = 1
        config.VIDEO_OPTION = ["disk"]
        config.TORCH_GPU_ID = 0
        config.TASK_CONFIG.DATASET.DATA_PATH: "data/datasets/maximuminfo/v4/{split}/{split}.json.gz"
        config.TASK_CONFIG.AREA_REWARD = area_reward_type
        config.freeze()

    elif run_type in ["collect"]:
        datadate = "24-10-19 21-11-05"
        current_ckpt = 206

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = "val"
        config.RL.PPO.num_mini_batch = 1
        config.NUM_PROCESSES = 1
        #config.TEST_EPISODE_COUNT = 220
        config.TEST_EPISODE_COUNT = 110
        config.VIDEO_OPTION = ["disk"]
        config.TORCH_GPU_ID = 0
        config.TASK_CONFIG.DATASET.DATA_PATH: "data/datasets/maximuminfo/v4/{split}/{split}.json.gz"
        config.TASK_CONFIG.AREA_REWARD = area_reward_type
        config.TASK_CONFIG.TASK.MEASUREMENTS.append('CI')
        config.freeze()
    
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    
    config.defrost()
    #config.TASK_CONFIG.DATASET.SPLIT = "train"
    #config.EVAL.SPLIT = "train"
    config.TRAINER_NAME = agent_type
    config.TASK_CONFIG.TRAINER_NAME = agent_type
    config.CHECKPOINT_FOLDER = "cpt/" + start_date
    config.EVAL_CKPT_PATH_DIR = "cpt/" + datadate 
    config.freeze()
    
    if agent_type in ["oracle", "oracle-ego", "no-map"]:
        if run_type in ["train2", "eval2", "random2"]:
            trainer_init = baseline_registry.get_trainer("oracle2")
        elif run_type in ["train3", "eval3", "random3"]:
            trainer_init = baseline_registry.get_trainer("oracle3")
            config.defrost()
            config.NUM_PROCESSES = 24
            config.freeze()
        elif run_type in ["train4", "eval4", "random4"]:
            trainer_init = baseline_registry.get_trainer("oracle4")
        elif run_type in ["train5", "eval5", "random5"]:
            trainer_init = baseline_registry.get_trainer("oracle5")
        else:
            trainer_init = baseline_registry.get_trainer("oracle")
            #trainer_init = PPOTrainerO
        
        config.defrost()
        config.RL.PPO.hidden_size = 512 if agent_type=="no-map" else 768
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.5
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
        config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
        config.freeze()
    else:
        trainer_init = baseline_registry.get_trainer("non-oracle")
        config.defrost()
        config.RL.PPO.hidden_size = 512
        config.freeze()
        
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    
    #ログファイルの設定   
    log_manager = LogManager()
    log_manager.setLogDirectory("./log/" + start_date + "/" + run_type)
    
    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info("-----------------------------------")
    logger.info("device:" + str(device))
    logger.info("-----------------------------------")

    try:
        if run_type in ["train", "train2", "train3", "train4", "train5"]:
            #フォルダがない場合は、作成
            p_dir = pathlib.Path(config.CHECKPOINT_FOLDER)
            if not p_dir.exists():
                p_dir.mkdir(parents=True)
                
            trainer.train(log_manager, start_date)
        elif run_type in ["eval", "eval2", "eval3", "eval4", "eval5"]:
            trainer.eval(log_manager, start_date, current_ckpt)
        
        elif run_type in ["random", "random2", "random3", "random4"]:
            trainer.random_eval(log_manager, start_date)
        elif run_type in ["collect"]:
            trainer.collect_images(current_ckpt)
    finally:
        end_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
        print("Start at " + start_date)
        print("End at " + end_date)

if __name__ == "__main__":
    main()
    #test()

    #MIN_DEPTH: 0.5
    #MAX_DEPTH: 5.0
