#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
sys.path.insert(0, "")
import argparse
import pathlib
import random
import datetime
import numpy as np
import torch
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config    
from utils.log_manager import LogManager
from habitat.core.logging import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "train2", "eval", "eval2", "eval3", "random", "random2"],
        default=None,
        help="run type of the experiment (train or eval)",
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
    

def test(run_type: str, opts=None):    
    exp_config = "habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    agent_type = "oracle-ego"
    
    if run_type is None:
        run_type = "train"
        run_type = "eval"
        run_type = "random"
        #run_type = "random2"
        
    logger.info("RUN TYPE: " + run_type)
    start_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    
    config = get_config(exp_config)
    
    if run_type in ["train", "train2"]:
        datadate = "" 
        config.defrost()
        config.NUM_PROCESSES = 40
        config.RL.PPO.num_mini_batch = 4
        config.freeze()
    elif run_type in ["eval", "eval2", "eval3"]:
        datadate = "24-03-08 14-49-01"
        #datadate = "24-03-09 01-05-42"
        #datadate = "24-03-13 01-04-07"
        #datadate = "24-03-16 20-40-27"

        config.defrost()
        config.NUM_PROCESSES = 60
        config.TEST_EPISODE_COUNT = 1100
        #config.VIDEO_OPTION = []
        config.VIDEO_OPTION = ["disk"]
        config.freeze()
    elif run_type=="random" or run_type=="random2":
        datadate = "" 
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = "val"
        config.NUM_PROCESSES = 60
        config.TEST_EPISODE_COUNT = 1100
        #config.VIDEO_OPTION = []
        config.VIDEO_OPTION = ["disk"]
        config.freeze()
    
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    
    config.defrost()
    #config.TASK_CONFIG.DATASET.DATA_PATH = "data/datasets/multinav/3_ON/{split}/{split}.json.gz"
    config.TASK_CONFIG.DATASET.DATA_PATH = "data/datasets/maximuminfo/v1/{split}/{split}.json.gz"
    config.TRAINER_NAME = agent_type
    config.TASK_CONFIG.TRAINER_NAME = agent_type
    config.CHECKPOINT_FOLDER = "cpt/" + start_date
    config.EVAL_CKPT_PATH_DIR = "cpt/" + datadate 
    config.freeze()
    
    if agent_type in ["oracle", "oracle-ego", "no-map"]:
        if run_type in ["train2", "eval2"]:
            trainer_init = baseline_registry.get_trainer("oracle2")
        elif run_type == "eval3":
            trainer_init = baseline_registry.get_trainer("oracle3")
        else:
            trainer_init = baseline_registry.get_trainer("oracle")
        
        config.defrost()
        config.RL.PPO.hidden_size = 512 if agent_type=="no-map" else 768
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.5
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
        config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
        if agent_type == "oracle-ego":
            config.TASK_CONFIG.TASK.MEASUREMENTS.append('FOW_MAP')
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

    if run_type in ["train", "train2"]:
        #フォルダがない場合は、作成
        p_dir = pathlib.Path(config.CHECKPOINT_FOLDER)
        if not p_dir.exists():
            p_dir.mkdir(parents=True)
            
        trainer.train(log_manager, start_date)
    elif run_type in ["eval", "eval2", "eval3"]:
        trainer.eval(log_manager, start_date)
    
    elif run_type == "random":
        trainer.random_eval(log_manager, start_date)
    
    elif run_type == "random2":
        trainer.random_eval2(log_manager, start_date)
       
    end_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    print("Start at " + start_date)
    print("End at " + end_date)

if __name__ == "__main__":
    main()
    #test()

    #MIN_DEPTH: 0.5
    #MAX_DEPTH: 5.0
