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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
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

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, agent_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    config.defrost()
    config.TRAINER_NAME = agent_type
    config.TASK_CONFIG.TRAINER_NAME = agent_type
    config.freeze()

    if agent_type in ["oracle", "oracle-ego", "no-map"]:
        trainer_init = baseline_registry.get_trainer("oracle")
        config.defrost()
        config.RL.PPO.hidden_size = 512 if agent_type=="no-map" else 768
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
    
    #ログファイルの設定   
    start_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    log_manager_train = LogManager()
    log_manager_train.setLogDirectory("./log/" + start_date + "/train")
    log_manager_val = LogManager()
    log_manager_val.setLogDirectory("./log/" + start_date + "/val")
    
    
    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("-----------------------------------")
    print("device:" + str(device))
    print("-----------------------------------")
    
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train(log_manager_train, start_date)
    elif run_type == "eval":
        trainer.eval(log_manager_val, start_date)
        
    end_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S')
    print("Start at " + start_date)
    print("End at " + end_date)
    

def test():
    exp_config = "habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    agent_type = "oracle-ego"
    run_type = "train"
    start_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    
    if run_type == "eval":
        datadate = "23-10-19 15-55-17"
    else:
       datadate = "" 
    
    config = get_config(exp_config)
    
    
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    
    config.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = "data/datasets/multinav/3_ON/{split}/{split}.json.gz"
    config.TRAINER_NAME = agent_type
    config.TASK_CONFIG.TRAINER_NAME = agent_type
    config.CHECKPOINT_FOLDER = "cpt/" + start_date
    config.EVAL_CKPT_PATH_DIR = "cpt/" + datadate 
    #config.TEST_EPISODE_COUNT = 1000
    config.VIDEO_OPTION = []
    #config.VIDEO_OPTION = ["disk"]
    config.freeze()
    
    if agent_type in ["oracle", "oracle-ego", "no-map"]:
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
    print("-----------------------------------")
    print("device:" + str(device))
    print("-----------------------------------")

    if run_type == "train":
        #フォルダがない場合は、作成
        p_dir = pathlib.Path(config.CHECKPOINT_FOLDER)
        if not p_dir.exists():
            p_dir.mkdir(parents=True)
            
        trainer.train(log_manager, start_date)
    elif run_type == "eval":
        trainer.eval(log_manager, start_date)
       
    end_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    print("Start at " + start_date)
    print("End at " + end_date)

if __name__ == "__main__":
    #main()
    test()

    #MIN_DEPTH: 0.5
    #MAX_DEPTH: 5.0
