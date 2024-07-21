#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import datetime
import pathlib

import numpy as np
import torch

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from utils.log_manager import LogManager
from habitat.core.logging import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        default="train",
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    #run_exp(**vars(args))
    test(**vars(args))


def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


def test(run_type: str, opts=None):    
    exp_config = "habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    agent_type = "oracle-ego"
    
    if run_type is None:
        run_type = "train"
        
    logger.info("RUN TYPE: " + run_type)
    start_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    
    config = get_config(exp_config)
    
    if run_type in ["train"]:
        datadate = "" 
        config.defrost()
        #config.NUM_PROCESSES = 24
        config.NUM_ENVIRONMENTS = 8
        config.RL.PPO.num_mini_batch = 4
        #config.NUM_PROCESSES = 1
        #config.NUM_ENVIRONMENTS = 1
        #config.RL.PPO.num_mini_batch = 1
        config.TORCH_GPU_ID = 0
        config.freeze()
    elif run_type in ["eval"]:
        if run_type == "eval2":
            datadate = "24-05-16 16-06-47"
            current_ckpt = 300
        else:
            datadate = "24-04-26 00-36-56"
            current_ckpt = 108

        config.defrost()
        #config.RL.PPO.num_mini_batch = 4
        #config.NUM_PROCESSES = 8
        config.RL.PPO.num_mini_batch = 1
        config.NUM_PROCESSES = 1
        config.TEST_EPISODE_COUNT = 220
        config.VIDEO_OPTION = ["disk"]
        config.TORCH_GPU_ID = 1
        config.freeze()
    elif run_type in ["random", "random2"]:
        datadate = "" 
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = "val"
        config.RL.PPO.num_mini_batch = 4
        config.NUM_PROCESSES = 24
        config.TEST_EPISODE_COUNT = 220
        config.VIDEO_OPTION = ["disk"]
        config.TORCH_GPU_ID = 1
        config.freeze()
    
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)
    
    config.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = "data/datasets/maximuminfo/v3/{split}/{split}.json.gz"
    config.TRAINER_NAME = agent_type
    config.TASK_CONFIG.TRAINER_NAME = agent_type
    config.CHECKPOINT_FOLDER = "cpt/" + start_date
    config.EVAL_CKPT_PATH_DIR = "cpt/" + datadate 
    config.freeze()
    
    if agent_type in ["oracle", "oracle-ego", "no-map"]:
        if run_type in ["train2", "eval2", "random2"]:
            trainer_init = baseline_registry.get_trainer("oracle2")
        elif run_type in ["train3", "eval3"]:
            trainer_init = baseline_registry.get_trainer("oracle3")
        else:
            trainer_init = baseline_registry.get_trainer("oracle")
            #trainer_init = PPOTrainerO
        
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

    try:
        if run_type in ["train", "train2", "train3"]:
            #フォルダがない場合は、作成
            p_dir = pathlib.Path(config.CHECKPOINT_FOLDER)
            if not p_dir.exists():
                p_dir.mkdir(parents=True)
                
            trainer.train(log_manager)
        elif run_type in ["eval", "eval2", "eval3"]:
            trainer.eval(log_manager, start_date, current_ckpt)
        
        elif run_type in ["random", "random2"]:
            trainer.random_eval(log_manager, start_date)
    finally:
        end_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
        print("Start at " + start_date)
        print("End at " + end_date)


if __name__ == "__main__":
    main()
