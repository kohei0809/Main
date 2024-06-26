#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ある程度多くの写真を探索中は保持し、探索終了後に取捨選択する 

import json
import os
import time
import pathlib
import sys
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw
import pandas as pd
import random

from einops import rearrange
from matplotlib import pyplot as plt
import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.optim.lr_scheduler import LambdaLR
import clip

from sentence_transformers import SentenceTransformer, util
from lavis.models import load_model_and_preprocess

from habitat import Config
from habitat.core.logging import logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainerOracle
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorageOracle
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo import PPOOracle, BaselinePolicyOracle, ProposedPolicyOracle
from log_manager import LogManager
from log_writer import LogWriter
from habitat.utils.visualizations import fog_of_war, maps

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer

@baseline_registry.register_trainer(name="oracle2")
class PPOTrainerO2(BaseRLTrainerOracle):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]
    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None
        
        self._num_picture = config.TASK_CONFIG.TASK.PICTURE.NUM_PICTURE
        self.save_picture_reward = 0.02


    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = ProposedPolicyOracle(
            agent_type = self.config.TRAINER_NAME,
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            device=self.device,
            object_category_embedding_size=self.config.RL.OBJECT_CATEGORY_EMBEDDING_SIZE,
            previous_action_embedding_size=self.config.RL.PREVIOUS_ACTION_EMBEDDING_SIZE,
            use_previous_action=self.config.RL.PREVIOUS_ACTION
        )
        
        logger.info("DEVICE: " + str(self.device))
        self.actor_critic.to(self.device)

        self.agent = PPOOracle(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def save_checkpoint(self, file_name: str, extra_state: Optional[Dict] = None) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "traj_metrics", "saliency"}

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict[str, Any]) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue
                
            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(cls, infos: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results
    
    def _create_caption(self, picture):
        # 画像からcaptionを生成する
        image = Image.fromarray(picture)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        generated_text = self.lavis_model.generate({"image": image}, use_nucleus_sampling=True,num_captions=1)[0]
        return generated_text
    
    def create_description(self, picture_list):
        # captionを連結してdescriptionを生成する
        description = ""
        for i in range(len(picture_list)):
            description += (self._create_caption(picture_list[i][1]) + ". ")
            
        return description
    
    def _create_new_description_embedding(self, caption):
        # captionのembeddingを作成
        embedding = self.bert_model.encode(caption, convert_to_tensor=True)
        return embedding
    
    def _create_new_image_embedding(self, obs):
        image = Image.fromarray(obs)
        image = self.preprocess(image)
        image = torch.tensor(image).clone().detach().to(self.device).unsqueeze(0)
        embetting = self.clip_model.encode_image(image).float()
        return embetting

    def calculate_similarity(self, pred_description, origin_description):
        # 文をSentence Embeddingに変換
        embedding1 = self.bert_model.encode(pred_description, convert_to_tensor=True)
        embedding2 = self.bert_model.encode(origin_description, convert_to_tensor=True)
    
        # コサイン類似度を計算
        sentence_sim = util.pytorch_cos_sim(embedding1, embedding2).item()
    
        return sentence_sim
    
    def _cal_remove_index(self, picture_list, new_emmbedding):
        # 削除する写真を決める
        # 他のsyasinnとの類似度を計算し、合計が最大のものを削除
        sim_list = [[-10 for _ in range(len(picture_list)+1)] for _ in range(len(picture_list)+1)]
        sim_list[len(picture_list)][len(picture_list)] = 0.0
        for i in range(len(picture_list)):
            emd = picture_list[i][2]
            sim_list[i][len(picture_list)] = util.pytorch_cos_sim(emd, new_emmbedding).item()
            sim_list[len(picture_list)][i] = sim_list[i][len(picture_list)]
            for j in range(i, len(picture_list)):
                if i == j:
                    sim_list[i][j] = 0.0
                    continue
                
                sim_list[i][j] = util.pytorch_cos_sim(emd, picture_list[j][2]).item()
                sim_list[j][i] = sim_list[i][j]
                
        total_sim = [sum(similarity_list) for similarity_list in sim_list]
        remove_index = total_sim.index(max(total_sim))
        return remove_index

    def _calculate_pic_sim(self, picture_list):
        if len(picture_list) <= 1:
            return 0.0
            
        sim_list = [[-10 for _ in range(len(picture_list))] for _ in range(len(picture_list))]

        for i in range(len(picture_list)):
            emd = picture_list[i][2]
            for j in range(i, len(picture_list)):
                if i == j:
                    sim_list[i][j] = 0.0
                    continue
                    
                sim_list[i][j] = util.pytorch_cos_sim(emd, picture_list[j][2]).item()
                sim_list[j][i] = sim_list[i][j]
                
        total_sim = np.sum(sim_list)
        total_sim /= (len(picture_list)*(len(picture_list)-1))
        return total_sim
                
    def _collect_rollout_step(
        self, rollouts, current_episode_reward, current_episode_exp_area, current_episode_picture_value, current_episode_similarity, current_episode_picsim, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        #logger.info("actions=" + str(actions))
        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        
        reward = []
        pic_val = []
        picture_value = []
        similarity = []
        pic_sim = []
        exp_area = [] # 探索済みのエリア()
        n_envs = self.envs.num_envs
        for n in range(n_envs):
            reward.append(rewards[n][0])
            pic_val.append(rewards[n][1])
            picture_value.append(0)
            similarity.append(0)
            pic_sim.append(0)
            exp_area.append(rewards[n][2]-rewards[n][3])
            
        current_episodes = self.envs.current_episodes()
        for n in range(len(observations)):
             #TAKE_PICTUREが呼び出されたかを検証(TAKE_PICTUREを行っても変な写真の場合は保存しない)
            if pic_val[n] == -1:
                continue
            # 2回連続でTAKE_PICTUREをした場合は保存しない
            # どうせ後から同じ写真は消すため、選別の際の計算量の削減のため
            if rollouts.prev_actions[rollouts.step][n] == actions[n]:
                continue

            image_emd = self._create_new_image_embedding(observations[n]["rgb"])
            self._taken_picture_list[n].append([pic_val[n], observations[n]["rgb"], image_emd])
            reward[n] += self.save_picture_reward
            
        reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
        exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.device).unsqueeze(1)
        picture_value = torch.tensor(picture_value, dtype=torch.float, device=self.device).unsqueeze(1)
        similarity = torch.tensor(similarity, dtype=torch.float, device=self.device).unsqueeze(1)
        pic_sim = torch.tensor(pic_sim, dtype=torch.float, device=self.device).unsqueeze(1)
            
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )
        
        # episode ended
        for n in range(len(observations)):
            if masks[n].item() == 0.0:
                # 写真の選別
                self._taken_picture_list[n], picture_value[n] = self._select_pictures(self._taken_picture_list[n])
                results_image = self._create_results_image(self._taken_picture_list[n])

                # Ground-Truth descriptionと生成文との類似度の計算 
                description = self.description_df[self.description_df["scene_id"]==current_episodes[n].scene_id[-15:-4]]["description"].item()
                #pred_description = self.create_description(self._taken_picture_list[n])
                pred_description = ""
                if results_image is not None:
                    pred_description = self.create_description_from_results_image(results_image)
                similarity[n] = self.calculate_similarity(pred_description, description)
                pic_sim[n] = self._calculate_pic_sim(self._taken_picture_list[n])                
                reward[n] += similarity[n]*10
    
                self._taken_picture_list[n] = []
                

        current_episode_reward += reward
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        current_episode_exp_area += exp_area
        running_episode_stats["exp_area"] += (1 - masks) * current_episode_exp_area
        current_episode_picture_value += picture_value
        running_episode_stats["picture_value"] += (1 - masks) * current_episode_picture_value
        current_episode_similarity += similarity
        running_episode_stats["similarity"] += (1 - masks) * current_episode_similarity
        current_episode_picsim += pic_sim
        running_episode_stats["pic_sim"] += (1 - masks) * current_episode_picsim
        running_episode_stats["count"] += 1 - masks

        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks
        current_episode_exp_area *= masks
        current_episode_picture_value *= masks
        current_episode_similarity *= masks
        current_episode_picsim *= masks
        
        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            reward,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )


    def train(self, log_manager, date) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        logger.info("########### PPO2 ##############")

        self.log_manager = log_manager
        
        #ログ出力設定
        #time, reward
        reward_logger = self.log_manager.createLogWriter("reward")
        #time, learning_rate
        learning_rate_logger = self.log_manager.createLogWriter("learning_rate")
        #time, found, forward, left, right, look_up, look_down
        action_logger = self.log_manager.createLogWriter("action_prob")
        #time, picture, episode_length
        metrics_logger = self.log_manager.createLogWriter("metrics")
        #time, losses_value, losses_policy
        loss_logger = self.log_manager.createLogWriter("loss")
        
        self.take_picture_writer = self.log_manager.createLogWriter("take_picture")
        self.picture_position_writer = self.log_manager.createLogWriter("picture_position")

        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        
        # picture_value, rgb_image, image_emb
        self._taken_picture_list = []
        for _ in range(self.envs.num_envs):
            self._taken_picture_list.append([])
            
        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # Sentence-BERTモデルの読み込み
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # lavisモデルの読み込み
        self.lavis_model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=self.device)
        self.bert_model.to(self.device)
        self.lavis_model.to(self.device)

        # Load the clip model
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        self._select_threthould = 0.9

        # LLaVA model
        load_4bit = True
        load_8bit = not load_4bit
        disable_torch_init()
        model_path = "liuhaotian/llava-v1.5-13b"
        self.llava_model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.llava_model, self.llava_image_processor, _ = load_pretrained_model(model_path, None, self.llava_model_name, load_8bit, load_4bit)
        
        # ファイルを読み込んで行ごとにリストに格納する
        with open('data/scene_datasets/mp3d/description.txt', 'r') as file:
            lines = file.readlines()

        # scene id と文章を抽出してデータフレームに変換する
        scene_ids = []
        descriptions = []
        for i in range(0, len(lines), 3):
            scene_ids.append(lines[i].strip())
            descriptions.append(lines[i+2].strip())

        self.description_df = pd.DataFrame({'scene_id': scene_ids, 'description': descriptions})
            
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorageOracle(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_exp_area = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picture_value = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_similarity = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picsim = torch.zeros(self.envs.num_envs, 1, device=self.device)
        
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            exp_area=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            picture_value=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            similarity=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            pic_sim=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        os.makedirs(self.config.TENSORBOARD_DIR + "/" + date, exist_ok=True)
        
        for update in range(self.config.NUM_UPDATES):
            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                    update, self.config.NUM_UPDATES
                )

            for step in range(ppo_cfg.num_steps):
                #logger.info(f"STEP: {step}")
                    
                # 毎ステップ初期化する
                """
                for n in range(self.envs.num_envs):
                    self._observed_object_ci_one[n] = [0, 0, 0]
                """
                    
                (
                    delta_pth_time,
                    delta_env_time,
                    delta_steps,
                ) = self._collect_rollout_step(
                    rollouts, current_episode_reward, current_episode_exp_area, current_episode_picture_value, current_episode_similarity, current_episode_picsim, running_episode_stats
                )
                pth_time += delta_pth_time
                env_time += delta_env_time
                count_steps += delta_steps

            (
                delta_pth_time,
                value_loss,
                action_loss,
                dist_entropy,
            ) = self._update_agent(ppo_cfg, rollouts)
            pth_time += delta_pth_time
                
            for k, v in running_episode_stats.items():
                window_episode_stats[k].append(v.clone())

            deltas = {
                k: (
                    (v[-1] - v[0]).sum().item()
                    if len(v) > 1
                    else v[0].sum().item()
                )
                for k, v in window_episode_stats.items()
            }
            deltas["count"] = max(deltas["count"], 1.0)
                
            #csv
            reward_logger.writeLine(str(count_steps) + "," + str(deltas["reward"] / deltas["count"]))
            learning_rate_logger.writeLine(str(count_steps) + "," + str(lr_scheduler._last_lr[0]))

            total_actions = rollouts.actions.shape[0] * rollouts.actions.shape[1]
            total_found_actions = int(torch.sum(rollouts.actions == 0).cpu().numpy())
            total_forward_actions = int(torch.sum(rollouts.actions == 1).cpu().numpy())
            total_left_actions = int(torch.sum(rollouts.actions == 2).cpu().numpy())
            total_right_actions = int(torch.sum(rollouts.actions == 3).cpu().numpy())
            total_look_up_actions = int(torch.sum(rollouts.actions == 4).cpu().numpy())
            total_look_down_actions = int(torch.sum(rollouts.actions == 5).cpu().numpy())
            assert total_actions == (total_found_actions + total_forward_actions + 
                total_left_actions + total_right_actions + total_look_up_actions + 
                total_look_down_actions
            )
                
            # csv
            action_logger.writeLine(
                str(count_steps) + "," + str(total_found_actions/total_actions) + ","
                + str(total_forward_actions/total_actions) + "," + str(total_left_actions/total_actions) + ","
                + str(total_right_actions/total_actions) + "," + str(total_look_up_actions/total_actions) + ","
                + str(total_look_down_actions/total_actions)
            )
            metrics = {
                k: v / deltas["count"]
                for k, v in deltas.items()
                if k not in {"reward", "count"}
            }

            if len(metrics) > 0:
                logger.info("COUNT: " + str(deltas["count"]))
                logger.info("Similarity: " + str(metrics["similarity"]))
                logger.info("Picture Value: " + str(metrics["picture_value"]))
                logger.info("Pic_Sim: " + str(metrics["pic_sim"]))
                logger.info("REWARD: " + str(deltas["reward"] / deltas["count"]))
                metrics_logger.writeLine(str(count_steps) + "," + str(metrics["exp_area"]) + "," + str(metrics["similarity"]) + "," + str(metrics["picture_value"]) + "," + str(metrics["pic_sim"]) + "," + str(metrics["raw_metrics.agent_path_length"]))
                    
                logger.info(metrics)
            
            loss_logger.writeLine(str(count_steps) + "," + str(value_loss) + "," + str(action_loss))
                

            # log stats
            if update > 0 and update % self.config.LOG_INTERVAL == 0:
                logger.info(
                    "update: {}\tfps: {:.3f}\t".format(
                        update, count_steps / (time.time() - t_start)
                    )
                )

                logger.info(
                    "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                    "frames: {}".format(
                        update, env_time, pth_time, count_steps
                    )
                )

                logger.info(
                    "Average window size: {}  {}".format(
                        len(window_episode_stats["count"]),
                        "  ".join(
                            "{}: {:.3f}".format(k, v / deltas["count"])
                            for k, v in deltas.items()
                            if k != "count"
                        ),
                    )
                )

            # checkpoint model
            if update % self.config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(
                    f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                )
                count_checkpoints += 1

        self.envs.close()

    def _select_pictures(self, taken_picture_list):
        results = []
        res_val = 0.0

        sorted_picture_list = sorted(taken_picture_list, key=lambda x: x[0], reverse=True)
        i = 0
        while True:
            if len(results) == self._num_picture:
                break
            if i == len(sorted_picture_list):
                break
            emd = sorted_picture_list[i][2]
            is_save = self._decide_save(emd, results)

            if is_save == True:
                results.append(sorted_picture_list[i])
                res_val += sorted_picture_list[i][0]
            i += 1

        res_val /= len(results)
        return results, res_val

    def _select_random_pictures(self, taken_picture_list):
        results = taken_picture_list
        num = len(taken_picture_list)
        if len(taken_picture_list) > self._num_picture:
            results = random.sample(taken_picture_list, self._num_picture)
            num = self._num_picture
        res_val = 0.0

        for i in range(num):
            res_val += results[i][0]

        return results, res_val


    def _decide_save(self, emd, results):
        for i in range(len(results)):
            check_emb = results[i][2]

            sim = util.pytorch_cos_sim(emd, check_emb).item()
            if sim >= self._select_threthould:
                return False
        return True


    def _create_results_image(self, picture_list):
        images = []
        if len(picture_list) == 0:
            return None

        for i in range(self._num_picture):
            idx = i%len(picture_list)
            images.append(Image.fromarray(picture_list[idx][1]))

        width, height = images[0].size
        result_width = width * 5
        result_height = height * 2
        result_image = Image.new("RGB", (result_width, result_height))

        for i, image in enumerate(images):
            x_offset = (i % 5) * width
            y_offset = (i // 5) * height
            result_image.paste(image, (x_offset, y_offset))
        
        draw = ImageDraw.Draw(result_image)
        for x in range(width, result_width, width):
            draw.line([(x, 0), (x, result_height)], fill="black", width=7)
        for y in range(height, result_height, height):
            draw.line([(0, y), (result_width, y)], fill="black", width=7)

        return result_image


    def create_description_from_results_image(self, results_image):
        input_text = "You are an excellent property writer. This picture consists of 10 pictures arranged in one picture, 5 horizontally and 2 vertically on one building. In addition, a black line separates the pictures from each other. From each picture, you should understand the details of this building's environment and describe this building's environment in detail in the form of a summary of these pictures. At this point, do not describe each picture one at a time, but rather in a summarized form. Also note that each picture was taken in a separate location, so successive pictures are not positionally close. Additionally, do not mention which picture you are quoting from or the black line separating each picture."
        response = self.generate_response(results_image, input_text)
        response = response[4:-4]
        return response


    def generate_response(self, image, input_text):
        if 'llama-2' in self.llava_model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.llava_model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.llava_model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        roles = conv.roles if "mpt" not in self.llava_model_name.lower() else ('user', 'assistant')

        image_tensor = self.llava_image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        inp = input_text
        if image is not None:
            if self.llava_model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.llava_model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=2048,
                streamer=streamer,
                use_cache=True,
            )

        outputs = self.tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs
        outputs = outputs.replace("\n\n", " ")
        return outputs


    def _eval_checkpoint(self, checkpoint_path: str, log_manager: LogManager, date: str, checkpoint_index: int = 0) -> None:
        logger.info("############### EAVL2 ##################")
        self.log_manager = log_manager
        #ログ出力設定
        #time, reward
        eval_reward_logger = self.log_manager.createLogWriter("reward")
        #time, exp_area, simlarity, each_sim, path_length
        eval_metrics_logger = self.log_manager.createLogWriter("metrics")
        #フォルダがない場合は、作成
        
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        logger.info(checkpoint_path)

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        
        # picture_value, rgb_image, image_emb
        self._taken_picture_list = []
        
        # Sentence-BERTモデルの読み込み
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # lavisモデルの読み込み
        self.lavis_model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=self.device)
        self.bert_model.to(self.device)
        self.lavis_model.to(self.device)

        # Load the clip model
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        self._select_threthould = 0.9

        # LLaVA model
        load_4bit = True
        load_8bit = not load_4bit
        disable_torch_init()
        model_path = "liuhaotian/llava-v1.5-13b"
        self.llava_model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.llava_model, self.llava_image_processor, _ = load_pretrained_model(model_path, None, self.llava_model_name, load_8bit, load_4bit)
        
        # ファイルを読み込んで行ごとにリストに格納する
        with open('data/scene_datasets/mp3d/description.txt', 'r') as file:
            lines = file.readlines()

        # scene id と文章を抽出してデータフレームに変換する
        scene_ids = []
        descriptions = []
        for i in range(0, len(lines), 3):
            scene_ids.append(lines[i].strip())
            descriptions.append(lines[i+2].strip())

        self.description_df = pd.DataFrame({'scene_id': scene_ids, 'description': descriptions})
        
        for i in range(self.envs.num_envs):
            self._taken_picture_list.append([])
        
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_exp_area = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picture_value = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_similarity = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picsim = torch.zeros(self.envs.num_envs, 1, device=self.device)
        
        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long)
        not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)
        stats_episodes = dict()  # dict of dicts that stores stats per episode
        raw_metrics_episodes = dict()

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR+"/"+date, exist_ok=True)

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):  

            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                pre_ac = torch.zeros(prev_actions.shape[0], 1, device=self.device, dtype=torch.long)
                for i in range(prev_actions.shape[0]):
                    pre_ac[i] = prev_actions[i]

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])
 
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            
            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            
            reward = []
            pic_val = []
            picture_value = []
            similarity = []
            pic_sim = []
            exp_area = [] # 探索済みのエリア()
            n_envs = self.envs.num_envs
            for n in range(n_envs):
                reward.append(rewards[n][0])
                pic_val.append(rewards[n][1])
                picture_value.append(0)
                similarity.append(0)
                pic_sim.append(0)
                exp_area.append(rewards[n][2]-rewards[n][3])
                
            for n in range(len(observations)):
                #TAKE_PICTUREが呼び出されたかを検証(TAKE_PICTUREを行っても変な写真の場合は保存しない)
                if pic_val[n] == -1:
                    continue
                # 2回連続でTAKE_PICTUREをした場合は保存しない
                # どうせ後から同じ写真は消すため、選別の際の計算量の削減のため
                if pre_ac[n].item() == actions[n].item():
                    continue
                
                image_emd = self._create_new_image_embedding(observations[n]["rgb"])
                self._taken_picture_list[n].append([pic_val[n], observations[n]["rgb"], image_emd])
                reward[n] += self.save_picture_reward
                
            reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
            exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.device).unsqueeze(1)
            picture_value = torch.tensor(picture_value, dtype=torch.float, device=self.device).unsqueeze(1)
            similarity = torch.tensor(similarity, dtype=torch.float, device=self.device).unsqueeze(1)
            
            current_episode_reward += reward
            current_episode_exp_area += exp_area
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []

            for n in range(n_envs):
                if len(stats_episodes) >= self.config.TEST_EPISODE_COUNT:
                    break
                """
                if (next_episodes[n].scene_id, next_episodes[n].episode_id) in stats_episodes:
                    envs_to_pause.append(n)
                """

                # episode ended
                if not_done_masks[n].item() == 0:
                    # use scene_id + episode_id as unique id for storing stats
                    _episode_id = current_episodes[n].episode_id
                    while (current_episodes[n].scene_id, _episode_id) in stats_episodes:
                        _episode_id = str(int(_episode_id) + 1)

                    # 写真の選別
                    self._taken_picture_list[n], picture_value[n] = self._select_pictures(self._taken_picture_list[n])
                    #self._taken_picture_list[n], picture_value[n] = self._select_random_pictures(self._taken_picture_list[n])
                    results_image = self._create_results_image(self._taken_picture_list[n])
                    
                    # Ground-Truth descriptionと生成文との類似度の計算 
                    description = self.description_df[self.description_df["scene_id"]==current_episodes[n].scene_id[-15:-4]]["description"].item()
                    #pred_description = self.create_description(self._taken_picture_list[n])
                    
                    pred_description = ""
                    if results_image is not None:
                        pred_description = self.create_description_from_results_image(results_image)
                     
                    similarity[n] = self.calculate_similarity(pred_description, description)
                    pic_sim[n] = self._calculate_pic_sim(self._taken_picture_list[n])
                    reward[n] += similarity[n]*10
                    current_episode_reward += similarity[n]*10
                    
                    # average of picture value par 1 picture
                    current_episode_picture_value[n] += picture_value[n]
                    current_episode_similarity[n] += similarity[n]
                    current_episode_picsim[n] += pic_sim[n]
                    
                    # save description
                    out_path = os.path.join("log/" + date + "/eval2/description.txt")
                    with open(out_path, 'a') as f:
                        # print関数でファイルに出力する
                        print(str(current_episodes[n].scene_id[-15:-4]) + "_" + str(_episode_id), file=f)
                        print(description, file=f)
                        print(pred_description,file=f)
                        print(similarity[i].item(),file=f)
                                    
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[n].item()
                    episode_stats["exp_area"] = current_episode_exp_area[n].item()
                    episode_stats["picture_value"] = current_episode_picture_value[n].item()
                    episode_stats["similarity"] = current_episode_similarity[n].item()
                    episode_stats["pic_sim"] = current_episode_picsim[n].item()
                    
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[n] = 0
                    current_episode_exp_area[n] = 0
                    current_episode_picture_value[n] = 0
                    current_episode_similarity[n] = 0
                    current_episode_picsim[n] = 0

                    stats_episodes[
                        (
                            current_episodes[n].scene_id,
                            _episode_id,
                        )
                    ] = episode_stats
                    
                    raw_metrics_episodes[
                        current_episodes[n].scene_id + '.' + 
                        _episode_id
                    ] = infos[n]["raw_metrics"]

                    if len(self.config.VIDEO_OPTION) > 0:
                        if len(rgb_frames[n]) == 0:
                            frame = observations_to_image(observations[n], infos[n], actions[n].cpu().numpy())
                            rgb_frames[n].append(frame)
                        picture = rgb_frames[n][-1]
                        for j in range(20):
                           rgb_frames[n].append(picture) 
                        metrics=self._extract_scalars_from_info(infos[n])
                        name_sim = similarity[n].item()
                        
                        name_sim = str(len(stats_episodes)) + "-" + str(name_sim)[:4] + "-" + str(episode_stats["exp_area"])[:4]
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR+"/"+date,
                            images=rgb_frames[n],
                            episode_id=_episode_id,
                            metrics=metrics,
                            name_ci=name_sim,
                        )
        
                        # Save taken picture                        
                        for j in range(len(self._taken_picture_list[n])):
                            value = self._taken_picture_list[n][j][0]
                            picture_name = f"episoede={_episode_id}-{len(stats_episodes)}-{j}-{value}"
                            dir_name = "./taken_picture/" + date 
                            os.makedirs(dir_name, exist_ok=True)
                        
                            picture = Image.fromarray(np.uint8(self._taken_picture_list[n][j][1]))
                            file_path = dir_name + "/" + picture_name + ".png"
                            picture.save(file_path)
                        
                        if results_image is not None:
                            results_image.save(f"/gs/fs/tga-aklab/matsumoto/Main/taken_picture/{date}/episoede={_episode_id}-{len(stats_episodes)}.png")    
                    
                    rgb_frames[n] = []
                    self._taken_picture_list[n] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[n], infos[n], actions[n].cpu().numpy())
                    rgb_frames[n].append(frame)
            
            """
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                current_episode_exp_area,
                current_episode_picture_value,
                current_episode_similarity,
                current_episode_picsim,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                current_episode_exp_area,
                current_episode_picture_value,
                current_episode_similarity,
                current_episode_picsim,
                prev_actions,
                batch,
                rgb_frames,
            )
            """

        num_episodes = len(stats_episodes)
        
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]
        
        eval_reward_logger.writeLine(str(step_id) + "," + str(aggregated_stats["reward"]))

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}

        logger.info("Similarity: " + str(metrics["similarity"]))
        logger.info("Pic_Sim: " + str(metrics["pic_sim"]))
        logger.info("Picture Value: " + str(metrics["picture_value"]))
        eval_metrics_logger.writeLine(str(step_id)+","+str(metrics["exp_area"])+","+str(metrics["similarity"])+","+str(metrics["picture_value"])+","+str(metrics["pic_sim"])+","+str(metrics["raw_metrics.agent_path_length"]))

        self.envs.close()
        
        
    def random_eval(self, log_manager: LogManager, date: str,) -> None:
        logger.info("RANDOM 2")
        self.log_manager = log_manager
        #ログ出力設定
        #time, reward
        eval_reward_logger = self.log_manager.createLogWriter("reward")
        #time, exp_area, distance. path_length
        eval_metrics_logger = self.log_manager.createLogWriter("metrics")

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.actor_critic = self.agent.actor_critic
        
        self._taken_picture_list = []
        
        # Sentence-BERTモデルの読み込み
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # lavisモデルの読み込み
        self.lavis_model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=self.device)
        self.bert_model.to(self.device)
        self.lavis_model.to(self.device)

        # Load the clip model
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        
        # LLaVA model
        load_4bit = True
        load_8bit = not load_4bit
        disable_torch_init()
        model_path = "liuhaotian/llava-v1.5-13b"
        self.llava_model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.llava_model, self.llava_image_processor, _ = load_pretrained_model(model_path, None, self.llava_model_name, load_8bit, load_4bit)

        # ファイルを読み込んで行ごとにリストに格納する
        with open('data/scene_datasets/mp3d/description.txt', 'r') as file:
            lines = file.readlines()

        # scene id と文章を抽出してデータフレームに変換する
        scene_ids = []
        descriptions = []
        for i in range(0, len(lines), 3):
            scene_ids.append(lines[i].strip())
            descriptions.append(lines[i+2].strip())

        self.description_df = pd.DataFrame({'scene_id': scene_ids, 'description': descriptions})
        
        for i in range(self.envs.num_envs):
            self._taken_picture_list.append([])
        
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_exp_area = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picture_value = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_similarity = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picsim = torch.zeros(self.envs.num_envs, 1, device=self.device)
        
        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long)
        not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)
        stats_episodes = dict()  # dict of dicts that stores stats per episode
        raw_metrics_episodes = dict()

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR+"/"+date, exist_ok=True)

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        episode_step = 0
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            episode_step += 1
            """
            if episode_step == 1:
                logger.info("####################")
                logger.info(f"stats_episode: {len(stats_episodes)}, count: {self.config.TEST_EPISODE_COUNT}, num_envs: {self.envs.num_envs}")
            """    
            current_episodes = self.envs.current_episodes()

            actions = []
            for _ in range(self.config.NUM_PROCESSES):
                a = random.randrange(4)
                actions.append([a])
                
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
                
            outputs = self.envs.step([a[0].item() for a in actions])
 
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            
            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            
            reward = []
            pic_val = []
            picture_value = []
            similarity = []
            pic_sim = []
            exp_area = [] # 探索済みのエリア()
            is_take_picture = []
            n_envs = self.envs.num_envs
            for n in range(n_envs):
                reward.append(rewards[n][0])
                pic_val.append(rewards[n][1])
                picture_value.append(0)
                similarity.append(0)
                pic_sim.append(0)
                exp_area.append(rewards[n][2]-rewards[n][3])
                is_take_picture.append(rewards[n][4])
                
            for n in range(len(observations)):
                #TAKE_PICTUREが呼び出されたかを検証
                if is_take_picture[n] == True:
                    image_emd = self._create_new_image_embedding(observations[n]["rgb"])
                    self._taken_picture_list[n].append([pic_val[n], observations[n]["rgb"], image_emd])
                    reward[n] += self.save_picture_reward
                        
            reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
            exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.device).unsqueeze(1)
            picture_value = torch.tensor(picture_value, dtype=torch.float, device=self.device).unsqueeze(1)
            similarity = torch.tensor(similarity, dtype=torch.float, device=self.device).unsqueeze(1)
            
            current_episode_reward += reward
            current_episode_exp_area += exp_area
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []

            for n in range(n_envs):
                if len(stats_episodes) >= self.config.TEST_EPISODE_COUNT:
                    break
                """
                if (next_episodes[n].scene_id, next_episodes[n].episode_id) in stats_episodes:
                    envs_to_pause.append(n)
                """

                # episode ended
                if not_done_masks[n].item() == 0:
                    # use scene_id + episode_id as unique id for storing stats
                    _episode_id = current_episodes[n].episode_id
                    while (current_episodes[n].scene_id, _episode_id) in stats_episodes:
                        _episode_id = str(int(_episode_id) + 1)

                    # 写真の選別
                    self._taken_picture_list[n], picture_value[n] = self._select_random_pictures(self._taken_picture_list[n])
                    results_image = self._create_results_image(self._taken_picture_list[n])
                        
                    # Ground-Truth descriptionと生成文との類似度の計算 
                    description = self.description_df[self.description_df["scene_id"]==current_episodes[n].scene_id[-15:-4]]["description"].item()
                    #pred_description = self.create_description(self._taken_picture_list[n])
                    
                    pred_description = ""
                    if results_image is not None:
                        pred_description = self.create_description_from_results_image(results_image)
                    
                    similarity[n] = self.calculate_similarity(pred_description, description)
                    pic_sim[n] = self._calculate_pic_sim(self._taken_picture_list[n])
                        
                    reward[n] += similarity[n]*10
                    current_episode_reward += similarity[n]*10
                        
                    current_episode_picture_value[n] += picture_value[n]
                    current_episode_similarity[n] += similarity[n]
                    current_episode_picsim[n] += pic_sim[n]

                    # save description
                    out_path = os.path.join("log/" + date + "/random2/description.txt")
                    with open(out_path, 'a') as f:
                        # print関数でファイルに出力する
                        print(str(current_episodes[n].scene_id[-15:-4]) + "_" + str(_episode_id), file=f)
                        print(description, file=f)
                        print(pred_description, file=f)
                        print(similarity[n].item(), file=f)
                                        
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[n].item()
                    episode_stats["exp_area"] = current_episode_exp_area[n].item()
                    episode_stats["picture_value"] = current_episode_picture_value[n].item()
                    episode_stats["similarity"] = current_episode_similarity[n].item()
                    episode_stats["pic_sim"] = current_episode_picsim[n].item()

                    episode_stats.update(
                        self._extract_scalars_from_info(infos[n])
                    )
                    current_episode_reward[n] = 0
                    current_episode_exp_area[n] = 0
                    current_episode_picture_value[n] = 0
                    current_episode_similarity[n] = 0
                    current_episode_picsim[n] = 0

                    stats_episodes[
                        (
                            current_episodes[n].scene_id,
                            _episode_id,
                        )
                    ] = episode_stats
                        
                    raw_metrics_episodes[
                        current_episodes[n].scene_id + '.' + 
                        _episode_id
                    ] = infos[n]["raw_metrics"]
                        

                    if len(self.config.VIDEO_OPTION) > 0:
                        if len(rgb_frames[n]) == 0:
                            frame = observations_to_image(observations[n], infos[n], actions[n].cpu().numpy())
                            rgb_frames[n].append(frame)
                        picture = rgb_frames[n][-1]
                        for j in range(20):
                            rgb_frames[n].append(picture) 
                        metrics=self._extract_scalars_from_info(infos[n])
                        name_sim = similarity[n].item()
                            
                        name_sim = str(len(stats_episodes)) + "-" + str(name_sim)[:4] + "-" + str(episode_stats["exp_area"])[:4]
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR+"/"+date,
                            images=rgb_frames[n],
                            episode_id=_episode_id,
                            metrics=metrics,
                            name_ci=name_sim,
                        )
            
                        # Save taken picture                        
                        for j in range(len(self._taken_picture_list[n])):
                            value = self._taken_picture_list[n][j][0]
                            picture_name = f"episoede={_episode_id}-{len(stats_episodes)}-{j}-{value}"
                            dir_name = "./taken_picture/" + date 
                            os.makedirs(dir_name, exist_ok=True)
                            
                            picture = Image.fromarray(np.uint8(self._taken_picture_list[n][j][1]))
                            file_path = dir_name + "/" + picture_name + ".png"
                            picture.save(file_path)
                            
                        if results_image is not None:
                            results_image.save(f"/gs/fs/tga-aklab/matsumoto/Main/taken_picture/{date}/episoede={_episode_id}-{len(stats_episodes)}.png")    
                    
                    rgb_frames[n] = []
                    self._taken_picture_list[n] = []
                    episode_step = 0

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[n], infos[n], actions[n].cpu().numpy())
                    rgb_frames[n].append(frame)

            """
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                current_episode_exp_area,
                current_episode_picture_value,
                current_episode_similarity,
                current_episode_picsim,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                current_episode_exp_area,
                current_episode_picture_value,
                current_episode_similarity,
                current_episode_picsim,
                prev_actions,
                batch,
                rgb_frames,
            )
            """

        num_episodes = len(stats_episodes)
        
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        
        step_id = -1
        
        eval_reward_logger.writeLine(str(step_id) + "," + str(aggregated_stats["reward"]))

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}

        logger.info("Similarity: " + str(metrics["similarity"]))
        logger.info("Pic_Sim: " + str(metrics["pic_sim"]))
        logger.info("Picture Value: " + str(metrics["picture_value"]))
        eval_metrics_logger.writeLine(str(step_id)+","+str(metrics["exp_area"])+","+str(metrics["similarity"])+","+str(metrics["picture_value"])+","+str(metrics["pic_sim"])+","+str(metrics["raw_metrics.agent_path_length"]))

        self.envs.close()
        
        
    def random_eval2(self, log_manager: LogManager, date: str,) -> None:
        #random action and select pictures by covered area
        logger.info("RANDOM 2")
        self.log_manager = log_manager
        #ログ出力設定
        #time, reward
        eval_reward_logger = self.log_manager.createLogWriter("reward")
        #time, exp_area, distance. path_length
        eval_metrics_logger = self.log_manager.createLogWriter("metrics")

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.actor_critic = self.agent.actor_critic
        
        self._taken_picture = []
        self._taken_picture_list = []
        
        # Sentence-BERTモデルの読み込み
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # lavisモデルの読み込み
        self.lavis_model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=self.device)
        self.bert_model.to(self.device)
        self.lavis_model.to(self.device)
        
        # ファイルを読み込んで行ごとにリストに格納する
        with open('data/scene_datasets/mp3d/description.txt', 'r') as file:
            lines = file.readlines()

        # scene id と文章を抽出してデータフレームに変換する
        scene_ids = []
        descriptions = []
        for i in range(0, len(lines), 3):
            scene_ids.append(lines[i].strip())
            descriptions.append(lines[i+2].strip())

        self.description_df = pd.DataFrame({'scene_id': scene_ids, 'description': descriptions})
        
        for i in range(self.envs.num_envs):
            self._taken_picture.append([])
            self._taken_picture_list.append([])
        
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_exp_area = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_similarity = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picsim = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_each_sim = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_sum_saliency = torch.zeros(self.envs.num_envs, 1, device=self.device)
        
        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long)
        not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)
        stats_episodes = dict()  # dict of dicts that stores stats per episode
        raw_metrics_episodes = dict()

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR+"/"+date, exist_ok=True)

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            actions = []
            for _ in range(self.config.NUM_PROCESSES):
                a = random.randrange(4)
                actions.append([a])
                
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
                
            pre_ac = torch.zeros(prev_actions.shape[0], 1, device=self.device, dtype=torch.long)
            for i in range(prev_actions.shape[0]):
                pre_ac[i] = prev_actions[i]

            prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])
 
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            
            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            
            reward = []
            saliency = []
            similarity = []
            pic_sim = []
            exp_area = [] # 探索済みのエリア()
            exp_area_pre = []
            fog_of_war_map = []
            top_down_map = [] 
            top_map = []
            each_sim = []
            sum_saliency = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                reward.append(rewards[i][0])
                saliency.append(rewards[i][1])
                similarity.append(0)
                pic_sim.append(0)
                exp_area.append(rewards[i][2]-rewards[i][3])
                exp_area_pre.append(rewards[i][3])
                fog_of_war_map.append(infos[i]["picture_range_map"]["fog_of_war_mask"])
                top_down_map.append(infos[i]["picture_range_map"]["map"])
                top_map.append(infos[i]["top_down_map"]["map"])
                each_sim.append(0)
                sum_saliency.append(0)
            
            for n in range(len(observations)):
                #TAKE_PICTUREが呼び出されたかを検証
                if saliency[n] == -1:
                    continue

                # 2回連続でTAKE_PICTUREをした場合は保存しない
                if pre_ac[n].item() == actions[n].item():
                    continue

                # 今回撮ったpicture(p_n)が保存してあるpicture(p_k)とかぶっているkを保存
                cover_list = [] 
                cover_per_list = []
                picture_range_map = self._create_picture_range_map(top_down_map[n], fog_of_war_map[n])
                
                picture_list = self._taken_picture_list[n]
                    
                description = self.description_df[self.description_df["scene_id"]==current_episodes[n].scene_id[-15:-4]]["description"].item()
                pred_description = self.create_description(picture_list)
                pre_sim = self.calculate_similarity(pred_description, description)

                caption = self._create_caption(observations[n]["rgb"])
                #new_emmbedding = self._create_new_description_embedding(caption)
                new_emmbedding = self._create_new_image_embedding(observations[n]["rgb"])

                # p_kのそれぞれのpicture_range_mapのリスト
                pre_fog_of_war_map = [sublist[1] for sublist in picture_list]

                # それぞれと閾値より被っているか計算
                idx = -1
                min_sal = saliency[n]

                for k in range(len(pre_fog_of_war_map)):
                    # 閾値よりも被っていたらcover_listにkを追加
                    check, per = self._check_percentage_of_fog(picture_range_map, pre_fog_of_war_map[k], threshold=0.1)
                    cover_per_list.append(per)
                    if check == True:
                        cover_list.append(k)

                    #saliencyの最小値の写真を探索(１つも被っていない時用)
                    if (idx == -1) and (min_sal == picture_list[idx][0]):
                        idx = -2
                    elif min_sal > picture_list[idx][0]:
                        idx = k
                        min_sal = picture_list[idx][0]

                # 今までの写真と多くは被っていない時
                if len(cover_list) == 0:
                    #範囲が多く被っていなくて、self._num_picture回未満写真を撮っていたらそのまま保存
                    if len(picture_list) != self._num_picture:
                        picture_list.append([saliency[n], picture_range_map, new_emmbedding, caption, steps])
                        self._taken_picture[n].append(observations[n]["rgb"])
                        self._taken_picture_list[n] = picture_list

                        # 説明文を生成し、similarityの差を計算する
                        pred_description = self.create_description(picture_list)
                        after_sim = self.calculate_similarity(pred_description, description)
                        each_sim[n] = (after_sim - pre_sim)*10
                        reward[n] += each_sim[n]
                        continue

                    #範囲が多く被っていなくて、self._num_picture回以上写真を撮っていたら
                    else:
                        # 今回の写真が保存している写真でsaliencyが最小のものと同じだった場合、写真の類似度が最大のものと交換
                        if idx == -2:
                            remove_index = self._cal_remove_index(picture_list, new_emmbedding)
                            # 入れ替えしない場合
                            if remove_index == len(picture_list):
                                continue
                            picture_list[remove_index] = [saliency[n], picture_range_map, new_emmbedding, caption]
                            self._taken_picture_list[n] = picture_list
                            self._taken_picture[n][remove_index] = observations[n]["rgb"]

                            # 説明文を生成し、similarityの差を計算する
                            pred_description = self.create_description(picture_list)
                            after_sim = self.calculate_similarity(pred_description, description)
                            each_sim[n] = (after_sim - pre_sim)*10
                            reward[n] += each_sim[n]
                            continue

                        # 今回の写真が保存してある写真の１つでもSaliencyが高かったらSaliencyが最小の保存写真と入れ替え
                        elif idx != -1:
                            sal_pre = picture_list[idx][0]
                            picture_list[idx] = [saliency[n], picture_range_map, new_emmbedding, caption]
                            self._taken_picture_list[n] = picture_list
                            self._taken_picture[n][idx] = observations[n]["rgb"]

                            # 説明文を生成し、similarityの差を計算する
                            pred_description = self.create_description(picture_list)
                            after_sim = self.calculate_similarity(pred_description, description)
                            each_sim[n] = (after_sim - pre_sim)*10
                            reward[n] += each_sim[n]
                            continue

                # 1つとでも多く被った場合
                else:
                    min_idx = -1
                    #min_sal_k = 1000
                    max_sal_k = 0.0
                    idx_sal = -1
                    # 多く被った写真のうち、saliencyが最小のものを計算
                    # 多く被った写真のうち、被っている割合が多い写真とsaliencyを比較
                    for k in range(len(cover_list)):
                        idx_k = cover_list[k]
                        """
                        if picture_list[idx_k][0] < min_sal_k:
                            min_sal_k = picture_list[idx_k][0]
                            min_idx = idx_k
                        """
                        if max_sal_k < cover_per_list[idx_k]:
                            max_sal_k = cover_per_list[idx_k]
                            min_idx = idx_k
                            idx_sal = picture_list[idx_k][0]

                    
                    # 被った割合分小さくなったCIでも保存写真の中の最小のCIより大きかったら交換
                    #if self._compareWithChangedSal(picture_range_map, pre_fog_of_war_map, cover_list, saliency[n], min_sal_k, min_idx) == True:
                    res = self._compareWithChangedSal(picture_range_map, pre_fog_of_war_map, cover_list, saliency[n], idx_sal, min_idx)
                    if res == 0:
                        picture_list[min_idx] = [saliency[n], picture_range_map, new_emmbedding, caption]
                        self._taken_picture_list[n] = picture_list
                        self._taken_picture[n][min_idx] = observations[n]["rgb"]   
                        
                        # 説明文を生成し、similarityの差を計算する
                        pred_description = self.create_description(picture_list)
                        after_sim = self.calculate_similarity(pred_description, description)
                        each_sim[n] = (after_sim - pre_sim)*10
                        reward[n] += each_sim[n]
                        continue
                    # 被った割合分小さくなったCIと保存写真の中の最小のCIが等しかったら写真の類似度が最大のものを削除
                    if res == 1:
                        remove_index = self._cal_remove_index(picture_list, new_emmbedding)
                        # 入れ替えしない場合
                        if remove_index == len(picture_list):
                            continue
                        picture_list[remove_index] = [saliency[n], picture_range_map, new_emmbedding, caption]
                        self._taken_picture_list[n] = picture_list
                        self._taken_picture[n][remove_index] = observations[n]["rgb"]

                        # 説明文を生成し、similarityの差を計算する
                        pred_description = self.create_description(picture_list)
                        after_sim = self.calculate_similarity(pred_description, description)
                        each_sim[n] = (after_sim - pre_sim)*10
                        reward[n] += each_sim[n]
                        continue
                
            reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
            exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.device).unsqueeze(1)
            similarity = torch.tensor(similarity, dtype=torch.float, device=current_episode_reward.device).unsqueeze(1)
            each_sim = torch.tensor(each_sim, dtype=torch.float, device=current_episode_reward.device).unsqueeze(1)

            current_episode_reward += reward
            current_episode_exp_area += exp_area
            current_episode_each_sim += each_sim
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []

            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    description = self.description_df[self.description_df["scene_id"]==current_episodes[i].scene_id[-15:-4]]["description"].item()
                    pred_description = self.create_description(self._taken_picture_list[i])
                    similarity[i] = self.calculate_similarity(pred_description, description)
                    current_episode_similarity[i] += similarity[i]
                    pic_sim[i] = self._calculate_pic_sim(self._taken_picture_list[i])
                    current_episode_picsim[i] += pic_sim[i]
                    
                    for j in range(len(self._taken_picture_list[i])):
                        sum_saliency[i] += self._taken_picture_list[i][j][0]
                    sum_saliency[i] /= len(self._taken_picture_list[i])
                    sum_saliency = torch.tensor(sum_saliency, dtype=torch.float, device=current_episode_reward.device).unsqueeze(1)
                    current_episode_sum_saliency[i] += sum_saliency[i][0].item()

                    # save description
                    out_path = os.path.join("log/" + date + "/eval/description.txt")
                    with open(out_path, 'a') as f:
                        # print関数でファイルに出力する
                        print(str(current_episodes[i].scene_id[-15:-4]) + "_" + str(current_episodes[i].episode_id), file=f)
                        print(description, file=f)
                        print(pred_description,file=f)
                        print(similarity[i].item(),file=f)
                                    
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["exp_area"] = current_episode_exp_area[i].item()
                    episode_stats["similarity"] = current_episode_similarity[i].item()
                    episode_stats["pic_sim"] = current_episode_picsim[i].item()
                    episode_stats["each_sim"] = current_episode_each_sim[i].item()
                    episode_stats["sum_saliency"] = current_episode_sum_saliency[i].item()
                    
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    current_episode_exp_area[i] = 0
                    current_episode_similarity[i] = 0
                    current_episode_picsim[i] = 0
                    current_episode_each_sim[i] = 0
                    current_episode_sum_saliency[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    
                    raw_metrics_episodes[
                        current_episodes[i].scene_id + '.' + 
                        current_episodes[i].episode_id
                    ] = infos[i]["raw_metrics"]

                    if len(self.config.VIDEO_OPTION) > 0:
                        if len(rgb_frames[i]) == 0:
                            frame = observations_to_image(observations[i], infos[i], actions[i].cpu().numpy())
                            rgb_frames[i].append(frame)
                        picture = rgb_frames[i][-1]
                        for j in range(50):
                           rgb_frames[i].append(picture) 
                        metrics=self._extract_scalars_from_info(infos[i])
                        name_sim = similarity[i].item()
                        
                        name_sim = str(len(stats_episodes)) + "-" + str(name_sim)[:4] + "-" + str(episode_stats["exp_area"])[:4]
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR+"/"+date,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            metrics=metrics,
                            name_ci=name_sim,
                        )
        
                        # Save taken picture                        
                        for j in range(len(self._taken_picture[i])):
                            picture_name = f"episode={current_episodes[i].episode_id}-{len(stats_episodes)}-{j}"
                            dir_name = "./taken_picture/" + date 
                            os.makedirs(dir_name, exist_ok=True)
                        
                            picture = Image.fromarray(np.uint8(self._taken_picture[i][j]))
                            file_path = dir_name + "/" + picture_name + ".png"
                            picture.save(file_path)

                            # Save picture range map
                            """
                            eval_range_logger = self.log_manager.createLogWriter(f"range_{current_episodes[i].episode_id}_{len(stats_episodes)}_{j}")
                            range_map = self._taken_picture_list[i][j][1]
                            for k in range(range_map.shape[0]):
                                for l in range(range_map.shape[1]):
                                    eval_range_logger.write(str(range_map[k][l]))
                                eval_range_logger.writeLine()
                            """
                            
                        rgb_frames[i] = []
                        
                    self._taken_picture[i] = []
                    self._taken_picture_list[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i], actions[i].cpu().numpy())
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                current_episode_exp_area,
                current_episode_similarity,
                current_episode_picsim,
                current_episode_each_sim,
                current_episode_sum_saliency,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                current_episode_exp_area,
                current_episode_similarity,
                current_episode_picsim,
                current_episode_each_sim,
                current_episode_sum_saliency,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]
        
        eval_reward_logger.writeLine(str(step_id) + "," + str(aggregated_stats["reward"]))

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}

        logger.info("Similarity: " + str(metrics["similarity"]))
        logger.info("Pic_Sim: " + str(metrics["pic_sim"]))
        logger.info("Sum Saliency: " + str(metrics["sum_saliency"]))
        eval_metrics_logger.writeLine(str(step_id)+","+str(metrics["exp_area"])+","+str(metrics["similarity"])+","+str(metrics["pic_sim"])+","+str(metrics["each_sim"])+","+str(metrics["sum_saliency"])+","+str(metrics["raw_metrics.agent_path_length"]))

        self.envs.close()
