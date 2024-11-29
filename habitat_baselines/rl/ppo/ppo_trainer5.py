#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 探索のみの報酬(既存研究)

import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw
import pandas as pd
import random
import csv
from collections import Counter
import warnings

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.nn.functional as F
from functools import lru_cache
from scipy.optimize import linear_sum_assignment

import clip
from sentence_transformers import SentenceTransformer, util
from lavis.models import load_model_and_preprocess

from habitat import Config
from habitat.core.logging import logger
from habitat.utils.visualizations.utils import observations_to_image, explored_to_image, create_each_image
from habitat_baselines.common.base_trainer import BaseRLTrainerOracle
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.rollout_storage import RolloutStorageOracle, RolloutStorageReconstruction
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo import PPOOracle, ProposedPolicyOracle
from log_manager import LogManager
from log_writer import LogWriter
from habitat.utils.visualizations import fog_of_war, maps

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer
#from transformers import AutoProcessor, LlavaNextForConditionalGeneration 

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score as Meteor_score
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from curiosity_model.curiosity import (
    ForwardDynamics,
    Phi,
    RunningMeanStd,
)

import h5py
from reconstruction_model.reconstruction import (
    FeatureReconstructionModule,
    FeatureNetwork,
    PoseEncoder,
    compute_reconstruction_rewards
)
from reconstruction_model.common import (
    process_image,
    flatten_two,
    unflatten_two,
    random_range,
    process_odometer,
)

# 必要なNLTKのリソースをダウンロード
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('stopwords')

warnings.simplefilter('ignore')


# SBERT + MLPによる回帰モデルの定義
class SBERTRegressionModel(nn.Module):
    def __init__(self, sbert_model, hidden_size1=512, hidden_size2=256, hidden_size3=128):
        super(SBERTRegressionModel, self).__init__()
        self.sbert = sbert_model
        
        # 6つの埋め込みベクトルを結合するため、入力サイズは6倍に
        embedding_size = self.sbert.get_sentence_embedding_dimension() * 6
        
        # 多層MLPの構造を定義
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size1),  # 結合ベクトルから第1隠れ層
            nn.ReLU(),  # 活性化関数
            nn.Linear(hidden_size1, hidden_size2),  # 第2隠れ層
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),  # 第3隠れ層
            nn.ReLU(),
            nn.Linear(hidden_size3, 1)  # 隠れ層からスカラー値出力
        )
        
    def forward(self, sentence_list):
        # 文章をSBERTで埋め込みベクトルに変換
        embeddings = [self.sbert.encode(sentence, convert_to_tensor=True).unsqueeze(0) for sentence in sentence_list]

        # 6つのベクトルを結合 (次元を6倍にする)
        combined_features = torch.cat(embeddings, dim=1)
        
        # MLPを通してスカラー値を予測
        output = self.mlp(combined_features)
        return output


@baseline_registry.register_trainer(name="oracle5")
class PPOTrainerO5(BaseRLTrainerOracle):
    # reward is added only from area reward
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    torch.cuda.empty_cache()
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

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.icm_fd = None # curiosity用
        self.decoder = None # reconstruction用
        self.recon_sensor = ["delta", "pose_estimation_mask", "pose_refs"]

        self.rec_step = 0
        self.NREF = 100

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
                
    def _collect_rollout_step(
        self, 
        rollouts, 
        current_episode_reward, 
        current_episode_area_rate, 
        running_episode_stats,
        step
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

        if self.icm_fd is not None:
            icm_feats = recurrent_hidden_states[0]
            self.all_icm_feats.append(icm_feats)
            self.all_icm_acts.append(actions)

        #logger.info("actions=" + str(actions))
        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        
        reward = []
        area_rate = [] # 探索済みエリアの割合
        n_envs = self.envs.num_envs
        for n in range(n_envs):
            if self.icm_fd is not None:
                reward.append(0)
            else:
                reward.append(rewards[n][0])
            area_rate.append(rewards[n][1])

        if (self.icm_fd is not None) and (step >= 1):
            phi_st = self.all_icm_feats[-2]
            phi_st1 = self.all_icm_feats[-1]
            self.action_onehot.zero_() 
            act = self.all_icm_acts[-2]
            self.action_onehot.scatter_(1, act, 1)
            with torch.no_grad():
                phi_st1_hat = self.icm_fd(phi_st, self.action_onehot)
            
            reward_curiosity = (
                F.mse_loss(phi_st1_hat, phi_st1, reduction="none")
                .sum(dim=1)
                .unsqueeze(1)
                .detach()
            )
            reward_curiosity *= 0.001
            reward = reward_curiosity.squeeze(-1).tolist()

        if (self.decoder is not None):
            # Processing environment inputs
            obs_im = self.get_obs(observations)  # (num_processes, 3, 84, 84)
            obs_delta = []
            for i in range(len(observations)):
                obs_delta.append(observations[i]["delta"])

            obs_delta = np.array(obs_delta)
            obs_odometer = process_odometer(obs_delta).to(self.device)  # (num_processes, 4)
                
            with torch.no_grad():
                obs_feat = self.feature_network(obs_im)
                # Compute similarity scores with all other clusters
                obs_feat = torch.matmul(obs_feat, self.cluster_centroids_t)  # (N, nclusters)

            # Always set masks to 1 (since this loop happens within one episode)
            rec_masks = torch.FloatTensor([[1.0] for _ in range(n_envs)]).to(self.device)

            # Accumulate odometer readings to give relative pose from the starting point
            obs_odometer = self.rollouts_recon.obs_odometer[self.rec_step] * rec_masks + obs_odometer

            # Update rollouts_recon
            self.rollouts_recon.insert(obs_feat, obs_odometer)
            
            
            if (self.rec_step+1) % self.rec_reward_interval == 0 or self.rec_step == 0:
                #logger.info(f"Reconstruction: {self.rec_step} Steps")
                rec_rewards = compute_reconstruction_rewards(
                    self.rollouts_recon.obs_feats[: (self.rec_step + 1)],
                    self.rollouts_recon.obs_odometer[: (self.rec_step + 1), :, :3],
                    self.rollouts_recon.tgt_feats,
                    self.rollouts_recon.tgt_poses,
                    self.cluster_centroids_t,
                    self.decoder,
                    self.pose_encoder,
                ).detach()  # (N, nRef)

                rec_rewards = rec_rewards * self.tgt_masks.squeeze(2)  # (N, nRef)
                rec_rewards = rec_rewards.sum(dim=1).unsqueeze(1)[0]  # / (tgt_masks.sum(dim=1) + 1e-8)
                #logger.info(f"rec_rewards = {rec_rewards}, prec_rec_rewards = {self.prev_rec_rewards}")

                reward = (rec_rewards - self.prev_rec_rewards) / 100
                #logger.info(f"Reconstruction Reward = {reward}")
                self.prev_rec_rewards = rec_rewards
            
        current_episodes = self.envs.current_episodes()
        
        reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
        area_rate = torch.tensor(area_rate, dtype=torch.float, device=self.device).unsqueeze(1)
        #logger.info(f"Reward={reward}, Area_Rate={area_rate}")

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        for n in range(len(observations)):
            if masks[n].item() == 0.0:
                #logger.info("###### Episode End !!! #####") 
                self.rec_step = 0
                
        current_episode_reward += reward
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        current_episode_area_rate += area_rate
        running_episode_stats["area_rate"] += (1 - masks) * current_episode_area_rate
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
        current_episode_area_rate *= masks
        
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

    def get_obs(self, obs):
        obs_im = []
        for i in range(len(obs)):
            obs_im.append(process_image(obs[i]["rgb"]))

        obs_im = torch.from_numpy(np.array(obs_im)).float().to(self.device)
        obs_im = obs_im.permute(0, 3, 1, 2)

        return obs_im

    def train(self, log_manager, date) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        logger.info("########### PPO5 ##############")

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
        #time, fd_loss
        curiosity_logger = self.log_manager.createLogWriter("curiosity")
        
        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        ppo_cfg = self.config.RL.PPO
            
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        checkpoint_path = "/gs/fs/tga-aklab/matsumoto/Main/cpt/24-10-23 02-03-03/ckpt.10.pth"
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        self.agent.load_state_dict(ckpt_dict["state_dict"])
        logger.info(f"########## LOAD CKPT at {checkpoint_path} ###########")

        if self.config.TASK_CONFIG.AREA_REWARD == "curiosity":
            #num = self.actor_critic.net.num_recurrent_layers
            #self.icm_fd = ForwardDynamics(self.envs.action_spaces[0].n, num)
            self.icm_fd = ForwardDynamics(self.envs.action_spaces[0].n)
            self.icm_fd.to(self.device)
            self.icm_fd.eval()
            self.icm_optimizer = optim.Adam(self.icm_fd.parameters(), lr=1e-4)

        if self.config.TASK_CONFIG.AREA_REWARD == "reconstruction":
            self.rec_reward_interval = 5
            self.rec_reward_interval = 4
            self.prev_rec_rewards = 0.0
            # =================== Load clusters =================
            clusters_path = "/gs/fs/tga-aklab/matsumoto/Main/reconstruction_model/imagenet_clusters/mp3d/clusters_00050_data.h5"
            clusters_h5 = h5py.File(clusters_path, "r")
            cluster_centroids = torch.Tensor(np.array(clusters_h5["cluster_centroids"])).to(self.device)
            self.cluster_centroids_t = cluster_centroids.t()
            self.nclusters = cluster_centroids.shape[0]
            clusters_h5.close()
            
            # =================== Create models ====================
            n_transformer_layers = 2
            self.decoder = FeatureReconstructionModule(self.nclusters, self.nclusters, nlayers=n_transformer_layers)
            self.feature_network = FeatureNetwork()
            self.pose_encoder = PoseEncoder()
            
            # =================== Load models ====================
            reconstruction_load_path = "/gs/fs/tga-aklab/matsumoto/Main/reconstruction_model/pretrained_reconstruction/ckpt.pth"
            decoder_state, pose_encoder_state = torch.load(reconstruction_load_path)[:2]
            # Remove DataParallel related strings
            new_decoder_state, new_pose_encoder_state = {}, {}
            for k, v in decoder_state.items():
                new_decoder_state[k.replace("module.", "")] = v
            for k, v in pose_encoder_state.items():
                new_pose_encoder_state[k.replace("module.", "")] = v
            self.decoder.load_state_dict(new_decoder_state)
            self.pose_encoder.load_state_dict(new_pose_encoder_state)
            #self.decoder = nn.DataParallel(self.decoder, dim=1)
            #self.pose_encoder = nn.DataParallel(self.pose_encoder, dim=0)

            self.decoder.to(self.device)
            self.feature_network.to(self.device)
            self.pose_encoder.to(self.device)

            # decoder, feature_network, pose_encoder are frozen during policy training
            self.decoder.eval()
            self.feature_network.eval()
            self.pose_encoder.eval()

            odometer_shape = (4,)
            num_pose_refs = 100
            self.rollouts_recon = RolloutStorageReconstruction(
                ppo_cfg.num_steps,
                self.envs.num_envs,
                (self.nclusters,),
                odometer_shape,
                num_pose_refs,
            )
            self.rollouts_recon.to(self.device)

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
            if sensor not in self.recon_sensor:
                rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        #batch = None
        #observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_area_rate = torch.zeros(self.envs.num_envs, 1, device=self.device)
        
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            area_rate=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
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

        for update in range(self.config.NUM_UPDATES):
            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                    update, self.config.NUM_UPDATES
                )
            if self.icm_fd is not None:
                self.icm_fd.eval()
            # Reset ICM data buffer
            self.all_icm_feats = []
            self.all_icm_acts = []
            self.action_onehot = torch.zeros(self.config.NUM_PROCESSES, self.envs.action_spaces[0].n).to(self.device)

            if self.decoder is not None:
                # Processing environment inputs
                #logger.info(f"##### observations = {observations}")
                #logger.info(f"obs.shape = {observations.shape}")
                obs_im = self.get_obs(observations)
                obs_odometer = []
                obs_delta = []
                for i in range(len(observations)):
                    x = observations[i]["delta"]
                    #logger.info(f"obs_delta={x}")
                    #logger.info(f"obs_delta={x.shape}")
                    obs_delta.append(x)
                    
                obs_delta = np.array(obs_delta)
                obs_odometer = process_odometer(obs_delta)  # (num_processes, 4)
                #obs_odometer = torch.from_numpy(np.array(obs_odometer)).to(self.device)

                # ============== Target poses and corresponding images ================
                # NOTE - these are constant throughout the episode.
                # (num_processes * num_pose_refs, 3) --- (y, x, t)
                obs_pose_regress = []
                obs_pose_refs = []
                pose_estimation_mask = []
                for i in range(len(observations)):
                    obs_pose_regress.append(observations[i]["pose_refs"][1])
                    obs_pose_refs.append(observations[i]["pose_refs"][0])
                    pose_estimation_mask.append(observations[i]["pose_estimation_mask"])
                obs_pose_regress = np.array(obs_pose_regress)
                obs_pose_refs = np.array(obs_pose_refs)
                pose_estimation_mask = torch.from_numpy(np.array(pose_estimation_mask))
                tgt_poses = process_odometer(flatten_two(obs_pose_regress))[:, :3]
                tgt_poses = unflatten_two(tgt_poses, self.config.NUM_PROCESSES, self.NREF)  # (N, nRef, 3)
                self.tgt_masks = pose_estimation_mask.unsqueeze(2).to(self.device)  # (N, nRef, 1)
                tgt_ims = process_image(flatten_two(obs_pose_refs))  # (N*nRef, C, H, W)
                # Initialize the memory of rollouts for reconstruction
                self.rollouts_recon.reset()
        
                with torch.no_grad():
                    #logger.info(f"obs_im={obs_im.shape}")
                    obs_feat = self.feature_network(obs_im)  # (N, 2048)
                    #logger.info(f"tgt_ims={tgt_ims.shape}")
                    tgt_ims = tgt_ims.float().to(self.device).permute(0, 3, 1, 2)
                    tgt_feat = self.feature_network(tgt_ims)  # (N*nRef, 2048)
                    # Compute similarity scores with all other clusters
                    obs_feat = torch.matmul(obs_feat, self.cluster_centroids_t)  # (N, nclusters)
                    tgt_feat = torch.matmul(
                        tgt_feat, self.cluster_centroids_t
                    )  # (N*nRef, nclusters)
                tgt_feat = unflatten_two(tgt_feat, self.config.NUM_PROCESSES, self.NREF)  # (N, nRef, nclusters)
                self.rollouts_recon.obs_feats[0].copy_(obs_feat)
                self.rollouts_recon.obs_odometer[0].copy_(obs_odometer)
                self.rollouts_recon.tgt_poses.copy_(tgt_poses)
                self.rollouts_recon.tgt_feats.copy_(tgt_feat)
                self.rollouts_recon.tgt_masks.copy_(self.tgt_masks)
                

            for step in range(ppo_cfg.num_steps):
                #logger.info(f"STEP: {step}")
                if step == 0:
                    self.rec_step = 0

                (
                    delta_pth_time,
                    delta_env_time,
                    delta_steps,
                ) = self._collect_rollout_step(
                    rollouts, 
                    current_episode_reward, 
                    current_episode_area_rate, 
                    running_episode_stats,
                    step
                )
                pth_time += delta_pth_time
                env_time += delta_env_time
                count_steps += delta_steps

                self.rec_step += 1

            (
                delta_pth_time,
                value_loss,
                action_loss,
                dist_entropy,
            ) = self._update_agent(ppo_cfg, rollouts)
            pth_time += delta_pth_time

            # ============ Update the ICM dynamics model using past data ===============
            if self.icm_fd is not None:
                self.icm_fd.train()
                self.action_onehot = torch.zeros(self.config.NUM_PROCESSES, self.envs.action_spaces[0].n).to(self.device)
                avg_fd_loss = 0
                avg_fd_loss_count = 0
                icm_update_count = 0
                for t in random_range(0, ppo_cfg.num_steps - 1):
                    phi_st = self.all_icm_feats[t]  # (N, 512)
                    phi_st1 = self.all_icm_feats[t + 1]  # (N, 512)
                    self.action_onehot.zero_()
                    at = self.all_icm_acts[t].long()  # (N, 1)
                    self.action_onehot.scatter_(1, at, 1)
                    # Forward pass
                    phi_st1_hat = self.icm_fd(phi_st, self.action_onehot)
                    fd_loss = F.mse_loss(phi_st1_hat, phi_st1)
                    # Backward pass
                    self.icm_optimizer.zero_grad()
                    fd_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.icm_fd.parameters(), 0.5)
                    # Update step
                    self.icm_optimizer.step()
                    avg_fd_loss += fd_loss.item()
                    avg_fd_loss_count += phi_st1_hat.shape[0]
                avg_fd_loss /= avg_fd_loss_count
                curiosity_logger.writeLine(str(count_steps)+","+str(avg_fd_loss))
                self.icm_fd.eval()
                
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
                logger.info("Area Rate: " + str(metrics["area_rate"]))
                logger.info("REWARD: " + str(deltas["reward"] / deltas["count"]))
                if self.icm_fd is not None:
                    logger.info("Curiosity Loss: " + str(avg_fd_loss))
                metrics_logger.writeLine(str(count_steps)+","+str(metrics["area_rate"])+","+str(metrics["raw_metrics.agent_path_length"]))
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

                # curiosity_model
                if self.icm_fd is not None:
                    curiosity_save_path = (f"/gs/fs/tga-aklab/matsumoto/Main/curiosity_model/curiosity_ckpt/{date}/curiosity_ckpt.{count_checkpoints}.pth")
                    os.makedirs(f"/gs/fs/tga-aklab/matsumoto/Main/curiosity_model/curiosity_ckpt/{date}", exist_ok=True)
                    icm_fd_state = self.icm_fd.state_dict()
                    torch.save(icm_fd_state, curiosity_save_path)

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
            emd = self._create_new_image_embedding(sorted_picture_list[i][1])
            is_save = self._decide_save(emd, results)

            if is_save == True:
                results.append(sorted_picture_list[i])
                res_val += sorted_picture_list[i][0]
            i += 1

        res_val /= len(results)
        return results, res_val

    def _select_pictures2(self, taken_picture_list):
        n = len(taken_picture_list)
        select_list = []
        noselect_list = [i for i in range(n)]

        # 類似度行列を作成
        #similarity_matrix = self.create_similarity_matrix(taken_picture_list)

        while len(select_list) < self._num_picture:
            max_value = float('-inf')
            best_picture = None

            for i in noselect_list:
                # select_listに入っている写真との平均類似度を計算
                #select_sim= self.calculate_avg_similarity(i, select_list, similarity_matrix)
                select_sim= self.calculate_avg_similarity(i, select_list, taken_picture_list)

                # select_listに入っていない写真との平均類似度を計算
                #noselect_sim = self.calculate_avg_similarity(i, noselect_list, similarity_matrix)
                noselect_sim = self.calculate_avg_similarity(i, noselect_list, taken_picture_list)

                # (noselect_sim - select_sim) を計算
                value = noselect_sim - select_sim

                # 最大のvalueを持つ写真を探す
                if value > max_value:
                    max_value = value
                    best_picture = i

            # 選ばれた写真をselect_listに追加し、noselect_listから削除
            if best_picture is not None:
                select_list.append(best_picture)
                noselect_list.remove(best_picture)

        # 選ばれたインデックスを元の写真に変換して返す
        selected_pictures = [taken_picture_list[i] for i in select_list]
        return selected_pictures, 0.0

    def create_similarity_matrix(self, picture_list):
        n = len(picture_list)
        similarity_matrix = np.zeros((n, n))

        # 各ペア間の類似度を計算し、行列に保存
        for i in range(n):
            for j in range(i+1, n):
                emd_i = self._create_new_image_embedding(picture_list[i][1])
                emd_j = self._create_new_image_embedding(picture_list[j][1])
                sim = util.pytorch_cos_sim(emd_i, emd_j).item()

                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim

        return similarity_matrix

    # select_listに入っている写真と入っていない写真との平均類似度の計算
    def calculate_avg_similarity(self, index, indices, picture_list, similarity_matrix=None):
        if not indices:
            return 0
        #return np.mean([similarity_matrix[index, i] for i in indices])
        similarity_sum = sum(self.cal_similarity(index, i, picture_list[index][i], picture_list[i][1]) for i in indices)
        return similarity_sum / len(indices)

    # 類似度を逐次計算し、キャッシュする関数
    #@lru_cache(maxsize=1000)  # キャッシュサイズを1000ペアに制限
    def cal_similarity(self, index, i, picture1, picture2):
        emd_index = self._create_new_image_embedding(picture1)
        emd_i = self._create_new_image_embedding(picture2)
        sim = util.pytorch_cos_sim(emd_index, emd_i).item()
        return sim

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
            check_emb = self._create_new_image_embedding(results[i][1])

            sim = util.pytorch_cos_sim(emd, check_emb).item()
            if sim >= self._select_threthould:
                return False
        return True


    def _create_results_image(self, picture_list, infos):
        images = []
        x_list = []
        y_list = []
    
        if len(picture_list) == 0:
            return None

        for i in range(self._num_picture):
            idx = i%len(picture_list)
            explored_map, fog_of_war_map = self.get_explored_picture(picture_list[idx][4])
            range_x = np.where(np.any(explored_map == maps.MAP_INVALID_POINT, axis=1))[0]
            range_y = np.where(np.any(explored_map == maps.MAP_INVALID_POINT, axis=0))[0]

            _ind_x_min = range_x[0]
            _ind_x_max = range_x[-1]
            _ind_y_min = range_y[0]
            _ind_y_max = range_y[-1]
            _grid_delta = 5
            clip_parameter = [_ind_x_min, _ind_x_max, _ind_y_min, _ind_y_max, _grid_delta]

            frame = create_each_image(picture_list[idx][1], explored_map, fog_of_war_map, infos, clip_parameter)
            
            images.append(frame)
            x_list.append(picture_list[idx][2])
            y_list.append(picture_list[idx][3])
            image = Image.fromarray(frame)
            image.save(f"/gs/fs/tga-aklab/matsumoto/Main/test_{i}.png")

        height, width, _ = images[0].shape
        result_width = width * 2
        result_height = height * 5
        result_image = Image.new("RGB", (result_width, result_height))

        for i, image in enumerate(images):
            x_offset = (i // 5) * width
            y_offset = (i % 5) * height
            image = Image.fromarray(image)
            result_image.paste(image, (x_offset, y_offset))
        
        draw = ImageDraw.Draw(result_image)
        for x in range(width, result_width, width):
            draw.line([(x, 0), (x, result_height)], fill="black", width=7)
        for y in range(height, result_height, height):
            draw.line([(0, y), (result_width, y)], fill="black", width=7)

        return result_image, x_list, y_list

    def _create_results_image2(self, picture_list, infos):
        images = []
    
        if len(picture_list) == 0:
            return None

        height, width, _ = picture_list[0][1].shape
        result_width = width * 5
        result_height = height * 2
        result_image = Image.new("RGB", (result_width, result_height))

        for i in range(self._num_picture):
            idx = i%len(picture_list)
            images.append(picture_list[idx][1])

        for i, image in enumerate(images):
            x_offset = (i % 5) * width
            y_offset = (i // 5) * height
            image = Image.fromarray(image)
            result_image.paste(image, (x_offset, y_offset))
        
        draw = ImageDraw.Draw(result_image)
        for x in range(width, result_width, width):
            draw.line([(x, 0), (x, result_height)], fill="black", width=7)
        for y in range(height, result_height, height):
            draw.line([(0, y), (result_width, y)], fill="black", width=7)

        return result_image, images


    def create_description_from_results_image(self, results_image, x_list, y_list, input_change=False):
        input_text = "<Instructions>\n"\
                    "You are an excellent property writer.\n"\
                    "The input image consists of 10 pictures of a building, 5 vertically and 2 horizontally, within a single picture.\n"\
                    "In addition, each picture is separated by a black line.\n"\
                    "\n"\
                    "From each picture, understand the details of this building's environment and summarize them in the form of a detailed description of this building's environment, paying attention to the <Notes>.\n"\
                    "In doing so, please also consider the location of each picture as indicated by <Location Information>.\n"\
                    "\n\n"\
                    "<Notes>\n"\
                    "・Note that adjacent pictures are not close in location.\n"\
                    "・When describing the environment, do not mention whether it was taken from that picture or the black line separating each picture.\n"\
                    "・Write a description of approximately 100 words in summary form without mentioning each individual picture."
        #logger.info("############## input_text ###############")
        #logger.info(input_text)
        if input_change == True:
            logger.info("############## Input Change ################")
            input_text = "You are an excellent property writer. This picture consists of 10 pictures arranged in one picture, 5 horizontally and 2 vertically on one building. In addition, a black line separates the pictures from each other. From each picture, you should understand the details of this building's environment and describe this building's environment in detail in the form of a summary of these pictures. At this point, do not describe each picture one at a time, but rather in a summarized form. Also note that each picture was taken in a separate location, so successive pictures are not positionally close. Additionally, do not mention which picture you are quoting from or the black line separating each picture."
        response = self.generate_response(results_image, input_text)
        response = response[4:-4]
        return response

    def create_description_sometimes(self, image_list, results_image):
        input_text1 = "# Instructions\n"\
                    "You are an excellent property writer.\n"\
                    "Please understand the details of the environment of this building from the pictures you have been given and explain what it is like to be in this environment as a person in this environment."

        image_descriptions = []
        for image in image_list:
        
            response = self.generate_response(image, input_text1)
            response = response[4:-4]
            
            #response = self._create_caption(image)
            image_descriptions.append(response)

        input_text2 = "# Instructions\n"\
                    "You are an excellent property writer.\n"\
                    "# Each_Description is a description of the building in the pictures you have entered. Please summarize these and write a description of the entire environment as if you were a person in this environment.\n"\
                    "\n"\
                    "# Each_Description\n"
        input_text3 = "# Notes\n"\
                    "・Please summarize # Each_Description and write a description of the entire environment as if you were a person in this environment.\n"\
                    "・Please write approximately 100 words.\n"\
                    "・Please note that the sentences in # Each_Description are not necessarily close in distance."

        for description in image_descriptions:
            each_description = "・" + description + "\n"
            input_text2 += each_description

        input_text = input_text2 + "\n" + input_text3

        #logger.info("############## input_text ###############")
        #logger.info(input_text)
        
        response = self.generate_response(results_image, input_text)
        response = response[4:-4]

        #logger.info("########### output_text ################")
        #logger.info(response)
        return response, image_descriptions

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


    # BLEUスコアの計算
    def calculate_bleu(self, reference, candidate):
        reference = [reference.split()]
        candidate = candidate.split()
        smoothie = SmoothingFunction().method4
        return sentence_bleu(reference, candidate, smoothing_function=smoothie)

    # ROUGEスコアの計算
    def calculate_rouge(self, reference, candidate):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores

    # METEORスコアの計算
    def calculate_meteor(self, reference, candidate):
        reference = reference.split()
        candidate = candidate.split()
        return Meteor_score([reference], candidate)

    def get_wordnet_pos(self, word):
        """WordNetの品詞タグを取得"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatize_and_filter(self, text):
        """ステミング処理を行い、ストップワードを除去"""
        tokens = word_tokenize(text.lower())
        filtered_tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) 
                        for token in tokens if token.isalpha() 
                        and token not in stopwords.words('english')]
        return filtered_tokens

    # 単語が一致しているかどうかを判断する
    def is_matching(self, word1, word2):
        # ステミングされた単語が一致するか
        lemma1 = self.lemmatizer.lemmatize(word1)
        lemma2 = self.lemmatizer.lemmatize(word2)
        
        if lemma1 == lemma2:
            return True
        
        # 類義語が存在するか
        synsets1 = wordnet.synsets(lemma1)
        synsets2 = wordnet.synsets(lemma2)
        
        if synsets1 and synsets2:
            # synsetsをリーマティックに比較
            return any(s1.wup_similarity(s2) >= 0.9 for s1 in synsets1 for s2 in synsets2)
        
        return False

    def calculate_pas(self, s_lemmatized, description):
        gt_lemmatized = self.lemmatize_and_filter(description)
        precision, recall, total_weight, total_gt_weight = 0.0, 0.0, 0.0, 0.0
        matched_words = set()

        for j, s_word in enumerate(s_lemmatized):
            weight = 1.0 / (j + 1)  # 単語の位置に応じた重み付け
            total_weight += weight
                
            if any(self.is_matching(s_word, gt_word) for gt_word in gt_lemmatized):
                precision += weight
                matched_words.add(s_word)

        for j, gt_word in enumerate(gt_lemmatized):
            weight = 1.0 / (j + 1)
            total_gt_weight += weight
            if any(self.is_matching(gt_word, s_word) for s_word in matched_words):
                recall += weight

        precision /= total_weight if total_weight > 0 else 1
        recall /= total_gt_weight if total_gt_weight > 0 else 1

        if precision + recall == 0:
            f_score = 0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)

        return f_score

    def get_explored_picture(self, infos):
        explored_map = infos["map"].copy()
        fog_of_war_map = infos["fog_of_war_mask"]

        explored_map[(fog_of_war_map == 1) & (explored_map == maps.MAP_VALID_POINT)] = maps.MAP_INVALID_POINT
        explored_map[(fog_of_war_map == 0) & ((explored_map == maps.MAP_VALID_POINT) | (explored_map == maps.MAP_INVALID_POINT))] = maps.MAP_BORDER_INDICATOR

        
        """
        y, x = explored_map.shape

        for i in range(y):
            for j in range(x):
                if fog_of_war_map[i][j] == 1:
                    if explored_map[i][j] == maps.MAP_VALID_POINT:
                        explored_map[i][j] = maps.MAP_INVALID_POINT
                else:
                    if explored_map[i][j] in [maps.MAP_VALID_POINT, maps.MAP_INVALID_POINT]:
                        explored_map[i][j] = maps.MAP_BORDER_INDICATOR 
        """

        return explored_map, fog_of_war_map

    def extract_after_inst(self, S: str) -> str:
        # '[/INST]' が見つかった場所を特定する
        inst_index = S.find('[/INST]')
        
        # '[/INST]' が見つかった場合、その後の文章を返す
        if inst_index != -1:
            return S[inst_index + len('[/INST]'):]
        
        # 見つからなかった場合は空の文字列を返す
        return ""

    def create_description_multi(self, image_list, results_image):
        input_text1 = "# Instructions\n"\
                    "You are an excellent property writer.\n"\
                    "Please understand the details of the environment of this building from the pictures you have been given and explain what it is like to be in this environment as a person in this environment."

        image_descriptions = []
        response = self.generate_multi_response(image_list, input_text1)
        for i in range(len(image_list)):
            output = self.extract_after_inst(response[i].strip().replace("\n\n", " "))
            image_descriptions.append(output)
            #logger.info(f"desc {i}")
            #logger.info(output)

        input_text2 = "# Instructions\n"\
                    "You are an excellent property writer.\n"\
                    "# Each_Description is a description of the building in the pictures you have entered. Please summarize these and write a description of the entire environment as if you were a person in this environment.\n"\
                    "\n"\
                    "# Each_Description\n"
        input_text3 = "# Notes\n"\
                    "・Please summarize # Each_Description and write a description of the entire environment as if you were a person in this environment.\n"\
                    "・Please write approximately 100 words.\n"\
                    "・Please note that the sentences in # Each_Description are not necessarily close in distance."

        for description in image_descriptions:
            each_description = "・" + description + "\n"
            input_text2 += each_description

        input_text = input_text2 + "\n" + input_text3

        response = self.generate_multi_response([results_image], input_text)
        response = self.extract_after_inst(response[0].strip().replace("\n\n", " "))
        #logger.info(f"response: ")
        #logger.info(response)

        return response

    def generate_multi_response(self, image_list, input_text):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": input_text},
                    ],
            },
        ]

        prompt = self.llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
        prompts = [prompt for _ in range(len(image_list))]
        
        inputs = self.llava_processor(images=image_list, text=prompts, padding=True, return_tensors="pt").to(self.llava_model.device)

        generate_ids = self.llava_model.generate(**inputs, max_new_tokens=2048)
        outputs = self.llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        #logger.info(f"device = {self.llava_model.device}")
        #logger.info(f"outputs_size = {len(outputs)}")
        #logger.info(f"image_list_size = {len(image_list)}")

        return outputs 

    def get_txt2dict(self, txt_path):
        data_dict = {}
        # ファイルを読み込み、行ごとにリストに格納
        with open(txt_path, 'r') as file:
            lines = file.readlines()

        # 奇数行目をキー、偶数行目を値として辞書に格納
        for i in range(0, len(lines), 2):
            scene_name = lines[i].strip()  # 奇数行目: scene名
            scene_data = lines[i + 1].strip().split(',')  # 偶数行目: コンマ区切りのデータ
            data_dict[scene_name] = scene_data
            
        return data_dict

        # sentence内の名詞のリストを取得
    def extract_nouns(self, sentence):
        tokens = word_tokenize(sentence)
        nouns = []

        for word in tokens:
            if word.isalpha() and word not in stopwords.words('english'):
                # 原型に変換
                lemma = self.lemmatizer.lemmatize(word)
                pos = self.get_wordnet_pos(word)
                if pos == wordnet.NOUN and self.is_valid_noun(lemma):  # 名詞に限定
                    if lemma not in nouns:
                        nouns.append(lemma)

        return nouns

    # 名詞であるかどうかを判断するための追加のフィルター
    def is_valid_noun(self, word):
        """単語が名詞であるかを確認する追加のフィルター"""
        # 除外したい名詞のリスト
        excluded_nouns = {"inside", "lead", "use", "look", "like", "lot", "clean", "middle", "walk", "gray"}

        if word in excluded_nouns:
            return False
        synsets = wordnet.synsets(word)
        return any(s.pos() == 'n' for s in synsets)

    def calculate_clip_score(self, image, text):
        # 画像の読み込み
        image = Image.fromarray(image)
        
        # 画像の前処理
        inputs = self.preprocess(image).unsqueeze(0).to(self.device)

        # テキストのトークン化とエンコード
        text_tokens = clip.tokenize([text]).to(self.device)

        # 画像とテキストの特徴ベクトルを計算
        with torch.no_grad():
            image_features = self.clip_model.encode_image(inputs)
            text_features = self.clip_model.encode_text(text_tokens)

        # 類似度（cosine similarity）を計算
        clip_score = torch.cosine_similarity(image_features, text_features)
        
        return clip_score.item()

    def calculate_iou(self, word1, word2):
        # word1, word2 の同義語集合を取得し、それらのJaccard係数を用いてIoU計算を行います。
        synsets1 = set(wordnet.synsets(word1))
        synsets2 = set(wordnet.synsets(word2))
        intersection = synsets1.intersection(synsets2)
        union = synsets1.union(synsets2)
        if not union:  # 同義語が全くない場合は0を返す
            return 0.0
        return len(intersection) / len(union)

    # IoU行列の生成
    def generate_iou_matrix(self, object_list1, object_list2):
        iou_matrix = np.zeros((len(object_list1), len(object_list2)))
        for i, obj1 in enumerate(object_list1):
            for j, obj2 in enumerate(object_list2):
                iou_matrix[i, j] = self.calculate_iou(obj1, obj2)
        return iou_matrix

    # Jonker-Volgenantアルゴリズム（線形代入問題の解法）で最適な対応を見つける
    def find_optimal_assignment(self, object_list1, object_list2):
        iou_matrix = self.generate_iou_matrix(object_list1, object_list2)
        # コスト行列はIoUの負の値を使う（最小コストの最大化）
        cost_matrix = -iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        optimal_iou = iou_matrix[row_ind, col_ind].sum() / min(len(object_list1), len(object_list2))
        return optimal_iou, list(zip(row_ind, col_ind))

    def calculate_ed(self, object_list, pred_sentence, area, picture_list, image_descriptions):
        #CLIP Scoreの平均の計算
        clip_score_list = []
        for i in range(len(picture_list)):
            pic_list = picture_list[i]
            clip_score_list.append(self.calculate_clip_score(pic_list[1], image_descriptions[i]))

        pred_object_list = self.extract_nouns(pred_sentence)

        if len(pred_object_list) == 0:
            logger.info(f"len(pred_object_list)=0")
            return 0.0
            
        optimal_iou, assignment = self.find_optimal_assignment(object_list, pred_object_list)

        ed_score = clip_score * optimal_iou * area
        #logger.info(f"ED-S: {ed_score}, CLIP Score: {clip_score}, IoU: {optimal_iou}, Area: {area}")

        return ed_score

    def _eval_checkpoint(self, checkpoint_path: str, log_manager: LogManager, date: str, checkpoint_index: int = 0) -> None:
        logger.info("############### EAVL5 ##################")
        self.log_manager = log_manager
        #ログ出力設定
        #time, reward
        eval_reward_logger = self.log_manager.createLogWriter("reward")
        #time, area_rate, simlarity, each_sim, path_length
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
        for i in range(self.envs.num_envs):
            self._taken_picture_list.append([])

        # Load the clip model
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        self._select_threthould = 0.9
        #self._select_threthould = 0.8

        # LLaVA model

        load_4bit = True
        load_8bit = not load_4bit
        disable_torch_init()
        model_path = "liuhaotian/llava-v1.5-13b"
        self.llava_model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.llava_model, self.llava_image_processor, _ = load_pretrained_model(model_path, None, self.llava_model_name, load_8bit, load_4bit)

        """
        # LLaVA NEXT model
        #model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
        model_path = "llava-hf/llava-v1.6-vicuna-13b-hf"
        # Load the model in half-precision
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.llava_processor = AutoProcessor.from_pretrained(model_path)
        """

        # Sentence-BERTモデルの読み込み
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        """
        # lavisモデルの読み込み
        self.lavis_model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=self.device)
        self.bert_model.to(self.device)
        self.lavis_model.to(self.device)
        """

        model_path = f"/gs/fs/tga-aklab/matsumoto/Main/SentenceBert_FineTuning/model_checkpoints_all/model_epoch_10000.pth"
        # SBERTモデルのロード
        sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.eval_model = SBERTRegressionModel(sbert_model).to(self.device)
        self.eval_model.load_state_dict(torch.load(model_path))
        self.eval_model.eval() 
        logger.info(f"Eval Model loaded from {model_path}")

        # 単語のステミング処理
        self.lemmatizer = WordNetLemmatizer()
        
        # ファイルを読み込んで行ごとにリストに格納する
        with open('data/scene_datasets/mp3d/Environment_Descriptions.txt', 'r') as file:
            lines = [line.strip() for line in file]
            
            """
            lines = file.readlines()

            # scene id と文章を抽出してデータフレームに変換する
            self.description_dict = {}
            for i in range(0, len(lines), 7):
                descriptions = []
                scene_id = lines[i].strip()
                desc_ind = i+2
                for j in range(5):
                    descriptions.append(lines[desc_ind+j].strip())
                self.description_dict[scene_id] = descriptions
            """
        
        # scene id と文章を辞書に変換
        self.description_dict = {
            lines[i]: lines[i+2:i+7]
            for i in range(0, len(lines), 7)
        }
        
        self.scene_object_dict = self.get_txt2dict("/gs/fs/tga-aklab/matsumoto/Main/scene_object_list.txt")

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_area_rate = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_similarity = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picsim = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_bleu_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_1_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_2_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_L_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_meteor_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_pas_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_hes_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_ed_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        
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
        self.step = 0
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):  

            if (self.step+1) % 100 == 0:
                logger.info(f"step={self.step+1}")
            self.step += 1

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
            similarity = []
            area_rate = [] # 探索済みのエリア()
            bleu_score = []
            rouge_1_score = []
            rouge_2_score = []
            rouge_L_score = []
            meteor_score = []
            pas_score = []
            hes_score = []
            ed_score = []

            n_envs = self.envs.num_envs
            for n in range(n_envs):
                reward.append(rewards[n][0])
                pic_val.append(rewards[n][2])
                similarity.append(0)
                area_rate.append(rewards[n][1])
                bleu_score.append(0)
                rouge_1_score.append(0)
                rouge_2_score.append(0)
                rouge_L_score.append(0)
                meteor_score.append(0)
                pas_score.append(0)
                hes_score.append(0)
                ed_score.append(0)
                
                self._taken_picture_list[n].append([rewards[n][2], observations[n]["rgb"], rewards[n][6], rewards[n][7], infos[n]["explored_map"]])
                    
            reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
            area_rate = torch.tensor(area_rate, dtype=torch.float, device=self.device).unsqueeze(1)
            similarity = torch.tensor(similarity, dtype=torch.float, device=self.device).unsqueeze(1)
            bleu_score = torch.tensor(bleu_score, dtype=torch.float, device=self.device).unsqueeze(1)
            rouge_1_score = torch.tensor(rouge_1_score, dtype=torch.float, device=self.device).unsqueeze(1)
            rouge_2_score = torch.tensor(rouge_2_score, dtype=torch.float, device=self.device).unsqueeze(1)
            rouge_L_score = torch.tensor(rouge_L_score, dtype=torch.float, device=self.device).unsqueeze(1)
            meteor_score = torch.tensor(meteor_score, dtype=torch.float, device=self.device).unsqueeze(1)
            pas_score = torch.tensor(pas_score, dtype=torch.float, device=self.device).unsqueeze(1)
            hes_score = torch.tensor(hes_score, dtype=torch.float, device=self.device).unsqueeze(1)
            ed_score = torch.tensor(ed_score, dtype=torch.float, device=self.device).unsqueeze(1)
            
            current_episode_reward += reward
            current_episode_area_rate += area_rate
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []

            for n in range(n_envs):
                if len(stats_episodes) >= self.config.TEST_EPISODE_COUNT:
                    break

                # episode ended
                if not_done_masks[n].item() == 0:
                    # use scene_id + episode_id as unique id for storing stats
                    _episode_id = current_episodes[n].episode_id
                    while (current_episodes[n].scene_id, _episode_id) in stats_episodes:
                        _episode_id = str(int(_episode_id) + 1)

                    # 写真の選別
                    self._taken_picture_list[n], _ = self._select_pictures(self._taken_picture_list[n])
                    results_image, image_list = self._create_results_image2(self._taken_picture_list[n], infos[n]["explored_map"])
                    
                    # Ground-Truth descriptionと生成文との類似度の計算 
                    similarity_list = []
                    bleu_list = []
                    rouge_1_list = []
                    rouge_2_list = []
                    rouge_L_list = []
                    meteor_list = []
                    pas_list = []

                    #pred_description = self.create_description(self._taken_picture_list[n])
                    pred_description = ""
                    if results_image is not None:
                        #pred_description = self.create_description_from_results_image(results_image, positions_x, positions_y)
                        pred_description, image_descriptions = self.create_description_sometimes(image_list, results_image)
                        #pred_description = self.create_description_multi(image_list, results_image)

                    s_lemmatized = self.lemmatize_and_filter(pred_description)                        
                    description_list = self.description_dict[current_episodes[n].scene_id[-15:-4]]
                    hes_sentence_list = [pred_description]
                        
                    for i in range(5):
                        description = description_list[i]
                        hes_sentence_list.append(description)

                        sim_score = self.calculate_similarity(pred_description, description)
                        bleu = self.calculate_bleu(description, pred_description)
                        rouge_scores = self.calculate_rouge(description, pred_description)
                        rouge_1 = rouge_scores['rouge1'].fmeasure
                        rouge_2 = rouge_scores['rouge2'].fmeasure
                        rouge_L = rouge_scores['rougeL'].fmeasure
                        meteor = self.calculate_meteor(description, pred_description)
                        pas = self.calculate_pas(s_lemmatized, description)

                        similarity_list.append(sim_score)
                        bleu_list.append(bleu)
                        rouge_1_list.append(rouge_1)
                        rouge_2_list.append(rouge_2)
                        rouge_L_list.append(rouge_L)
                        meteor_list.append(meteor)
                        pas_list.append(pas)
                        
                    similarity[n] = sum(similarity_list) / len(similarity_list)
                    
                    bleu_score[n] = sum(bleu_list) / len(bleu_list)
                    rouge_1_score[n] = sum(rouge_1_list) / len(rouge_1_list)
                    rouge_2_score[n] = sum(rouge_2_list) / len(rouge_2_list)
                    rouge_L_score[n] = sum(rouge_L_list) / len(rouge_L_list)
                    meteor_score[n] = sum(meteor_list) / len(meteor_list)
                    pas_score[n] = sum(pas_list) / len(pas_list)    
                    hes_score[n] = self.eval_model(hes_sentence_list).item()

                    # ED-Sの計算
                    scene_name = current_episodes[n].scene_id[-15:-4]
                    area = current_episode_exp_area[n].item()
                    object_list = self.scene_object_dict[scene_name]
                    ed_score[n] = self.calculate_ed(object_list, pred_description, area, self._taken_picture_list[n], image_descriptions)

                    current_episode_similarity[n] += similarity[n]
                    current_episode_bleu_score[n] += bleu_score[n]
                    current_episode_rouge_1_score[n] += rouge_1_score[n]
                    current_episode_rouge_2_score[n] += rouge_2_score[n]
                    current_episode_rouge_L_score[n] += rouge_L_score[n]
                    current_episode_meteor_score[n] += meteor_score[n]
                    current_episode_pas_score[n] += pas_score[n]
                    current_episode_hes_score[n] += hes_score[n]
                    current_episode_ed_score[n] += ed_score[n]
                    
                    # save description
                    out_path = os.path.join("log/" + date + "/eval5/description.txt")
                    with open(out_path, 'a') as f:
                        # print関数でファイルに出力する
                        print(str(current_episodes[n].scene_id[-15:-4]) + "_" + str(_episode_id), file=f)
                        print(description, file=f)
                        print(pred_description,file=f)
                        print(similarity[n].item(),file=f)
                        print(hes_score[n].item(), file=f)
                        
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[n].item()
                    episode_stats["area_rate"] = current_episode_area_rate[n].item()
                    episode_stats["similarity"] = current_episode_similarity[n].item()
                    episode_stats["bleu_score"] = current_episode_bleu_score[n].item()
                    episode_stats["rouge_1_score"] = current_episode_rouge_1_score[n].item()
                    episode_stats["rouge_2_score"] = current_episode_rouge_2_score[n].item()
                    episode_stats["rouge_L_score"] = current_episode_rouge_L_score[n].item()
                    episode_stats["meteor_score"] = current_episode_meteor_score[n].item()
                    episode_stats["pas_score"] = current_episode_pas_score[n].item()
                    episode_stats["hes_score"] = current_episode_hes_score[n].item()
                    episode_stats["ed_score"] = current_episode_ed_score[n].item()
                    
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[n])
                    )
                    current_episode_reward[n] = 0
                    current_episode_area_rate[n] = 0
                    current_episode_similarity[n] = 0
                    current_episode_bleu_score[n] = 0
                    current_episode_rouge_1_score[n] = 0
                    current_episode_rouge_2_score[n] = 0
                    current_episode_rouge_L_score[n] = 0
                    current_episode_meteor_score[n] = 0
                    current_episode_pas_score[n] = 0
                    current_episode_hes_score[n] = 0
                    current_episode_ed_score[n] = 0

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

                        name_hes = str(len(stats_episodes)) + "-" + str(hes_score[n].item())[:4] + "-" + str(episode_stats["area_rate"])[:4]
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR+"/"+date,
                            images=rgb_frames[n],
                            episode_id=_episode_id,
                            metrics=metrics,
                            name_ci=name_hes,
                        )
        
                        # Save taken picture                        
                        for j in range(len(self._taken_picture_list[n])):
                            value = self._taken_picture_list[n][j][0]
                            picture_name = f"episode={_episode_id}-{len(stats_episodes)}-{j}-{value}"
                            dir_name = "./taken_picture/" + date 
                            os.makedirs(dir_name, exist_ok=True)
                        
                            picture = Image.fromarray(np.uint8(self._taken_picture_list[n][j][1]))
                            file_path = dir_name + "/" + picture_name + ".png"
                            picture.save(file_path)
                        
                        if results_image is not None:
                            results_image.save(f"/gs/fs/tga-aklab/matsumoto/Main/taken_picture/{date}/episode={_episode_id}-{len(stats_episodes)}.png")    
                    
                    rgb_frames[n] = []
                    self._taken_picture_list[n] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[n], infos[n], actions[n].cpu().numpy())
                    rgb_frames[n].append(frame)

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

        logger.info("HES Score: " + str(metrics["hes_score"]))
        logger.info("Similarity: " + str(metrics["similarity"]))
        logger.info("PAS Score: " + str(metrics["pas_score"]))
        logger.info("ED Score: " + str(metrics["ed_score"]))
        logger.info("BLUE: " + str(metrics["bleu_score"]) + ", ROUGE-1: " + str(metrics["rouge_1_score"]) + ", ROUGE-2: " + str(metrics["rouge_2_score"]) + ", ROUGE-L: " + str(metrics["rouge_L_score"]) + ", METEOR: " + str(metrics["meteor_score"]))
        eval_metrics_logger.writeLine(str(step_id)+","+str(metrics["exp_area"])+","+str(metrics["similarity"])+","+str(metrics["picture_value"])+","+str(metrics["pic_sim"])+","+str(metrics["bleu_score"])+","+str(metrics["rouge_1_score"])+","+str(metrics["rouge_2_score"])+","+str(metrics["rouge_L_score"])+","+str(metrics["meteor_score"])+","+str(metrics["pas_score"])+","+str(metrics["hes_score"])+","+str(metrics["ed_score"])+","+str(metrics["raw_metrics.agent_path_length"]))

        self.envs.close()
