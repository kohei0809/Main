#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 探索のみの報酬

import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw
import pandas as pd
import random
import csv
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim.lr_scheduler import LambdaLR
from functools import lru_cache

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
from habitat_baselines.common.rollout_storage import RolloutStorageOracle
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

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score as Meteor_score
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# 必要なNLTKのリソースをダウンロード
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


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

    def _calculate_pic_sim(self, picture_list):
        if len(picture_list) <= 1:
            return 0.0
            
        sim_list = [[-10 for _ in range(len(picture_list))] for _ in range(len(picture_list))]

        for i in range(len(picture_list)):
            emd = self._create_new_image_embedding(picture_list[i][1])
            for j in range(i, len(picture_list)):
                if i == j:
                    sim_list[i][j] = 0.0
                    continue
                emd2 = self._create_new_image_embedding(picture_list[j][1])
                sim_list[i][j] = util.pytorch_cos_sim(emd, emd2).item()
                sim_list[j][i] = sim_list[i][j]
                
        total_sim = np.sum(sim_list)
        total_sim /= (len(picture_list)*(len(picture_list)-1))
        return total_sim

    def _load_subgoal_list(self, current_episodes, n, semantic_scene_df):
        self.subgoal_list[n] = []
        self.subgoal_num_list[n] = []
        scene_name = current_episodes[n].scene_id[-15:-4]
        
        """
        file_path = f"/gs/fs/tga-aklab/matsumoto/Main/data/scene_datasets/mp3d/{scene_name}/pickup_object.csv"
        with open(file_path, mode="r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                self.subgoal_list[n].append(int(row[2]))
                self.subgoal_num_list[n].append(0)
        """
        category_file_path = f"/gs/fs/tga-aklab/matsumoto/Main/data/scene_datasets/mp3d/{scene_name}/category.txt"
        with open(category_file_path, mode='r') as f:
            content = f.read()
            object_names = content.split(',')
            for _, row in semantic_scene_df.iterrows():
                if row['object_name'] in object_names:
                    if row['id'] not in self.subgoal_list[n]:
                        self.subgoal_list[n].append(row['id'])
                        self.subgoal_num_list[n].append(0)

        #logger.info("########## load_subgoal: " + scene_name + ", " + str(len(self.subgoal_list[n])) + "###########")

    def _calculate_subgoal_reward(self, semantic_obs, n):
        H, W = semantic_obs.shape
        threshold = H*W*0.05
        r = 0.0
        subgoals = np.array(self.subgoal_list[n])
        if subgoals.size == 0:
            return 0.0

        flat_obs = semantic_obs.ravel()
        obs_counter = Counter(flat_obs)

        subgoal_counts = np.zeros(len(subgoals), dtype=int)
        for idx, subgoal in enumerate(subgoals):
            subgoal_counts[idx] = obs_counter[subgoal]

        for i in range(len(subgoal_counts)):
            if subgoal_counts[i] > threshold:
                if self.subgoal_num_list[n][i] < self.threshold_subgoal:
                    r += self.each_subgoal_reward
                    self.subgoal_num_list[n][i] += 1

        """
        subgoal_num = []
        for i in range(len(self.subgoal_list[n])):
            subgoal_num.append(0)
        if len(subgoal_num) == 0:
            return r

        for i in range(H):
            for j in range(W):
                obs = semantic_obs[i][j]
                if obs in self.subgoal_list[n]:
                    subgoal_num[self.subgoal_list[n].index(obs)] += 1

        for i in range(len(self.subgoal_list[n])):
            if subgoal_num[i] > threshold:
                if self.subgoal_num_list[n][i] < self.threshold_subgoal:
                    r += self.each_subgoal_reward
                    self.subgoal_num_list[n][i] += 1
        """

        return r

                
    def _collect_rollout_step(
        self, 
        rollouts, 
        current_episode_reward, 
        current_episode_exp_area, 
        current_episode_picture_value, 
        current_episode_similarity, 
        current_episode_picsim, 
        current_episode_subgoal_reward, 
        current_episode_bleu_score,
        current_episode_rouge_1_score,
        current_episode_rouge_2_score,
        current_episode_rouge_L_score,
        current_episode_meteor_score,
        current_episode_pas_score,
        current_episode_hes_score,
        running_episode_stats
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
        semantic_obs = []
        subgoal_reward = []
        bleu_score = []
        rouge_1_score = []
        rouge_2_score = []
        rouge_L_score = []
        meteor_score = []
        pas_score = []
        hes_score = []
        n_envs = self.envs.num_envs
        for n in range(n_envs):
            reward.append(rewards[n][0])
            pic_val.append(rewards[n][1])
            picture_value.append(0)
            similarity.append(0)
            pic_sim.append(0)
            exp_area.append(rewards[n][2])
            semantic_obs.append(observations[n]["semantic"])
            subgoal_reward.append(0)
            bleu_score.append(0)
            rouge_1_score.append(0)
            rouge_2_score.append(0)
            rouge_L_score.append(0)
            meteor_score.append(0)
            pas_score.append(0)
            hes_score.append(0)
            
        current_episodes = self.envs.current_episodes()
        for n in range(len(observations)):
            if len(self._taken_picture_list[n]) == 0:
                self._load_subgoal_list(current_episodes, n, rewards[n][5])
            
            self._taken_picture_list[n].append([pic_val[n], observations[n]["rgb"], rewards[n][6], rewards[n][7], infos[n]["explored_map"]])
                
            subgoal_reward[n] = self._calculate_subgoal_reward(semantic_obs[n], n)
            #logger.info("########## subgoal_reward: " + str(subgoal_reward[n]))
            reward[n] += subgoal_reward[n]

        reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
        exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.device).unsqueeze(1)
        picture_value = torch.tensor(picture_value, dtype=torch.float, device=self.device).unsqueeze(1)
        similarity = torch.tensor(similarity, dtype=torch.float, device=self.device).unsqueeze(1)
        pic_sim = torch.tensor(pic_sim, dtype=torch.float, device=self.device).unsqueeze(1)
        subgoal_reward = torch.tensor(subgoal_reward, dtype=torch.float, device=self.device).unsqueeze(1)
        bleu_score = torch.tensor(bleu_score, dtype=torch.float, device=self.device).unsqueeze(1)
        rouge_1_score = torch.tensor(rouge_1_score, dtype=torch.float, device=self.device).unsqueeze(1)
        rouge_2_score = torch.tensor(rouge_2_score, dtype=torch.float, device=self.device).unsqueeze(1)
        rouge_L_score = torch.tensor(rouge_L_score, dtype=torch.float, device=self.device).unsqueeze(1)
        meteor_score = torch.tensor(meteor_score, dtype=torch.float, device=self.device).unsqueeze(1)
        pas_score = torch.tensor(pas_score, dtype=torch.float, device=self.device).unsqueeze(1)
        hes_score = torch.tensor(hes_score, dtype=torch.float, device=self.device).unsqueeze(1)
            
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
                #results_image, positions_x, positions_y = self._create_results_image(self._taken_picture_list[n], infos[n]["explored_map"])
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
                    pred_description = self.create_description_sometimes(image_list, results_image)
                
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
                #pic_sim[n] = self._calculate_pic_sim(self._taken_picture_list[n])                

                bleu_score[n] = sum(bleu_list) / len(bleu_list)
                rouge_1_score[n] = sum(rouge_1_list) / len(rouge_1_list)
                rouge_2_score[n] = sum(rouge_2_list) / len(rouge_2_list)
                rouge_L_score[n] = sum(rouge_L_list) / len(rouge_L_list)
                meteor_score[n] = sum(meteor_list) / len(meteor_list)
                pas_score[n] = sum(pas_list) / len(pas_list)    
                hes_score[n] = self.eval_model(hes_sentence_list).item()

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
        current_episode_subgoal_reward += subgoal_reward
        running_episode_stats["subgoal_reward"] += (1 - masks) * current_episode_subgoal_reward
        current_episode_bleu_score += bleu_score
        running_episode_stats["bleu_score"] += (1 - masks) * current_episode_bleu_score
        current_episode_rouge_1_score += rouge_1_score
        running_episode_stats["rouge_1_score"] += (1 - masks) * current_episode_rouge_1_score
        current_episode_rouge_2_score += rouge_2_score
        running_episode_stats["rouge_2_score"] += (1 - masks) * current_episode_rouge_2_score
        current_episode_rouge_L_score += rouge_L_score
        running_episode_stats["rouge_L_score"] += (1 - masks) * current_episode_rouge_L_score
        current_episode_meteor_score += meteor_score
        running_episode_stats["meteor_score"] += (1 - masks) * current_episode_meteor_score
        current_episode_pas_score += pas_score
        running_episode_stats["pas_score"] += (1 - masks) * current_episode_pas_score
        current_episode_hes_score += hes_score
        running_episode_stats["hes_score"] += (1 - masks) * current_episode_hes_score
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
        current_episode_subgoal_reward *= masks
        current_episode_bleu_score *= masks
        current_episode_rouge_1_score *= masks
        current_episode_rouge_2_score *= masks
        current_episode_rouge_L_score *= masks
        current_episode_meteor_score *= masks
        current_episode_pas_score *= masks
        current_episode_hes_score *= masks
        
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
        logger.info("########### PPO4 ##############")

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
        self.subgoal_list = []
        self.subgoal_num_list = []
        for _ in range(self.envs.num_envs):
            self._taken_picture_list.append([])
            self.subgoal_list.append([])
            self.subgoal_num_list.append([])

        self.each_subgoal_reward = 0.05
        self.threshold_subgoal = 5
            
        ppo_cfg = self.config.RL.PPO
            
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        ################
        #heckpoint_path = "/gs/fs/tga-aklab/matsumoto/Main/cpt/24-06-28 04-00-29/ckpt.44.pth"
        #checkpoint_path = "/gs/fs/tga-aklab/matsumoto/Main/cpt/24-06-28 04-51-59/ckpt.48.pth"
        #ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        #self.agent.load_state_dict(ckpt_dict["state_dict"])
        #logger.info(f"########## LOAD CKPT at {checkpoint_path} ###########")
        #############

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
        current_episode_subgoal_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_bleu_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_1_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_2_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_L_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_meteor_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_pas_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_hes_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            exp_area=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            picture_value=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            similarity=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            pic_sim=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            subgoal_reward=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            bleu_score=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            rouge_1_score=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            rouge_2_score=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            rouge_L_score=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            meteor_score=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            pas_score=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
            hes_score=torch.zeros(self.envs.num_envs, 1, device=current_episode_reward.device),
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
                    rollouts, 
                    current_episode_reward, 
                    current_episode_exp_area, 
                    current_episode_picture_value, 
                    current_episode_similarity, 
                    current_episode_picsim, 
                    current_episode_subgoal_reward, 
                    current_episode_bleu_score,
                    current_episode_rouge_1_score,
                    current_episode_rouge_2_score,
                    current_episode_rouge_L_score,
                    current_episode_meteor_score,
                    current_episode_pas_score,
                    current_episode_hes_score,
                    running_episode_stats
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
                logger.info("HES Score: " + str(metrics["hes_score"]))
                logger.info("PAS Score: " + str(metrics["pas_score"]))
                logger.info("Similarity: " + str(metrics["similarity"]))
                #logger.info("Picture Value: " + str(metrics["picture_value"]))
                #logger.info("Pic_Sim: " + str(metrics["pic_sim"]))
                logger.info("SubGoal_Reward: " + str(metrics["subgoal_reward"]))
                logger.info("BLUE: " + str(metrics["bleu_score"]) + ", ROUGE-1: " + str(metrics["rouge_1_score"]) + ", ROUGE-2: " + str(metrics["rouge_2_score"]) + ", ROUGE-L: " + str(metrics["rouge_L_score"]) + ", METEOR: " + str(metrics["meteor_score"]))
                logger.info("REWARD: " + str(deltas["reward"] / deltas["count"]))
                metrics_logger.writeLine(str(count_steps)+","+str(metrics["exp_area"])+","+str(metrics["similarity"])+","+str(metrics["picture_value"])+","+str(metrics["pic_sim"])+","+str(metrics["subgoal_reward"])+","+str(metrics["bleu_score"])+","+str(metrics["rouge_1_score"])+","+str(metrics["rouge_2_score"])+","+str(metrics["rouge_L_score"])+","+str(metrics["meteor_score"])+","+str(metrics["pas_score"])+","+str(metrics["hes_score"])+","+str(metrics["raw_metrics.agent_path_length"]))
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
            """
            response = self.generate_response(image, input_text1)
            response = response[4:-4]
            """
            response = self._create_caption(image)
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
        explored_map = infos["map"]
        fog_of_war_map = infos["fog_of_war_mask"]
        y, x = explored_map.shape

        for i in range(y):
            for j in range(x):
                if fog_of_war_map[i][j] == 1:
                    if explored_map[i][j] == maps.MAP_VALID_POINT:
                        explored_map[i][j] = maps.MAP_INVALID_POINT
                else:
                    if explored_map[i][j] in [maps.MAP_VALID_POINT, maps.MAP_INVALID_POINT]:
                        explored_map[i][j] = maps.MAP_BORDER_INDICATOR
            
        return explored_map, fog_of_war_map

    def _eval_checkpoint(self, checkpoint_path: str, log_manager: LogManager, date: str, checkpoint_index: int = 0) -> None:
        logger.info("############### EAVL3 ##################")
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
        
        # ファイルを読み込んで行ごとにリストに格納する
        with open('data/scene_datasets/mp3d/Environment_Descriptions.txt', 'r') as file:
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
        
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_exp_area = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picture_value = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_similarity = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picsim = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_bleu_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_1_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_2_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_L_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_meteor_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_pas_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_hes_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        
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
            picture_value = []
            similarity = []
            pic_sim = []
            exp_area = [] # 探索済みのエリア()
            bleu_score = []
            rouge_1_score = []
            rouge_2_score = []
            rouge_L_score = []
            meteor_score = []
            pas_score = []
            hes_score = []
            n_envs = self.envs.num_envs
            for n in range(n_envs):
                reward.append(rewards[n][0])
                pic_val.append(rewards[n][1])
                picture_value.append(0)
                similarity.append(0)
                pic_sim.append(0)
                exp_area.append(rewards[n][2])
                bleu_score.append(0)
                rouge_1_score.append(0)
                rouge_2_score.append(0)
                rouge_L_score.append(0)
                meteor_score.append(0)
                pas_score.append(0)
                hes_score.append(0)
                
            for n in range(len(observations)):
                self._taken_picture_list[n].append([pic_val[n], observations[n]["rgb"], rewards[n][6], rewards[n][7], infos[n]["explored_map"]])
                
            reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
            exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.device).unsqueeze(1)
            picture_value = torch.tensor(picture_value, dtype=torch.float, device=self.device).unsqueeze(1)
            similarity = torch.tensor(similarity, dtype=torch.float, device=self.device).unsqueeze(1)
            bleu_score = torch.tensor(bleu_score, dtype=torch.float, device=self.device).unsqueeze(1)
            rouge_1_score = torch.tensor(rouge_1_score, dtype=torch.float, device=self.device).unsqueeze(1)
            rouge_2_score = torch.tensor(rouge_2_score, dtype=torch.float, device=self.device).unsqueeze(1)
            rouge_L_score = torch.tensor(rouge_L_score, dtype=torch.float, device=self.device).unsqueeze(1)
            meteor_score = torch.tensor(meteor_score, dtype=torch.float, device=self.device).unsqueeze(1)
            pas_score = torch.tensor(pas_score, dtype=torch.float, device=self.device).unsqueeze(1)
            hes_score = torch.tensor(hes_score, dtype=torch.float, device=self.device).unsqueeze(1)
            
            current_episode_reward += reward
            current_episode_exp_area += exp_area
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
                    self._taken_picture_list[n], picture_value[n] = self._select_pictures(self._taken_picture_list[n])
                    #self._taken_picture_list[n], picture_value[n] = self._select_pictures2(self._taken_picture_list[n])
                    #self._taken_picture_list[n], picture_value[n] = self._select_random_pictures(self._taken_picture_list[n])
                    #results_image, positions_x, positions_y = self._create_results_image(self._taken_picture_list[n], infos[n]["explored_map"])
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
                        pred_description = self.create_description_sometimes(image_list, results_image)

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
                    #pic_sim[n] = self._calculate_pic_sim(self._taken_picture_list[n])                
                    
                    bleu_score[n] = sum(bleu_list) / len(bleu_list)
                    rouge_1_score[n] = sum(rouge_1_list) / len(rouge_1_list)
                    rouge_2_score[n] = sum(rouge_2_list) / len(rouge_2_list)
                    rouge_L_score[n] = sum(rouge_L_list) / len(rouge_L_list)
                    meteor_score[n] = sum(meteor_list) / len(meteor_list)
                    pas_score[n] = sum(pas_list) / len(pas_list)    
                    hes_score[n] = self.eval_model(hes_sentence_list).item()

                    reward[n] += similarity[n]*10
                    current_episode_reward += similarity[n]*10
                    
                    # average of picture value par 1 picture
                    current_episode_picture_value[n] += picture_value[n]
                    current_episode_similarity[n] += similarity[n]
                    current_episode_picsim[n] += pic_sim[n]
                    current_episode_bleu_score[n] += bleu_score[n]
                    current_episode_rouge_1_score[n] += rouge_1_score[n]
                    current_episode_rouge_2_score[n] += rouge_2_score[n]
                    current_episode_rouge_L_score[n] += rouge_L_score[n]
                    current_episode_meteor_score[n] += meteor_score[n]
                    current_episode_pas_score[n] += pas_score[n]
                    current_episode_hes_score[n] += hes_score[n]
                    
                    # save description
                    out_path = os.path.join("log/" + date + "/eval4/description.txt")
                    with open(out_path, 'a') as f:
                        # print関数でファイルに出力する
                        print(str(current_episodes[n].scene_id[-15:-4]) + "_" + str(_episode_id), file=f)
                        print(description, file=f)
                        print(pred_description,file=f)
                        print(similarity[n].item(),file=f)
                        print(hes_score[n].item(), file=f)
                        #print(location_input, file=f)

                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[n].item()
                    episode_stats["exp_area"] = current_episode_exp_area[n].item()
                    episode_stats["picture_value"] = current_episode_picture_value[n].item()
                    episode_stats["similarity"] = current_episode_similarity[n].item()
                    episode_stats["pic_sim"] = current_episode_picsim[n].item()
                    episode_stats["bleu_score"] = current_episode_bleu_score[n].item()
                    episode_stats["rouge_1_score"] = current_episode_rouge_1_score[n].item()
                    episode_stats["rouge_2_score"] = current_episode_rouge_2_score[n].item()
                    episode_stats["rouge_L_score"] = current_episode_rouge_L_score[n].item()
                    episode_stats["meteor_score"] = current_episode_meteor_score[n].item()
                    episode_stats["pas_score"] = current_episode_pas_score[n].item()
                    episode_stats["hes_score"] = current_episode_hes_score[n].item()
                    
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[n])
                    )
                    current_episode_reward[n] = 0
                    current_episode_exp_area[n] = 0
                    current_episode_picture_value[n] = 0
                    current_episode_similarity[n] = 0
                    current_episode_picsim[n] = 0
                    current_episode_bleu_score[n] = 0
                    current_episode_rouge_1_score[n] = 0
                    current_episode_rouge_2_score[n] = 0
                    current_episode_rouge_L_score[n] = 0
                    current_episode_meteor_score[n] = 0
                    current_episode_pas_score[n] = 0
                    current_episode_hes_score[n] = 0

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
                        
                        name_sim = str(len(stats_episodes)) + "-" + str(hes_score[n].item())[:4] + "-" + str(name_sim)[:4] + "-" + str(episode_stats["exp_area"])[:4]
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
        logger.info("BLUE: " + str(metrics["bleu_score"]) + ", ROUGE-1: " + str(metrics["rouge_1_score"]) + ", ROUGE-2: " + str(metrics["rouge_2_score"]) + ", ROUGE-L: " + str(metrics["rouge_L_score"]) + ", METEOR: " + str(metrics["meteor_score"]))
        eval_metrics_logger.writeLine(str(step_id)+","+str(metrics["exp_area"])+","+str(metrics["similarity"])+","+str(metrics["picture_value"])+","+str(metrics["pic_sim"])+","+str(metrics["bleu_score"])+","+str(metrics["rouge_1_score"])+","+str(metrics["rouge_2_score"])+","+str(metrics["rouge_L_score"])+","+str(metrics["meteor_score"])+","+str(metrics["pas_score"])+","+str(metrics["hes_score"])+","+str(metrics["raw_metrics.agent_path_length"]))

        self.envs.close()
        
        
    def random_eval(self, log_manager: LogManager, date: str,) -> None:
        logger.info("RANDOM 3")
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
        for i in range(self.envs.num_envs):
            self._taken_picture_list.append([])
        
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_exp_area = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picture_value = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_similarity = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_picsim = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_bleu_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_1_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_2_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_rouge_L_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_meteor_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_pas_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_hes_score = torch.zeros(self.envs.num_envs, 1, device=self.device)
        
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
        self.step = 0
        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            #logger.info(f"######### step={self.step} #########")
            self.step += 1
            current_episodes = self.envs.current_episodes()

            actions = []
            for _ in range(self.config.NUM_PROCESSES):
                a = random.randrange(3)
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
            bleu_score = []
            rouge_1_score = []
            rouge_2_score = []
            rouge_L_score = []
            meteor_score= []
            pas_score = []
            hes_score = []
            n_envs = self.envs.num_envs

            for n in range(n_envs):
                reward.append(rewards[n][0])
                pic_val.append(rewards[n][1])
                picture_value.append(0)
                similarity.append(0)
                pic_sim.append(0)
                exp_area.append(rewards[n][2])
                bleu_score.append(0)
                rouge_1_score.append(0)
                rouge_2_score.append(0)
                rouge_L_score.append(0)
                meteor_score.append(0)
                pas_score.append(0)
                hes_score.append(0)
                
            for n in range(len(observations)):
                self._taken_picture_list[n].append([pic_val[n], observations[n]["rgb"], rewards[n][6], rewards[n][7]])

            reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
            exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.device).unsqueeze(1)
            picture_value = torch.tensor(picture_value, dtype=torch.float, device=self.device).unsqueeze(1)
            similarity = torch.tensor(similarity, dtype=torch.float, device=self.device).unsqueeze(1)
            bleu_score = torch.tensor(bleu_score, dtype=torch.float, device=self.device).unsqueeze(1)
            rouge_1_score = torch.tensor(rouge_1_score, dtype=torch.float, device=self.device).unsqueeze(1)
            rouge_2_score = torch.tensor(rouge_2_score, dtype=torch.float, device=self.device).unsqueeze(1)
            rouge_L_score = torch.tensor(rouge_L_score, dtype=torch.float, device=self.device).unsqueeze(1)
            meteor_score = torch.tensor(meteor_score, dtype=torch.float, device=self.device).unsqueeze(1)
            pas_score = torch.tensor(pas_score, dtype=torch.float, device=self.device).unsqueeze(1)
            hes_score = torch.tensor(hes_score, dtype=torch.float, device=self.device).unsqueeze(1)
            
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
                    #results_image, positions_x, positions_y = self._create_results_image(self._taken_picture_list[n], infos[n]["explored_map"])
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
                        pred_description = self.create_description_sometimes(image_list, results_image)
                    
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
                    #pic_sim[n] = self._calculate_pic_sim(self._taken_picture_list[n])                
                    
                    bleu_score[n] = sum(bleu_list) / len(bleu_list)
                    rouge_1_score[n] = sum(rouge_1_list) / len(rouge_1_list)
                    rouge_2_score[n] = sum(rouge_2_list) / len(rouge_2_list)
                    rouge_L_score[n] = sum(rouge_L_list) / len(rouge_L_list)
                    meteor_score[n] = sum(meteor_list) / len(meteor_list)
                    pas_score[n] = sum(pas_list) / len(pas_list)
                    hes_score[n] = self.eval_model(hes_sentence_list).item()

                    reward[n] += similarity[n]*10
                    current_episode_reward += similarity[n]*10
                        
                    current_episode_picture_value[n] += picture_value[n]
                    current_episode_similarity[n] += similarity[n]
                    current_episode_picsim[n] += pic_sim[n]
                    current_episode_bleu_score[n] += bleu_score[n]
                    current_episode_rouge_1_score[n] += rouge_1_score[n]
                    current_episode_rouge_2_score[n] += rouge_2_score[n]
                    current_episode_rouge_L_score[n] += rouge_L_score[n]
                    current_episode_meteor_score[n] += meteor_score[n]
                    current_episode_pas_score[n] += pas_score[n]
                    current_episode_hes_score[n] += hes_score[n]
                    
                    # save description
                    out_path = os.path.join("log/" + date + "/random4/description.txt")
                    with open(out_path, 'a') as f:
                        # print関数でファイルに出力する
                        print(str(current_episodes[n].scene_id[-15:-4]) + "_" + str(_episode_id), file=f)
                        print(description, file=f)
                        print(pred_description, file=f)
                        print(similarity[n].item(), file=f)
                        #print(location_input, file=f)
                                        
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[n].item()
                    episode_stats["exp_area"] = current_episode_exp_area[n].item()
                    episode_stats["picture_value"] = current_episode_picture_value[n].item()
                    episode_stats["similarity"] = current_episode_similarity[n].item()
                    episode_stats["pic_sim"] = current_episode_picsim[n].item()
                    episode_stats["bleu_score"] = current_episode_bleu_score[n].item()
                    episode_stats["rouge_1_score"] = current_episode_rouge_1_score[n].item()
                    episode_stats["rouge_2_score"] = current_episode_rouge_2_score[n].item()
                    episode_stats["rouge_L_score"] = current_episode_rouge_L_score[n].item()
                    episode_stats["meteor_score"] = current_episode_meteor_score[n].item()
                    episode_stats["pas_score"] = current_episode_pas_score[n].item()
                    episode_stats["hes_score"] = current_episode_hes_score[n].item()
                    
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[n])
                    )
                    current_episode_reward[n] = 0
                    current_episode_exp_area[n] = 0
                    current_episode_picture_value[n] = 0
                    current_episode_similarity[n] = 0
                    current_episode_picsim[n] = 0
                    current_episode_bleu_score[n] = 0
                    current_episode_rouge_1_score[n] = 0
                    current_episode_rouge_2_score[n] = 0
                    current_episode_rouge_L_score[n] = 0
                    current_episode_meteor_score[n] = 0
                    current_episode_pas_score[n] = 0
                    current_episode_hes_score[n] = 0

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
                            
                        name_sim = str(len(stats_episodes)) + "-" + str(hes_score[n].item())[:4] + "-" + str(name_sim)[:4] + "-" + str(episode_stats["exp_area"])[:4]
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
        
        step_id = -1
        
        eval_reward_logger.writeLine(str(step_id) + "," + str(aggregated_stats["reward"]))

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}

        logger.info("HES Score: " + str(metrics["hes_score"]))
        logger.info("Similarity: " + str(metrics["similarity"]))
        logger.info("PAS Score: " + str(metrics["pas_score"]))
        logger.info("BLUE: " + str(metrics["bleu_score"]) + ", ROUGE-1: " + str(metrics["rouge_1_score"]) + ", ROUGE-2: " + str(metrics["rouge_2_score"]) + ", ROUGE-L: " + str(metrics["rouge_L_score"]) + ", METEOR: " + str(metrics["meteor_score"]))
        eval_metrics_logger.writeLine(str(step_id)+","+str(metrics["exp_area"])+","+str(metrics["similarity"])+","+str(metrics["picture_value"])+","+str(metrics["pic_sim"])+","+str(metrics["bleu_score"])+","+str(metrics["rouge_1_score"])+","+str(metrics["rouge_2_score"])+","+str(metrics["rouge_L_score"])+","+str(metrics["meteor_score"])+","+str(metrics["pas_score"])+","+str(metrics["hes_score"])+","+str(metrics["raw_metrics.agent_path_length"]))

        self.envs.close()
