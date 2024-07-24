#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ある程度多くの写真を探索中は保持し、探索終了後に取捨選択する 

import contextlib
import os
import random
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
import tqdm
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

import clip
from sentence_transformers import SentenceTransformer, util
from lavis.models import load_model_and_preprocess

from habitat import Config, VectorEnv, logger
from habitat.utils import profiling_wrapper
from habitat.utils.env_utils import construct_envs
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image, explored_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainerOracle
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorageOracle
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ppo import PPOOracle, ProposedPolicyOracle
from utils.log_manager import LogManager
from utils.log_writer import LogWriter
from habitat.utils.visualizations import fog_of_war, maps
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_array_to_dict,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
)

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer


@baseline_registry.register_trainer(name="ddppo")
@baseline_registry.register_trainer(name="oracle")
class PPOTrainerO(BaseRLTrainerOracle):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    agent: PPOOracle
    actor_critic: ProposedPolicyOracle

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        self._num_picture = config.TASK_CONFIG.TASK.PICTURE.NUM_PICTURE

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        self._obs_batching_cache = ObservationBatchingCache()

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

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

        #self.obs_space = observation_space

        logger.info("DEVICE: " + str(self.device))
        self.actor_critic.to(self.device)

        self.agent = (DDPPO if self._is_distributed else PPOOracle)(
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

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
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
            if not isinstance(k, str) or k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if isinstance(subk, str)
                        and k + "." + subk not in cls.METRICS_BLACKLIST
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

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=is_slurm_batch_job(),
        )

    def _init_train(self, log_manager):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

                logger.info("########### PPO3 ##############")

                self.log_manager = log_manager
                
                #ログ出力設定
                #time, reward
                self.reward_logger = self.log_manager.createLogWriter("reward")
                #time, learning_rate
                #self.learning_rate_logger = self.log_manager.createLogWriter("learning_rate")
                #time, found, forward, left, right, look_up, look_down
                self.action_logger = self.log_manager.createLogWriter("action_prob")
                #time, picture, episode_length
                self.metrics_logger = self.log_manager.createLogWriter("metrics")
                #time, losses_value, losses_policy
                self.loss_logger = self.log_manager.createLogWriter("loss")
                
            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        # picture_value, rgb_image, image_emb
        self._taken_picture_list = []
        self.subgoal_list = []
        self.subgoal_num_list = []
        for _ in range(self.envs.num_envs):
            self._taken_picture_list.append([])
            self.subgoal_list.append([])
            self.subgoal_num_list.append([])

        self.each_subgoal_reward = 0.1
        self.threshold_subgoal = 5

        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space
        if is_continuous_action_space(action_space):
            # Assume ALL actions are NOT discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = None
            discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

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

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        #obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorageOracle(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device, cache=self._obs_batching_cache)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.current_episode_exp_area = torch.zeros(self.envs.num_envs, 1)
        self.current_episode_picture_value = torch.zeros(self.envs.num_envs, 1)
        self.current_episode_similarity = torch.zeros(self.envs.num_envs, 1)
        self.current_episode_picsim = torch.zeros(self.envs.num_envs, 1)
        self.current_episode_subgoal_reward = torch.zeros(self.envs.num_envs, 1)

        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
            exp_area=torch.zeros(self.envs.num_envs, 1),
            picture_value=torch.zeros(self.envs.num_envs, 1),
            similarity=torch.zeros(self.envs.num_envs, 1),
            pic_sim=torch.zeros(self.envs.num_envs, 1),
            subgoal_reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()


    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if act.shape[0] > 1:
                step_action = action_array_to_dict(
                    self.policy_action_space, act
                )
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        #logger.info("######## before output ###########")
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]
        #logger.info("######## after output ###########")

        observations, rewards, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        reward = []
        pic_val = []
        picture_value = []
        similarity = []
        pic_sim = []
        exp_area = [] # 探索済みのエリア()
        semantic_obs = []
        subgoal_reward = []
        for n in range(num_envs):
            reward.append(rewards[n][0])
            pic_val.append(rewards[n][1])
            picture_value.append(0)
            similarity.append(0)
            pic_sim.append(0)
            exp_area.append(rewards[n][2]-rewards[n][3])
            semantic_obs.append(observations[n]["semantic"])
            subgoal_reward.append(0)

        current_episodes = self.envs.current_episodes()
        for n in range(len(observations)):
            if len(self._taken_picture_list[n]) == 0:
                self._load_subgoal_list(current_episodes, n, rewards[n][5])
            self._taken_picture_list[n].append([pic_val[n], observations[n]["rgb"], rewards[n][6], rewards[n][7]])
            subgoal_reward[n] = self._calculate_subgoal_reward(semantic_obs[n], n)
            reward[n] += subgoal_reward[n]

        
        reward = torch.tensor(reward, dtype=torch.float, device=self.current_episode_reward.device).unsqueeze(1)
        exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.current_episode_reward.device).unsqueeze(1)
        picture_value = torch.tensor(picture_value, dtype=torch.float, device=self.current_episode_reward.device).unsqueeze(1)
        similarity = torch.tensor(similarity, dtype=torch.float, device=self.current_episode_reward.device).unsqueeze(1)
        pic_sim = torch.tensor(pic_sim, dtype=torch.float, device=self.current_episode_reward.device).unsqueeze(1)
        subgoal_reward = torch.tensor(subgoal_reward, dtype=torch.float, device=self.current_episode_reward.device).unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        # episode ended
        for n in range(len(observations)):
            if done_masks[n].item() == True:   
                # tannsakuzumino kannkyouno syasinnwo syutoku
                explored_picture, start_position = self.get_explored_picture(infos[n]["explored_map"])
                explored_picture = explored_to_image(explored_picture, infos[n])
                explored_picture = Image.fromarray(np.uint8(explored_picture))

                # 写真の選別
                self._taken_picture_list[n], picture_value[n] = self._select_pictures(self._taken_picture_list[n])
                results_image = None
                results_image, positions_x, positions_y = self._create_results_image(self._taken_picture_list[n], explored_picture)

                # Ground-Truth descriptionと生成文との類似度の計算 
                description = self.description_df[self.description_df["scene_id"]==current_episodes[n].scene_id[-15:-4]]["description"].item()
                pred_description = self.create_description(self._taken_picture_list[n])
                pred_description = ""
                if results_image is not None:
                    pred_description, location_input = self.create_description_from_results_image(results_image, start_position, positions_x, positions_y)
                
                similarity[n] = self.calculate_similarity(pred_description, description)
                pic_sim[n] = self._calculate_pic_sim(self._taken_picture_list[n])                
                reward[n] += similarity[n]*10
    
                self._taken_picture_list[n] = []

        self.current_episode_reward[env_slice] += reward
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore

        self.current_episode_exp_area[env_slice] += exp_area
        current_ep_exp_area = self.current_episode_exp_area[env_slice]
        self.running_episode_stats["exp_area"][env_slice] += current_ep_exp_area.where(done_masks, current_ep_exp_area.new_zeros(()))  # type: ignore
        
        self.current_episode_picture_value[env_slice] += picture_value
        current_ep_picture_value = self.current_episode_picture_value[env_slice]
        self.running_episode_stats["picture_value"][env_slice] += current_ep_picture_value.where(done_masks, current_ep_picture_value.new_zeros(()))  # type: ignore
        
        self.current_episode_similarity[env_slice] += similarity
        current_ep_similarity = self.current_episode_similarity[env_slice]
        self.running_episode_stats["similarity"][env_slice] += current_ep_similarity.where(done_masks, current_ep_similarity.new_zeros(()))  # type: ignore
        
        self.current_episode_picsim[env_slice] += pic_sim
        current_ep_picsim = self.current_episode_picsim[env_slice]
        self.running_episode_stats["pic_sim"][env_slice] += current_ep_picsim.where(done_masks, current_ep_picsim.new_zeros(()))  # type: ignore
        
        self.current_episode_subgoal_reward[env_slice] += subgoal_reward
        current_ep_subgoal_reward = self.current_episode_subgoal_reward[env_slice]
        self.running_episode_stats["subgoal_reward"][env_slice] += current_ep_subgoal_reward.where(done_masks, current_ep_subgoal_reward.new_zeros(()))  # type: ignore
        
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k, dtype=torch.float, device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore
            
        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)
        self.current_episode_exp_area[env_slice].masked_fill_(done_masks, 0.0)
        self.current_episode_picture_value[env_slice].masked_fill_(done_masks, 0.0)
        self.current_episode_similarity[env_slice].masked_fill_(done_masks, 0.0)
        self.current_episode_picsim[env_slice].masked_fill_(done_masks, 0.0)
        self.current_episode_subgoal_reward[env_slice].masked_fill_(done_masks, 0.0)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.insert(
            next_observations=batch,
            rewards=reward,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        self.agent.train()

        value_loss, action_loss, dist_entropy = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        #csv
        self.reward_logger.writeLine(str(self.num_steps_done) + "," + str(deltas["reward"] / deltas["count"]))

        total_actions = self.rollouts.buffers["actions"].shape[0] * self.rollouts.buffers["actions"].shape[1]
        total_found_actions = int(torch.sum(self.rollouts.buffers["actions"] == 0).cpu().numpy())
        total_forward_actions = int(torch.sum(self.rollouts.buffers["actions"] == 1).cpu().numpy())
        total_left_actions = int(torch.sum(self.rollouts.buffers["actions"] == 2).cpu().numpy())
        total_right_actions = int(torch.sum(self.rollouts.buffers["actions"] == 3).cpu().numpy())
        total_look_up_actions = int(torch.sum(self.rollouts.buffers["actions"] == 4).cpu().numpy())
        total_look_down_actions = int(torch.sum(self.rollouts.buffers["actions"] == 5).cpu().numpy())
        assert total_actions == (total_found_actions + total_forward_actions + 
            total_left_actions + total_right_actions + total_look_up_actions + 
            total_look_down_actions
        )

        # csv
        self.action_logger.writeLine(
            str(self.num_steps_done) + "," + str(total_found_actions/total_actions) + ","
            + str(total_forward_actions/total_actions) + "," + str(total_left_actions/total_actions) + ","
            + str(total_right_actions/total_actions) + "," + str(total_look_up_actions/total_actions) + ","
            + str(total_look_down_actions/total_actions)
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
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
            logger.info("SubGoal_Reward: " + str(metrics["subgoal_reward"]))
            logger.info("REWARD: " + str(deltas["reward"] / deltas["count"]))
            self.metrics_logger.writeLine(str(self.num_steps_done) + "," + str(metrics["exp_area"]) + "," + str(metrics["similarity"]) + "," + str(metrics["picture_value"]) + "," + str(metrics["pic_sim"]) + "," + str(metrics["subgoal_reward"]) + "," + str(metrics["raw_metrics.agent_path_length"]))
                    
            logger.info(metrics)    

        self.loss_logger.write(str(self.num_steps_done))
        for k, v in losses.items():
            self.loss_logger.write(str(v))
        self.loss_logger.writeLine()
        

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.RL.PPO.num_steps * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.RL.DDPPO.sync_frac * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self, log_manager) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train(log_manager)

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        ppo_cfg = self.config.RL.PPO

        while not self.is_done():
            profiling_wrapper.on_start_step()
            profiling_wrapper.range_push("train update")

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * (
                    1 - self.percent_done()
                )

            if rank0_only() and self._should_save_resume_state():
                requeue_stats = dict(
                    env_time=self.env_time,
                    pth_time=self.pth_time,
                    count_checkpoints=count_checkpoints,
                    num_steps_done=self.num_steps_done,
                    num_updates_done=self.num_updates_done,
                    _last_checkpoint_percent=self._last_checkpoint_percent,
                    prev_time=(time.time() - self.t_start) + prev_time,
                    running_episode_stats=self.running_episode_stats,
                    window_episode_stats=dict(self.window_episode_stats),
                )

                save_resume_state(
                    dict(
                        state_dict=self.agent.state_dict(),
                        optim_state=self.agent.optimizer.state_dict(),
                        lr_sched_state=lr_scheduler.state_dict(),
                        config=self.config,
                        requeue_stats=requeue_stats,
                    ),
                    self.config,
                )

            if EXIT.is_set():
                profiling_wrapper.range_pop()  # train update

                self.envs.close()

                requeue_job()

                return

            self.agent.eval()
            count_steps_delta = 0
            profiling_wrapper.range_push("rollouts loop")

            profiling_wrapper.range_push("_collect_rollout_step")
            for buffer_index in range(self._nbuffers):
                self._compute_actions_and_step_envs(buffer_index)

            for step in range(ppo_cfg.num_steps):
                """
                if rank0_only():
                    logger.info(f"STEP={step}")
                """
                is_last_step = (
                    self.should_end_early(step + 1)
                    or (step + 1) == ppo_cfg.num_steps
                )

                for buffer_index in range(self._nbuffers):
                    count_steps_delta += self._collect_environment_result(
                        buffer_index
                    )

                    if (buffer_index + 1) == self._nbuffers:
                        profiling_wrapper.range_pop()  # _collect_rollout_step

                    if not is_last_step:
                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_push(
                                "_collect_rollout_step"
                            )

                        self._compute_actions_and_step_envs(buffer_index)

                if is_last_step:
                    break

            profiling_wrapper.range_pop()  # rollouts loop

            if self._is_distributed:
                self.num_rollouts_done_store.add("num_done", 1)

            (
                value_loss,
                action_loss,
                dist_entropy,
            ) = self._update_agent()

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()  # type: ignore

            self.num_updates_done += 1
            losses = self._coalesce_post_step(
                dict(
                    value_loss=value_loss,
                    action_loss=action_loss,
                    entropy=dist_entropy,
                ),
                count_steps_delta,
            )

            self._training_log(losses, prev_time)

            # checkpoint model
            #if rank0_only() and self.should_checkpoint():
            if rank0_only():
                self.save_checkpoint(
                    f"ckpt.{count_checkpoints}.pth",
                    dict(
                        step=self.num_steps_done,
                        wall_time=(time.time() - self.t_start) + prev_time,
                    ),
                )
                count_checkpoints += 1

            profiling_wrapper.range_pop()  # train update

        self.envs.close()


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
        file_path = f"/gs/fs/tga-aklab/matsumoto/Main/data/scene_datasets/mp3d/{scene_name}/pickup_object.csv"

        with open(file_path, mode="r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                self.subgoal_list[n].append(int(row[2]))
                self.subgoal_num_list[n].append(0)
        category_file_path = f"/gs/fs/tga-aklab/matsumoto/Main/data/scene_datasets/mp3d/{scene_name}/category.txt"
        with open(category_file_path, mode='r') as f:
            content = f.read()
            object_names = content.split(',')
            for _, row in semantic_scene_df.iterrows():
                if row['object_name'] in object_names:
                    if row['id'] not in self.subgoal_list[n]:
                        self.subgoal_list[n].append(row['id'])
                        self.subgoal_num_list[n].append(0)


    def _calculate_subgoal_reward(self, semantic_obs, n):
        H, W, _ = semantic_obs.shape
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
        return r


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


    def _create_results_image(self, picture_list, explored_picture):
        images = []
        x_list = []
        y_list = []
    
        if len(picture_list) == 0:
            return None

        for i in range(self._num_picture):
            idx = i%len(picture_list)
            images.append(Image.fromarray(picture_list[idx][1]))
            x_list.append(picture_list[idx][2])
            y_list.append(picture_list[idx][3])

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

        # explored_pictureの新しい横幅を計算
        if explored_picture.height != 0:
            aspect_ratio = explored_picture.width / explored_picture.height
        else:
            aspect_ratio = explored_picture.width
        new_explored_picture_width = int(result_height * aspect_ratio)

        # explored_pictureをリサイズ
        explored_picture = explored_picture.resize((new_explored_picture_width, result_height))

        # 最終画像の幅を計算
        final_width = result_width + new_explored_picture_width

        # 最終画像を作成
        final_image = Image.new('RGB', (final_width, result_height), color=(255, 255, 255))

        # result_imageを貼り付け
        final_image.paste(result_image, (0, 0))

        # リサイズしたexplored_pictureを貼り付け
        final_image.paste(explored_picture, (result_width, 0))

        return final_image, x_list, y_list


    def create_description_from_results_image(self, results_image, start_position, x_list, y_list, input_change=False):
        blue_x, blue_y = start_position
        red_x = blue_x + 1.0
        red_y = blue_y + 1.0
        input_text_1 = "<Instructions>\n"\
                        "You are an excellent property writer.\n"\
                        "The picture you have entered consists of 10 pictures of a building, 5 horizontally and 2 vertically placed in a single picture, with a map drawing of the building to the right of the pictures.\n"\
                        "Each picture is also separated by a black line.\n"\
                        "From each picture, understand the details of this building's environment, and in the form of a summary of these pictures, describe this building's environment in detail, paying attention to the <Notes>.\n"\
                        "In doing so, please also consider the location of each picture as indicated by <Location Information>.\n"\
                        "\n\n"\
                        "<Location Information>\n"\
                        "The top leftmost picture is picture_1, and from its right to left are picture_2, picture_3, picture_4, and picture_5.\n"\
                        "Similarly, the bottom-left corner is picture_6, and from its right, picture_7, picture_8, picture_9, and picture_10.\n"\
                        "The following is the location information for each picture.\n\n"
        input_text_2 = "<Notes>\n"\
                        "・Note that each picture is taken at the location indicated by <Location Information>, and that adjacent pictures are not close in location.\n"\
                        f"・In the map diagram on the right, there is a blue dot and a red dot. The coordinates of the blue point are ({blue_x}, {blue_y}), and the coordinates of the red point are ({red_x}, {red_y}), so please use this as a reference to output the location information for each picture from <Location Information>.\n"\
                        "・When describing the environment, do not mention whether it was taken from that picture or the black line separating each picture.\n"\
                        "・Only refer to the structure of the description from <Example of output>, and do not use your imagination to describe things not shown in the picture.\n"\
                        "\n\n"\
                        "<Example of output>\n"\
                        "This building features a spacious layout with multiple living rooms, bedrooms, and bathrooms. A living space with a fireplace is next to a fully equipped kitchen. There are also three bedrooms on the left side of the building, with a bathroom nearby. There are plenty of books to work with.\n"\
                        "Overall, the apartment is spacious and well-equipped, with many paintings on the walls."
        
        location_input = ""
        for i in range(self._num_picture):
            idx = i+1
            location_input += "picture_"+str(idx)+" : ("+str(x_list[i])+", "+str(y_list[i])+")\n"
        location_input+="\n"
        input_text = input_text_1 + location_input + input_text_2
        #logger.info("############## input_text ###############")
        #logger.info(input_text)
        if input_change == True:
            logger.info("############## Input Change ################")
            input_text = "You are an excellent property writer. This picture consists of 10 pictures arranged in one picture, 5 horizontally and 2 vertically on one building. In addition, a black line separates the pictures from each other. From each picture, you should understand the details of this building's environment and describe this building's environment in detail in the form of a summary of these pictures. At this point, do not describe each picture one at a time, but rather in a summarized form. Also note that each picture was taken in a separate location, so successive pictures are not positionally close. Additionally, do not mention which picture you are quoting from or the black line separating each picture."
        response = self.generate_response(results_image, input_text)
        response = response[4:-4]
        return response, location_input


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


    def get_explored_picture(self, infos):
        explored_map = infos["map"]
        fog_of_war_map = infos["fog_of_war_mask"]
        start_position = infos["start_position"]
        y, x = explored_map.shape

        for i in range(y):
            for j in range(x):
                if fog_of_war_map[i][j] == 1:
                    if explored_map[i][j] == maps.MAP_VALID_POINT:
                        explored_map[i][j] = maps.MAP_INVALID_POINT
                else:
                    if explored_map[i][j] in [maps.MAP_VALID_POINT, maps.MAP_INVALID_POINT]:
                        explored_map[i][j] = maps.MAP_BORDER_INDICATOR

        range_x = np.where(~np.all(explored_map == maps.MAP_BORDER_INDICATOR, axis=1))[0]
        range_y = np.where(~np.all(explored_map == maps.MAP_BORDER_INDICATOR, axis=0))[0]

        _ind_x_min = range_x[0]
        _ind_x_max = range_x[-1]
        _ind_y_min = range_y[0]
        _ind_y_max = range_y[-1]
        _grid_delta = 3

        explored_map = explored_map[
            _ind_x_min - _grid_delta : _ind_x_max + _grid_delta,
            _ind_y_min - _grid_delta : _ind_y_max + _grid_delta,
        ]
            
        return explored_map, start_position


    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.EVAL.SHOULD_LOAD_CKPT:
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
        else:
            ckpt_dict = {}

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if (
            len(self.config.VIDEO_OPTION) > 0
            and self.config.VIDEO_RENDER_TOP_DOWN
        ):
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        action_space = self.envs.action_spaces[0]
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            if is_continuous_action_space(action_space):
                # Assume NONE of the actions are discrete
                action_shape = (get_num_actions(action_space),)
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = (1,)
                discrete_actions = True

        self._setup_actor_critic_agent(ppo_cfg)

        if self.agent.actor_critic.should_load_agent_state:
            self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
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

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if actions[0].shape[0] > 1:
                step_data = [
                    action_array_to_dict(self.policy_action_space, a)
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.VIDEO_FPS,
                            keys_to_include_in_name=self.config.EVAL_KEYS_TO_INCLUDE_IN_NAME,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if self.config.VIDEO_RENDER_ALL_INFO:
                        frame = overlay_frame(frame, infos[i])

                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
