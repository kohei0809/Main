def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        log_manager: LogManager,
        date: str,
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
        
        #ログ出力設定
        #time, reward
        eval_reward_logger = log_manager.createLogWriter("reward")
        #time, ci, exp_area, path_length
        eval_metrics_logger = log_manager.createLogWriter("metrics")
        eval_take_picture_writer = log_manager.createLogWriter("take_picture")
        eval_picture_position_writer = log_manager.createLogWriter("picture_position")
        
        #フォルダがない場合は、作成
        p_dir = pathlib.Path("./log/" + date + "/eval/Matrics")
        if not p_dir.exists():
            p_dir.mkdir(parents=True)
        
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        print("PATH")
        print(checkpoint_path)

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
        
        self._taken_picture = []
        self._taken_picture_list = []
        for i in range(self.envs.num_envs):
            self._taken_picture.append([])
            self._taken_picture_list.append([])

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_exp_area = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_ci = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        
        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
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
            ci = []
            exp_area = [] # 探索済みのエリア()
            exp_area_pre = []
            matrics = []
            fog_of_war_map = []
            top_down_map = [] 
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                reward.append(rewards[i][0][0])
                ci.append(rewards[i][0][1])
                exp_area.append(rewards[i][0][2]-rewards[i][0][3])
                exp_area_pre.append(rewards[i][0][3])
                matrics.append(rewards[i][1])
                fog_of_war_map.append(infos[i]["picture_range_map"]["fog_of_war_mask"])
                top_down_map.append(infos[i]["picture_range_map"]["map"])
            
            for n in range(len(observations)):
            #TAKE_PICTUREが呼び出されたかを検証
                if ci[n] != -sys.float_info.max:
                    # 今回撮ったpicture(p_n)が保存してあるpicture(p_k)とかぶっているkを保存
                    cover_list = [] 
                    picture_range_map = self._create_picture_range_map(top_down_map[n], fog_of_war_map[n])
                    
                    # p_kのそれぞれのpicture_range_mapのリスト
                    pre_fog_of_war_map = [sublist[2] for sublist in self._taken_picture_list[n]]
                        
                    # それぞれと閾値より被っているか計算
                    idx = -1
                    min_ci = ci[n]
                    for k in range(len(pre_fog_of_war_map)):
                        # 閾値よりも被っていたらcover_listにkを追加
                        if self._check_percentage_of_fog(picture_range_map, pre_fog_of_war_map[k]) == True:
                            cover_list.append(k)
                                
                        #ciの最小値の写真を探索(１つも被っていない時用)
                        if min_ci < self._taken_picture_list[n][idx][0]:
                            idx = k
                            min_ci = self._taken_picture_list[n][idx][0]
                            
                    # 今までの写真と多くは被っていない時
                    if len(cover_list) == 0:
                        #範囲が多く被っていなくて、self._num_picture回未満写真を撮っていたらそのまま保存
                        if len(self._taken_picture_list[n]) != self._num_picture:
                            self._taken_picture_list[n].append([ci[n], observations[n]["agent_position"], picture_range_map])
                            self._taken_picture[n].append(observations[n]["rgb"])
                            reward[n] += ci[n]
                                
                        #範囲が多く被っていなくて、self._num_picture回以上写真を撮っていたら
                        else:
                            # 今回の写真が保存してある写真の１つでもCIが高かったらCIが最小の保存写真と入れ替え
                            if idx != -1:
                                ci_pre = self._taken_picture_list[n][idx][0]
                                self._taken_picture_list[n][idx] = [ci[n], observations[n]["agent_position"], picture_range_map]
                                self._taken_picture[n][idx] = observations[n]["rgb"]   
                                reward[n] += (ci[n] - ci_pre) 
                                ci[n] -= ci_pre
                            
                            ci[n] = 0.0    
                            
                    # 1つとでも多く被っていた時    
                    else:
                        min_idx = -1
                        min_ci_k = 1000
                        # 多く被った写真のうち、ciが最小のものを計算
                        for k in range(len(cover_list)):
                            idx_k = cover_list[k]
                            if self._taken_picture_list[n][idx_k][0] < min_ci_k:
                                min_ci_k = self._taken_picture_list[n][idx_k][0]
                                min_idx = idx_k
                                    
                        # 被った割合分小さくなったCIでも保存写真の中の最小のCIより大きかったら交換
                        if self._compareWithChangedCI(picture_range_map, pre_fog_of_war_map, cover_list, ci[n], min_ci_k, min_idx) == True:
                            self._taken_picture_list[n][min_idx] = [ci[n], observations[n]["agent_position"], picture_range_map]
                            self._taken_picture[n][min_idx] = observations[n]["rgb"]   
                            reward[n] += (ci[n] - min_ci_k)  
                            ci[n] -= min_ci_k
                        
                        ci[n] = 0.0
                else:
                    ci[n] = 0.0
                
            
            reward = torch.tensor(
                reward, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            exp_area = np.array(exp_area)
            exp_area = torch.tensor(
                exp_area, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            ci = np.array(ci)
            ci= torch.tensor(
                ci, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)

            current_episode_reward += reward
            current_episode_exp_area += exp_area
            current_episode_ci += ci
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
                    #logger.info("Exp_area: " + str(current_episode_exp_area[i]))
                    eval_take_picture_writer.write(str(len(stats_episodes)) + "," + str(current_episodes[i].episode_id) + "," + str(n))
                    eval_picture_position_writer.write(str(len(stats_episodes)) + "," + str(current_episodes[i].episode_id) + "," + str(n))
                    for j in range(self._num_picture):
                        if j < len(self._taken_picture_list[i]):
                            eval_take_picture_writer.write(str(self._taken_picture_list[i][j][0]))
                            eval_picture_position_writer.write(str(self._taken_picture_list[i][j][1][0]) + "," + str(self._taken_picture_list[i][j][1][1]) + "," + str(self._taken_picture_list[i][j][1][2]))
                        else:
                            eval_take_picture_writer.write(" ")
                            eval_picture_position_writer.write(" ")
                        
                    eval_take_picture_writer.writeLine()
                    eval_picture_position_writer.writeLine()

                    next_episodes = self.envs.current_episodes()
                    envs_to_pause = []
                    
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["exp_area"] = current_episode_exp_area[i].item()
                    
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    current_episode_exp_area[i] = 0
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
                        name_ci = 0.0
                        
                        for j in range(len(self._taken_picture_list[i])):
                            name_ci += self._taken_picture_list[i][j][0]
                        
                        name_ci = str(name_ci) + "-" + str(len(stats_episodes))
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR+"/"+date,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=metrics,
                            tb_writer=writer,
                            name_ci=name_ci,
                        )
        
                        # Save taken picture
                        metric_strs = []
                        in_metric = ['exp_area', 'ci']
                        for k, v in metrics.items():
                            if k in in_metric:
                                metric_strs.append(f"{k}={v:.2f}")
                        
                        #logger.info(infos[i])
                        name_p = 0.0  
                            
                        for j in range(len(self._taken_picture_list[i])):
                            eval_picture_top_logger = log_manager.createLogWriter("picture_top_" + str(len(stats_episodes)) + "_" + str(current_episodes[i].episode_id) + "_" + str(j) + "_" + str(n) + "_" + str(checkpoint_index))
                
                            for k in range(len(self._taken_picture_list[i][j][2])):
                                for l in range(len(self._taken_picture_list[i][j][2][0])):
                                    eval_picture_top_logger.write(str(self._taken_picture_list[i][j][2][k][l]))
                                eval_picture_top_logger.writeLine()
                                
                            name_p = self._taken_picture_list[i][j][0]
                            picture_name = "episode=" + str(len(stats_episodes))+ "-" + str(current_episodes[i].episode_id)+ "-ckpt=" + str(checkpoint_index) + "-" + str(j) + "-" + str(name_p)
                            dir_name = "./taken_picture/" + date 
                            if not os.path.exists(dir_name):
                                os.makedirs(dir_name)
                        
                            picture = self._taken_picture[i][j]
                            plt.figure()
                            ax = plt.subplot(1, 1, 1)
                            ax.axis("off")
                            plt.imshow(picture)
                            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)
                            path = dir_name + "/" + picture_name + ".png"
                        
                            plt.savefig(path)
                            logger.info("Picture created: " + path)
                        
                        #Save score_matrics
                        if matrics[i] is not None:
                            eval_matrics_logger = log_manager.createLogWriter("Matrics/matrics_" + str(current_episodes[i].episode_id) + "_" + str(checkpoint_index))
                            for j in range(matrics[i].shape[0]):
                                for k in range(matrics[i].shape[1]):
                                    eval_matrics_logger.write(str(matrics[i][j][k]))
                                eval_matrics_logger.writeLine("")
                            
                            
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

        logger.info("CI:" + str(metrics["ci"]))
        eval_metrics_logger.writeLine(str(step_id) + "," + str(metrics["ci"]) + "," + str(metrics["exp_area"]) + "," + str(metrics["raw_metrics.agent_path_length"]))

        self.envs.close()