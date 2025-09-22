"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import time
import datetime
import shutil
import json
import h5py
import imageio
import numpy as np
import traceback
from copy import deepcopy
from collections import OrderedDict

import torch

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils

from robomimic.utils.dataset import SequenceDataset, R2D2Dataset, MetaDataset
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from tianshou.env import SubprocVectorEnv

from robocasa.demos.kitchen_scenes import capture_depth_camera_data
from termcolor import colored
import open3d as o3d
from robomimic.utils.navi_utils import NaviPolicy, obs_to_SE2, obs_to_SE3
from robomimic.utils.sample_utils import TargetHelper, get_target_helper_for_rollout_collection
from robomimic.utils.mode_utils import arm_fake_controller
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
from robomimic.utils.misc_util import convert_extrinsic_to_pos_and_quat, qpos_command_wrapper

from nav2man.utils.prediction import predict_SIR_target_point
from nav2man.utils.visualizer import save_gmm_visualization_se2

def numpy_encoder(obj):
    """Helper function to convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return str(obj)

def get_exp_dir(config, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    base_output_dir = os.path.join(os.getcwd(), base_output_dir)  # modify by kx
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            # ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
            print("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
            print(colored("directly put 'n' as default to skip this annoying message. This is a hardcode by kx\n", "yellow"))
            ans = "n"
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)

    # vis directory
    vis_dir = os.path.join(base_output_dir, time_str, "vis")
    os.makedirs(vis_dir)

    # rollout directory modified by hj
    rollout_dir = os.path.join(base_output_dir, time_str, "rollout")
    os.makedirs(rollout_dir)
    
    return log_dir, output_dir, video_dir, vis_dir, rollout_dir


def load_data_for_training(config, obs_keys, lang_encoder=None):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    train_filter_by_attribute = config.train.hdf5_filter_key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert config.experiment.validate, "specified validation filter key {}, but config.experiment.validate is not set".format(valid_filter_by_attribute)

    # load the dataset into memory
    if config.experiment.validate:
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        assert (train_filter_by_attribute is not None) and (valid_filter_by_attribute is not None), \
            "did not specify filter keys corresponding to train and valid split in dataset" \
            " - please fill config.train.hdf5_filter_key and config.train.hdf5_validation_filter_key"
        train_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=train_filter_by_attribute,
        )
        valid_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=valid_filter_by_attribute,
        )
        assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), "training demonstrations overlap with " \
            "validation demonstrations!"
        train_dataset = dataset_factory(
            config, obs_keys,
            filter_by_attribute=train_filter_by_attribute,
            lang_encoder=lang_encoder,
        )
        valid_dataset = dataset_factory(
            config, obs_keys,
            filter_by_attribute=valid_filter_by_attribute,
            lang_encoder=lang_encoder,
        )
    else:
        train_dataset = dataset_factory(
            config, obs_keys,
            filter_by_attribute=train_filter_by_attribute,
            lang_encoder=lang_encoder,
        )
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None, lang_encoder=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        action_keys=config.train.action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=config.train.action_config,
        load_next_obs=config.train.hdf5_load_next_obs, # whether to load next observations (s') from dataset
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=True,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute,
        shuffled_obs_key_groups=config.train.shuffled_obs_key_groups,
        lang_encoder=lang_encoder,
    )

    ds_kwargs["hdf5_path"] = [ds_cfg["path"] for ds_cfg in config.train.data]
    ds_kwargs["filter_by_attribute"] = [ds_cfg.get("filter_key", filter_by_attribute) for ds_cfg in config.train.data]
    ds_weights = [ds_cfg.get("weight", 1.0) for ds_cfg in config.train.data]
    ds_langs = [ds_cfg.get("lang", None) for ds_cfg in config.train.data]

    meta_ds_kwargs = dict()

    dataset = get_dataset(
        ds_class=R2D2Dataset if config.train.data_format == "r2d2" else SequenceDataset,
        ds_kwargs=ds_kwargs,
        ds_weights=ds_weights,
        ds_langs=ds_langs,
        normalize_weights_by_ds_size=False,
        meta_ds_class=MetaDataset,
        meta_ds_kwargs=meta_ds_kwargs,
    )

    return dataset


def get_dataset(
    ds_class,
    ds_kwargs,
    ds_weights,
    ds_langs,
    normalize_weights_by_ds_size,
    meta_ds_class=MetaDataset,
    meta_ds_kwargs=None,
):
    ds_list = []
    for i in range(len(ds_weights)):
        
        ds_kwargs_copy = deepcopy(ds_kwargs)

        keys = ["hdf5_path", "filter_by_attribute"]

        for k in keys:
            ds_kwargs_copy[k] = ds_kwargs[k][i]

        ds_kwargs_copy["dataset_lang"] = ds_langs[i]
        
        ds_list.append(ds_class(**ds_kwargs_copy))
    
    if len(ds_weights) == 1:
        ds = ds_list[0]
    else:
        if meta_ds_kwargs is None:
            meta_ds_kwargs = dict()
        ds = meta_ds_class(
            datasets=ds_list,
            ds_weights=ds_weights,
            normalize_weights_by_ds_size=normalize_weights_by_ds_size,
            **meta_ds_kwargs
        )

    return ds


def batchify_obs(obs_list):
    """
    TODO: add comments
    """
    keys = list(obs_list[0].keys())
    obs = {
        k: np.stack([obs_list[i][k] for i in range(len(obs_list))]) for k in keys
    }
    
    return obs


def run_rollout(
        policy, 
        env, 
        horizon,
        ob_dict=None,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
    ):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper) or isinstance(env, SubprocVectorEnv)

    batched = isinstance(env, SubprocVectorEnv)
    flag_is_sir = ob_dict is not None
    if not flag_is_sir:
        ob_dict = env.reset()
    
    if batched:
        lang = env.get_env_attr("_ep_lang_str", id=list(range(len(env))))
    else:
        lang = env._ep_lang_str
    policy.start_episode(lang=lang)

    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    rews = []
    success = None #{ k: False for k in env.is_success() } # success metrics

    if batched:
        end_step = [None for _ in range(len(env))]
    else:
        end_step = None

    if batched:
        video_frames = [[] for _ in range(len(env))]
    else:
        video_frames = []
    
    # last_arm_pos = None
    for step_i in range(horizon): #LogUtils.tqdm(range(horizon)):
        # get action from policy
        if batched:
            policy_ob = batchify_obs(ob_dict)
            ac = policy(ob=policy_ob, goal=goal_dict, batched=True) #, return_ob=True)
        else:
            policy_ob = ob_dict
            ac = policy(ob=policy_ob, goal=goal_dict) #, return_ob=True)

        # play action
        # print("*"*100)
        # print("ac", ac)
        ob_dict, r, done, info = env.step(ac)
        # arm_pos = env.env.env.sim.data.qpos[env.env.env.robots[0]._ref_arm_joint_pos_indexes]
        # print("arm_pos: ", arm_pos)
        # if last_arm_pos is not None:
        #     print("diff: ", arm_pos - last_arm_pos)
        # last_arm_pos = arm_pos
        
        # render to screen
        if render:
            if flag_is_sir:
                env.render(mode="human",camera_name='free')
            else:
                env.render(mode="human")

        # compute reward
        rews.append(r)

        # cur_success_metrics = env.is_success()
        if batched:
            cur_success_metrics = TensorUtils.list_of_flat_dict_to_dict_of_list([info[i]["is_success"] for i in range(len(info))])
            cur_success_metrics = {k: np.array(v) for (k, v) in cur_success_metrics.items()}
        else:
            cur_success_metrics = info["is_success"]

        if success is None:
            success = deepcopy(cur_success_metrics)
        else:
            for k in success:
                success[k] = success[k] | cur_success_metrics[k]

        # visualization
        if video_writer is not None:
            if video_count % video_skip == 0:
                if batched:
                    # frames = env.render(mode="rgb_array", height=video_height, width=video_width)
                    
                    frames = []
                    policy_ob = deepcopy(policy_ob)
                    for env_i in range(len(env)):
                        cam_imgs = []
                        for im_name in ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"]:
                            im = TensorUtils.to_numpy(
                                policy_ob[im_name][env_i, -1]
                            )
                            im = np.transpose(im, (1, 2, 0))
                            cam_imgs.append(im)
                        frame = np.concatenate(cam_imgs, axis=1)
                        frame = (frame * 255.0).astype(np.uint8)
                        frames.append(frame)
                    
                    for env_i in range(len(env)):
                        frame = frames[env_i]
                        video_frames[env_i].append(frame)
                else:
                    frame = env.render(mode="rgb_array", height=512, width=512)
                    
                    # cam_imgs = []
                    # for im_name in ["robot0_eye_in_hand_image", "robot0_agentview_right_image", "robot0_agentview_left_image"]:
                    #     im_input = TensorUtils.to_numpy(
                    #         policy_ob_dict[im_name][0,-1]
                    #     )
                    #     im_ret = TensorUtils.to_numpy(
                    #         policy_ob_dict["ret"]["obs"][im_name][0,:,-1]
                    #     )
                    #     im_input = np.transpose(im_input, (1, 2, 0))
                    #     im_input = add_border_to_frame(im_input, border_size=3, color="black")
                    #     im_ret = np.transpose(im_ret, (0, 2, 3, 1))
                    #     im = np.concatenate((im_input, *im_ret), axis=1)
                    #     cam_imgs.append(im)

                    # frame = np.concatenate(cam_imgs, axis=0)
                    video_frames.append(frame)

            video_count += 1

        # break if done
        if batched:
            for env_i in range(len(env)):
                if end_step[env_i] is not None:
                    continue
                
                if done[env_i] or (terminate_on_success and success["task"][env_i]):
                    end_step[env_i] = step_i
        else:
            if done or (terminate_on_success and success["task"]):
                end_step = step_i
                break


    if video_writer is not None:
        if batched:
            for env_i in range(len(video_frames)):
                for frame in video_frames[env_i]:
                    video_writer.append_data(frame)
        else:
            for frame in video_frames:
                video_writer.append_data(frame)

    if batched:
        total_reward = np.zeros(len(env))
        rews = np.array(rews)
        for env_i in range(len(env)):
            end_step_env_i = end_step[env_i] or step_i
            total_reward[env_i] = np.sum(rews[:end_step_env_i+1, env_i])
            end_step[env_i] = end_step_env_i
        
        results["Return"] = total_reward
        results["Horizon"] = np.array(end_step) + 1
        results["Success_Rate"] = success["task"].astype(float)
    else:
        end_step = end_step or step_i
        total_reward = np.sum(rews[:end_step + 1])
        
        results["Return"] = total_reward
        results["Horizon"] = end_step + 1
        results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            if batched:
                results["{}_Success_Rate".format(k)] = success[k].astype(float)
            else:
                results["{}_Success_Rate".format(k)] = float(success[k])

    return results


def rollout_with_stats(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
        del_envs_after_rollouts=False,
        data_logger=None,
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout
    
    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...) 
            averaged across all rollouts 

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)

    if isinstance(horizon, list):
        horizon_list = horizon
    else:
        horizon_list = [horizon]

    for env, horizon in zip(envs, horizon_list):
        if env is None:
            continue

        batched = isinstance(env, SubprocVectorEnv)

        if batched:
            env_name = env.get_env_attr(key="name", id=0)[0]
        else:
            env_name = env.name

        if video_dir is not None:
            # video is written per env
            video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
            video_path = os.path.join(video_dir, "{}{}".format(env_name, video_str))
            video_writer = imageio.get_writer(video_path, fps=20)
            
        env_video_writer = None
        if write_video:
            print("video writes to " + video_path)
            env_video_writer = imageio.get_writer(video_path, fps=20)

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env_name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        if batched:
            iterator = range(0, num_episodes, len(env))
        else:
            iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=len(iterator))

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            try:
                rollout_info = run_rollout(
                    policy=policy,
                    env=env,
                    horizon=horizon,
                    render=render,
                    use_goals=use_goals,
                    video_writer=env_video_writer,
                    video_skip=video_skip,
                    terminate_on_success=terminate_on_success,
                )
            except Exception as e:
                print("Rollout exception at episode number {}!".format(ep_i))
                print(traceback.format_exc())
                break

            if batched:
                rollout_info["time"] = [(time.time() - rollout_timestamp) / len(env)] * len(env)

                for env_i in range(len(env)):
                    rollout_logs.append({k: rollout_info[k][env_i] for k in rollout_info})
                num_success += np.sum(rollout_info["Success_Rate"])
            else:
                rollout_info["time"] = time.time() - rollout_timestamp

                rollout_logs.append(rollout_info)
                num_success += rollout_info["Success_Rate"]
            
            if verbose:
                if batched:
                    raise NotImplementedError
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        if len(rollout_logs) > 0:
            rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
            rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
            rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
            all_rollout_logs[env_name] = rollout_logs_mean
        else:
            all_rollout_logs[env_name] = {"Time_Episode": -1, "Return": -1, "Success_Rate": -1, "time": -1}

        if del_envs_after_rollouts:
            # delete the environment after use
            env.close()
            del env

        if data_logger is not None:
            # summarize results from rollouts to tensorboard and terminal
            rollout_logs = all_rollout_logs[env_name]
            for k, v in rollout_logs.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                else:
                    data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

            print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
            print('Env: {}'.format(env_name))
            print(json.dumps(rollout_logs, sort_keys=True, indent=4))

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, None


def rollout_with_stats_for_SIR(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
        del_envs_after_rollouts=False,
        data_logger=None,
        SIR_predictor=None,
        SIR_config=None,
        randomize_base=False,
        inference_mode=False,
        SIR_sample_num=1,
        rollout_dir=None,
        robot_centric=False,
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout
    
    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...) 
            averaged across all rollouts 

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)

    if isinstance(horizon, list):
        horizon_list = horizon
    else:
        horizon_list = [horizon]

    # define flags
    is_act_policy = policy.policy.global_config.ALGO_NAME == "act"
    is_sir_prediction = SIR_predictor is not None

    for env, horizon in zip(envs, horizon_list):
        if env is None:
            continue

        batched = isinstance(env, SubprocVectorEnv)

        if batched:
            env_name = env.get_env_attr(key="name", id=0)[0]
        else:
            env_name = env.name

        if video_dir is not None:
            # video is written per env
            video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
            video_path = os.path.join(video_dir, "{}{}".format(env_name, video_str))
            video_writer = imageio.get_writer(video_path, fps=20)
            
        env_video_writer = None
        if write_video:
            print("video writes to " + video_path)
            env_video_writer = imageio.get_writer(video_path, fps=20)

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env_name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        if batched:
            iterator = range(0, num_episodes, len(env))
        else:
            iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=len(iterator))

        num_success = 0
        
        ##################################################
        # STEP1: main directory for saving rollout results
        capture = True
        # exp_folder = "./datasets/rollout_exp/exp1_applepnp"
        # if SIR_predictor is None:
        #     exp_folder = os.path.join(exp_folder, "rollout")
        # else:
        #     exp_folder = os.path.join(exp_folder, "sirInference")
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_folder = os.path.join(exp_folder, timestamp)
        save_folder = os.path.relpath(rollout_dir)
        os.makedirs(save_folder, exist_ok=True)

        if is_act_policy:
            seed = env.env.seed
        else:
            seed = env.env.env.seed
        
        misc_info = {
            "seed": seed,
            # "timestamp": timestamp,
            "config": policy.policy.global_config,
            "trial_count": None,    
            "success_count": None,
            "success_rate": None,
        }
        # save misc_info as json file
        meta_path = os.path.join(save_folder, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(misc_info, f, indent=4)
        
        full_data = {
            "meta": misc_info,
            "episodes": [],
        }

        ##################################################
        # STEP2: rollout loop
        success_count = 0
        detect_cam_name = "robot0_front_depth"
        for ep_i in iterator:
            rollout_timestamp = time.time()
            try:
                ##################################################
                # STEP3: reset environment
                ob_dict = env.reset()
                navi_policy = NaviPolicy(navi_mode="omni", max_vel=2.5)

                # STEP 3-1: variables and directory
                if is_act_policy:
                    easy_env = env.env
                else:
                    easy_env = env.env.env
                easy_robot = easy_env.robots[0]
                se2_origin = obs_to_SE2(ob_dict, algorithm_name=policy.policy.global_config.ALGO_NAME)
                ac = np.zeros(12)

                if render:
                    env.render(mode="human",camera_name='free')
                    env.step(ac)

                if not is_sir_prediction:
                    ##################################################
                    # CHOICE 1: rollout data collection
                    ##################################################
                    # STEP 3-2: move robot away
                    easy_env.sim.data.qpos[easy_robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(np.array([0.0, -50.0, 0.0]))
                    easy_env.sim.forward()
                    env.step(ac)
                    
                    layout = easy_env.layout_id
                    style = easy_env.style_id
                    furniture_name = easy_env.init_robot_base_pos.name
                    furniture_pos = easy_env.fixtures[furniture_name].pos

                    # STEP 3-3: capture PCL + RGBD + INTRINSIC_EXTRINSIC MATRIX
                    pcl_folder = os.path.join(save_folder, "pcl")
                    img_folder = os.path.join(save_folder, "img")
                    depth_folder = os.path.join(save_folder, "depth")
                    intrinsic_extrinsic_matrix_folder = os.path.join(save_folder, "info")
                    os.makedirs(pcl_folder, exist_ok=True)
                    os.makedirs(img_folder, exist_ok=True)
                    os.makedirs(depth_folder, exist_ok=True)
                    os.makedirs(intrinsic_extrinsic_matrix_folder, exist_ok=True)
                    pcd1 = capture_depth_camera_data(easy_env, camera_name='depth_camera1')
                    pcd2 = capture_depth_camera_data(easy_env, camera_name='depth_camera2')
                    pcd3 = capture_depth_camera_data(easy_env, camera_name='depth_camera3')
                    pcd4 = capture_depth_camera_data(easy_env, camera_name='depth_camera4')
                    pcd5 = capture_depth_camera_data(easy_env, camera_name='depth_camera5')
                    _ = capture_depth_camera_data(easy_env, camera_name=detect_cam_name, save_dir=save_folder, id=ep_i)
                    all_pcd = pcd1+pcd2+pcd3+pcd4+pcd5
                    o3d.io.write_point_cloud(os.path.join(pcl_folder, f"{ep_i}.pcd"), all_pcd)

                    # STEP 3-4: determine target_se2
                    target_helper = get_target_helper_for_rollout_collection(inference_mode=inference_mode, all_pcd=all_pcd, se2_origin=se2_origin, vis=False)
                    furniture_name = easy_env.init_robot_base_pos.name
                    furniture_pos = easy_env.fixtures[furniture_name].pos[:2]
                    target_se2 = target_helper.get_random_target_se2_with_reachability_check(furniture_pos)
                    # continue  # for debug
                    
                    # STEP 3-5: move robot to target_se2
                    if randomize_base:
                        easy_env.sim.data.qpos[easy_robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(target_se2)
                    else:
                        easy_env.sim.data.qpos[easy_robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(np.array([0.0, 0.0, 0.0]))
                    easy_env.sim.forward()
                    env.step(ac)
                    
                else:
                    ##################################################
                    # CHOICE 2: SIR module inference
                    ##################################################
                    # STEP 3-2: DETECT mode
                    arm_fake_controller(easy_env, "DETECT")
                    
                    # STEP 3-3: Initialize target helper
                    camera_height = easy_env.camera_heights[easy_env.camera_names.index(detect_cam_name)]
                    camera_width = easy_env.camera_widths[easy_env.camera_names.index(detect_cam_name)]
                    intrinsic_matrix = get_camera_intrinsic_matrix(easy_env.sim, detect_cam_name, camera_height, camera_width)
                    intrinsic_list = [intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2], camera_width, camera_height]

                    easy_env.sim.data.qpos[easy_robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(np.array([0.0, -50.0, 0.0]))
                    easy_env.sim.forward()
                    env.step(ac)

                    pcd1 = capture_depth_camera_data(easy_env, camera_name='depth_camera1')
                    pcd2 = capture_depth_camera_data(easy_env, camera_name='depth_camera2')
                    pcd3 = capture_depth_camera_data(easy_env, camera_name='depth_camera3')
                    pcd4 = capture_depth_camera_data(easy_env, camera_name='depth_camera4')
                    pcd5 = capture_depth_camera_data(easy_env, camera_name='depth_camera5')
                    all_pcd = pcd1+pcd2+pcd3+pcd4+pcd5
                    target_helper = TargetHelper(all_pcd, se2_origin, x_half_range=1.0, y_half_range=1.0, theta_half_range_deg=30, vis=False, camera_intrinsic=intrinsic_list)

                    # STEP 3-4: move robot to the initial position
                    furniture_name = easy_env.init_robot_base_pos.name
                    furniture_pos = easy_env.fixtures[furniture_name].pos[:2]
                    initial_base_se2, _ = target_helper.get_random_target_se2_with_visibility_check(furniture_pos)

                    easy_env.sim.data.qpos[easy_robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(initial_base_se2)
                    easy_env.sim.forward()
                    env.step(ac)

                    # STEP 3-5: Sample a point from the SIR region that doesn't collide with the furniture
                    point_cloud_from_initial_position = capture_depth_camera_data(easy_env, camera_name=detect_cam_name)
                    pc_numpy = np.concatenate([point_cloud_from_initial_position.points, point_cloud_from_initial_position.colors], axis=1)

                    prediction_folder = os.path.join(save_folder, "prediction")
                    os.makedirs(prediction_folder, exist_ok=True)
                    task_name = f"{easy_env.layout_id:02d}_{easy_env.style_id:02d}"
                    target_SIR_prediction, model_output_se2, model_output_means, model_output_covs, model_output_weights, model_input_pcl, SIR_prediction_success = predict_SIR_target_point(
                        SIR_predictor=SIR_predictor,
                        SIR_config=SIR_config, 
                        pc_numpy=pc_numpy,
                        target_helper=target_helper, 
                        SIR_sample_num=SIR_sample_num,
                        robot_centric=robot_centric,
                        abs_base_se2=initial_base_se2+se2_origin,
                        task_name=task_name,
                        # save_dir=save_folder,
                        # id=ep_i,
                    )
                    if SIR_prediction_success:
                        print(f"SIR prediction success. Using {target_SIR_prediction} as the target position")
                        target_SIR_relative_position = target_SIR_prediction - se2_origin
                    else:
                        target_SIR_relative_position = np.array([0.0, 0.0, 0.0])

                    # while True:
                    #     prediction_folder = os.path.join(save_folder, "prediction")
                    #     os.makedirs(prediction_folder, exist_ok=True)
                    #     layout = easy_env.layout_id
                    #     style = easy_env.style_id
                    #     task_name = f"{layout:02d}_{style:02d}"
                    #     # target_SIR_prediction, means, covs, weights = predict_SIR_target_point(SIR_predictor, SIR_config, pc_numpy, task_name=task_name, save_dir=save_folder, id=ep_i)
                    #     target_SIR_prediction, model_output_means, model_output_covs, model_output_weights, model_input_pcl = predict_SIR_target_point(SIR_predictor, SIR_config, pc_numpy, task_name=task_name)
                    #     target_SIR_relative_position = target_SIR_prediction - se2_origin
                    #     # break
                    #     if not target_helper.check_collision(target_SIR_prediction): # target_helper is based on the relative position
                    #         print("target_SIR_prediction is collided with the furniture, try again")
                    #     elif not target_helper.check_boundary(target_SIR_prediction): # target_helper is based on the relative position
                    #         print("target_SIR_prediction is out of the boundary, try again")
                    #     else:
                    #         break

                    # STEP 3-6: move robot to target_SIR_prediction
                    easy_env.sim.data.qpos[easy_robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(target_SIR_relative_position)
                    # easy_env.sim.data.qpos[easy_robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(np.array([0.0, 0.0, 0.0]))
                    easy_env.sim.forward()
                    env.step(ac)

                    arm_fake_controller(easy_env, "MANIPULATION")

                # STEP 4: wait until the robot is stable
                for _ in range(5):
                    ac = np.zeros(12)
                    ob_dict, r, done, info = env.step(ac)
                
                if is_act_policy:
                    tmp_pos = ob_dict["robot0_base_pos"].tolist()      # [x, y, z]
                    tmp_quat = ob_dict["robot0_base_quat"].tolist()    # [x, y, z, w]
                else:
                    tmp_pos = ob_dict["robot0_base_pos"][-1].tolist()      # [x, y, z]
                    tmp_quat = ob_dict["robot0_base_quat"][-1].tolist()    # [x, y, z, w]
                tmp_se2 = obs_to_SE2(ob_dict, algorithm_name=policy.policy.global_config.ALGO_NAME).tolist()

                layout = easy_env.layout_id
                style = easy_env.style_id
                furniture_name = easy_env.init_robot_base_pos.name
                furniture_pos = easy_env.fixtures[furniture_name].pos
                
                episode_data = {
                    "id": ep_i,
                    "is_success": None,
                    "pose": {
                        "pos": tmp_pos,      # [x, y, z]
                        "quat": tmp_quat,    # [x, y, z, w]
                        "se2": tmp_se2
                    },
                    "file_path": os.path.join("pcl", f"{ep_i}.pcd"),
                    "task_name": f"{layout:02d}_{style:02d}",
                    
                    "meta_info": {
                        "layout": layout,
                        "style": style,
                        "se2_origin": se2_origin,
                        "furniture_name": furniture_name,
                        "furniture_pos": furniture_pos,
                        "seed": easy_env.seed,
                        # "timestamp": timestamp,
                    }
                }

                ##################################################
                # STEP 5: call manipulation policy
                rollout_info = run_rollout(
                    policy=policy,
                    env=env,
                    horizon=horizon,
                    ob_dict=ob_dict,
                    render=render,
                    use_goals=use_goals,
                    video_writer=env_video_writer,
                    video_skip=video_skip,
                    terminate_on_success=terminate_on_success,
                )
                if is_sir_prediction and not SIR_prediction_success:
                    rollout_info['Success_Rate'] = 0.0
                
                if is_sir_prediction:
                    save_gmm_visualization_se2(
                        point_cloud=model_input_pcl,
                        target_se2=model_output_se2,
                        label=rollout_info['Success_Rate'],
                        means=model_output_means,
                        covs=model_output_covs,
                        weights=model_output_weights,
                        output_path=os.path.join(save_folder, "prediction", f"{ep_i}.ply"),
                    )
                
                # STEP 6: save rollout results
                episode_data["is_success"] = rollout_info['Success_Rate']
                full_data["episodes"].append(episode_data)
                
                full_data["meta"]["trial_count"] = ep_i + 1
                success_count += rollout_info['Success_Rate']
                full_data["meta"]["success_count"] = success_count
                full_data["meta"]["success_rate"] = success_count / (ep_i + 1)
                print(f"success: {success_count} / {ep_i + 1}")
                
                with open(meta_path, "w") as f:
                    json.dump(full_data, f, default=numpy_encoder, indent=4)

            except Exception as e:
                print("Rollout exception at episode number {}!".format(ep_i))
                print(traceback.format_exc())
                break
            
            if batched:
                rollout_info["time"] = [(time.time() - rollout_timestamp) / len(env)] * len(env)

                for env_i in range(len(env)):
                    rollout_logs.append({k: rollout_info[k][env_i] for k in rollout_info})
                num_success += np.sum(rollout_info["Success_Rate"])
            else:
                rollout_info["time"] = time.time() - rollout_timestamp

                rollout_logs.append(rollout_info)
                num_success += rollout_info["Success_Rate"]
            
            if verbose:
                if batched:
                    raise NotImplementedError
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        if len(rollout_logs) > 0:
            rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
            rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
            rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
            all_rollout_logs[env_name] = rollout_logs_mean
        else:
            all_rollout_logs[env_name] = {"Time_Episode": -1, "Return": -1, "Success_Rate": -1, "time": -1}

        if del_envs_after_rollouts:
            # delete the environment after use
            env.close()
            del env

        if data_logger is not None:
            # summarize results from rollouts to tensorboard and terminal
            rollout_logs = all_rollout_logs[env_name]
            for k, v in rollout_logs.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                else:
                    data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

            print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
            print('Env: {}'.format(env_name))
            print(json.dumps(rollout_logs, sort_keys=True, indent=4))

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, None


def should_save_from_rollout_logs(
        all_rollout_logs,
        best_return,
        best_success_rate,
        epoch_ckpt_name,
        save_on_best_rollout_return,
        save_on_best_rollout_success_rate,
    ):
    """
    Helper function used during training to determine whether checkpoints and videos
    should be saved. It will modify input attributes appropriately (such as updating
    the best returns and success rates seen and modifying the epoch ckpt name), and
    returns a dict with the updated statistics.

    Args:
        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

        epoch_ckpt_name (str): what to name the checkpoint file - this name might be modified
            by this function

        save_on_best_rollout_return (bool): if True, should save checkpoints that achieve a 
            new best rollout return

        save_on_best_rollout_success_rate (bool): if True, should save checkpoints that achieve a 
            new best rollout success rate

    Returns:
        save_info (dict): dictionary that contains updated input attributes @best_return,
            @best_success_rate, @epoch_ckpt_name, along with two additional attributes
            @should_save_ckpt (True if should save this checkpoint), and @ckpt_reason
            (string that contains the reason for saving the checkpoint)
    """
    should_save_ckpt = False
    ckpt_reason = None
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if rollout_logs["Return"] > best_return[env_name]:
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                epoch_ckpt_name += "_{}_return_{}".format(env_name, best_return[env_name])
                should_save_ckpt = True
                ckpt_reason = "return"

        if rollout_logs["Success_Rate"] > best_success_rate[env_name]:
            best_success_rate[env_name] = rollout_logs["Success_Rate"]
            if save_on_best_rollout_success_rate:
                # save checkpoint if achieve new best success rate
                epoch_ckpt_name += "_{}_success_{}".format(env_name, best_success_rate[env_name])
                should_save_ckpt = True
                ckpt_reason = "success"

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )


def save_model(model, config, env_meta, shape_meta, ckpt_path, obs_normalization_stats=None, action_normalization_stats=None):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

        action_normalization_stats (dict): TODO
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        obs_normalization_stats = deepcopy(obs_normalization_stats)
        params["obs_normalization_stats"] = TensorUtils.to_list(obs_normalization_stats)
    if action_normalization_stats is not None:
        action_normalization_stats = deepcopy(action_normalization_stats)
        params["action_normalization_stats"] = TensorUtils.to_list(action_normalization_stats)
    torch.save(params, ckpt_path)
    print("save checkpoint to {}".format(ckpt_path))


def run_epoch(model, data_loader, epoch, validate=False, num_steps=None, obs_normalization_stats=None):
    """
    Run an epoch of training or validation.

    Args:
        model (Algo instance): model to train

        data_loader (DataLoader instance): data loader that will be used to serve batches of data
            to the model

        epoch (int): epoch number

        validate (bool): whether this is a training epoch or validation epoch. This tells the model
            whether to do gradient steps or purely do forward passes.

        num_steps (int): if provided, this epoch lasts for a fixed number of batches (gradient steps),
            otherwise the epoch is a complete pass through the training dataset

        obs_normalization_stats (dict or None): if provided, this should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

    Returns:
        step_log_all (dict): dictionary of logged training metrics averaged across all batches
    """
    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        num_steps = len(data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = iter(data_loader)
    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # load next batch from data loader
        try:
            t = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            t = time.time()
            batch = next(data_loader_iter)
        timing_stats["Data_Loading"].append(time.time() - t)

        # process batch for training
        t = time.time()
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
        timing_stats["Process_Batch"].append(time.time() - t)

        # forward and backward pass
        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        timing_stats["Train_Batch"].append(time.time() - t)

        # tensorboard logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    # add in timing stats
    for k in timing_stats:
        # sum across all training steps, and convert from seconds to minutes
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.

    return step_log_all


def is_every_n_steps(interval, current_step, skip_zero=False):
    """
    Convenient function to check whether current_step is at the interval. 
    Returns True if current_step % interval == 0 and asserts a few corner cases (e.g., interval <= 0)
    
    Args:
        interval (int): target interval
        current_step (int): current step
        skip_zero (bool): whether to skip 0 (return False at 0)

    Returns:
        is_at_interval (bool): whether current_step is at the interval
    """
    if interval is None:
        return False
    assert isinstance(interval, int) and interval > 0
    assert isinstance(current_step, int) and current_step >= 0
    if skip_zero and current_step == 0:
        return False
    return current_step % interval == 0
