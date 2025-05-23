# measure the sucessrate of a low level agent

from collections import OrderedDict, deque
from typing import List

import hydra
from utils.ppo_buffer import PPOBuffer
from utils.generate_prompt import generate_prompt
from utils.scoring_utils import scores_stacking
import torch
import numpy as np
import logging

from tqdm import tqdm
import time
import pickle
import math
import os
import json
import matplotlib.pyplot as plt

import gym

from babyai.babyai.levels.verifier import ObjDesc, GoToInstr
import babyai.babyai.utils as utils

@hydra.main(config_path='config', config_name='config')
def main(config_args):

    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    name_env = config_args.rl_script_args.task
    env = gym.make(name_env)
    env.seed(int(1e9 * seed))

    # upload the low level agent
    logging.info("loading ACModel")
    acmodel = utils.load_model(config_args.rl_script_args.model, raise_not_found=False)
    if acmodel is None:
        if config_args.rl_script_args.pretrained_model:
            acmodel = utils.load_model(config_args.rl_script_args.pretrained_model, raise_not_found=True)
        else:
            logging.warning("There is an issue with the loading of the low level agent")
    logging.info("loaded ACModel")
    obss_preprocessor = utils.ObssPreprocessor(config_args.rl_script_args.model,
                                               env.observation_space,
                                               config_args.rl_script_args.pretrained_model)

    sucesss = []
    rewards = []
    lengths = []
    goals = []

    for i in range(config_args.rl_script_args.nbr_tests):

        memory = torch.zeros(1, acmodel.memory_size, device="cpu")
        obs = env.reset()
        obs = obs[0]
        goals.append(obs["mission"])
        done = False
        length = 0
        while not done:

            preprocessed_obs = obss_preprocessor([obs], device="cpu")
            with torch.no_grad():
                model_results = acmodel(preprocessed_obs, memory)
                dist = model_results['dist']
                memory = model_results['memory']

            action = dist.sample()

            a = action.cpu().numpy()
            obs, reward, done, env_info = env.step(a)
            length += 1

            if done:
                rewards.append(reward)
                lengths.append(length)
                if reward > 0:
                    sucesss.append(1)
                else:
                    sucesss.append(0)

    sucesss_arr = np.array(sucesss)
    rewards_arr = np.array(rewards)
    lengths_arr = np.array(lengths)

    print("success rate: {}".format(np.average(sucesss_arr)))
    print("reward average: {}".format(np.average(rewards_arr)))
    print("length average: {}".format(np.average(lengths_arr)))
    length_success = []
    length_fail = []
    for sa, la in zip(sucesss_arr, lengths_arr):
        if sa == 1:
            length_success.append(la)
        else:
            length_fail.append(la)
    print("length success average: {}".format(np.average(length_success)))
    print("length fail average: {}".format(np.average(length_fail)))
    plt.hist(lengths_arr, bins=20)
    plt.show()

    plt.hist(length_success, bins=20)
    plt.show()

    # save the results

    if config_args.rl_script_args.save_results:
        if not os.path.exists(config_args.rl_script_args.save_results_path):
            os.makedirs(config_args.rl_script_args.save_results_path)
        with open(os.path.join(config_args.rl_script_args.save_results_path, "sucesss.pkl"), "wb") as f:
            pickle.dump(sucesss, f)
        with open(os.path.join(config_args.rl_script_args.save_results_path, "rewards.pkl"), "wb") as f:
            pickle.dump(rewards, f)
        with open(os.path.join(config_args.rl_script_args.save_results_path, "lengths.pkl"), "wb") as f:
            pickle.dump(lengths, f)
        with open(os.path.join(config_args.rl_script_args.save_results_path, "goals.pkl"), "wb") as f:
            pickle.dump(goals, f)






if __name__ == '__main__':
    main()