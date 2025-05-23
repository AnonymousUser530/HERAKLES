# This is a test file with a simple version of DLP in a BabyAI env
# We use a pretrain a low level policy on the "Go To" tasks and
# train the high level policy (LLM) to succeed in the "pick up task"
# we also train the low level agent on the successful trajectories using IQL
# We fine tune the LLM using PPO and LoRA
import os
import shutil
visible_device = str(max(0, int(os.environ.get("RANK")) - 1))
print(f"Setting visible devices to be: {visible_device}")
os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "0"


working_dir = os.getcwd() + "/rank_" + os.environ.get("RANK")
if os.path.exists(working_dir):
    shutil.rmtree(working_dir)
os.makedirs(working_dir)
os.chdir(working_dir)

os.environ["TMPDIR"] = working_dir

from unsloth import FastLanguageModel

import shutil
import time
import pickle
import copy
import hydra
import functools as f
import torch
import numpy as np
import logging

from collections import deque
from operator import add
from transformers import set_seed, AutoTokenizer

# import Env
from textcrafter_env import CrafterTextGoalCondition

# Import HL and LL related modules
# HL
from utils.HL_Updater import HLUpdater
from utils.model_heads import (LogScoringModuleFn, ValueHeadModuleFn, SR_LL_Estimator_ModuleFn, SR_HL_Estimator_ModuleFn,
                               SequentialInitializer, WeightsLoaderInitializer, PeftInitializer)
from utils.constraints import TokenLattice, prefix_fn_global_2
from utils.ppo_buffer import PPOBuffer
from tests_HL_sr_estimator import test_high_level_sr_estimator
# LL
from model import AC_model, Actor_model, Critic_model
from utils.loading_LL_model import LL_loader_separated_actor_critic, LL_loader_shared_actor_critic
from utils.low_level_updater import AWR_low_level_separated_actor_critic, AWR_low_level_shared_encoder
from utils.awr_buffer import AWRBuffer, AWRBuffer_store_trajectories
from tests_LL_sr_estimator import test_low_level_sr_estimator, test_true_sr_from_reset_env

# HIERARCHIC AGENT
from test_hierarchical_agent import test_hierarchical_agent_sr_from_reset_env

# Goal sampling related modules
from utils.goal_sampler import MALPGoalSampler, SRDiffGoalSampler, MAGELLANGoalSampler
from utils.preprocessor import get_obss_preprocessor

# Import lamorel to parallelize the computation with multiple LLM
from lamorel import Caller

# Useful fuctions
from utils.utils import format_goal, push_to_tensor

def extract_base_sr_ll_estimator_prompt(prompt):
    first_split=prompt.split("for the goal: ")
    second_split=first_split[-1].split("\n\nYour coordinates:")
    return first_split[0]+"for the goal: {}"+"\n\nYour coordinates:"+second_split[1]


def proba_subgoal_calculation(observations, update_subgoal, history_sr_ll_estimator, lm, task_space, rl_script_args):

    sigm = torch.nn.Sigmoid()
    output = lm.custom_module_fns(['sr_ll_estimator'],
                                   contexts=observations,
                                   candidates=[[" "] for _ in range(len(observations))],
                                   require_grad=False,
                                   peft_adapter='sr_LL_adapters',
                                   add_eos_on_candidates=False)
    logits_sr = torch.stack([_o['sr_ll_estimator']["sr_ll_estimated"] for _o in output])

    freq_update_per_task = {}
    for tsk in task_space:
        if len(update_subgoal[tsk])<5:
            freq_update_per_task[tsk] = 0
        else:
            freq_update_per_task[tsk] = np.mean(update_subgoal[tsk])

    proba_subgoal = {tsk: max(sigm(logit_sr.squeeze()).detach().cpu().numpy(),
                              min(freq_update_per_task[tsk],
                                  rl_script_args.nn_approximation.explo_noise)) for logit_sr, tsk in zip(logits_sr, task_space)}

    history_sr_ll_estimator.append([extract_base_sr_ll_estimator_prompt(observations[0]), sigm(logits_sr.squeeze()).detach().cpu().numpy()])

    return proba_subgoal

def reset_history():
    return {
        "ep_len": [],
        "ep_interaction_len": [],
        "ep_ret": [],
        "ep_adv": [],
        "ep_logp": [],
        "ep_values": [],
        "ep_ret_with_bootstrap": [],
        "goal": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "ratio": [],
        "kl_penalty":[],
        "high_level_action": [],
        "actions": [],
        "dist_high_level": [],
        "prompts": [],
        "model_head": [],
        "value_rep": [],
        "action_space_hl" : [],
        "ep_ret_low_level": [],
        "policy_loss_low_level": dict(),
        "value_loss_low_level": dict(),
        "ratio_low_level": dict(),
        "entropy_low_level": dict(),
        "max_weight_low_level": dict(),
        "min_weight_low_level": dict(),
        "exp_w_low_level": dict(),
        "nbr_saturated_weights_low_level": dict(),
        "success_rate_low_level": dict(),
        "success_rate_low_level_from_reset_env": dict(),
        "stop_training_ll": dict(),
        "reward_low_level": dict(),
        "length_low_level": dict(),
        "length_low_level_from_reset_env": dict(),
        "success_rate_hierarchical_agent_from_reset_env": dict(),
        "length_hierarchical_agent_from_reset_env": dict(),
        "subgoal_instructions_dict": dict(),
        "achievements": [],
        'epsilon_gs': [],
        "obs_ll_sr_estimator":[],
        "update_subgoal":dict(),
        "proba_subgoal":dict(),
        "ll_sr_estimator_loss": [],
        "MAGELLAN_loss": [],
        "obs_hl_sr_estimator":[],
    }




@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # Random seed
    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    task_space= config_args.rl_script_args.task_space
    print("task space: ", task_space)

    num_envs = config_args.rl_script_args.number_envs
    name_experiment = config_args.rl_script_args.name_experiment

    # create saving path for high level and low level agents
    if config_args.rl_script_args.name_experiment is not None:

        saving_path_high_level = f"{config_args.rl_script_args.output_dir}/{name_experiment}/high_level/seed_{config_args.rl_script_args.seed}"
        saving_path_low_level = f"{config_args.rl_script_args.output_dir}/{name_experiment}/low_level/seed_{config_args.rl_script_args.seed}"

    else:
        ValueError("The name of the experiment is not defined")

    # Create LLM agent (High level agent)
    # print("handler: {}".format(config_args.lamorel_args.llm_configs.main_llm.handler))
    lm_server = Caller(config_args.lamorel_args,
                       custom_updater={"main_llm":HLUpdater(config_args.lamorel_args.llm_configs.main_llm.model_type,
                                                  config_args.rl_script_args.minibatch_size,
                                                  config_args.rl_script_args.gradient_batch_size,
                                                  config_args.lamorel_args.llm_configs.main_llm.handler=="unsloth")},
                       custom_model_initializer={"main_llm":SequentialInitializer([
                           PeftInitializer(config_args.lamorel_args.llm_configs.main_llm.model_type,
                                           config_args.lamorel_args.llm_configs.main_llm.model_path,
                                           config_args.lamorel_args.llm_configs.main_llm.handler=="unsloth",
                                           config_args.rl_script_args.use_lora,
                                           config_args.lamorel_args.llm_configs.main_llm.load_in_4bit,
                                           config_args.rl_script_args.lora_r,
                                           config_args.rl_script_args.lora_alpha,
                                           config_args.lamorel_args.llm_configs.main_llm.pre_encode_inputs),
                           WeightsLoaderInitializer(config_args.rl_script_args.loading_path_high_level,
                                                    config_args.rl_script_args.load_state_dict_strict)
                       ])},
                       custom_module_functions={"main_llm":{
                           'score': LogScoringModuleFn(config_args.lamorel_args.llm_configs.main_llm.model_type,
                                                       config_args.lamorel_args.llm_configs.main_llm.pre_encode_inputs),
                           'value': ValueHeadModuleFn(config_args.lamorel_args.llm_configs.main_llm.model_type,
                                                      config_args.lamorel_args.llm_configs.main_llm.pre_encode_inputs),
                           'sr_ll_estimator': SR_LL_Estimator_ModuleFn(config_args.lamorel_args.llm_configs.main_llm.model_type,
                                                        config_args.lamorel_args.llm_configs.main_llm.pre_encode_inputs,
                                                        name="sr_ll_estimator", adapters="sr_LL_adapters"),
                           "sr_hl_estimator": SR_HL_Estimator_ModuleFn(config_args.lamorel_args.llm_configs.main_llm.model_type,
                                                        config_args.lamorel_args.llm_configs.main_llm.pre_encode_inputs,
                                                        name="sr_hl_estimator", adapters="sr_HL_adapters"),
                           "sr_hl_estimator_delayed": SR_HL_Estimator_ModuleFn(config_args.lamorel_args.llm_configs.main_llm.model_type,
                                                        config_args.lamorel_args.llm_configs.main_llm.pre_encode_inputs,
                                                        name="sr_hl_estimator_delayed", adapters="sr_HL_adapters_delayed"),

                       }})
    if os.path.exists(saving_path_high_level + "/last/model.checkpoint"):
        # if model.checkpoint already exists that means update =! 0 and we reload the weights of the fine-tuned model
        lm_server.update([None for _ in range(config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes)],
                         [[None] for _ in range(config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes)],
                         load_fine_tuned_version=True,
                         saving_path_model=saving_path_high_level,
                         sr_hl_estimator=True,
                         buffer_size=int(config_args.magellan_args.N/config_args.magellan_args.recompute_freq +1))
    else:
        try:
            os.makedirs(os.path.join(saving_path_high_level, 'last'))
            os.makedirs(os.path.join(saving_path_high_level, 'backup'))
        except FileExistsError:
            pass
        lm_server.update([None for _ in range(config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes)],
                         [[None] for _ in range(config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes)],
                         save_first_last=True,
                         lr=config_args.rl_script_args.lr,
                         lr_ll_sr_estimator=config_args.rl_script_args.nn_approximation.lr_ll_sr_estimator,
                         lr_hl_sr_estimator=config_args.magellan_args.lr_hl_sr_estimator,
                         saving_path_model=saving_path_high_level,
                         sr_hl_estimator=True,
                         buffer_size=int(config_args.magellan_args.N/config_args.magellan_args.recompute_freq +1))

    # ========================= Prevent bug from unsloth =========================
    # print("Prevent bug from unsloth")
    output = lm_server.custom_module_fns(['value'],
                                                 contexts=["None"] * config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes,
                                                 candidates=[["None"]] * config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes,
                                                 add_eos_on_candidates=config_args.rl_script_args.add_eos_on_candidates,
                                                 peft_adapter="default"
                                                 )
    # ===========================================================================

    # instantiate goal sampler
    if config_args.rl_script_args.goal_sampler == "MALPGoalSampler":
        goal_sampler = MALPGoalSampler([format_goal(tsk) for tsk in task_space], config_args.malp_args)
    elif config_args.rl_script_args.goal_sampler == "SRDiffGoalSampler":
        goal_sampler = SRDiffGoalSampler([format_goal(tsk) for tsk in task_space], config_args.srdiff_args)
    elif config_args.rl_script_args.goal_sampler == "MAGELLANGoalSampler":
        goal_sampler = MAGELLANGoalSampler([format_goal(tsk) for tsk in task_space], lm_server, saving_path_high_level, config_args)
    else:
        print("/!\ No goal sampler specified /!\ ")

    # action space high_level
    # instantiate with elementary actions
    # when the space of subgoal is built it can be completed with other subgoals that can be done by the low-level agent
    # the construction of the subgoal space is done inside the env at each steps
    action_space_hl = ["move left", "move right", "move up", "move down", "sleep", 'consume cow', 'consume plant',
                    'attack zombie', 'attack skeleton', 'attack cow', 'chop tree', 'chop bush', 'chop grass',
                    'extract stone', 'extract coal', 'extract iron', 'extract diamond',
                    'drink water', 'put stone', 'build table', 'build furnace', 'put plant',
                    'craft wood pickaxe', 'craft stone pickaxe', 'craft iron pickaxe', 'craft wood sword',
                    'craft stone sword', 'craft iron sword']

    # elementary action that can be done by the hl agent
    hl_elementary_actions = ["move left", "move right", "move up", "move down", "sleep", 'consume cow', 'consume plant',
                    'attack zombie', 'attack skeleton', 'attack cow', 'chop tree', 'chop bush', 'chop grass',
                    'extract stone', 'extract coal', 'extract iron', 'extract diamond',
                    'drink water', 'put stone', 'build table', 'build furnace', 'put plant',
                    'craft wood pickaxe', 'craft stone pickaxe', 'craft iron pickaxe', 'craft wood sword',
                    'craft stone sword', 'craft iron sword']

    ll_action_space_id_to_act = {
        0: "move left",
        1: "move right",
        2: "move up",
        3: "move down",
        4: "do",
        5: "sleep",
        6: 'put stone',
        7: 'build table',
        8: 'build furnace',
        9: 'put plant',
        10: 'craft wood pickaxe',
        11: 'craft stone pickaxe',
        12: 'craft iron pickaxe',
        13: 'craft wood sword',
        14: 'craft stone sword',
        15: 'craft iron sword'}

    ll_action_space_act_to_id = {"move left": 0, "move right": 1, "move up": 2, "move down": 3, "sleep": 5,
                                 "do":4, 'consume cow': 4, 'consume plant': 4,'attack zombie': 4, 'attack skeleton': 4,
                                 'attack cow': 4, 'chop tree': 4, 'chop bush': 4, 'chop grass': 4, 'extract stone': 4,
                                 'extract coal': 4, 'extract iron': 4, 'extract diamond': 4, 'drink water': 4, 'put stone': 6,
                                 'build table': 7, 'build furnace': 8, 'put plant': 9,
                                 'craft wood pickaxe': 10, 'craft stone pickaxe': 11, 'craft iron pickaxe': 12,
                                 'craft wood sword': 13, 'craft stone sword': 14, 'craft iron sword':15}

    history = reset_history()
    update_subgoal = {tsk: deque(maxlen=20) for tsk in task_space}
    # sr_history_per_task = {tsk: deque(maxlen=3) for tsk in task_space}
    # Stop training low level agent if its sr is above a threshold (we do that for efficiency reasons)
    stop_training_ll = {tsk: False for tsk in task_space}
    # we measure the success rate of the subgoal averaged over the starting state to determine if we continue training the low level agent
    success_subgoal_averaged_over_state = {tsk: deque(maxlen=50) for tsk in task_space}

    history = reset_history()
    # The parameters from where the training is resumed if necessary
    begin_epoch = 0
    next_path = f"{config_args.rl_script_args.output_dir}/{name_experiment}/epochs_{begin_epoch}-{begin_epoch + 1}/seed_{config_args.rl_script_args.seed}/history.pkl"
    if os.path.exists(next_path):
        with open(next_path, 'rb') as file:
            old_hist = pickle.load(file)
        while len(old_hist["loss"]) > 0 and begin_epoch < config_args.rl_script_args.epochs:
            begin_epoch += 1
            next_path = f"{config_args.rl_script_args.output_dir}/{name_experiment}/epochs_{begin_epoch}-{begin_epoch + 1}/seed_{config_args.rl_script_args.seed}/history.pkl"
            if os.path.exists(next_path):
                with open(next_path, 'rb') as file:
                    old_hist = pickle.load(file)
            else:
                break
    if begin_epoch > 0:
        print(f"Resuming training from epoch {begin_epoch}")
        last_epoch = begin_epoch - 1
        next_path = f"{config_args.rl_script_args.output_dir}/{name_experiment}/epochs_{last_epoch}-{begin_epoch}/seed_{config_args.rl_script_args.seed}/history.pkl"
        with open(next_path, 'rb') as file:
            old_hist = pickle.load(file)
            # action_space_hl = old_hist["action_space_hl"]
            success_subgoal_averaged_over_state = old_hist["success_subgoal_averaged_over_state"]
            update_subgoal = old_hist["update_subgoal"]
            if config_args.rl_script_args.update_low_level:
                # sr_history_per_task = old_hist["sr_history_per_task"]
                if "stop_training_ll" in old_hist.keys():
                    stop_training_ll = old_hist["stop_training_ll"]
                else:
                    stop_training_ll = {tsk: False for tsk in task_space}

        goal_sampler.load(f"{config_args.rl_script_args.output_dir}/{name_experiment}/epochs_{last_epoch}-{begin_epoch}/seed_{config_args.rl_script_args.seed}/goal_sampler.pkl")

    # Instantiate the len max of a high level trajectory
    if isinstance(config_args.rl_script_args.hl_traj_len_max, int):
        hl_traj_len_max = np.ones(num_envs) * config_args.rl_script_args.hl_traj_len_max
    else:
        hl_traj_len_max = np.ones(num_envs) * config_args.rl_script_args.hl_traj_len_max[0]
    # Instantiate the len max of a low level trajectory
    ll_traj_len_max = np.ones(num_envs) * config_args.rl_script_args.ll_traj_len_max



    # Create the buffer for the ll_sr_estimator and load it if they exist
    buffer_ll_sr_estimator = {"obs": deque(maxlen=config_args.rl_script_args.nn_approximation.memory_depth),
                              "success": deque(maxlen=config_args.rl_script_args.nn_approximation.memory_depth)}

    buffer_ll_sr_estimator_path = saving_path_low_level + "/buffer_ll_sr_estimator_last.pkl"
    if os.path.exists(buffer_ll_sr_estimator_path):
        try:
            with open(buffer_ll_sr_estimator_path, 'rb') as file:
                buffer_ll_sr_estimator = pickle.load(file)
        except:
            with open(buffer_ll_sr_estimator_path.replace("last","backup"), 'rb') as file:
                print("/!\ load the back up file for buffer_ll_sr_estimator_path/!\ ")
                buffer_ll_sr_estimator = pickle.load(file)

    # Create the buffer for the hl_sr_estimator and load it if they exist
    buffer_hl_sr_estimator = {"obs": deque(maxlen=config_args.magellan_args.memory_depth),
                                "success": deque(maxlen=config_args.magellan_args.memory_depth)}
    buffer_hl_sr_estimator_path = saving_path_high_level + "/buffer_hl_sr_estimator_last.pkl"
    if os.path.exists(buffer_hl_sr_estimator_path):
        try:
            with open(buffer_hl_sr_estimator_path, 'rb') as file:
                buffer_hl_sr_estimator = pickle.load(file)
        except:
            with open(buffer_hl_sr_estimator_path.replace("last","backup"), 'rb') as file:
                print("/!\ load the back up file for buffer_hl_sr_estimator_path/!\ ")
                buffer_hl_sr_estimator = pickle.load(file)

    if config_args.rl_script_args.ll_traj_len_max >= 10:
        nbr_obs_per_traj_for_sr_ll_estimator_training = np.floor(0.1*config_args.rl_script_args.ll_traj_len_max)
    else:
        nbr_obs_per_traj_for_sr_ll_estimator_training = 1

    if config_args.rl_script_args.hl_traj_len_max >= 10:
        nbr_obs_per_traj_for_sr_hl_estimator_training = np.floor(0.1*config_args.rl_script_args.hl_traj_len_max)
    else:
        nbr_obs_per_traj_for_sr_hl_estimator_training = 1

    # create the function that will be used to estimate the probability of sampling a subgoal in the action space of the high level agent
    proba_subgoal_estimator = f.partial(proba_subgoal_calculation, lm=lm_server,
                                                                 task_space=task_space,
                                                                 rl_script_args=config_args.rl_script_args)

    """test_low_level_sr_estimator(lm=lm_server)
    observations = ['Your task is to evaluate your success rate for the goal: collect wood',
                    'Your task is to evaluate your success rate for the goal: place table']
    output = lm_server.custom_module_fns(['sr_ll_estimator'],
                                   contexts=observations,
                                   candidates=[[" "] for _ in range(len(observations))],
                                   require_grad=False,
                                   peft_adapter='sr_LL_adapters',
                                   add_eos_on_candidates=False)"""


    # Instantiate the len max of a trajectory to inf because we want to have a manual control over it using
    # hl_traj_len_max that can be change more easily during training
    if isinstance(config_args.rl_script_args.env_max_step, str):
        if config_args.rl_script_args.env_max_step=="inf":
            env_max_step = np.inf
        else:
            ValueError("The env_max_step is not a number or inf")
    else:
        env_max_step = config_args.rl_script_args.env_max_step
    dict_config = {"number_envs": num_envs,
                   "seed": config_args.rl_script_args.seed,
                   "elementary_action_space":hl_elementary_actions,
                   "action_space": action_space_hl,
                   "length":env_max_step,
                   "goal_sampler":goal_sampler,
                   "proba_subgoal_estimator":proba_subgoal_estimator,
                   "hl_traj_len_max": hl_traj_len_max,
                   "long_description": config_args.rl_script_args.long_description,
                   "reset_word_only_at_episode_termination": config_args.rl_script_args.reset_word_only_at_episode_termination,
                   "MAGELLAN_goal_sampler":config_args.rl_script_args.goal_sampler == "MAGELLANGoalSampler",
                       }
    envs = CrafterTextGoalCondition(dict_config)

    # Prepare for interaction with environment
    (obs, infos), ep_ret, ep_len, ep_interaction_len = envs.reset(task_space=task_space,
                                                                  update_subgoal=update_subgoal,
                                                                  history_sr_ll_estimator=history["obs_ll_sr_estimator"],
                                                                  history_sr_hl_estimator=history["obs_hl_sr_estimator"]), \
        np.zeros(num_envs), \
        np.zeros(num_envs), \
        np.zeros(num_envs)

    obs_ll_sr_estimator = [None] * num_envs
    obs_hl_sr_estimator = [[] for _ in range(num_envs)]

    # you can manually fix to one *goal*
    # envs.set_goals(*goal*)

    # get obeservation preprocessor
    obs_space = envs._env.envs[0].observation_space
    # obs_space["instr"] = obs_space["task_enc"]
    obs_space.spaces.pop("task_enc")
    obs_space, obss_preprocessor = get_obss_preprocessor(obs_space)

    obs_text = [ob["textual_obs"] for ob in obs]
    obs_vis = [ob["image"] for ob in obs]
    obs_task_enc = [ob["task_enc"] for ob in obs]

    # upload the low level agents one per low level tasks
    logging.info("loading ACModels")
    memory_size_low_level = config_args.rl_script_args.ll_traj_len_max
    if config_args.rl_script_args.actor_critic_separated:
        low_level_agents = LL_loader_separated_actor_critic(Actor_model, Critic_model,
                                         obs_space, task_space, ll_action_space_id_to_act,
                                         memory_size_low_level, saving_path_low_level, config_args)
    else: # actor_critic_shared
        low_level_agents = LL_loader_shared_actor_critic(AC_model,
                                         obs_space, task_space, ll_action_space_id_to_act,
                                         memory_size_low_level, saving_path_low_level, config_args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info("loaded ACModel")

    memory_actor_hindsight = torch.zeros(num_envs, memory_size_low_level, len(ll_action_space_id_to_act), device=device)
    memory_critic_hindsight = torch.zeros(num_envs, memory_size_low_level, len(ll_action_space_id_to_act), device=device)


    sub_goal_instructions = [None] * num_envs  # the instruction passed to the low-level agent

    # Prepare for constrained decoding
    tokenizer = AutoTokenizer.from_pretrained(config_args.lamorel_args.llm_configs.main_llm.model_path)
    assert tokenizer.encode(tokenizer.eos_token) != 0  # verify eos token has not the value we use for padding
    lattice = [TokenLattice(tokenizer=tokenizer, valid_actions=infs["action_space"]) for infs in infos]
    # print("action space: ", [infs["action_space"] for infs in infos])

    # we calculate the max_len_tokenized_actions on the full HL action space
    full_action_space_hl = copy.deepcopy(action_space_hl)
    for tsk in task_space:
        parts = tsk.split('_')
        name_sg = parts[0]
        for p in parts[1:]:
            name_sg += ' '+p
        full_action_space_hl.append(name_sg)
    tokenized_actions = {_a: tokenizer.tokenize(_a) for _a in full_action_space_hl}
    max_len_tokenized_actions = max(len(_a) for _a in tokenized_actions.values())

    # Set up experience buffer for HL

    # Set up experience buffer for LL
    buffers_hindsight = {tsk: [
        AWRBuffer_store_trajectories(config_args.rl_script_args.steps_per_epoch // num_envs)
        for _ in range(num_envs)
    ] for tsk, _ in low_level_agents.items()}

    buffers_low_level_action = {tsk: [
        AWRBuffer_store_trajectories(config_args.rl_script_args.steps_per_epoch // num_envs)
        for _ in range(num_envs)
    ] for tsk, _ in low_level_agents.items()}

    # Set up updater for LL
    if config_args.rl_script_args.actor_critic_separated:
        low_level_updater = {tsk: AWR_low_level_separated_actor_critic(acm["actor"], acm["critic"],
                                                saving_path_low_level + "/{}/last/awr_buffer.pkl".format(tsk),
                                                config_args, device) for tsk, acm in low_level_agents.items()}
    else: # actor_critic_shared
        low_level_updater = {tsk: AWR_low_level_shared_encoder(acm,
                                                saving_path_low_level + "/{}/last/awr_buffer.pkl".format(tsk),
                                                config_args, device) for tsk, acm in low_level_agents.items()}

    """
    # test on env
    if config_args.rl_script_args.actor_critic_separated:
        tested_agent = low_level_agents["collect_wood"]["actor"]
    else:
        tested_agent = low_level_agents["collect_wood"]
    test_true_sr_from_reset_env("collect_wood", 0, ll_action_space_id_to_act, tested_agent,
                         obss_preprocessor, device, 1, config_args.rl_script_args)"""
    for tsk  in task_space:
        buffer_tsk_awr_path = saving_path_low_level + "/{}".format(tsk)
        low_level_updater[tsk].perform_update(saving_path_model=buffer_tsk_awr_path,
                                              lr_low_level_actor=config_args.rl_script_args.lr_low_level_actor,
                                              lr_low_level_critic=config_args.rl_script_args.lr_low_level_critic)


    trajectories_low_level = {tsk: [] for tsk, _ in low_level_agents.items()}
    nbr_transition_collected_ll_agent = {tsk: 0 for tsk, _ in low_level_agents.items()}

    for epoch in range(begin_epoch, config_args.rl_script_args.epochs):

        print(f"Epoch number {epoch}")

        # generate the saving path
        save_model_and_history = (epoch % config_args.rl_script_args.save_freq == 0 or
                                  epoch == config_args.rl_script_args.epochs - 1) and epoch != 0
        start_epoch = epoch - config_args.rl_script_args.save_freq

        saving_path = f"{config_args.rl_script_args.output_dir}/{name_experiment}/epochs_{start_epoch}-{epoch}/seed_{config_args.rl_script_args.seed}"

        if save_model_and_history:
            os.makedirs(saving_path, exist_ok=True)
            os.makedirs(saving_path_high_level, exist_ok=True)
            os.makedirs(saving_path_low_level, exist_ok=True)

        time_begin_epoch = time.time()
        steps_counter = 0
        token_generated = 0

        low_level_updated = {tsk: False for tsk, _ in low_level_agents.items()}
        low_level_nbr_update = {tsk: 0 for tsk, _ in low_level_agents.items()}
        buffer_ll_sr_estimator["obs"].append([])
        buffer_ll_sr_estimator["success"].append([])
        old_len_obs_ll_sr_estimator_buffer = 0

        buffer_hl_sr_estimator["obs"].append([])
        buffer_hl_sr_estimator["success"].append([])
        old_len_obs_hl_sr_estimator_buffer = 0
        hl_sr_estimator_updated = False

        # we stop the while loop when we reach the number of steps per epoch or if we reach the max number of tokens
        while steps_counter < (config_args.rl_script_args.steps_per_epoch // num_envs) and token_generated < config_args.rl_script_args.max_tokens_per_epoch:
            steps_counter += 1
            ep_len += 1

            # fill the hl sr estimator buffer
            for env_n in range(num_envs):
                if len(obs_hl_sr_estimator[env_n])<nbr_obs_per_traj_for_sr_hl_estimator_training:
                    obs_hl_sr_estimator[env_n].append(infos[env_n]["description_proba_goal_estimator"].format(infos[env_n]["goal"]))

            tokenized_prompts = [tokenizer.encode(_o, add_special_tokens=False) for _o in obs_text]
            prefix_fxn = f.partial(prefix_fn_global_2, lattice, tokenizer, tokenized_prompts)
            max_new_tkn = max([max(len(key) for key in latt.lattice.keys()) + 1 for latt in lattice])
            gen_output = lm_server.generate(contexts=obs_text,
                                             return_logprobs=True,
                                             num_return_sequences=1,
                                             num_beams=1,
                                             num_beam_groups=1,
                                             max_new_tokens=max_new_tkn + 1,
                                             prefix_allowed_tokens_fn=prefix_fxn,
                                             early_stopping=True,
                                             do_sample=True,
                                             top_k=0,
                                             peft_adapter="default"
                                             )

            action_strs = [[gp[0]['text']] for gp in gen_output]
            actions = [act[0] for act in action_strs]
            # print("actions_str: {}".format(actions_str))
            action_tokens = [[gp[0]['tokens'][gpt].item() for gpt in range(len(gp[0]['tokens']))] for gp in gen_output]
            obs_act_prompts = [[p + tokenizer.decode(gp[0]['tokens'][:end]) for end in range(len(gp[0]['tokens']))] for
                               gp, p in zip(gen_output, obs_text)]
            obs_act_token = [[p + gp[0]['tokens'][:end].tolist() for end in range(len(gp[0]['tokens']))] for
                             gp, p in zip(gen_output, tokenized_prompts)]

            # print("tokenized_prompts: {}".format(tokenized_prompts[0]))
            # obs_act_token = [[p for _ in range(len(gp[0]['tokens']))] for gp, p in zip(gen_output, prompts)]
            token_to_str = [[str(tk.item()) for tk in gp[0]['tokens']] for gp in gen_output]
            idx_lat = []
            for tk_str in token_to_str:
                str_tk = '('
                list_str = [str_tk]
                for tk in tk_str:
                    str_tk += tk + ','
                    list_str.append(str_tk)
                idx_lat.append(list_str)
            # print("idx_lat: {}".format(idx_lat))
            # possible_token_actions = [[[tokenizer.tokenize(tokenizer.decode(tk))[0] for tk in lattice.token_lattice[eval(_il+')')]] for _il in il[:-1]] for il in idx_lat]
            possible_token_actions = [[[tk for tk in lattice[nbr].token_lattice[eval(_il + ')')]] for _il in il[:-1]] for nbr, il
                                          in enumerate(idx_lat)]

            token_action_id = [[pat[idx_act].index(act) for idx_act, act in enumerate(at)] for at, pat in
                               zip(action_tokens, possible_token_actions)]

            output = lm_server.custom_module_fns(['value'],
                                                 contexts=obs_text,
                                                 candidates=action_strs,
                                                 add_eos_on_candidates=config_args.rl_script_args.add_eos_on_candidates,
                                                 peft_adapter="default"
                                                 )

            values = [t["value"]["value"].squeeze()[:len(gp[0]['tokens'])] for t, gp in zip(output, gen_output)]
            model_heads = [t["value"]["model_head"].squeeze()[:len(gp[0]['tokens'])] for t, gp in zip(output, gen_output)]
            value_reps = [t["value"]["value_rep"].squeeze()[:len(gp[0]['tokens'])] for t, gp in zip(output, gen_output)]
            log_probs = [gp[0]['tokens_logprob'] for gp in gen_output]

            ##########################
            """print("beginning of the test")
            from torch.distributions import Categorical
            import math
            for test_i in range(num_envs-2):
                _contexts = obs_act_token[test_i] + obs_act_token[test_i+1] + obs_act_token[test_i+2]
                # _contexts = [[1763, 1228, 5311, 1032, 21902, 6774, 1505, 2807, 29491, 1763, 1309, 1706, 25035, 7536, 1210, 29493, 1281, 3400, 29493, 1448, 9864, 3627, 29501, 5172, 10854, 29491, 781, 11927, 4406, 29515, 3955, 1066, 5486, 1065, 29473, 29552, 29549, 29491, 29502, 6712, 781, 781, 2744, 1274, 2909, 2971, 29473, 29502, 4475, 29491, 781, 781, 11927, 16402, 29515, 1093, 29538, 29518, 29493, 29538, 29518, 29499, 781, 781, 2744, 1800, 29515, 781, 29501, 10877, 29473, 29508, 4475, 1066, 1342, 8403, 781, 29501, 5486, 29473, 29555, 6712, 1066, 1342, 6888, 29501, 8239, 781, 781, 2744, 2873, 10877, 1206, 1342, 3546, 29491, 781, 781, 2744, 1274, 3279, 1065, 1342, 20698, 29491, 781, 781, 3125, 1396, 7536, 1136, 1309, 2156, 29515, 781, 29501, 3086, 2517, 781, 29501, 3086, 1871, 781, 29501, 3086, 1350, 781, 29501, 3086, 1828, 781, 29501, 5057, 781, 29501, 24095, 12840, 781, 29501, 24095, 5868, 781, 29501, 4285, 29205, 1180, 781, 29501, 4285, 12517, 9992, 781, 29501, 4285, 12840, 781, 29501, 17579, 5486, 1093, 7119, 12749, 5486, 29499, 781, 29501, 17579, 26284, 1093, 7119, 12749, 26284, 29499, 781, 29501, 17579, 10877, 1093, 7119, 12749, 10877, 29499, 781, 29501, 9899, 8021, 1093, 7119, 29473, 29508, 5536, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 8021, 29499, 781, 29501, 9899, 12640, 1093, 7119, 29473, 29508, 5536, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 12640, 29499, 781, 29501, 9899, 8843, 1093, 7119, 29473, 29508, 8021, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 8843, 29499, 781, 29501, 9899, 25389, 1093, 7119, 29473, 29508, 8843, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 25389, 29499, 781, 29501, 5431, 2898, 781, 29501, 2426, 8021, 1093, 7119, 29473, 29508, 8021, 1065, 1342, 20698, 29499, 781, 29501, 2581, 3169, 1093, 7119, 29473, 29518, 17516, 1065, 1342, 20698, 29499, 781, 29501, 2581, 9324, 1329, 1093, 7119, 29473, 29549, 18156, 1065, 1342, 20698, 29499, 781, 29501, 2426, 5868, 1093, 7119, 29473, 29508, 23829, 2673, 1065, 1342, 20698, 29499, 781, 29501, 10717, 5536, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8021, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 8021, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8843, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 12640, 29493, 29473, 29508, 8843, 1065, 1342, 20698, 2080, 12749, 1032, 9324, 1329, 29499, 781, 29501, 10717, 5536, 14211, 1093, 7119, 29473, 29508, 5536, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8021, 14211, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 8021, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8843, 14211, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 12640, 29493, 29473, 29508, 8843, 1065, 1342, 20698, 2080, 12749, 1032, 9324, 1329, 29499, 781, 781, 11927, 3760, 29515, 29473], [1763, 1228, 5311, 1032, 21902, 6774, 1505, 2807, 29491, 1763, 1309, 1706, 25035, 7536, 1210, 29493, 1281, 3400, 29493, 1448, 9864, 3627, 29501, 5172, 10854, 29491, 781, 11927, 4406, 29515, 3955, 1066, 5486, 1065, 29473, 29552, 29549, 29491, 29502, 6712, 781, 781, 2744, 1274, 2909, 2971, 29473, 29502, 4475, 29491, 781, 781, 11927, 16402, 29515, 1093, 29538, 29518, 29493, 29538, 29518, 29499, 781, 781, 2744, 1800, 29515, 781, 29501, 10877, 29473, 29508, 4475, 1066, 1342, 8403, 781, 29501, 5486, 29473, 29555, 6712, 1066, 1342, 6888, 29501, 8239, 781, 781, 2744, 2873, 10877, 1206, 1342, 3546, 29491, 781, 781, 2744, 1274, 3279, 1065, 1342, 20698, 29491, 781, 781, 3125, 1396, 7536, 1136, 1309, 2156, 29515, 781, 29501, 3086, 2517, 781, 29501, 3086, 1871, 781, 29501, 3086, 1350, 781, 29501, 3086, 1828, 781, 29501, 5057, 781, 29501, 24095, 12840, 781, 29501, 24095, 5868, 781, 29501, 4285, 29205, 1180, 781, 29501, 4285, 12517, 9992, 781, 29501, 4285, 12840, 781, 29501, 17579, 5486, 1093, 7119, 12749, 5486, 29499, 781, 29501, 17579, 26284, 1093, 7119, 12749, 26284, 29499, 781, 29501, 17579, 10877, 1093, 7119, 12749, 10877, 29499, 781, 29501, 9899, 8021, 1093, 7119, 29473, 29508, 5536, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 8021, 29499, 781, 29501, 9899, 12640, 1093, 7119, 29473, 29508, 5536, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 12640, 29499, 781, 29501, 9899, 8843, 1093, 7119, 29473, 29508, 8021, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 8843, 29499, 781, 29501, 9899, 25389, 1093, 7119, 29473, 29508, 8843, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 25389, 29499, 781, 29501, 5431, 2898, 781, 29501, 2426, 8021, 1093, 7119, 29473, 29508, 8021, 1065, 1342, 20698, 29499, 781, 29501, 2581, 3169, 1093, 7119, 29473, 29518, 17516, 1065, 1342, 20698, 29499, 781, 29501, 2581, 9324, 1329, 1093, 7119, 29473, 29549, 18156, 1065, 1342, 20698, 29499, 781, 29501, 2426, 5868, 1093, 7119, 29473, 29508, 23829, 2673, 1065, 1342, 20698, 29499, 781, 29501, 10717, 5536, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8021, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 8021, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8843, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 12640, 29493, 29473, 29508, 8843, 1065, 1342, 20698, 2080, 12749, 1032, 9324, 1329, 29499, 781, 29501, 10717, 5536, 14211, 1093, 7119, 29473, 29508, 5536, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8021, 14211, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 8021, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8843, 14211, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 12640, 29493, 29473, 29508, 8843, 1065, 1342, 20698, 2080, 12749, 1032, 9324, 1329, 29499, 781, 781, 11927, 3760, 29515, 29473, 3086], [1763, 1228, 5311, 1032, 21902, 6774, 1505, 2807, 29491, 1763, 1309, 1706, 25035, 7536, 1210, 29493, 1281, 3400, 29493, 1448, 9864, 3627, 29501, 5172, 10854, 29491, 781, 11927, 4406, 29515, 3955, 1066, 5486, 1065, 29473, 29552, 29549, 29491, 29502, 6712, 781, 781, 2744, 1274, 2909, 2971, 29473, 29502, 4475, 29491, 781, 781, 11927, 16402, 29515, 1093, 29538, 29518, 29493, 29538, 29518, 29499, 781, 781, 2744, 1800, 29515, 781, 29501, 10877, 29473, 29508, 4475, 1066, 1342, 8403, 781, 29501, 5486, 29473, 29555, 6712, 1066, 1342, 6888, 29501, 8239, 781, 781, 2744, 2873, 10877, 1206, 1342, 3546, 29491, 781, 781, 2744, 1274, 3279, 1065, 1342, 20698, 29491, 781, 781, 3125, 1396, 7536, 1136, 1309, 2156, 29515, 781, 29501, 3086, 2517, 781, 29501, 3086, 1871, 781, 29501, 3086, 1350, 781, 29501, 3086, 1828, 781, 29501, 5057, 781, 29501, 24095, 12840, 781, 29501, 24095, 5868, 781, 29501, 4285, 29205, 1180, 781, 29501, 4285, 12517, 9992, 781, 29501, 4285, 12840, 781, 29501, 17579, 5486, 1093, 7119, 12749, 5486, 29499, 781, 29501, 17579, 26284, 1093, 7119, 12749, 26284, 29499, 781, 29501, 17579, 10877, 1093, 7119, 12749, 10877, 29499, 781, 29501, 9899, 8021, 1093, 7119, 29473, 29508, 5536, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 8021, 29499, 781, 29501, 9899, 12640, 1093, 7119, 29473, 29508, 5536, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 12640, 29499, 781, 29501, 9899, 8843, 1093, 7119, 29473, 29508, 8021, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 8843, 29499, 781, 29501, 9899, 25389, 1093, 7119, 29473, 29508, 8843, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 25389, 29499, 781, 29501, 5431, 2898, 781, 29501, 2426, 8021, 1093, 7119, 29473, 29508, 8021, 1065, 1342, 20698, 29499, 781, 29501, 2581, 3169, 1093, 7119, 29473, 29518, 17516, 1065, 1342, 20698, 29499, 781, 29501, 2581, 9324, 1329, 1093, 7119, 29473, 29549, 18156, 1065, 1342, 20698, 29499, 781, 29501, 2426, 5868, 1093, 7119, 29473, 29508, 23829, 2673, 1065, 1342, 20698, 29499, 781, 29501, 10717, 5536, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8021, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 8021, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8843, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 12640, 29493, 29473, 29508, 8843, 1065, 1342, 20698, 2080, 12749, 1032, 9324, 1329, 29499, 781, 29501, 10717, 5536, 14211, 1093, 7119, 29473, 29508, 5536, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8021, 14211, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 8021, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8843, 14211, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 12640, 29493, 29473, 29508, 8843, 1065, 1342, 20698, 2080, 12749, 1032, 9324, 1329, 29499, 781, 781, 11927, 3760, 29515, 29473, 3086, 2517],[1763, 1228, 5311, 1032, 21902, 6774, 1505, 2807, 29491, 1763, 1309, 1706, 25035, 7536, 1210, 29493, 1281, 3400, 29493, 1448, 9864, 3627, 29501, 5172, 10854, 29491, 781, 11927, 4406, 29515, 3955, 1066, 5486, 1065, 29473, 29552, 29549, 29491, 29502, 6712, 781, 781, 2744, 1274, 2909, 2971, 29473, 29502, 4475, 29491, 781, 781, 11927, 16402, 29515, 1093, 29538, 29518, 29493, 29538, 29518, 29499, 781, 781, 2744, 1800, 29515, 781, 29501, 10877, 29473, 29508, 4475, 1066, 1342, 8403, 781, 29501, 5486, 29473, 29555, 6712, 1066, 1342, 6888, 29501, 8239, 781, 781, 2744, 2873, 10877, 1206, 1342, 3546, 29491, 781, 781, 2744, 1274, 3279, 1065, 1342, 20698, 29491, 781, 781, 3125, 1396, 7536, 1136, 1309, 2156, 29515, 781, 29501, 3086, 2517, 781, 29501, 3086, 1871, 781, 29501, 3086, 1350, 781, 29501, 3086, 1828, 781, 29501, 5057, 781, 29501, 24095, 12840, 781, 29501, 24095, 5868, 781, 29501, 4285, 29205, 1180, 781, 29501, 4285, 12517, 9992, 781, 29501, 4285, 12840, 781, 29501, 17579, 5486, 1093, 7119, 12749, 5486, 29499, 781, 29501, 17579, 26284, 1093, 7119, 12749, 26284, 29499, 781, 29501, 17579, 10877, 1093, 7119, 12749, 10877, 29499, 781, 29501, 9899, 8021, 1093, 7119, 29473, 29508, 5536, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 8021, 29499, 781, 29501, 9899, 12640, 1093, 7119, 29473, 29508, 5536, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 12640, 29499, 781, 29501, 9899, 8843, 1093, 7119, 29473, 29508, 8021, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 8843, 29499, 781, 29501, 9899, 25389, 1093, 7119, 29473, 29508, 8843, 3856, 1665, 29474, 1065, 1342, 20698, 2080, 12749, 25389, 29499, 781, 29501, 5431, 2898, 781, 29501, 2426, 8021, 1093, 7119, 29473, 29508, 8021, 1065, 1342, 20698, 29499, 781, 29501, 2581, 3169, 1093, 7119, 29473, 29518, 17516, 1065, 1342, 20698, 29499, 781, 29501, 2581, 9324, 1329, 1093, 7119, 29473, 29549, 18156, 1065, 1342, 20698, 29499, 781, 29501, 2426, 5868, 1093, 7119, 29473, 29508, 23829, 2673, 1065, 1342, 20698, 29499, 781, 29501, 10717, 5536, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8021, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 8021, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8843, 3856, 1665, 29474, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 12640, 29493, 29473, 29508, 8843, 1065, 1342, 20698, 2080, 12749, 1032, 9324, 1329, 29499, 781, 29501, 10717, 5536, 14211, 1093, 7119, 29473, 29508, 5536, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8021, 14211, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 8021, 1065, 1342, 20698, 2080, 12749, 1032, 3169, 29499, 781, 29501, 10717, 8843, 14211, 1093, 7119, 29473, 29508, 5536, 29493, 29473, 29508, 12640, 29493, 29473, 29508, 8843, 1065, 1342, 20698, 2080, 12749, 1032, 9324, 1329, 29499, 781, 781, 11927, 3760, 29515, 29473, 3086, 2517] ]
                # print("_contexts: {}".format(_contexts))
                _candidates = possible_token_actions[test_i] + possible_token_actions[test_i+1] + possible_token_actions[test_i+2]
                # _candidates = [[10717, 5057, 17579, 9899, 3086, 2581, 5431, 2426, 4285, 24095], [1828, 2517, 1350, 1871], [2], [2]]
                max_len = max([len(c) for c in _candidates])
                # print("_candidates: {}".format(_candidates))
                chosen_actions = token_action_id[test_i] + token_action_id[test_i+1] + token_action_id[test_i+2]
                log_probs_generated = torch.tensor(np.concatenate([log_probs[test_i], log_probs[test_i+1], log_probs[test_i+2]], axis=0))
                dict_tokens = {"{}".format(
                            tokenizer("{}".format(idx_c), add_special_tokens=False, return_token_type_ids=False)[
                                "input_ids"]):
                                           _cands for idx_c, _cands in enumerate(_candidates)}
                print("dict_tokens: {}".format(dict_tokens))
                candidates_keys = [["{}".format(i)] for i in range(len(_candidates))]
                # print("candidates_keys: {}".format(candidates_keys))
                # dict_tokens = {'[15]': [2651], '[16]': [50256]}
                output = lm_server.custom_module_fns(['score'], contexts=_contexts,
                                                     candidates=candidates_keys,
                                                     add_eos_on_candidates=False,
                                                     peft_adapter="default",
                                                     dict_tokens=dict_tokens)

                list_scores= []
                for _o in output:
                    # print("tokens_logprobs before:{}".format(_o['score']["tokens_logprobs"]))
                    while _o['score']["tokens_logprobs"].dim() != 1:
                        if _o['score']["tokens_logprobs"].dim() == 0:
                            _o['score']["tokens_logprobs"] = _o['score']["tokens_logprobs"].unsqueeze(0)
                        elif _o['score']["tokens_logprobs"].dim() > 1:
                            _o['score']["tokens_logprobs"] = _o['score']["tokens_logprobs"].squeeze()
                    # print("tokens_logprobs after:{}".format(_o['score']["tokens_logprobs"]))
                    list_scores.append(torch.cat([_o['score']["tokens_logprobs"], torch.full_like(torch.ones(1+max_len-_o['score']["tokens_logprobs"].size()[-1]), -math.inf)]))
                    # print("list_score[-1]: {}".format(list_scores[-1]))
                # print("list_scores: {}".format(list_scores))
                scores = torch.stack(list_scores)
                # print(scores)

                # scores = torch.stack([torch.cat([_o['score']["tokens_logprobs"], torch.full_like(torch.ones(1+max_len-_o['score']["tokens_logprobs"].size()[-1]), -math.inf)]) for _o in output])
                scores_over_voc = torch.stack([_o['score']["log_prob_full_distrib"] for _o in output]).squeeze()
                if scores.dim() < 3:
                    scores = scores.unsqueeze(1)
                distrib = Categorical(logits=scores)
                # Compute policy loss
                # print("token_action_id: {}".format(chosen_actions))
                # print("log_probs_generated: {}".format(log_probs_generated))
                # print("probs: {}".format(distrib.probs))
                log_prob = distrib.log_prob(torch.tensor(chosen_actions).unsqueeze(
                            -1)).squeeze()  # Use logprobs from dist as they were normalized
                ratio = torch.exp(log_prob - log_probs_generated)
                print("test_{} ratio: {}".format(test_i, ratio))
            print("end_test")"""
            ###################"

            end_token_seq = []
            for gp in gen_output:
                end_token_seq_act = np.zeros(len(gp[0]['tokens']), dtype=bool)
                end_token_seq_act[-1] = True
                end_token_seq.append(end_token_seq_act)

            for idx_env in range(num_envs):
                if action_strs[idx_env][0] not in hl_elementary_actions:
                    # the hl agent has chosen a subgoal to be done by the ll agent
                    sub_goal_instructions[idx_env]=action_strs[idx_env][0].replace(" ", "_")
            """
            # for testing sub_goal_instructions=["collect_wood", "collect_wood"]
            goals = [infs["goal"] for infs in infos]
            print([(idx_env, goal, sgi) for idx_env, (sgi,goal) in enumerate(zip(sub_goal_instructions,goals))])"""

            for idx_env in range(num_envs):
                if sub_goal_instructions[idx_env] is not None:
                    # the observation used to predict the success of the subgoal is the one before the execution of the subgoal
                    # obs_ll_sr_estimator[idx_env] = infos[idx_env]["description_proba_subgoal_estimator"].format(action_strs[idx_env][0])
                    obs_ll_sr_estimator[idx_env] = []

            # if stop_mask[i] == True, no step is executed in env i
            envs.update_stop_mask([False for _ in range(num_envs)])

            # d the table that register if the env is done after the execution of the action decided by high level agent
            d = [False for _ in range(num_envs)]

            # return high level agent
            return_hl = [0.0 for _ in range(num_envs)]

            # instantiate memory for the low level agent
            memory_actor_low_level = torch.zeros(num_envs, memory_size_low_level, len(ll_action_space_id_to_act), device=device)
            memory_critic_low_level = torch.zeros(num_envs, memory_size_low_level, len(ll_action_space_id_to_act), device=device)

            # len trajectory low level agent
            len_low_level = [0 for _ in range(num_envs)]
            old_infos = copy.deepcopy(infos)

            while not all(envs.get_stop_mask()):

                old_stop_mask = envs.get_stop_mask()
                # print("enter while loop with stop_mask: {}".format(old_stop_mask))
                # print("old_stop_mask: {}".format(old_stop_mask))
                for idx, new_stop in enumerate(old_stop_mask):
                    if not new_stop:
                        len_low_level[idx] += 1

                # instantiate the low level action ("" is just placeholder)
                actions_id = ["" for _ in range(num_envs)]

                # look for not already executed low level actions decided by the high level agent
                for a_j in range(num_envs):
                    if sub_goal_instructions[a_j] is None and not old_stop_mask[a_j]:
                        actions_id[a_j] = action_strs[a_j][0]

                # treat the subgoal decided by the high level agent
                # the low level agent used preprocessed image
                """obs_low_level = []
                obs_hindsight = []
                positions = []"""

                old_memory_actor_hindsight = memory_actor_hindsight.clone()
                old_memory_critic_hindsight = memory_critic_hindsight.clone()
                old_memory_actor = memory_actor_low_level.clone()
                old_memory_critic = memory_critic_low_level.clone()

                preprocessed_obs_hindsight = []
                preprocessed_obs = []
                for a_j in range(num_envs):
                    action_low_level_agent=None
                    if sub_goal_instructions[a_j] is not None and not old_stop_mask[a_j]:
                        # memory_acts[a_j][-1][1:] to suppress the space at the beginning of the command
                        if len(obs_ll_sr_estimator[a_j])<nbr_obs_per_traj_for_sr_ll_estimator_training:
                            obs_ll_sr_estimator[a_j].append(infos[a_j]["description_proba_subgoal_estimator"].format(sub_goal_instructions[a_j].replace("_", " ")))
                        """preprocessed_obs.append(obss_preprocessor([{"instr": task_encoder(sub_goal_instructions[a_j], ENC_ORDER),
                                              "image": obs_vis[a_j]}], device=device))"""
                        preprocessed_obs.append(obss_preprocessor([{"image": obs_vis[a_j]}], device=device))
                        with torch.no_grad():
                            if config_args.rl_script_args.actor_critic_separated:
                                actor_model_results = low_level_agents[sub_goal_instructions[a_j]]["actor"](preprocessed_obs[-1], memory_actor_low_level[a_j].unsqueeze(0))
                            else:
                                actor_model_results = low_level_agents[sub_goal_instructions[a_j]](preprocessed_obs[-1], memory_actor_low_level[a_j].unsqueeze(0))
                            action_low_level_agent = actor_model_results["dist"].sample()
                            actions_id[a_j] = ll_action_space_id_to_act[int(action_low_level_agent)]

                            # the memory is composed of the passed n actions
                            one_hot_action = torch.zeros(len(ll_action_space_id_to_act), device=device)
                            one_hot_action[int(action_low_level_agent)] = 1
                            memory_actor_low_level[a_j]= push_to_tensor(memory_actor_low_level[a_j], one_hot_action.unsqueeze(0), device=device)
                            memory_critic_low_level[a_j] = push_to_tensor(memory_critic_low_level[a_j],one_hot_action.unsqueeze(0), device=device)

                    else:
                        #  it is bad to add this example in the buffer for two reasons:
                        #  - first you have double the trajectories for hindsight without special reason for doing so
                        #  - second you generate a state (obs_ll, memory_ll) where obs_ll=obs_hl but potentially
                        #    memory_ll!=memory_hl, because memory_ll is reinitialised at each turn of the high level.
                        #    Thus, you learn for a state that you will never comme across in real settings
                        preprocessed_obs.append(None)
                    if not old_stop_mask[a_j]:
                        preprocessed_obs_hindsight.append(obss_preprocessor([{"image": obs_vis[a_j]}], device=device))
                        with torch.no_grad():
                            if sub_goal_instructions[a_j] is None :
                                one_hot_action_hindsight = torch.zeros(len(ll_action_space_id_to_act), device=device)
                                one_hot_action_hindsight[ll_action_space_act_to_id[actions_id[a_j]]] = 1
                                memory_actor_hindsight[a_j] = push_to_tensor(memory_actor_hindsight[a_j], one_hot_action_hindsight.unsqueeze(0), device=device)
                                memory_critic_hindsight[a_j] = push_to_tensor(memory_critic_hindsight[a_j],one_hot_action_hindsight.unsqueeze(0), device=device)
                            else:
                                one_hot_action_hindsight = torch.zeros(len(ll_action_space_id_to_act), device=device)
                                one_hot_action_hindsight[int(action_low_level_agent)] = 1
                                memory_actor_hindsight[a_j] = push_to_tensor(memory_actor_hindsight[a_j], one_hot_action_hindsight.unsqueeze(0), device=device)
                                memory_critic_hindsight[a_j] = push_to_tensor(memory_critic_hindsight[a_j],one_hot_action_hindsight.unsqueeze(0), device=device)


                    else:
                        preprocessed_obs_hindsight.append(None)


                # print("copied old goals: {}".format([(env_idx, oi["goal"]) for env_idx, oi in enumerate(old_infos)]))
                return_o, return_r, return_d, return_infos = envs.step(actions=actions_id,
                                                                       steps_high_level_policy=ep_len,
                                                                       hl_traj_len_max=hl_traj_len_max,
                                                                       sub_goal_instructions=sub_goal_instructions,
                                                                       len_low_level=len_low_level,
                                                                       ll_traj_len_max=ll_traj_len_max,
                                                                       task_space=task_space,
                                                                       update_subgoal=update_subgoal,
                                                                       history_sr_ll_estimator=history["obs_ll_sr_estimator"],
                                                                       history_sr_hl_estimator=history["obs_hl_sr_estimator"])


                new_stop_mask = envs.get_stop_mask()
                # "return_r: {} ".format(return_r))

                # calculate the preprocessed obs for the next state
                reward_low_level = []
                reward_low_level_hindsight = return_r
                stop_subgoal = [False for _ in range(num_envs)]
                old_subgoals = copy.deepcopy(sub_goal_instructions)
                for idx, (new_o, new_d, new_info) in enumerate(zip(return_o, return_d, return_infos)):
                    if not old_stop_mask[idx]:
                        ep_interaction_len[idx] += 1
                        obs_text[idx] = new_o["textual_obs"]
                        if sub_goal_instructions[idx] is not None and "The last action you took: " in obs_text[idx]:
                            fp, sp = obs_text[idx].split("The last action you took: ")
                            _, tp = sp.split("\n\n")
                            obs_text[idx] = fp + "The last action you took: "+action_strs[idx][0]+"\n\n"+tp
                        obs_vis[idx] = new_o["image"]
                        obs_task_enc[idx] = new_o["task_enc"]
                        infos[idx] = new_info
                        d[idx] = new_d
                        return_hl[idx] = return_r[idx]

                        if new_d:
                            lattice[idx] = TokenLattice(tokenizer=tokenizer, valid_actions=new_info["action_space"])
                            new_stop_mask[idx] = True
                            if sub_goal_instructions[idx] is not None:
                                sub_goal_instructions[idx] = None
                                stop_subgoal[idx] = True
                                reward_low_level.append(return_infos[idx]["reward_ll"])
                                if return_infos[idx]["done_ll"]:
                                    history["ep_ret_low_level"].append(return_infos[idx]["reward_ll"])
                            else:
                                reward_low_level.append(None)

                        else:
                            if sub_goal_instructions[idx] is not None:
                                reward_low_level.append(return_infos[idx]["reward_ll"])
                                if return_infos[idx]["done_ll"]:
                                    lattice[idx] = TokenLattice(tokenizer=tokenizer, valid_actions=new_info["action_space"])
                                    history["ep_ret_low_level"].append(return_infos[idx]["reward_ll"])
                                    new_stop_mask[idx] = True
                                    sub_goal_instructions[idx] = None
                                    stop_subgoal[idx] = True
                                    if old_subgoals[idx] not in history["success_rate_low_level"].keys():
                                        history["success_rate_low_level"][old_subgoals[idx]] = []
                                        history["length_low_level"][old_subgoals[idx]] = []
                                    if return_infos[idx]["reward_ll"] > 0.0:
                                        success_subgoal_averaged_over_state[old_subgoals[idx]].append(1.0)
                                        history["success_rate_low_level"][old_subgoals[idx]].append(1.0)
                                        # history["obs_ll_sr_estimator"].extend(obs_ll_sr_estimator[idx])
                                        buffer_ll_sr_estimator["obs"][-1].extend(obs_ll_sr_estimator[idx])
                                        buffer_ll_sr_estimator["success"][-1].extend([1.0]*len(obs_ll_sr_estimator[idx]))
                                        obs_ll_sr_estimator[idx]=None
                                    else:
                                        success_subgoal_averaged_over_state[old_subgoals[idx]].append(0.0)
                                        history["success_rate_low_level"][old_subgoals[idx]].append(0.0)
                                        # history["obs_ll_sr_estimator"].append(obs_ll_sr_estimator[idx])
                                        buffer_ll_sr_estimator["obs"][-1].extend(obs_ll_sr_estimator[idx])
                                        buffer_ll_sr_estimator["success"][-1].extend([0.0]*len(obs_ll_sr_estimator[idx]))
                                        obs_ll_sr_estimator[idx]=None
                                    history["length_low_level"][old_subgoals[idx]].append(len_low_level[idx])
                                    if (len(success_subgoal_averaged_over_state[old_subgoals[idx]])==success_subgoal_averaged_over_state[old_subgoals[idx]].maxlen and
                                            np.mean(success_subgoal_averaged_over_state[old_subgoals[idx]])>0.95):
                                        # the low level agent has mastered the task
                                       stop_training_ll[old_subgoals[idx]] = True
                                    else:
                                        stop_training_ll[old_subgoals[idx]] = False

                            else:
                                # just one step has been done in the env by the high level agent
                                new_stop_mask[idx] = True
                                lattice[idx] = TokenLattice(tokenizer=tokenizer, valid_actions=new_info["action_space"])
                                reward_low_level.append(None)
                    else:
                        # no low level task executed so no reward
                        reward_low_level.append(None)

                # Save transition in buffer for low level
                for idx in range(num_envs):
                    if not old_stop_mask[idx]:
                        # Hindsight transition
                        goal_hindsight = old_infos[idx]["goal"].lower().replace(" ", "_")
                        if not stop_training_ll[goal_hindsight]:
                            # do not train if the low level agent has already learned the task
                            preprocessed_obs_hindsight[idx].image = preprocessed_obs_hindsight[idx].image.squeeze(0)
                            if not config_args.rl_script_args.actor_critic_separated:
                                obs_h = [preprocessed_obs_hindsight[idx],
                                       old_memory_actor_hindsight[idx]]
                            else:
                                obs_h=[preprocessed_obs_hindsight[idx],
                                       old_memory_actor_hindsight[idx],
                                       old_memory_critic_hindsight[idx]]
                            buffers_hindsight[goal_hindsight][idx].store(obs=obs_h,
                                                         act=ll_action_space_act_to_id[actions_id[idx]],
                                                         rew=reward_low_level_hindsight[idx])

                            if return_d[idx]:
                                if reward_low_level_hindsight[idx]>0.0:
                                    buffers_hindsight[goal_hindsight][idx].finish_path(bootstrap=False)
                                else:
                                    # infinite trajectory arbitrarily stopped
                                    buffers_hindsight[goal_hindsight][idx].finish_path(bootstrap=True)
                                if buffers_hindsight[goal_hindsight][idx].ptr == buffers_hindsight[goal_hindsight][idx].max_size:
                                    trajectories_low_level[goal_hindsight].append(buffers_hindsight[goal_hindsight][idx].get())

                            elif buffers_hindsight[goal_hindsight][idx].ptr == buffers_hindsight[goal_hindsight][idx].max_size:
                                buffers_hindsight[goal_hindsight][idx].finish_path(bootstrap=True)
                                trajectories_low_level[goal_hindsight].append(buffers_hindsight[goal_hindsight][idx].get())

                        # Transition done by the low level agent
                        if sub_goal_instructions[idx] is not None or stop_subgoal[idx]:
                            # the low-level agent has a subgoal and has not reached the subgoal yet
                            # or
                            # the low-level agent has reached the subgoal/or failed
                            sub_goal = old_subgoals[idx]
                            if not stop_training_ll[sub_goal]:
                                # do not train if the low level agent has already learned the task
                                preprocessed_obs[idx].image = preprocessed_obs[idx].image.squeeze(0)
                                if not config_args.rl_script_args.actor_critic_separated:
                                    obs_sg = [preprocessed_obs[idx],
                                              old_memory_actor[idx]]
                                else:
                                    obs_sg = [preprocessed_obs[idx],
                                              old_memory_actor[idx],
                                              old_memory_critic[idx]]
                                buffers_low_level_action[sub_goal][idx].store(
                                    obs=obs_sg,
                                    act=ll_action_space_act_to_id[actions_id[idx]],
                                    rew=reward_low_level[idx])

                                if stop_subgoal[idx]:
                                    if reward_low_level[idx] > 0.0:
                                        buffers_low_level_action[sub_goal][idx].finish_path(bootstrap=False)
                                    else:
                                        # infinite trajectory arbitrarily stopped
                                        buffers_low_level_action[sub_goal][idx].finish_path(bootstrap=True)
                                    if buffers_low_level_action[sub_goal][idx].ptr == buffers_low_level_action[sub_goal][idx].max_size:
                                        trajectories_low_level[sub_goal].append(buffers_low_level_action[sub_goal][idx].get())
                                elif buffers_low_level_action[sub_goal][idx].ptr == buffers_low_level_action[sub_goal][idx].max_size:
                                    buffers_low_level_action[sub_goal][idx].finish_path(bootstrap=True)
                                    trajectories_low_level[sub_goal].append(buffers_low_level_action[sub_goal][idx].get())
                envs.update_stop_mask(new_stop_mask)

            epoch_ended = not(steps_counter < (config_args.rl_script_args.steps_per_epoch // num_envs) and token_generated < config_args.rl_script_args.max_tokens_per_epoch)
            bootstrap_dict = {
                "ids": [],
                "contexts": []
            }

            # Update Low Level Agent
            for tsk in task_space:
                len_trajectories = 0
                for t_ll in trajectories_low_level[tsk]:
                    len_trajectories += len(t_ll["act"])
                    # print("len trajectories_low_level for {} : {}".format(tsk, len_trajectories))
                if len_trajectories>=config_args.rl_script_args.update_low_level_after_n_transitions:
                    time_update_low_level_begin = time.time()
                    print("Performed update for task {} size trajectories_low_level: {}".format(tsk, len_trajectories))

                    # update the low level
                    buffer_awr = AWRBuffer(actor_critic_shared_encoder=not(config_args.rl_script_args.actor_critic_separated),
                                           use_memory=config_args.rl_script_args.memory_ll,
                                           size=config_args.rl_script_args.max_size_awr_buffer,
                                           gamma=config_args.rl_script_args.gamma_awr,
                                           lam=config_args.rl_script_args.lam_awr,
                                           device=device)
                    buffer_tsk_awr_path = saving_path_low_level + "/{}/last/awr_buffer.pkl".format(tsk)
                    if os.path.exists(buffer_tsk_awr_path):
                        buffer_awr.load(buffer_tsk_awr_path)
                    buffer_awr.store(trajectories_low_level[tsk])
                    buffer_awr.save(saving_path_low_level + "/{}/last/awr_buffer.pkl".format(tsk))
                    del buffer_awr
                    if config_args.rl_script_args.update_low_level:
                        update_low_level_results = low_level_updater[tsk].perform_update(save_after_update=save_model_and_history,
                                                                                         saving_path_model=saving_path_low_level + "/{}".format(tsk))
                        low_level_updated[tsk] = True
                        low_level_nbr_update[tsk] += 1

                        if tsk not in history["value_loss_low_level"].keys():
                            history["value_loss_low_level"][tsk] = [update_low_level_results['value_loss']]
                            history["policy_loss_low_level"][tsk] = [update_low_level_results['policy_loss']]
                            history["entropy_low_level"][tsk] = [update_low_level_results['entropy']]
                            if config_args.rl_script_args.normalisation_coef_awr:
                                history["max_weight_low_level"][tsk] = [update_low_level_results['max_weight']]
                                history["min_weight_low_level"][tsk] = [update_low_level_results['min_weight']]
                            else:
                                history["exp_w_low_level"][tsk] = [update_low_level_results['exp_w']]
                                history["nbr_saturated_weights_low_level"][tsk] = [update_low_level_results['nbr_saturated_weights']]
                        else:
                            history["value_loss_low_level"][tsk].append(update_low_level_results['value_loss'])
                            history["policy_loss_low_level"][tsk].append(update_low_level_results['policy_loss'])
                            history["entropy_low_level"][tsk].append(update_low_level_results['entropy'])
                            if config_args.rl_script_args.normalisation_coef_awr:
                                history["max_weight_low_level"][tsk].append(update_low_level_results['max_weight'])
                                history["min_weight_low_level"][tsk].append(update_low_level_results['min_weight'])
                            else:
                                history["exp_w_low_level"][tsk].append(update_low_level_results['exp_w'])
                                history["nbr_saturated_weights_low_level"][tsk].append(update_low_level_results['nbr_saturated_weights'])

                        print("update low level agent for task {}".format(tsk))
                        print("value_loss_low_level: {}".format(update_low_level_results['value_loss']))
                        print("policy_loss_low_level: {}".format(update_low_level_results['policy_loss']))
                        print("entropy_low_level: {}".format(update_low_level_results['entropy']))
                        if config_args.rl_script_args.normalisation_coef_awr:
                            print("max_weight_low_level: {}".format(update_low_level_results['max_weight']))
                            print("min_weight_low_level: {}".format(update_low_level_results['min_weight']))
                        else:
                            print("exp_w_low_level: {}".format(update_low_level_results['exp_w']))
                            print("nbr_saturated_weights_low_level: {}".format(update_low_level_results['nbr_saturated_weights']))
                    nbr_transition_collected_ll_agent[tsk] += len_trajectories
                    trajectories_low_level[tsk] = []
                    time_update_low_level_end = time.time()
                    history["time_update_low_level"] = time_update_low_level_end - time_update_low_level_begin

            history["nbr_transition_collected_ll_agent"] = nbr_transition_collected_ll_agent

            # update ll_sr_estimator if enough new samples
            buffer_ll_sr_estimator_contain_obs = False
            for obs_ll_sr in buffer_ll_sr_estimator["obs"]:
                if len(obs_ll_sr)>0:
                    buffer_ll_sr_estimator_contain_obs = True
                    break
            if buffer_ll_sr_estimator_contain_obs and len(buffer_ll_sr_estimator["obs"][-1])-old_len_obs_ll_sr_estimator_buffer>config_args.rl_script_args.nn_approximation.update_ll_sr_estimator_after_n_transitions:
                old_len_obs_ll_sr_estimator_buffer = len(buffer_ll_sr_estimator["obs"][-1])
                # save the buffer
                buffer_ll_sr_estimator_path = saving_path_low_level + "/buffer_ll_sr_estimator_last.pkl"
                if os.path.exists(buffer_ll_sr_estimator_path):
                    shutil.copyfile(buffer_ll_sr_estimator_path, buffer_ll_sr_estimator_path.replace("last","backup"))
                with open(buffer_ll_sr_estimator_path, 'wb') as file:
                    pickle.dump(buffer_ll_sr_estimator, file)

                weights = []
                contexts_list = []
                candidates_list = []
                for i in range(len(buffer_ll_sr_estimator["obs"])):
                    contexts_list.extend(buffer_ll_sr_estimator["obs"][i])
                    candidates_list.extend(buffer_ll_sr_estimator["success"][i])
                    weights.extend([i+1] * len(buffer_ll_sr_estimator["obs"][i]))
                weights = np.array(weights)
                # shuffling the data
                index_shuffling = np.random.choice(len(contexts_list), len(contexts_list),
                                           replace=False)
                contexts_list = [contexts_list[i] for i in index_shuffling]
                candidates_list = [[candidates_list[i]] for i in index_shuffling]
                weights = weights[index_shuffling]
                # update the estimator
                print("update ll_sr_estimator with {} samples".format(len(contexts_list)))
                update_results_ll_sr_estimator = lm_server.update(contexts_list,
                                                                  candidates_list,
                                                                  sr_ll_estimator_update=True,
                                                                  batch_size=config_args.rl_script_args.nn_approximation.batch_size_ll_sr_estimator,
                                                                  weights=weights,
                                                                  sr_estimator_epochs=config_args.rl_script_args.nn_approximation.epochs_ll_sr_estimator)

                history["ll_sr_estimator_loss"].extend(update_results_ll_sr_estimator[0]['loss'])

             # update hl_sr_estimator if enough new samples
            buffer_hl_sr_estimator_contain_obs = False
            for obs_hl_sr in buffer_hl_sr_estimator["obs"]:
                if len(obs_hl_sr)>0:
                    buffer_hl_sr_estimator_contain_obs = True
                    break
            if buffer_hl_sr_estimator_contain_obs and len(buffer_hl_sr_estimator["obs"][-1])-old_len_obs_hl_sr_estimator_buffer>config_args.magellan_args.update_hl_sr_estimator_after_n_transitions:
                old_len_obs_hl_sr_estimator_buffer = len(buffer_hl_sr_estimator["obs"][-1])
                buffer_hl_sr_estimator_path = saving_path_high_level + "/buffer_hl_sr_estimator_last.pkl"
                if os.path.exists(buffer_hl_sr_estimator_path):
                    shutil.copyfile(buffer_hl_sr_estimator_path, buffer_hl_sr_estimator_path.replace("last","backup"))
                with open(buffer_hl_sr_estimator_path, 'wb') as file:
                    pickle.dump(buffer_hl_sr_estimator, file)
                # update the goal sampler
                weights = []
                contexts_list = []
                candidates_list = []
                for i in range(len(buffer_hl_sr_estimator["obs"])):
                    contexts_list.extend(buffer_hl_sr_estimator["obs"][i])
                    candidates_list.extend(buffer_hl_sr_estimator["success"][i])
                    weights.extend([i+1] * len(buffer_hl_sr_estimator["obs"][i]))
                weights = np.array(weights)
                # shuffling the data
                index_shuffling = np.random.choice(len(contexts_list), len(contexts_list),
                                            replace=False)
                contexts_list = [contexts_list[i] for i in index_shuffling]
                candidates_list = [[candidates_list[i]] for i in index_shuffling]
                weights = weights[index_shuffling]
                # update the estimator
                print("update hl_sr_estimator with {} samples".format(len(contexts_list)))
                update_results_hl_sr_estimator = lm_server.update(contexts_list,
                                                                  candidates_list,
                                                                  sr_hl_estimator_update=True,
                                                                  batch_size=config_args.magellan_args.batch_size_hl_sr_estimator,
                                                                  weights=weights,
                                                                  sr_estimator_epochs=config_args.magellan_args.epochs_hl_sr_estimator)
                history["MAGELLAN_loss"].extend(update_results_hl_sr_estimator[0]['loss'])
                print("hl_sr_estimator loss: {}".format(update_results_hl_sr_estimator[0]['loss']))
                hl_sr_estimator_updated = True

            for i in range(num_envs):
                for obs_act, tact_id, v, mh, v_rep, lp, pta, ets, obs_act_p in zip(obs_act_token[i], token_action_id[i],
                                                                        values[i], model_heads[i], value_reps[i],
                                                                        log_probs[i], possible_token_actions[i],
                                                                        end_token_seq[i],
                                                                        obs_act_prompts[i]):
                    token_generated += 1

                ep_ret[i] += return_hl[i]
                timeout = ep_len[i] == config_args.rl_script_args.max_ep_len
                terminal = d[i] or timeout

                if terminal or epoch_ended:
                    if not terminal:
                        bootstrap_dict["ids"].append(i)
                        bootstrap_dict["contexts"].append(obs_text[i])
                    else:
                        if ep_ret[i]>0.0:
                            # the task have been achieved
                            # print("B env number:{} previous goal {} with reward {} previous goal (old goal):{}".format(i, old_infos[i]["goal"].replace(" ", "_").lower(), ep_ret[i], infos[i]["old_goals"]))
                            g = old_infos[i]["goal"].replace(" ", "_").lower()
                            old_infos[i]["achievements"][g] += 1
                        history["achievements"].append(old_infos[i]["achievements"])
                        history["goal"].append(old_infos[i]["goal"])
                        history["ep_len"].append(ep_len[i])
                        history["ep_interaction_len"].append(ep_interaction_len[i])
                        history["ep_ret"].append(ep_ret[i])

                        buffer_hl_sr_estimator["obs"][-1].extend(obs_hl_sr_estimator[i])
                        if ep_ret[i]>0.0:
                            buffer_hl_sr_estimator["success"][-1].extend([1.0]*len(obs_hl_sr_estimator[i]))
                        else:
                            buffer_hl_sr_estimator["success"][-1].extend([0.0]*len(obs_hl_sr_estimator[i]))
                        obs_hl_sr_estimator[i] = []
                        ep_len[i], ep_interaction_len[i], ep_ret[i] = 0, 0, 0
                        memory_actor_hindsight[i] = torch.zeros(memory_size_low_level, len(ll_action_space_id_to_act), device=device)
                        memory_critic_hindsight[i] = torch.zeros(memory_size_low_level, len(ll_action_space_id_to_act), device=device)


        history["test_low_level_sr_estimator"] = test_low_level_sr_estimator(lm=lm_server)
        test_sr_hl, test_lp_hl = test_high_level_sr_estimator(lm=lm_server)
        history["test_high_level_sr_estimator"] = test_sr_hl
        history["test_high_level_lp_estimator"] = test_lp_hl

        time_end_env_interaction = time.time()
        print("number of steps: {}, number of tokens: {}".format(steps_counter, token_generated))

        # Verification of the success rate of the low level agent when starting from reset env
        if save_model_and_history:
            if config_args.rl_script_args.update_low_level and config_args.rl_script_args.test_low_level:
                print("Test low level agent")
                time_test_ll_beginning = time.time()
                for name_env in task_space:
                    if low_level_updated[name_env]: #and "go_to" not in name_env:
                        # test the low level only if updated
                        if config_args.rl_script_args.actor_critic_separated:
                            tested_agent = low_level_agents[name_env]["actor"]
                        else:
                            tested_agent = low_level_agents[name_env]
                        successs_, rewards_n, lengths_n = test_true_sr_from_reset_env(name_env=name_env,
                                                                                      seed=seed,
                                                                                      ll_action_space_id_to_act=ll_action_space_id_to_act,
                                                                                      low_level_agent=tested_agent,
                                                                                      obss_preprocessor=obss_preprocessor,
                                                                                      device=device,
                                                                                      nbr_tests=config_args.rl_script_args.nbr_tests_low_level,
                                                                                      rl_script_args=config_args.rl_script_args)

                        if name_env not in history["success_rate_low_level_from_reset_env"].keys():
                            history["success_rate_low_level_from_reset_env"][name_env] = successs_
                            history["length_low_level_from_reset_env"][name_env] = lengths_n
                time_test_ll_end = time.time()
                print("time to test low level agent: {}\n".format(time_test_ll_end - time_test_ll_beginning))

        for tsk in task_space:
            update_subgoal[tsk].append(low_level_nbr_update[tsk])


        history["success_subgoal_averaged_over_state"] = success_subgoal_averaged_over_state
        history["update_subgoal"] = update_subgoal

        """
        history["proba_subgoal"] = proba_subgoal_calculation(success_subgoal, update_subgoal, task_space, config_args.rl_script_args)
        """

        for tsk in task_space:
            if len(success_subgoal_averaged_over_state[tsk])>1:
                sr_tsk = np.mean(success_subgoal_averaged_over_state[tsk])
            else:
                sr_tsk = np.nan
            if len(update_subgoal[tsk])>1:
                upd_freq_tsk = np.mean(update_subgoal[tsk])
            else:
                upd_freq_tsk = np.nan

            print("task {}: sr {} (len_success {}) update_frequency {} (len_update {})".format(tsk, sr_tsk,
                                                                                              len(success_subgoal_averaged_over_state[tsk]),
                                                                                              upd_freq_tsk,
                                                                                              len(update_subgoal[tsk])))

        # history["action_space_hl"]= action_space_hl
        history["stop_training_ll"] = stop_training_ll


        # Saving LL agent  to test him later inside the hierarchical agent
        for tsk in task_space:
            low_level_updater[tsk].perform_update(save_models_at_epoch_k=epoch,
                                                  saving_path_model=saving_path_low_level + "/{}".format(tsk))


        # Test the hierarchical agent on task in reset environment
        """
        if (epoch % config_args.rl_script_args.test_hierarchical_agent_freq == 0 or epoch == config_args.rl_script_args.epochs - 1):
            time_test_hierarchic_agent_beginning = time.time()
            for name_env in task_space:
                time_begin_task = time.time()
                successs_, lengths_n, subgoal_instructions_dict = test_hierarchical_agent_sr_from_reset_env(name_env=name_env,
                                                                                 seed=seed,
                                                                                 ll_action_space_id_to_act=ll_action_space_id_to_act,
                                                                                 high_level_agent=lm_server,
                                                                                 low_level_agents=low_level_agents,
                                                                                 tokenizer=tokenizer,
                                                                                 proba_subgoal_estimator=proba_subgoal_estimator,
                                                                                 update_subgoal=update_subgoal,
                                                                                 obss_preprocessor=obss_preprocessor,
                                                                                 device=device,
                                                                                 nbr_tests=config_args.rl_script_args.nbr_tests_low_level,
                                                                                 rl_script_args=config_args.rl_script_args,
                                                                                 task_space=task_space,
                                                                                 action_space_hl=action_space_hl)
                time_end_task = time.time()
                print("time to test hierarchical agent on task {}: {}".format(name_env, time_end_task - time_begin_task))

                if name_env not in history["success_rate_hierarchical_agent_from_reset_env"].keys():
                    history["success_rate_hierarchical_agent_from_reset_env"][name_env] = [successs_]
                    history["length_hierarchical_agent_from_reset_env"][name_env] = [lengths_n]
                    history["subgoal_instructions_dict"][name_env] = [subgoal_instructions_dict]
                else:
                    history["success_rate_hierarchical_agent_from_reset_env"][name_env].append(successs_)
                    history["length_hierarchical_agent_from_reset_env"][name_env].append(lengths_n)
                    history["subgoal_instructions_dict"][name_env].append(subgoal_instructions_dict)

            time_test_hierarchic_agent_end = time.time()
            print("time to test hierarchical agent: {}\n".format(time_test_hierarchic_agent_end - time_test_hierarchic_agent_beginning))
            """

        # Update LLM

        sucess_rate = np.mean([1 if _ret > 0 else 0 for _ret in history["ep_ret"]])
        print(f"Success rate HL: {sucess_rate}")

        for tsk in task_space:
            sr_task = []
            for achievements, goal in zip(history["achievements"],history["goal"]):
                if tsk==goal.replace(" ", "_").lower():
                    if achievements[tsk]>0.0:
                        sr_task.append(1)
                    else:
                        sr_task.append(0)
            print("Success rate HL for {} is {} with {} traj completed :{}".format(tsk, np.mean(sr_task), len(sr_task), sr_task))

        print("goal sampler epsilon: {} ".format(envs._env.goal_sampler.epsilon))
        history['epsilon_gs'].append(envs._env.goal_sampler.epsilon)

        history["time_env_interaction"] = time_end_env_interaction - time_begin_epoch

        # save the high level agent to test him later inside the hierarchical agent
        lm_server.update([""] * config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes,
                          [[""]] * config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes,
                         saving_path_model=saving_path_high_level,
                         save_model_at_epoch_k=epoch)


        # update the estimator
        if hl_sr_estimator_updated:
            if epoch>begin_epoch:
                # Update the delayed sr estimator
                goal_sampler.update(saving_path_buffer_sr_hl_estimator_weights=saving_path_high_level)
                lm_server.update([""] * config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes,
                                 [[""]] * config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes,
                                 set_weights_hl_sr_estimator=True,
                                 )
        else:
            print("no samples to update hl_sr_estimator")
            history["MAGELLAN_loss"].append(np.nan)


        if save_model_and_history:
            if len(history["ll_sr_estimator_loss"])==0:
                history["ll_sr_estimator_loss"].append(np.nan)
            if len(history["MAGELLAN_loss"])==0:
                history["MAGELLAN_loss"].append(np.nan)
            # Save history
            nbr_transition_collected_ll_agent = {tsk: 0 for tsk, _ in low_level_agents.items()}
            with open(f"{saving_path}/history.pkl", "wb") as file:
                pickle.dump(history, file)
            history = reset_history()
            # save goal sampler
            goal_sampler.save(saving_path)

    start_epoch = epoch - config_args.rl_script_args.save_freq
    saving_path = f"{config_args.rl_script_args.output_dir}/{name_experiment}/epochs_{start_epoch}-{epoch}/seed_{config_args.rl_script_args.seed}"
    os.makedirs(saving_path, exist_ok=True)
    
    # save history
    with open(f"{saving_path}/history.pkl", "wb") as file:
        pickle.dump(history, file)

if __name__ == '__main__':
    main()
