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
    # at the beginning action_space_hl=hl_elementary_actions
    # after enough training the action space also has subgoal for instance "collect wood"
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

        goal_sampler.load(f"{config_args.rl_script_args.output_dir}/{name_experiment}/epochs_{last_epoch}-{begin_epoch}/seed_{config_args.rl_script_args.seed}/goal_sampler.pkl")

    # Instantiate the len max of a high level trajectory
    if isinstance(config_args.rl_script_args.hl_traj_len_max, int):
        hl_traj_len_max = np.ones(num_envs) * config_args.rl_script_args.hl_traj_len_max
    else:
        hl_traj_len_max = np.ones(num_envs) * config_args.rl_script_args.hl_traj_len_max[0]
    
    ll_traj_len_max = np.zeros(num_envs)


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

    # nbr observation to train sr predictor for high level
    if config_args.rl_script_args.hl_traj_len_max >= 10:
        nbr_obs_per_traj_for_sr_hl_estimator_training = np.floor(0.1*config_args.rl_script_args.hl_traj_len_max)
    else:
        nbr_obs_per_traj_for_sr_hl_estimator_training = 1

    # create the function that will be used to estimate the probability of sampling a subgoal in the action space of the high level agent
    # Here None because we only use a HL agent
    proba_subgoal_estimator = None

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
                                                                  update_subgoal={},
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

    # Prepare for constrained decoding
    tokenizer = AutoTokenizer.from_pretrained(config_args.lamorel_args.llm_configs.main_llm.model_path)
    assert tokenizer.encode(tokenizer.eos_token) != 0  # verify eos token has not the value we use for padding
    lattice = [TokenLattice(tokenizer=tokenizer, valid_actions=infs["action_space"]) for infs in infos]

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
    # we add +1 to the max_len_tokenized_actions because we also add a eos token at the end of the action
    buffers = [
        PPOBuffer((config_args.rl_script_args.steps_per_epoch // num_envs) * (max_len_tokenized_actions + 1),
                  config_args.rl_script_args.gamma, config_args.rl_script_args.lam)
        for _ in range(num_envs)
    ]

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

            end_token_seq = []
            for gp in gen_output:
                end_token_seq_act = np.zeros(len(gp[0]['tokens']), dtype=bool)
                end_token_seq_act[-1] = True
                end_token_seq.append(end_token_seq_act)

            for idx_env in range(num_envs):
                if action_strs[idx_env][0] not in hl_elementary_actions:
                    # the hl agent has chosen a subgoal to be done by the ll agent
                    ValueError("The hl agent has chosen a subgoal to be done by the ll agent")
            """
            # for testing sub_goal_instructions=["collect_wood", "collect_wood"]
            goals = [infs["goal"] for infs in infos]
            print([(idx_env, goal, sgi) for idx_env, (sgi,goal) in enumerate(zip(sub_goal_instructions,goals))])"""

            # if stop_mask[i] == True, no step is executed in env i
            envs.update_stop_mask([False for _ in range(num_envs)])

            # d the table that register if the env is done after the execution of the action decided by high level agent
            d = [False for _ in range(num_envs)]

            # return high level agent
            return_hl = [0.0 for _ in range(num_envs)]

            # instantiate memory for the low level agent

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

                for a_j in range(num_envs):
                    if not old_stop_mask[a_j]:
                        actions_id[a_j] = action_strs[a_j][0]

                # print("copied old goals: {}".format([(env_idx, oi["goal"]) for env_idx, oi in enumerate(old_infos)]))
                return_o, return_r, return_d, return_infos = envs.step(actions=actions_id,
                                                                       steps_high_level_policy=ep_len,
                                                                       hl_traj_len_max=hl_traj_len_max,
                                                                       sub_goal_instructions=[None] * num_envs ,
                                                                       len_low_level=len_low_level,
                                                                       ll_traj_len_max=ll_traj_len_max,
                                                                       task_space=task_space,
                                                                       update_subgoal={},
                                                                       history_sr_ll_estimator=history["obs_ll_sr_estimator"],
                                                                       history_sr_hl_estimator=history["obs_hl_sr_estimator"])


                new_stop_mask = envs.get_stop_mask()
                # "return_r: {} ".format(return_r))

                # calculate the preprocessed obs for the next state
                reward_low_level_hindsight = return_r
                stop_subgoal = [False for _ in range(num_envs)]
                for idx, (new_o, new_d, new_info) in enumerate(zip(return_o, return_d, return_infos)):
                    if not old_stop_mask[idx]:
                        ep_interaction_len[idx] += 1
                        obs_text[idx] = new_o["textual_obs"]
                        obs_vis[idx] = new_o["image"]
                        obs_task_enc[idx] = new_o["task_enc"]
                        infos[idx] = new_info
                        d[idx] = new_d
                        return_hl[idx] = return_r[idx]

                        if new_d:
                            lattice[idx] = TokenLattice(tokenizer=tokenizer, valid_actions=new_info["action_space"])
                            new_stop_mask[idx] = True

                        else:
                            # just one step has been done in the env by the high level agent
                            new_stop_mask[idx] = True
                            lattice[idx] = TokenLattice(tokenizer=tokenizer, valid_actions=new_info["action_space"])

                envs.update_stop_mask(new_stop_mask)

            epoch_ended = not(steps_counter < (config_args.rl_script_args.steps_per_epoch // num_envs) and token_generated < config_args.rl_script_args.max_tokens_per_epoch)
            bootstrap_dict = {
                "ids": [],
                "contexts": []
            }

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
                    if ets:
                        token_generated += 1
                        buffers[i].store(obs_act, tact_id, return_hl[i], v, lp,
                                         possible_act=pta,
                                         end_token_seq=ets,
                                         action_str=action_strs[i][0],
                                         obs_act_p=obs_act_p,
                                         model_head=mh,
                                         value_rep=v_rep)
                    else:
                        token_generated += 1
                        buffers[i].store(obs_act, tact_id, 0, v, lp,
                                         possible_act=pta,
                                         end_token_seq=ets,
                                         action_str=action_strs[i][0],
                                         obs_act_p=obs_act_p,
                                         model_head=mh,
                                         value_rep=v_rep)
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
                        buffers[i].finish_path(0)
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


            if len(bootstrap_dict["ids"]) > 0:
                # print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                output = lm_server.custom_module_fns(
                    module_function_keys=['value'],
                    contexts=bootstrap_dict["contexts"],
                    candidates=[[" "] for _ in range(len(bootstrap_dict["contexts"]))],
                    peft_adapter="default"
                )
                for _i in range(len(output)):
                    if output[_i]["value"]["value"].squeeze().dim() == 0:
                        # if the value is a scalar
                        buffers[bootstrap_dict["ids"][_i]].finish_path(output[_i]["value"]["value"].squeeze().item())
                    else:
                        buffers[bootstrap_dict["ids"][_i]].finish_path(output[_i]["value"]["value"].squeeze()[0].item())


        test_sr_hl, test_lp_hl = test_high_level_sr_estimator(lm=lm_server)
        history["test_high_level_sr_estimator"] = test_sr_hl
        history["test_high_level_lp_estimator"] = test_lp_hl

        time_end_env_interaction = time.time()
        print("number of steps: {}, number of tokens: {}".format(steps_counter, token_generated))

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
                                                                                 low_level_agents=None,
                                                                                 tokenizer=tokenizer,
                                                                                 proba_subgoal_estimator=proba_subgoal_estimator,
                                                                                 update_subgoal={},
                                                                                 obss_preprocessor=obss_preprocessor,
                                                                                 device=None,
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
        time_update_high_level_begin = time.time()
        # Stack trajectories for all envs
        trajectories = [buf.get() for buf in buffers]

        collected_trajectories = {
            k: torch.cat([traj[k] for traj in trajectories]) if isinstance(trajectories[0][k], torch.Tensor)
            else list(f.reduce(add, [traj[k] for traj in trajectories]))
            for k, _ in trajectories[0].items()
        }
        del trajectories
        print("update high level agent")

        print("size collected_trajectories: {}".format(len(collected_trajectories['obs'])))
        ############### REMOVE TRANSITIONS WHERE THE HIGH LEVEL AGENT DID NOT DECIDE (ONLY ONE TOKEN POSSIBLE) ################
        # thought: diminish size of process batches + better use of gradient
        if config_args.rl_script_args.hl_only_useful_tokens:
            idx_to_conserve = []
            for idx_logp, _logp in enumerate(collected_trajectories['logp']):
                if _logp != 0:
                    idx_to_conserve.append(idx_logp)
            print("total token used to update HL: {}".format(len(idx_to_conserve)))
            collected_trajectories['obs'] = [collected_trajectories['obs'][idx] for idx in idx_to_conserve]
            collected_trajectories['possible_act'] = [collected_trajectories['possible_act'][idx] for idx in idx_to_conserve]
            collected_trajectories['act'] = collected_trajectories['act'][idx_to_conserve]
            collected_trajectories['ret'] = collected_trajectories['ret'][idx_to_conserve]
            collected_trajectories['adv'] = collected_trajectories['adv'][idx_to_conserve]
            collected_trajectories['logp'] = collected_trajectories['logp'][idx_to_conserve]
            collected_trajectories['val'] = collected_trajectories['val'][idx_to_conserve]
            collected_trajectories['action_str'] = [collected_trajectories['action_str'][idx] for idx in idx_to_conserve]
            collected_trajectories['end_token_seq'] = [collected_trajectories['end_token_seq'][idx] for idx in idx_to_conserve]
            collected_trajectories['rew'] = collected_trajectories['rew'][idx_to_conserve]
            collected_trajectories["obs_act_p"] = [collected_trajectories["obs_act_p"][idx] for idx in idx_to_conserve]

        print("to_remove: {}".format(
            len(collected_trajectories['obs']) % config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes))
        if len(collected_trajectories['obs']) % config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes != 0:
            to_remove = len(
                collected_trajectories['obs']) % config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes
            collected_trajectories['obs'] = collected_trajectories['obs'][:-to_remove]
            collected_trajectories['possible_act'] = collected_trajectories['possible_act'][:-to_remove]
            collected_trajectories['act'] = collected_trajectories['act'][:-to_remove]
            collected_trajectories['ret'] = collected_trajectories['ret'][:-to_remove]
            collected_trajectories['adv'] = collected_trajectories['adv'][:-to_remove]
            collected_trajectories['logp'] = collected_trajectories['logp'][:-to_remove]
            collected_trajectories['val'] = collected_trajectories['val'][:-to_remove]
            collected_trajectories['action_str'] = collected_trajectories['action_str'][:-to_remove]
            collected_trajectories['end_token_seq'] = collected_trajectories['end_token_seq'][:-to_remove]
            collected_trajectories['rew'] = collected_trajectories['rew'][:-to_remove]
            collected_trajectories["obs_act_p"] = collected_trajectories["obs_act_p"][:-to_remove]

        history["high_level_action"].extend(collected_trajectories['action_str'])
        history["actions"].extend([
            _poss_act[int(_a.item())] for _poss_act, _a in
            zip(collected_trajectories['possible_act'], collected_trajectories['act'])])
        # history["dist_high_level"].extend(collected_trajectories['dist_high_level'])
        history["ep_adv"].extend(collected_trajectories['adv'].cpu().numpy())
        history["ep_values"].extend(collected_trajectories['val'].cpu().numpy())
        history["ep_ret_with_bootstrap"].extend(collected_trajectories['ret'].cpu().numpy())
        history["ep_logp"].extend(collected_trajectories['logp'].cpu().numpy())
        history["prompts"].extend(collected_trajectories["obs_act_p"])
        # mh = torch.cat([mh.unsqueeze(0) for mh in  collected_trajectories["model_head"]]).cpu().numpy()
        # history["model_head"].extend(mh[np.random.choice(len(mh), min(len(mh), 2*len(mh[0])), replace=False)])
        # vr = torch.cat([mh.unsqueeze(0) for mh in  collected_trajectories["value_rep"]]).cpu().numpy()
        # history["value_rep"].extend(mh[np.random.choice(len(vr), min(len(vr), 2*len(vr[0])), replace=False)])

        # random shuffling to avoid correlation between trajectories
        index_shuffling = np.random.choice(len(collected_trajectories['obs']), len(collected_trajectories['obs']),
                                           replace=False)
        collected_trajectories['obs'] = [collected_trajectories['obs'][i] for i in index_shuffling]
        collected_trajectories['possible_act'] = [collected_trajectories['possible_act'][i] for i in index_shuffling]
        collected_trajectories['act'] = collected_trajectories['act'][index_shuffling]
        collected_trajectories['ret'] = collected_trajectories['ret'][index_shuffling]
        collected_trajectories['adv'] = collected_trajectories['adv'][index_shuffling]
        collected_trajectories['logp'] = collected_trajectories['logp'][index_shuffling]
        collected_trajectories['val'] = collected_trajectories['val'][index_shuffling]
        collected_trajectories['action_str'] = [collected_trajectories['action_str'][i] for i in index_shuffling]
        collected_trajectories['end_token_seq'] = [collected_trajectories['end_token_seq'][i] for i in index_shuffling]
        collected_trajectories['rew'] = collected_trajectories['rew'][index_shuffling]

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

        adv_mean, adv_std = torch.mean(collected_trajectories['adv']), torch.std(collected_trajectories['adv'])
        collected_trajectories['adv'] = (collected_trajectories['adv'] - adv_mean) / adv_std

        # save the high level agent to test him later inside the hierarchical agent
        lm_server.update([""] * config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes,
                          [[""]] * config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes,
                         saving_path_model=saving_path_high_level,
                         save_model_at_epoch_k=epoch,)

        update_results=None
        update_results = lm_server.update(collected_trajectories['obs'],
                                              collected_trajectories['possible_act'],
                                              actions=collected_trajectories['act'],
                                              returns=collected_trajectories['ret'],
                                              advantages=collected_trajectories['adv'],
                                              logprobs=collected_trajectories['logp'],
                                              values=collected_trajectories['val'],
                                              lr=config_args.rl_script_args.lr,
                                              clip_eps=config_args.rl_script_args.clip_eps,
                                              entropy_coef=config_args.rl_script_args.entropy_coef,
                                              value_loss_coef=config_args.rl_script_args.value_loss_coef,
                                              max_grad_norm=config_args.rl_script_args.max_grad_norm,
                                              ppo_epochs=config_args.rl_script_args.ppo_epochs,
                                              compute_kl_with_original_model=config_args.rl_script_args.compute_kl_with_original_model,
                                              kl_coef_with_original_model=config_args.rl_script_args.kl_coef_with_original_model,
                                              save_after_update=save_model_and_history,
                                              saving_path_model=saving_path_high_level,
                                              tokenizer=tokenizer
                                              )

        del collected_trajectories
        print("update high level agent done")
        time_update_high_level_end = time.time()

        # history["time_test_low_level"] = time_test_low_level_end - time_test_low_level_begin
        history["time_update_high_level"] = time_update_high_level_end - time_update_high_level_begin
        history["time_env_interaction"] = time_end_env_interaction - time_begin_epoch

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


        # history["new_action_added"] = new_action_added

        avg_loss = np.mean([_r['loss'] for _r in update_results], axis=0)
        avg_policy_loss = np.mean([_r['policy_loss'] for _r in update_results], axis=0)
        avg_value_loss = np.mean([_r['value_loss'] for _r in update_results], axis=0)
        avg_entropy = np.mean([_r['entropy'] for _r in update_results], axis=0)
        avg_ratio = np.mean([_r['ratio'] for _r in update_results], axis=0)
        avg_kl_penalty = np.mean([_r['kl_penalty'] for _r in update_results], axis=0)

        history["loss"].append(avg_loss)
        history["policy_loss"].append(avg_policy_loss)
        history["value_loss"].append(avg_value_loss)
        history["entropy"].append(avg_entropy)
        history["ratio"].append(avg_ratio)
        history["kl_penalty"].append(avg_kl_penalty)

        print(f"Update loss: {avg_loss}")
        print(f"Update policy loss: {avg_policy_loss}")
        print(f"Update value loss: {avg_value_loss}")
        print(f"Update entropy: {avg_entropy}")
        if config_args.rl_script_args.compute_kl_with_original_model:
            print(f"Update kl_penalty: {avg_kl_penalty}")

        print(" ")

        if save_model_and_history:
            if len(history["ll_sr_estimator_loss"])==0:
                history["ll_sr_estimator_loss"].append(np.nan)
            if len(history["MAGELLAN_loss"])==0:
                history["MAGELLAN_loss"].append(np.nan)

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
