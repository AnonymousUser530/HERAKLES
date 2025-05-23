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

try:
    from unsloth import FastLanguageModel
    print(f"Successfully imported unsloth!")
except Exception as err:
    print("Failed to import unsloth")
# from unsloth import FastLanguageModel

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

# LL
from model import AC_model, Actor_model, Critic_model
from utils.loading_LL_model import LL_loader_separated_actor_critic, LL_loader_shared_actor_critic
from utils.low_level_updater import AWR_low_level_separated_actor_critic, AWR_low_level_shared_encoder

# Goal sampling related modules
from utils.preprocessor import get_obss_preprocessor

# Import lamorel to parallelize the computation with multiple LLM
from lamorel import Caller

# Useful fuctions
from utils.utils import format_goal, push_to_tensor
from Text_crafter_DLP import proba_subgoal_calculation

hl_elementary_actions = ["move left", "move right", "move up", "move down", "sleep", 'consume cow', 'consume plant',
                    'attack zombie', 'attack skeleton', 'attack cow', 'chop tree', 'chop bush', 'chop grass',
                    'extract stone', 'extract coal', 'extract iron', 'extract diamond',
                    'drink water', 'put stone', 'build table', 'build furnace', 'put plant',
                    'craft wood pickaxe', 'craft stone pickaxe', 'craft iron pickaxe', 'craft wood sword',
                    'craft stone sword', 'craft iron sword']

def test_hierarchical_agent_sr_from_reset_env(name_env, seed, ll_action_space_id_to_act,
                                              high_level_agent, low_level_agents, tokenizer,
                                              proba_subgoal_estimator, update_subgoal, obss_preprocessor, device,
                                              nbr_tests, rl_script_args, task_space, action_space_hl):
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_envs = min(nbr_tests, rl_script_args.number_envs * 2)
    if isinstance(rl_script_args.hl_traj_len_max, int):
        hl_traj_len_max = np.ones(num_envs) * rl_script_args.hl_traj_len_max
    else:
        hl_traj_len_max = np.ones(num_envs) * rl_script_args.hl_traj_len_max[0]

    dict_config = {"number_envs": num_envs,
                   "seed": int(1e9 * (seed + 1)),
                   "elementary_action_space":hl_elementary_actions,
                   "action_space": action_space_hl,
                   "length":hl_traj_len_max[0],
                   "proba_subgoal_estimator":proba_subgoal_estimator,
                   "hl_traj_len_max": hl_traj_len_max,
                   "long_description": rl_script_args.long_description,}

    envs = CrafterTextGoalCondition(dict_config)
    envs.set_goals([format_goal(name_env)]*num_envs)
    ep_interaction_len = np.zeros(num_envs)

    successs = []
    lengths = []
    subgoal_instructions_dict = {tsk: 0 for tsk in task_space}

    (obs_env, infos_env), ep_len = envs.reset(task_space=task_space,
                                              update_subgoal=update_subgoal,
                                              history_sr_ll_estimator=[],
                                              history_sr_hl_estimator=[]), np.zeros(num_envs)

    obs_vis = [ob["image"] for ob in obs_env]
    obs_text = [ob["textual_obs"] for ob in obs_env]
    # obs_task_enc = [ob["task_enc"] for ob in obs_env]
    lattice = [TokenLattice(tokenizer=tokenizer, valid_actions=infs["action_space"]) for infs in infos_env]

    sub_goal_instructions = [None] * num_envs
    memory_size_low_level= rl_script_args.ll_traj_len_max



    ll_traj_len_max = np.ones(num_envs) * rl_script_args.ll_traj_len_max

    while len(successs) < nbr_tests:
        ep_len += 1
        tokenized_prompts = [tokenizer.encode(_o, add_special_tokens=False) for _o in obs_text]
        prefix_fxn = f.partial(prefix_fn_global_2, lattice, tokenizer, tokenized_prompts)
        max_new_tkn = max([max(len(key) for key in latt.lattice.keys()) + 1 for latt in lattice])
        gen_output = high_level_agent.generate(contexts=obs_text,
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
        for idx_env in range(num_envs):
            if action_strs[idx_env][0] not in hl_elementary_actions:
                # the hl agent has chosen a subgoal to be done by the ll agent
                sub_goal_instructions[idx_env]=action_strs[idx_env][0].replace(" ", "_")
                subgoal_instructions_dict[sub_goal_instructions[idx_env]] += 1

        envs.update_stop_mask([False for _ in range(num_envs)])

        # d the table that register if the env is done after the execution of the action decided by high level agent
        d = [False for _ in range(num_envs)]

        # return high level agent
        return_hl = [0.0 for _ in range(num_envs)]

        # instantiate memory for the low level agent
        if device is not None:
            memory_actor_low_level = torch.zeros(num_envs, memory_size_low_level, len(ll_action_space_id_to_act), device=device)
            memory_critic_low_level = torch.zeros(num_envs, memory_size_low_level, len(ll_action_space_id_to_act), device=device)


        # len trajectory low level agent
        len_low_level = [0 for _ in range(num_envs)]

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

            preprocessed_obs_hindsight = []
            preprocessed_obs = []
            for a_j in range(num_envs):
                action_low_level_agent=None
                if sub_goal_instructions[a_j] is not None and not old_stop_mask[a_j]:
                    # memory_acts[a_j][-1][1:] to suppress the space at the beginning of the command

                    preprocessed_obs.append(obss_preprocessor([{"image": obs_vis[a_j]}], device=device))
                    with torch.no_grad():
                        if rl_script_args.actor_critic_separated:
                            try:
                                actor_model_results = low_level_agents[sub_goal_instructions[a_j]]["actor"](preprocessed_obs[-1], memory_actor_low_level[a_j].unsqueeze(0))
                            except:
                                print("subgoal: {}".format(sub_goal_instructions[a_j]))
                                raise ValueError("Error in actor model results")
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
                    preprocessed_obs.append(None)

                # print("copied old goals: {}".format([(env_idx, oi["goal"]) for env_idx, oi in enumerate(old_infos)]))
            return_o, return_r, return_d, return_infos = envs.step(actions=actions_id,
                                                                   steps_high_level_policy=ep_len,
                                                                   hl_traj_len_max=hl_traj_len_max,
                                                                   sub_goal_instructions=sub_goal_instructions,
                                                                   len_low_level=len_low_level,
                                                                   ll_traj_len_max=ll_traj_len_max,
                                                                   task_space=task_space,
                                                                   update_subgoal=update_subgoal,
                                                                   history_sr_ll_estimator=[],
                                                                   history_sr_hl_estimator=[])


            new_stop_mask = envs.get_stop_mask()

            # calculate the preprocessed obs for the next state
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
                    infos_env[idx] = new_info
                    d[idx] = new_d
                    return_hl[idx] = return_r[idx]

                    if new_d:

                        lattice[idx] = TokenLattice(tokenizer=tokenizer, valid_actions=new_info["action_space"])
                        new_stop_mask[idx] = True
                        if sub_goal_instructions[idx] is not None:
                            sub_goal_instructions[idx] = None
                            stop_subgoal[idx] = True

                    else:
                        if sub_goal_instructions[idx] is not None:
                            if return_infos[idx]["done_ll"]:
                                lattice[idx] = TokenLattice(tokenizer=tokenizer, valid_actions=new_info["action_space"])
                                new_stop_mask[idx] = True
                                sub_goal_instructions[idx] = None
                                stop_subgoal[idx] = True

                        else:
                            # just one step has been done in the env by the high level agent
                            new_stop_mask[idx] = True
                            lattice[idx] = TokenLattice(tokenizer=tokenizer, valid_actions=new_info["action_space"])

            envs.update_stop_mask(new_stop_mask)

        for i in range(num_envs):
            if d[i]:
                successs.append(1 if return_hl[i] > 0 else 0)
                lengths.append(ep_len[i])
                ep_len[i] = 0

    successs_arr = np.array(successs[:nbr_tests])
    lengths_arr = np.array(lengths[:nbr_tests])

    subgoal_instructions_dict = {k: v/len(successs) for k, v in subgoal_instructions_dict.items()}

    print("Test hierarchical agent from reset for task: {}".format(name_env))
    print("success rate: {}".format(np.average(successs_arr)))
    print("length average: {}".format(np.average(lengths_arr)))
    print("subgoal instructions usage ratio: {}".format(subgoal_instructions_dict))

    return successs, lengths, subgoal_instructions_dict

@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # Random seed
    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    task_space= config_args.rl_script_args.task_space
    print("task space: ", task_space)
    epoch_k = config_args.test_training_script.epoch_k
    print("epoch_k: ", epoch_k)
    goal_space = config_args.test_training_script.goal_space

    num_envs = config_args.rl_script_args.number_envs
    name_experiment = config_args.rl_script_args.name_experiment

    # create saving path for high level and low level agents
    if config_args.rl_script_args.name_experiment is not None:

        loading_path_high_level = f"{config_args.rl_script_args.output_dir}/{name_experiment}/high_level/seed_{config_args.rl_script_args.seed}/save_epoch_{epoch_k}"
        loading_path_low_level = f"{config_args.rl_script_args.output_dir}/{name_experiment}/low_level/seed_{config_args.rl_script_args.seed}/"+"{}"+f"/save_epoch_{epoch_k}"
        save_testing_path = f"{config_args.rl_script_args.output_dir}/{name_experiment}/epochs_{epoch_k}-{epoch_k+1}/seed_{config_args.rl_script_args.seed}"
    else:
        ValueError("The name of the experiment is not defined")

    if not os.path.exists(save_testing_path+"/history.pkl"):
        raise ValueError("The history file does not exist, for path {}".format(save_testing_path+"/history.pkl"))

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
                           WeightsLoaderInitializer(loading_path_high_level,
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

    # ========================= Prevent bug from unsloth =========================
    print("Prevent bug from unsloth")
    output = lm_server.custom_module_fns(['value'],
                                                  contexts=["None"] * config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes,
                                                  candidates=[["None"]] * config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes,
                                                  add_eos_on_candidates=config_args.rl_script_args.add_eos_on_candidates,
                                                  peft_adapter="default"
                                                  )
    # ===========================================================================


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


    print(f"Testing for epoch {epoch_k}")
    path_history = f"{config_args.rl_script_args.output_dir}/{name_experiment}/epochs_{epoch_k}-{epoch_k+1}/seed_{config_args.rl_script_args.seed}/history.pkl"
    with open(path_history, 'rb') as file:
        old_hist = pickle.load(file)
        update_subgoal = old_hist["update_subgoal"]

    # Instantiate the len max of a high level trajectory
    if isinstance(config_args.rl_script_args.hl_traj_len_max, int):
        hl_traj_len_max = np.ones(num_envs) * config_args.rl_script_args.hl_traj_len_max
    else:
        hl_traj_len_max = np.ones(num_envs) * config_args.rl_script_args.hl_traj_len_max[0]



    # create the function that will be used to estimate the probability of sampling a subgoal in the action space of the high level agent
    proba_subgoal_estimator = f.partial(proba_subgoal_calculation, lm=lm_server,
                                                                 task_space=task_space,
                                                                 rl_script_args=config_args.rl_script_args)

    # Instantiate the len max of a trajectory to inf because we want to have a manual control over it using
    # hl_traj_len_max that can be change more easily during training
    if isinstance(config_args.rl_script_args.env_max_step, str):
        if config_args.rl_script_args.env_max_step=="inf":
            env_max_step = np.inf
        else:
            ValueError("The env_max_step is not a number or inf")
    else:
        env_max_step = config_args.rl_script_args.env_max_step

    dict_config = {"number_envs": 2,
                   "seed": config_args.rl_script_args.seed,
                   "elementary_action_space":hl_elementary_actions,
                   "action_space": action_space_hl,
                   "length":env_max_step,
                   "proba_subgoal_estimator":proba_subgoal_estimator,
                   "hl_traj_len_max": hl_traj_len_max,
                   "long_description": False,}
    envs = CrafterTextGoalCondition(dict_config)
    envs.set_goals([format_goal("collect_wood")]*2)

    # Prepare for interaction with environment
    (obs, infos)= envs.reset(task_space=task_space,
                           update_subgoal=update_subgoal,
                           history_sr_ll_estimator=[],
                           history_sr_hl_estimator=[])


    # get obeservation preprocessor
    obs_space = envs._env.envs[0].observation_space
    # obs_space["instr"] = obs_space["task_enc"]
    obs_space.spaces.pop("task_enc")
    obs_space, obss_preprocessor = get_obss_preprocessor(obs_space)


    # upload the low level agents one per low level tasks
    logging.info("loading ACModels")
    memory_size_low_level = config_args.rl_script_args.ll_traj_len_max
    loading_path_low_level
    if config_args.rl_script_args.actor_critic_separated:
        low_level_agents = LL_loader_separated_actor_critic(Actor_model, Critic_model,
                                         obs_space, task_space, ll_action_space_id_to_act,
                                         memory_size_low_level,
                                         saving_path_low_level=None,
                                         config_args=config_args,
                                         loading_path_epoch_k_low_level=loading_path_low_level,)
    else: # actor_critic_shared
        low_level_agents = LL_loader_shared_actor_critic(AC_model,
                                         obs_space, task_space, ll_action_space_id_to_act,
                                         memory_size_low_level,
                                         saving_path_low_level=None,
                                         config_args=config_args,
                                         loading_path_epoch_k_low_level=loading_path_low_level)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info("loaded ACModel")


    # Prepare for constrained decoding
    tokenizer = AutoTokenizer.from_pretrained(config_args.lamorel_args.llm_configs.main_llm.model_path)
    assert tokenizer.encode(tokenizer.eos_token) != 0  # verify eos token has not the value we use for padding
    lattice = [TokenLattice(tokenizer=tokenizer, valid_actions=infs["action_space"]) for infs in infos]

    test_results = {"success_rate_hierarchical_agent_from_reset_env": dict(),
                    "length_hierarchical_agent_from_reset_env": dict(),
                    "subgoal_instructions_dict": dict()
                    }
    for name_env in goal_space:
        time_begin_task = time.time()
        print("Testing task: {}".format(name_env))
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
        test_results["success_rate_hierarchical_agent_from_reset_env"][name_env] = [successs_]
        test_results["length_hierarchical_agent_from_reset_env"][name_env] = [lengths_n]
        test_results["subgoal_instructions_dict"][name_env] = [subgoal_instructions_dict]

    # save history
    with open(f"{save_testing_path}/{config_args.test_training_script.test_name}.pkl", "wb") as file:
        pickle.dump(test_results, file)


if __name__ == '__main__':
    main()
