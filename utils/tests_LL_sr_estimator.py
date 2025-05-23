import torch
import numpy as np

from DLP.textcrafter_env import CrafterTextGoalCondition
from utils import format_goal, push_to_tensor

TESTS_COLLECT_WOOD = ["You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: collect wood\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYour success rate is: ",
                      "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: collect wood\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYour success rate is: ",
                      "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: collect wood\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 1\n\nYour success rate is: ",
                      "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: collect wood\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 1\n\nYour success rate is: "
                      ]

TESTS_PLACE_TABLE = ["You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: place table\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYour success rate is: ",
                     "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: place table\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYour success rate is: ",
                     "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: place table\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 1\n\nYour success rate is: ",
                     "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: place table\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 1\n\nYour success rate is: ",
                     "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: place table\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 2\n\nYour success rate is: ",
                     "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: place table\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 2\n\nYour success rate is: ",
                     "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: place table\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n- stone 1 step to your west\n- path 2 steps to your south-west\n- tree 6 steps to your north-west\n- coal 2 steps to your west\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYour success rate is: ",
                     "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: place table\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 1 step to your north\n\nYou face tree at your front.\n\nYou have nothing in your inventory.\n\nYour success rate is: ",
                     "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: place table\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n- table 1 step to your north\n\nYou face table at your front.\n\nYou have nothing in your inventory.\n\nYou placed table at (32,31)\n\nYour success rate is: ",
                     "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: place table\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- coal 1 step to your north\n\nYou face coal at your front.\n\nYou have nothing in your inventory.\n\nYour success rate is: "
                     ]

TESTS_MAKE_WOOD_PICKAXE = ["You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 1\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 1\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 2\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 2\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 3\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 3\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n- table 2 steps to your north\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYou placed table at (32, 30)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n- table 2 steps to your north\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYou placed table at (32, 30)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n- table 2 steps to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 1\n\nYou placed table at (32, 30)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n- table 2 steps to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 1\n\nYou placed table at (32, 30)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n- table 2 steps to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 2\n\nYou placed table at (32, 30)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n- table 2 steps to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 2\n\nYou placed table at (32, 30)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n- table 2 steps to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 3\n\nYou placed table at (32, 30)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n- table 2 steps to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 3\n\nYou placed table at (32, 30)\n\nYour success rate is:",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYou placed table at (20,21)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYou have nothing in your inventory.\n\nYou placed table at (20,21)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 1\n\nYou placed table at (20,21)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 1\n\nYou placed table at (20,21)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 2\n\nYou placed table at (20,21)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 2\n\nYou placed table at (20,21)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 3\n\nYou placed table at (20,21)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n\nYou face grass at your front.\n\nYour inventory:\n- wood: 3\n\nYou placed table at (20,21)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n- table 1 step to your north\\n\\nYou face table at your front.\n\nYou have nothing in your inventory.\n\nYou placed table at (32, 31)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n- table 1 step to your north\n\nYou face table at your front.\n\nYou have nothing in your inventory.\n\nYou placed table at (32, 31)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n- table 1 step to your north\n\nYou face table at your front.\n\nYour inventory:\n- wood: 1\n\nYou placed table at (32, 31)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n- table 1 step to your north\n\nYou face table at your front.\n\nYour inventory:\n- wood: 1\n\nYou placed table at (32, 31)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n- table 1 step to your north\n\nYou face table at your front.\n\nYour inventory:\n- wood: 2\n\nYou placed table at (32, 31)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n- table 1 step to your north\n\nYou face table at your front.\n\nYour inventory:\n- wood: 2\n\nYou placed table at (32, 31)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your north\n- table 1 step to your north\n\nYou face table at your front.\n\nYour inventory:\n- wood: 3\n\nYou placed table at (32, 31)\n\nYour success rate is: ",
                           "You are playing a Minecraft like game.\nYour task is to evaluate your success rate for the goal: make wood pickaxe\n\nYour coordinates: (32,32)\n\nYou see:\n- grass 1 step to your west\n- tree 6 steps to your north-west\n- table 1 step to your north\n\nYou face table at your front.\n\nYour inventory:\n- wood: 3\n\nYou placed table at (32, 31)\n\nYour success rate is: "
                           ]

def test_low_level_sr_estimator(lm):
    estimated_sr = dict()
    for test_list, tsk in zip([TESTS_COLLECT_WOOD, TESTS_PLACE_TABLE, TESTS_MAKE_WOOD_PICKAXE], ["collect_wood", "place_table", "make_wood_pickaxe"]):
        sigm = torch.nn.Sigmoid()
        output = lm.custom_module_fns(['sr_ll_estimator'],
                                       contexts=test_list,
                                       candidates=[[" "] for _ in range(len(test_list))],
                                       require_grad=False,
                                       peft_adapter='sr_LL_adapters',
                                       add_eos_on_candidates=False)
        logits_sr = torch.stack([_o['sr_ll_estimator']["sr_ll_estimated"] for _o in output])

        estimated_sr[tsk] = [sigm(logit_sr.squeeze()).detach().cpu().numpy() for logit_sr in logits_sr]

    print("Estimated success rate for collect wood: {}".format(estimated_sr["collect_wood"]))
    print("Estimated success rate for place table: {}".format(estimated_sr["place_table"]))
    print("Estimated success rate for make wood pickaxe: {}".format(estimated_sr["make_wood_pickaxe"]))
    return estimated_sr

def test_true_sr_from_reset_env(name_env, seed, ll_action_space_id_to_act,
                         low_level_agent, obss_preprocessor, device, nbr_tests, rl_script_args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_envs = rl_script_args.number_envs
    # thought: when you learn ll the traj you copy is ll(sg_1) + ... + ll(sg_n)
    #  len(ll(sg_1) + ... + ll(sg_n)) <= len(goal_test) if not during learning you have more step than in testing
    dict_config = {"number_envs": num_envs,
                   "seed": int(1e9 * (seed + 1)),
                   "action_space": [v for v in ll_action_space_id_to_act.values()],
                   "length": rl_script_args.ll_traj_len_max}
    envs = CrafterTextGoalCondition(dict_config)
    envs.set_goals([format_goal(name_env)]*num_envs)

    successs = []
    rewards = []
    lengths = []

    if rl_script_args.memory_ll:
        memory = torch.zeros(num_envs, low_level_agent.memory_size, len(ll_action_space_id_to_act), device=device)
    (obs_env, infos_env), ep_ret, ep_len = envs.reset(), \
        [0 for _ in range(num_envs)], \
        [0 for _ in range(num_envs)]

    obs_vis = [ob["image"] for ob in obs_env]
    # obs_task_enc = [ob["task_enc"] for ob in obs_env]

    while len(successs) < nbr_tests:

        obs = []
        for idx in range(num_envs):
            obs.append({"image": obs_vis[idx]})
        preprocessed_obs = obss_preprocessor(obs, device=device)

        with torch.no_grad():
            if rl_script_args.memory_ll:
                actor_model_results = low_level_agent(preprocessed_obs, memory)
            else:
                actor_model_results = low_level_agent(preprocessed_obs)
            dist = actor_model_results['dist']

        action = dist.sample()
        if rl_script_args.memory_ll:
            for i in range(num_envs):
                one_hot_action = torch.zeros(len(ll_action_space_id_to_act), device=device)
                one_hot_action[int(action[i])] = 1
                memory[i] = push_to_tensor(memory[i], one_hot_action.unsqueeze(0), device=device)

        acts = action.cpu().numpy()
        a = np.array([ll_action_space_id_to_act[int(act)] for act in acts])
        obs_env, reward, done, infos_env = envs.step(a)

        obs_vis = [ob["image"] for ob in obs_env]
        for i in range(num_envs):
            ep_len[i] += 1
            ep_ret[i] += reward[i]

            if done[i]:
                successs.append(1 if reward[i] > 0 else 0)
                rewards.append(ep_ret[i])
                lengths.append(ep_len[i])
                ep_len[i] = 0
                ep_ret[i] = 0
                if rl_script_args.memory_ll:
                    memory[i] = torch.zeros(low_level_agent.memory_size, len(ll_action_space_id_to_act), device=device)

    successs_arr = np.array(successs[:nbr_tests])
    rewards_arr = np.array(rewards[:nbr_tests])
    lengths_arr = np.array(lengths[:nbr_tests])

    print("Test from reset for task: {}".format(name_env))
    print("success rate: {}".format(np.average(successs_arr)))
    print("reward average: {}".format(np.average(rewards_arr)))
    print("length average: {}".format(np.average(lengths_arr)))

    return successs, rewards, lengths
