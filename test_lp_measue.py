import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import numpy as np
import torch
import re
from itertools import zip_longest

from dataclasses import dataclass

from utils.goal_sampler import MALPGoalSampler, SRDiffGoalSampler

list_color_perso = ["purple", "green", "sky", "brown", "red", "teal", "orange", "magenta", "violet", "turquoise", "tan",
                    "bright green", "maroon", "olive", "salmon", "royal blue", "hot pink", "light brown", "dark pink",
                    "indigo", "lime", "mustard", "rose", "bright blue", "neon green", "burnt orange", "yellow green",
                    "brick red", "gold", "bright pink", "electric blue", "red_orange", "purplish blue", "azur",
                    "neon purple", "bright red", "pinkish red", "emerald"]

list_color_seed = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
list_line_style = ["--", "-.", ":"]

task_space = ['collect wood', "place table"]


def filtration_high_level_action(action_list, high_level_action_list):
    new_high_level_action_list = []
    switch=True
    for tk, act in zip(action_list, high_level_action_list):
        if switch:
            new_high_level_action_list.append(act)
            switch=False
        if tk==2:
            switch=True
    return new_high_level_action_list
def proportion_of_skills_used(task_space, action_list):
    proportions = {tsk: 0.0 for tsk in task_space}
    for ac in action_list:
        for tsk in task_space:
            if ac == tsk:
                proportions[tsk] += 1
                break
    proportions = {tsk: p / len(action_list) for tsk, p in proportions.items()}
    return proportions


def load_results(path, file_name="history.pkl", filter=None):
    results = {}

    for expe_dir in os.listdir(path):
        if "epochs" in expe_dir:
            dir_path = os.path.join(path, expe_dir)
            if os.path.isdir(dir_path):
                if filter is not None and not filter in expe_dir:
                    continue
                else:
                    results[expe_dir] = {}
                    for seed_dir in os.listdir(dir_path):
                        try:
                            seed = seed_dir.split("seed_")[1]
                            with open(os.path.join(dir_path, seed_dir, file_name), 'rb') as file:
                                results[expe_dir][int(seed)] = pickle.load(file)
                                if "value_rep" and "model_head" in results[expe_dir][int(seed)].keys():
                                    results[expe_dir][int(seed)]["value_rep"] = []
                                    results[expe_dir][int(seed)]["model_head"] = []
                                """if "value_rep" and "model_head" in results[expe_dir][int(seed)].keys():
                                    value_rep = np.array(results[expe_dir][int(seed)]["value_rep"])
                                    results[expe_dir][int(seed)]["value_rep"] = value_rep[np.random.choice(len(value_rep), min(len(value_rep), 2*len(value_rep[0])), replace=False)]
                                    model_head = np.array(results[expe_dir][int(seed)]["model_head"])
                                    results[expe_dir][int(seed)]["model_head"] = model_head[np.random.choice(len(model_head), min(len(model_head), 2*len(model_head[0])), replace=False)]"""
                            """if "value_rep" and "model_head" in results[expe_dir][int(seed)].keys():
                                with open(os.path.join(dir_path, seed_dir, file_name), 'wb') as fp:
                                    pickle.dump(results[expe_dir][int(seed)], fp)"""
                        except Exception as err:
                            print(f"Error in {seed_dir} for experiment {expe_dir}: {err}")
    return results


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    arr = np.ma.masked_invalid(arr)
    return arr.mean(axis=-1), arr.std(axis=-1)

def format_goal(goal_str:str):
    parts = goal_str.split('_')
    name_g = parts[0].capitalize()
    for p in parts[1:]:
         name_g += ' '+p
    return name_g


@dataclass
class malp_args:
    epsilon_start: int
    epsilon_end: int
    epsilon_decay: int
    buffer_size: int
    alpha: int

@dataclass
class srdiff_arg:
    epsilon_start: int
    epsilon_end: int
    epsilon_decay: int
    buffer_size: int

experiments = load_results("./results/AGG/textcrafter_DLP_test_3_ac_separated_simple_sr_06_cw_pt_mnbts_256_lr_actor_1e-4_lr_critic_1e-3_useful_tkn_kl_penalty_001/", "history.pkl")
len_expe = len(experiments)



for s in range(4):
    probe_lp_malp = MALPGoalSampler([format_goal(tsk) for tsk in task_space], malp_args(1.0, 0.1, 320, 100, 0.1))
    probe_lp_srdiff = SRDiffGoalSampler([format_goal(tsk) for tsk in task_space], srdiff_arg(1.0, 0.1, 320, 400))

    sampling_malp = {format_goal(tsk): [] for tsk in task_space}
    sampling_srdiff = {format_goal(tsk): [] for tsk in task_space}
    sampling_observed = {format_goal(tsk): [] for tsk in task_space}

    for idx in range(len_expe):
        exp = experiments["epochs_{}-{}".format(idx, idx + 1)]

        if s in exp.keys():
            lp = probe_lp_malp.update(goals=exp[s]["goal"], returns=exp[s]["ep_ret"])
            lp_values = np.array(list(probe_lp_malp.lp.values()))
            sum_lp = np.sum(lp_values)
            for tsk in task_space:
                sampling_malp[format_goal(tsk)].append(lp['lp'][format_goal(tsk)]/sum_lp)

            lp = probe_lp_srdiff.update(goals=exp[s]["goal"], returns=exp[s]["ep_ret"])
            lp_values = np.array(list(probe_lp_srdiff.lp.values()))
            sum_lp = np.sum(lp_values)
            for tsk in task_space:
                sampling_srdiff[format_goal(tsk)].append(lp['lp'][format_goal(tsk)]/sum_lp)

            sum_lp = np.sum(list(exp[s]['lp_goals'].values()))
            for tsk in task_space:
                sampling_observed[format_goal(tsk)].append(exp[s]['lp_goals'][format_goal(tsk)]/sum_lp)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    for idx, tsk in enumerate(task_space):
        ax[idx].plot(sampling_malp[format_goal(tsk)], label="MALP")
        ax[idx].plot(sampling_srdiff[format_goal(tsk)], label="SRDiff")
        ax[idx].plot(sampling_observed[format_goal(tsk)], label="Observed")
        ax[idx].set_xlabel("Epochs")
        ax[idx].set_ylabel("probability sampling")
        ax[idx].legend()
    ax[0].set_title("Collect wood")
    ax[1].set_title("Place table")

    plt.show()



print()