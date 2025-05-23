import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import numpy as np
import math
import torch
import re
from itertools import zip_longest


list_color_perso = ["purple", "green", "sky", "brown", "red", "teal", "orange", "magenta", "violet", "turquoise", "tan",
                    "bright green", "maroon", "olive", "salmon", "royal blue", "hot pink", "light brown", "dark pink",
                    "indigo", "lime", "mustard", "rose", "bright blue", "neon green", "burnt orange", "yellow green",
                    "brick red", "gold", "bright pink", "electric blue", "red_orange", "purplish blue", "azur",
                    "neon purple", "bright red", "pinkish red", "emerald"]

list_color_seed = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
list_line_style = ["--", "-.", ":"]

task_space = ['collect_wood', "place_table"]


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

def plot_mastering_task_space(experiments, experiments_names, task_space, measuring_time, difficulty_scale, master_tsk_level, label=None, avg_window=10, title=None):
    dict_expe_step_master_tsk = {exp_name: {tsk: dict() for tsk in task_space} for exp_name in experiments_names}

    for exp_name, expe in zip(experiments_names, experiments):
        if "FuN" not in exp_name:
            len_expe = len(expe[0])
            nbr_steps = None
            for idx in range(len_expe):
                exp_test = expe[0]["epochs_{}-{}".format(idx, idx + 1)]
                exp_hist = expe[1]["epochs_{}-{}".format(idx, idx + 1)]
                if idx == 0:
                    nbr_steps = np.array([np.nan] * len(exp_test))
                for seed in exp_test.keys():
                    if len(exp_test[seed]) > 0:
                        if np.isnan(nbr_steps[seed]):
                            nbr_steps[seed] = 0
                        nbr_steps[seed] += np.sum(exp_hist[seed][measuring_time])
                        for tsk in task_space:
                            sr = np.mean(exp_test[seed]["success_rate_hierarchical_agent_from_reset_env"][tsk])
                            if sr > master_tsk_level and seed not in dict_expe_step_master_tsk[exp_name][tsk]:
                                dict_expe_step_master_tsk[exp_name][tsk][seed] = nbr_steps[seed]

    if "FuN" in experiments_names:
        idx_funs = experiments_names.index("FuN")
        for tsk in task_space:
            for seed in experiments[idx_funs].keys():
                steps = np.nan
                for v in experiments[idx_funs][seed]:
                    v = {_k.lower().replace(" ", "_"): _v for _k, _v in v.items()}
                    if v[tsk] > master_tsk_level and seed not in dict_expe_step_master_tsk[experiments_names[idx_funs]][tsk]:
                        if np.isnan(steps):
                            steps = 0
                        val = (steps + 1) * (16000 if measuring_time == "ep_interaction_len" else 1600)
                        dict_expe_step_master_tsk[experiments_names[idx_funs]][tsk][seed] = val
                    else:
                        if np.isnan(steps):
                            steps = 0
                        steps += 1

    num_tasks = len(task_space)
    num_experiments = len(experiments_names)

    # Compute average and std
    averages = np.zeros((num_experiments, num_tasks))
    stds = np.zeros((num_experiments, num_tasks))

    for i, exp in enumerate(experiments_names):
        for j, task in enumerate(task_space):
            seeds = list(dict_expe_step_master_tsk[exp].get(task, {}).values())
            if seeds:
                averages[i, j] = np.nanmean(seeds)
                stds[i, j] = np.nanstd(seeds)
            else:
                averages[i, j] = np.nan
                stds[i, j] = 0

    x = np.array([difficulty_scale[tsk] for tsk in task_space])
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.set_xlim(left=0, right=max(x) + 0.5)
    # ax.set_ylim(top=35000)

    # Plot with error bars and lines
    for i, exp in enumerate(experiments_names):
        # x_offset = x + (i - num_experiments / 2) * 0.05
        x_offset=x
        ax.errorbar(x_offset, averages[i], yerr=stds[i], fmt='o-', label=exp, capsize=4)

    # Axis labels
    ax.set_xlabel('Difficulty Scale', fontsize=34, fontweight='bold')
    ylabel = 'Average Steps to Master '
    ylabel += 'environment interactions' if measuring_time == "ep_interaction_len" else 'high level steps'
    ax.set_ylabel(ylabel, fontsize=34, fontweight='bold')
    if title:
        ax.set_title(title)

    # Custom ticks and task names
    tick_positions = x
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(difficulty_scale[tsk]) for tsk in task_space], fontsize=32, fontweight='bold')


    # Make tick lines thicker and longer
    ax.tick_params(axis='both', which='both', width=4, length=6)

    global_max = np.nanmax(averages + stds)

    for j, task in enumerate(task_space):
        if j%2 == 0:
            label_y_pos = global_max + 0.05 * global_max
        else:
            label_y_pos = global_max + 0.1 * global_max
        max_height = np.nanmax(averages[:, j] + stds[:, j])
        ax.text(tick_positions[j], label_y_pos, task,
                ha='center', va='bottom', fontsize=20, fontweight='bold')

    for label in ax.get_yticklabels():
        label.set_fontsize(32)
        label.set_fontweight('bold')

    ax.legend(fontsize=30)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def crafter_scorer(list_sr_per_task):
    results = []
    for ls in list_sr_per_task:
        s=0
        for tsk in ls.keys():
            s += np.log(ls[tsk]*100+1)
        s = np.exp(s/len(ls))-1
        results.append(s)
    return results



def plot_crafter_score_evolution(experiments, experiments_names, task_space, random_score=None,
                                 measuring_time="ep_len", max_len=None, label=None, avg_window=10, title=None):
    #All experiments need to have the same task space

    dict_expe = {exp_name : dict() for exp_name in experiments_names}
    nbr_steps_expe = {exp_name : dict() for exp_name in experiments_names}

    for exp_name, expe in zip(experiments_names, experiments):
        if exp_name!="FuN":
            len_expe = len(expe[0])
            nbr_steps = None
            for idx in range(len_expe):
                exp_test = expe[0]["epochs_{}-{}".format(idx, idx + 1)]
                exp_hist = expe[1]["epochs_{}-{}".format(idx, idx + 1)]
                if idx==0:
                    for seed in exp_test.keys():
                        if len(exp_test[seed])>0:
                            dict_expe[exp_name][seed] = [{tsk: np.mean(exp_test[seed]["success_rate_hierarchical_agent_from_reset_env"][tsk]) for tsk in task_space}]
                            nbr_steps_expe[exp_name][seed] = [np.sum(exp_hist[seed][measuring_time])]
                else:
                    for seed in exp_test.keys():
                        try:
                            if len(exp_test[seed]) > 0:
                                if max_len is None:
                                    dict_expe[exp_name][seed].append({tsk: np.mean(exp_test[seed]["success_rate_hierarchical_agent_from_reset_env"][tsk]) for tsk in task_space})
                                    nbr_steps_expe[exp_name][seed].append(nbr_steps_expe[exp_name][seed][-1]+np.sum(exp_hist[seed][measuring_time]))
                                else:
                                    if nbr_steps_expe[exp_name][seed][-1]+np.sum(exp_hist[seed][measuring_time])<max_len:
                                        dict_expe[exp_name][seed].append({tsk: np.mean(exp_test[seed]["success_rate_hierarchical_agent_from_reset_env"][tsk]) for tsk in task_space})
                                        nbr_steps_expe[exp_name][seed].append(nbr_steps_expe[exp_name][seed][-1]+np.sum(exp_hist[seed][measuring_time]))
                        except:
                            print(" ")
    if "FuN" in experiments_names:
        idx_funs = experiments_names.index("FuN")
        for seed in experiments[idx_funs].keys():

            if max_len is not None:
                max_epochs = 0
                if measuring_time=="ep_interaction_len":
                    max_epochs = max_len/16000
                elif measuring_time=="ep_len":
                    max_epochs = max_len/1600
            if max_len is None:
                dict_expe[experiments_names[idx_funs]][seed] = experiments[idx_funs][seed]
            else:
                dict_expe[experiments_names[idx_funs]][seed] = experiments[idx_funs][seed][:int(max_epochs+1)]
            if max_len is None:
                if measuring_time=="ep_interaction_len":
                    nbr_steps_expe[experiments_names[idx_funs]][seed] = list(np.cumsum([16000]*len(experiments[idx_funs][seed])))
                elif measuring_time=="ep_len":
                    nbr_steps_expe[experiments_names[idx_funs]][seed] = list(np.cumsum([1600]*len(experiments[idx_funs][seed])))
                else:
                    raise ValueError("measuring_time should be ep_interaction_len or ep_len")
            else:
                if measuring_time=="ep_interaction_len":
                    nbr_steps_expe[experiments_names[idx_funs]][seed] = list(np.cumsum([16000]*len(dict_expe[experiments_names[idx_funs]][seed])))
                elif measuring_time=="ep_len":
                    nbr_steps_expe[experiments_names[idx_funs]][seed] = list(np.cumsum([1600]*len(dict_expe[experiments_names[idx_funs]][seed])))
                else:
                    raise ValueError("measuring_time should be ep_interaction_len or ep_len")
    # generate crafter score
    dict_expe_score = {exp_name : dict() for exp_name in experiments_names}
    for exp_name in dict_expe.keys():
        for i in list(dict_expe[exp_name].keys()):
            dict_expe_score[exp_name][i] = [crafter_scorer(dict_expe[exp_name][i])]


    avg_scores = []
    avg_scores_std = []
    avg_steps = []

    # padding right for the scores and steps
    for exp_n in experiments_names:
        scores = list(dict_expe_score[exp_n].values())
        max_len_sc = 0
        for sc in scores:
            if len(sc[0])>max_len_sc:
                max_len_sc = len(sc[0])
        for sc in dict_expe_score[exp_n].values():
            if len(sc[0])<max_len_sc:
                sc[0].extend([np.nan]*(max_len_sc-len(sc[0])))

        for st in nbr_steps_expe[exp_n].values():
            if len(st)<max_len_sc:
                st.extend([np.nan]*(max_len_sc-len(st)))

        list_keys = list(dict_expe_score[exp_n].keys())
        list_keys.sort()
        avg_scores.append(np.nanmean([dict_expe_score[exp_n][sc_k] for sc_k in list_keys], axis=0)[0])
        avg_scores_std.append(np.nanstd([dict_expe_score[exp_n][sc_k] for sc_k in list_keys], axis=0)[0])
        avg_steps.append(np.nanmean([nbr_steps_expe[exp_n][sc_k] for sc_k in list_keys], axis=0))

    # calculating integral for each curve
    for idx_exp_n, exp_n in enumerate(experiments_names):
        # Remove points where y is nan
        mask = ~np.isnan(avg_scores[idx_exp_n])
        x_clean = avg_steps[idx_exp_n][mask]
        y_clean = avg_scores[idx_exp_n][mask]

        if len(x_clean) < 2:
            return 0.0  # Not enough points to integrate

        # Use the trapezoidal rule
        integral = np.trapz(y_clean, x_clean)/np.max(x_clean)
        print("Integral for {}: {}".format(exp_n, integral))

    # Plotting
    plt.figure(figsize=(20, 15))
    # plt.xlim(left=0, right=30000)

    if random_score is not None:
        plt.axhline(y=random_score, color='r', linestyle='--', label="Random score")

    for stp, scores, score_std, exp_n in zip(avg_steps, avg_scores, avg_scores_std, experiments_names):
        plt.plot(stp, scores, label=exp_n)
        plt.fill_between(stp, scores - score_std, scores + score_std, alpha=0.2)


    plt.ylabel('Crafter score', fontsize=34, fontweight='bold')
    if measuring_time == "ep_interaction_len":
        plt.xlabel('Number of interaction with environment', fontsize=34, fontweight='bold')
    elif measuring_time == "ep_len":
        plt.xlabel('Number of high level steps', fontsize=34, fontweight='bold')

    plt.xticks(fontsize=32, fontweight='bold')
    plt.yticks(fontsize=32, fontweight='bold')
    plt.legend(fontsize=32)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def moving_average(data, window_size):
    """Compute moving average with a given window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_sr_evolution(experiments, experiments_names, task_space,
                                 measuring_time="ep_len", max_len=None, label=None, title=None):
    #All experiments need to have the same task space

    dict_expe = {exp_name : dict() for exp_name in experiments_names}
    nbr_steps_expe = {exp_name : dict() for exp_name in experiments_names}

    for exp_name, expe in zip(experiments_names, experiments):
        if exp_name!="FuN":
            len_expe = len(expe[0])
            nbr_steps = None
            for idx in range(len_expe):
                exp_test = expe[0]["epochs_{}-{}".format(idx, idx + 1)]
                exp_hist = expe[1]["epochs_{}-{}".format(idx, idx + 1)]
                if idx==0:
                    for seed in exp_test.keys():
                        if len(exp_test[seed])>0:
                            dict_expe[exp_name][seed] = [{tsk: np.mean(exp_test[seed]["success_rate_hierarchical_agent_from_reset_env"][tsk]) for tsk in task_space}]
                            nbr_steps_expe[exp_name][seed] = [np.sum(exp_hist[seed][measuring_time])]
                else:
                    for seed in exp_test.keys():
                        try:
                            if len(exp_test[seed]) > 0:
                                if max_len is None:
                                    dict_expe[exp_name][seed].append({tsk: np.mean(exp_test[seed]["success_rate_hierarchical_agent_from_reset_env"][tsk]) for tsk in task_space})
                                    nbr_steps_expe[exp_name][seed].append(nbr_steps_expe[exp_name][seed][-1]+np.sum(exp_hist[seed][measuring_time]))
                                else:
                                    if nbr_steps_expe[exp_name][seed][-1]+np.sum(exp_hist[seed][measuring_time])<max_len:
                                        dict_expe[exp_name][seed].append({tsk: np.mean(exp_test[seed]["success_rate_hierarchical_agent_from_reset_env"][tsk]) for tsk in task_space})
                                        nbr_steps_expe[exp_name][seed].append(nbr_steps_expe[exp_name][seed][-1]+np.sum(exp_hist[seed][measuring_time]))
                        except:
                            print(" ")
    if "FuN" in experiments_names:
        idx_funs = experiments_names.index("FuN")
        for seed in experiments[idx_funs].keys():

            if max_len is not None:
                max_epochs = 0
                if measuring_time=="ep_interaction_len":
                    max_epochs = max_len/16000
                elif measuring_time=="ep_len":
                    max_epochs = max_len/1600
            if max_len is None:
                expe_selected = experiments[idx_funs][seed][:int(max_epochs+1)]
                for id_exp_s, exp_s in enumerate(expe_selected):
                    expe_selected[id_exp_s] = {_k.lower().replace(" ", "_"): _v for _k, _v in exp_s.items()}
                dict_expe[experiments_names[idx_funs]][seed] = experiments[idx_funs][seed]
            else:
                # modify keys to match the other experiments
                expe_selected = experiments[idx_funs][seed][:int(max_epochs+1)]
                for id_exp_s, exp_s in enumerate(expe_selected):
                    expe_selected[id_exp_s] = {_k.lower().replace(" ", "_"): _v for _k, _v in exp_s.items()}
                dict_expe[experiments_names[idx_funs]][seed] = expe_selected
            if max_len is None:
                if measuring_time=="ep_interaction_len":
                    nbr_steps_expe[experiments_names[idx_funs]][seed] = list(np.cumsum([16000]*len(experiments[idx_funs][seed])))
                elif measuring_time=="ep_len":
                    nbr_steps_expe[experiments_names[idx_funs]][seed] = list(np.cumsum([1600]*len(experiments[idx_funs][seed])))
                else:
                    raise ValueError("measuring_time should be ep_interaction_len or ep_len")
            else:
                if measuring_time=="ep_interaction_len":
                    nbr_steps_expe[experiments_names[idx_funs]][seed] = list(np.cumsum([16000]*len(dict_expe[experiments_names[idx_funs]][seed])))
                elif measuring_time=="ep_len":
                    nbr_steps_expe[experiments_names[idx_funs]][seed] = list(np.cumsum([1600]*len(dict_expe[experiments_names[idx_funs]][seed])))
                else:
                    raise ValueError("measuring_time should be ep_interaction_len or ep_len")

    # Plotting
    n_tasks = len(task_space)
    n_cols = 3
    n_rows = math.ceil(len(task_space) / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False)

    # Flatten axs to 1D array for easy indexing
    axs = axs.flatten()

    for idx, task in enumerate(task_space):
        ax = axs[idx]
        for exp_name in experiments_names:
            all_curves = []
            all_steps = []
            for seed in dict_expe[exp_name]:
                try:
                    scores = [entry[task] for entry in dict_expe[exp_name][seed]]
                except:
                    continue
                steps = nbr_steps_expe[exp_name][seed]
                all_curves.append(scores)
                all_steps.append(steps)
            if all_curves:
                min_len = min(len(s) for s in all_curves)
                curves = np.array([curve[:min_len] for curve in all_curves])
                steps = np.array([step[:min_len] for step in all_steps])
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0)
                mean_steps = np.mean(steps, axis=0)

                ax.plot(mean_steps, mean_curve, label=exp_name)
                ax.fill_between(mean_steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)

        ax.set_title(f"Task: {task}")
        ax.set_ylabel("Success Rate")
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for i in range(len(task_space), len(axs)):
        fig.delaxes(axs[i])

    axs[-1].set_xlabel("Steps ({})".format(measuring_time))
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def results_generalisation(experiments, experiments_names, gene_task_space, epoch):
    dict_expe = {exp_name : [] for exp_name in experiments_names}

    for exp_name, expe in zip(experiments_names, experiments):
        if exp_name!="FuN":
            len_expe = len(expe)
            nbr_steps = None
            exp_gene = expe["epochs_{}-{}".format(epoch, epoch + 1)]
            nbr_seed = len(exp_gene)
            for seed in exp_gene.keys():
                if len(exp_gene[seed])>0:
                    if len(dict_expe[exp_name])==0:
                        dict_expe[exp_name] = {tsk: exp_gene[seed]["success_rate_hierarchical_agent_from_reset_env"][tsk] for tsk in gene_task_space}
                    else:
                        for tsk in gene_task_space:
                            dict_expe[exp_name][tsk][0].extend(exp_gene[seed]["success_rate_hierarchical_agent_from_reset_env"][tsk][0])

    if "FuN" in experiments_names:
        idx_funs = experiments_names.index("FuN")
        for tsk in gene_task_space:
            for seed in experiments[idx_funs].keys():
                if len(experiments[idx_funs][seed])>int(epoch*2496/1600):
                    lower_case_dict = {_k.lower().replace(" ", "_"): _v for _k, _v in experiments[idx_funs][seed][int(epoch*2496/1600)].items()}
                else:
                    lower_case_dict = {_k.lower().replace(" ", "_"): _v for _k, _v in experiments[idx_funs][seed][-1].items()}
                if len(dict_expe[experiments_names[idx_funs]])==0:
                    dict_expe[experiments_names[idx_funs]] = {tsk: [lower_case_dict[tsk]]}
                else:
                    if tsk not in dict_expe[experiments_names[idx_funs]].keys():
                        dict_expe[experiments_names[idx_funs]][tsk] = [lower_case_dict[tsk]]
                    else:
                        dict_expe[experiments_names[idx_funs]][tsk].append(lower_case_dict[tsk])

    for exp_name in dict_expe.keys():
        print(f"Experiment {exp_name} ")
        for tsk in gene_task_space:
            mean_tsk = np.mean(dict_expe[exp_name][tsk])
            std_tsk = np.std(dict_expe[exp_name][tsk])
            print(f"Task {tsk}: Mean = {mean_tsk}, Std = {std_tsk}")
        print(" ")


def sampling_evolution(experiment, task_space, track_ea, track_sg,measuring_time="ep_len"):

    dict_sg = {tsk : dict() for tsk in task_space}
    dict_ea = {tsk : dict() for tsk in task_space}
    nbr_steps_expe = {tsk : dict() for tsk in task_space}

    len_expe = len(experiment[0])
    for idx in range(len_expe):
        exp_test = experiment[0]["epochs_{}-{}".format(idx, idx + 1)]
        exp_hist = experiment[1]["epochs_{}-{}".format(idx, idx + 1)]
        for task_name in task_space:
            if idx==0:
                for seed in exp_test.keys():
                    if len(exp_test[seed])>0:
                        dict_sg[task_name][seed] = [exp_test[seed]["subgoal_instructions_dict"][task_name]]
                        dict_ea[task_name][seed] = [exp_test[seed]["elementary_actions_dict"][task_name]]
                        nbr_steps_expe[task_name][seed] = [np.sum(exp_hist[seed][measuring_time])]
            else:
                for seed in exp_test.keys():
                    if len(exp_test[seed]) > 0:
                        dict_sg[task_name][seed].extend([exp_test[seed]["subgoal_instructions_dict"][task_name]])
                        dict_ea[task_name][seed].extend([exp_test[seed]["elementary_actions_dict"][task_name]])
                        try:
                            nbr_steps_expe[task_name][seed].append(nbr_steps_expe[task_name][seed][-1]+np.sum(exp_hist[seed][measuring_time]))
                        except:
                            print(" ")

    for tsk in task_space:
        max_epoch = 0
        for seed in dict_sg[tsk].keys():
            if len(dict_sg[tsk][seed])>max_epoch:
                max_epoch = len(dict_sg[tsk][seed])

        sampling_HL_dict = {k_H: dict() for k_H in track_sg+track_ea}

        for seed in dict_sg[tsk].keys():
            for epoch in dict_sg[tsk][seed]:
                untracked_sg = []
                for sg_k, sg_v in epoch[0].items():
                    if sg_k not in track_sg:
                        untracked_sg.append(sg_v)
                    else:
                        if seed not in sampling_HL_dict[sg_k].keys():
                            sampling_HL_dict[sg_k][seed] = [sg_v]
                        else:
                            sampling_HL_dict[sg_k][seed].append(sg_v)
                if len(untracked_sg) > 0:
                    if "untracked_sg" not in sampling_HL_dict.keys():
                        sampling_HL_dict["untracked_sg"] = {seed: [np.mean(untracked_sg)]}
                    else:
                        if seed not in sampling_HL_dict["untracked_sg"].keys():
                            sampling_HL_dict["untracked_sg"][seed] = [np.mean(untracked_sg)]
                        else:
                            sampling_HL_dict["untracked_sg"][seed].append(np.mean(untracked_sg))


            for epoch in dict_ea[tsk][seed]:
                untracked_ea = []
                move = []
                for ea_k, ea_v in epoch[0].items():
                    if "move" not in ea_k and ea_k not in track_ea:
                        untracked_ea.append(ea_v)
                    elif "move" in ea_k:
                        move.append(ea_v)
                    else:
                        if seed not in sampling_HL_dict[ea_k].keys():
                            sampling_HL_dict[ea_k][seed] = [ea_v]
                        else:
                            sampling_HL_dict[ea_k][seed].append(ea_v)
                if len(untracked_ea) > 0:
                    if "untracked_ea" not in sampling_HL_dict.keys():
                        sampling_HL_dict["untracked_ea"] = {seed: [np.mean(untracked_ea)]}
                    else:
                        if seed not in sampling_HL_dict["untracked_ea"].keys():
                            sampling_HL_dict["untracked_ea"][seed] = [np.mean(untracked_ea)]
                        else:
                            sampling_HL_dict["untracked_ea"][seed].append(np.mean(untracked_ea))
                if len(move) > 0:
                    if "move" not in sampling_HL_dict.keys():
                        sampling_HL_dict["move"] = {seed: [np.mean(move)]}
                    else:
                        if seed not in sampling_HL_dict["move"].keys():
                            sampling_HL_dict["move"][seed] = [np.mean(move)]
                        else:
                            sampling_HL_dict["move"][seed].append(np.mean(move))


        # padding right for the scores and steps
        for sg_n in sampling_HL_dict.keys():
            for seed_sg_n in sampling_HL_dict[sg_n].keys():
                if len(sampling_HL_dict[sg_n][seed_sg_n])<max_epoch:
                    sampling_HL_dict[sg_n][seed_sg_n].extend([np.nan]*(max_epoch-len(sampling_HL_dict[sg_n][seed_sg_n])))
        for st in nbr_steps_expe[tsk].keys():
            if len(nbr_steps_expe[tsk][st])<max_epoch:
                nbr_steps_expe[tsk][st].extend([np.nan]*(max_epoch-len(nbr_steps_expe[tsk][st])))

        mean_sampling_HL_dict = {k_H: np.nanmean([v_HL for v_HL in sampling_HL_dict[k_H].values()], axis=0) for k_H in sampling_HL_dict.keys()}
        std_sampling_HL_dict = {k_H: np.nanstd([v_HL for v_HL in sampling_HL_dict[k_H].values()], axis=0) for k_H in sampling_HL_dict.keys()}
        mean_sampling_HL_dict["nbr_steps"] = np.nanmean([v_HL for v_HL in nbr_steps_expe[tsk].values()], axis=0)

        # plot for each sampling HL
        plt.figure(figsize=(20, 15))
        for k_H in mean_sampling_HL_dict.keys():
            if k_H != "nbr_steps":
                plt.plot(mean_sampling_HL_dict["nbr_steps"], mean_sampling_HL_dict[k_H], label=k_H)
                plt.fill_between(mean_sampling_HL_dict["nbr_steps"], mean_sampling_HL_dict[k_H] - std_sampling_HL_dict[k_H],
                                     mean_sampling_HL_dict[k_H] + std_sampling_HL_dict[k_H], alpha=0.2)
        plt.xlabel('Number of high level steps', fontsize=44, fontweight='bold')
        plt.ylabel('Mean number of sampling by HL', fontsize=44, fontweight='bold')
        plt.xticks(fontsize=42, fontweight='bold')
        plt.yticks(fontsize=42, fontweight='bold')
        plt.legend(fontsize=42)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(tsk, fontsize=44, fontweight='bold')
        plt.tight_layout()
        plt.show()


results_gtt_to_pf_ll_sr_update_256_HL_1e5_history = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "history.pkl")
results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_sr = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "test_sr.pkl")
results_gtt_to_pf_ll_sr_update_256_HL_1e5_generalisation = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "generalisation.pkl")
results_gtt_to_pf_ll_sr_update_256_HL_1e5_generalisation_synonym = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "generalisation_synonym.pkl")
results_gtt_to_pf_ll_sr_update_256_HL_1e5_generalisation_compositional2 = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "generalisation_compositional2.pkl")
results_gtt_to_pf_ll_sr_update_256_HL_1e5_generalisation_compositional3 = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "generalisation_compositional3.pkl")
results_gtt_to_pf_ll_sr_update_256_HL_1e5_generalisation_compositional4 = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "generalisation_compositional4.pkl")
results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_HL_sampling = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "test_HL_sampling.pkl")


# results_ablation_HL_only_gtt_to_pf = load_results("./results/AGG/DLP_textcrafter_ablation_HL_only_gtt_to_pf_unsloth/", "history.pkl")
results_ablation_no_update_HL_gtt_to_pf_history = load_results("./results/AGG/DLP_textcrafter_ablation_no_update_HL_gtt_to_pf_unsloth/", "history.pkl")
results_ablation_no_update_HL_gtt_to_pf_test_sr = load_results("./results/AGG/DLP_textcrafter_ablation_no_update_HL_gtt_to_pf_unsloth/", "test_sr.pkl")
results_ablation_no_update_HL_gtt_to_pf_generalisation = load_results("./results/AGG/DLP_textcrafter_ablation_no_update_HL_gtt_to_pf_unsloth/", "generalisation.pkl")

results_ablation_HL_only_gtt_to_pf_history = load_results("./results/AGG/DLP_textcrafter_ablation_HL_only_gtt_to_pf_unsloth/", "history.pkl")
results_ablation_HL_only_gtt_to_pf_test_sr = load_results("./results/AGG/DLP_textcrafter_ablation_HL_only_gtt_to_pf_unsloth/", "test_sr.pkl")
results_ablation_HL_only_gtt_to_pf_generalisation = load_results("./results/AGG/DLP_textcrafter_ablation_HL_only_gtt_to_pf_unsloth/", "generalisation.pkl")
results_ablation_HL_only_gtt_to_pf_generalisation_synonym = load_results("./results/AGG/DLP_textcrafter_ablation_HL_only_gtt_to_pf_unsloth/", "generalisation_synonym.pkl")
results_ablation_HL_only_gtt_to_pf_generalisation_compositional2 = load_results("./results/AGG/DLP_textcrafter_ablation_HL_only_gtt_to_pf_unsloth/", "generalisation_compositional2.pkl")
results_ablation_HL_only_gtt_to_pf_generalisation_compositional3 = load_results("./results/AGG/DLP_textcrafter_ablation_HL_only_gtt_to_pf_unsloth/", "generalisation_compositional3.pkl")
results_ablation_HL_only_gtt_to_pf_generalisation_compositional4 = load_results("./results/AGG/DLP_textcrafter_ablation_HL_only_gtt_to_pf_unsloth/", "generalisation_compositional4.pkl")


results_ablation_all_subgoals_accepted_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_history = load_results("./results/AGG/DLP_textcrafter_ablation_all_subgoals_accepted_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "history.pkl")
results_ablation_all_subgoals_accepted_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_sr = load_results("./results/AGG/DLP_textcrafter_ablation_all_subgoals_accepted_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "test_sr.pkl")
results_ablation_all_subgoals_accepted_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_generalisation = load_results("./results/AGG/DLP_textcrafter_ablation_all_subgoals_accepted_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "generalisation.pkl")

results_gtt_to_pf_ll_sr_update_256_HL_1e5_AWR_buffer_25000_history = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_awr_buffer_25000/", "history.pkl")
results_gtt_to_pf_ll_sr_update_256_HL_1e5_AWR_buffer_25000_test_sr = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_awr_buffer_25000/", "test_sr.pkl")

results_gtt_to_pf_ll_sr_update_256_HL_1e5_AWR_buffer_50000_test_sr = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_awr_buffer_50000/", "test_sr.pkl")
results_gtt_to_pf_ll_sr_update_256_HL_1e5_AWR_buffer_50000_history = load_results("./results/AGG/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_awr_buffer_50000/", "history.pkl")

results_ablation_no_linguistic_cues_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_history = load_results("./results/AGG/DLP_textcrafter_ablation_no_linguistic_cues_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "history.pkl")
results_ablation_no_linguistic_cues_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_sr = load_results("./results/AGG/DLP_textcrafter_ablation_no_linguistic_cues_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5/", "test_sr.pkl")

with open("/home/tcarta/Automatic_Goal_Generation/DLP/results/AGG/FuNs/eval_results_fun.pkl", 'rb') as file:
    results_FuNs_test = pickle.load(file)
with open("/home/tcarta/Automatic_Goal_Generation/DLP/results/AGG/FuNs/test_results_fun.pkl", 'rb') as file:
    results_FuNs_generalisation = pickle.load(file)

"""task_space=["go_to_tree","collect_wood","place_table","make_wood_pickaxe","collect_stone","collect_coal","place_furnace"]
difficulty_scale = {"go_to_tree": 0,
                    "collect_wood": 1,
                    "place_table": 2.5,
                    "make_wood_pickaxe": 4,
                    "collect_stone": 5,
                    "collect_coal": 5,
                    "place_furnace":8.5}"""
task_space=["go_to_tree","collect_wood","place_table","make_wood_pickaxe","collect_stone","place_furnace"]
difficulty_scale = {"go_to_tree": 0.5,
"collect_wood": 1,
                    "place_table": 2.5,
                    "make_wood_pickaxe": 4,
                    "collect_stone": 5,
                    "place_furnace":8.5}
"""task_space=["go_to_tree", "collect_wood","place_table","make_wood_pickaxe"]
difficulty_scale = {"go_to_tree": 0.5,
                    "collect_wood": 1,
                    "place_table": 2.5,
                    "make_wood_pickaxe": 4
                    }"""
"""plot_mastering_task_space([(results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_sr, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                              (results_ablation_HL_only_gtt_to_pf_test_sr, results_ablation_HL_only_gtt_to_pf_history),
                           results_FuNs_test,
                           ],
                          ["HERAKLES", "POAD", "FuN",],
                          task_space,
                          measuring_time="ep_len",
                          difficulty_scale=difficulty_scale,
                          master_tsk_level=0.8, label=None, avg_window=10, title=None)

plot_mastering_task_space([(results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_sr, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                              (results_ablation_no_update_HL_gtt_to_pf_test_sr, results_ablation_no_update_HL_gtt_to_pf_history),
                           (results_ablation_all_subgoals_accepted_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_sr,
                            results_ablation_all_subgoals_accepted_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_history),
                           (results_ablation_no_linguistic_cues_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_sr,
                            results_ablation_no_linguistic_cues_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_history)],
                          ["HERAKLES", "POAD", "FuN"],
                          task_space,
                          measuring_time="ep_len",
                          difficulty_scale=difficulty_scale,
                          master_tsk_level=0.8, label=None, avg_window=10, title=None)"""


task_space=["go_to_tree","collect_wood","place_table", "go_to_table", "make_wood_pickaxe","go_to_stone", "collect_stone","place_furnace", "go_to_furnace"]


plot_sr_evolution([(results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_sr, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                              (results_ablation_HL_only_gtt_to_pf_test_sr, results_ablation_HL_only_gtt_to_pf_history),
                              results_FuNs_test,
                              ],
                          ["HERAKLES", "POAD", "FuN"],
                             task_space,
                             measuring_time="ep_len",
                             max_len=30000,
                             label=None, title=None)
task_space=["go_to_tree","collect_wood","place_table", "go_to_table", "make_wood_pickaxe","go_to_stone", "collect_stone","go_to_coal", "collect_coal","place_furnace", "go_to_furnace"]

plot_crafter_score_evolution([(results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_sr, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                              (results_ablation_HL_only_gtt_to_pf_test_sr, results_ablation_HL_only_gtt_to_pf_history),
                              results_FuNs_test,
                              ],
                          ["HERAKLES", "POAD", "FuN"],
                             task_space,
                             random_score = 7.69,
                             measuring_time="ep_len",
                             max_len=30000,
                             label=None, avg_window=10, title=None)

# ablation size awr buffer
plot_crafter_score_evolution([(results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_sr, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                              (results_gtt_to_pf_ll_sr_update_256_HL_1e5_AWR_buffer_25000_test_sr, results_gtt_to_pf_ll_sr_update_256_HL_1e5_AWR_buffer_25000_history),
                           (results_gtt_to_pf_ll_sr_update_256_HL_1e5_AWR_buffer_50000_test_sr, results_gtt_to_pf_ll_sr_update_256_HL_1e5_AWR_buffer_50000_history),],
                          ["HERAKLES buffer AWR 1e5", "HERAKLES buffer AWR 2.5e4", "HERAKLES buffer AWR 5e4"],
                          task_space,
                          measuring_time="ep_len",
                             max_len=30000,
                             label=None, avg_window=10, title=None)


# ablation architecture
plot_crafter_score_evolution([(results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_sr, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                              (results_ablation_no_update_HL_gtt_to_pf_test_sr, results_ablation_no_update_HL_gtt_to_pf_history),
                              (results_ablation_all_subgoals_accepted_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_sr,
                            results_ablation_all_subgoals_accepted_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_history),
                           (results_ablation_no_linguistic_cues_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_sr,
                            results_ablation_no_linguistic_cues_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_history)],
                          ["HERAKLES", "HERAKLES_no_update_HL",
                           "HERAKLES_all_subgoals_accepted", "HERAKLES_no_linguistic_cues"],
                             task_space,
                             measuring_time="ep_len",
                             max_len=30000,
                             label=None, avg_window=10, title=None)

# ablation synonym
plot_crafter_score_evolution([(results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_sr, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                              (results_gtt_to_pf_ll_sr_update_256_HL_1e5_generalisation_synonym, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                              ],
                             ["HERAKLES", "HERAKLES_synonym"],
                             task_space,
                             measuring_time="ep_len",
                             label=None, avg_window=10, title=None)

plot_crafter_score_evolution([(results_ablation_HL_only_gtt_to_pf_test_sr, results_ablation_HL_only_gtt_to_pf_history),
                              (results_ablation_HL_only_gtt_to_pf_generalisation_synonym, results_ablation_HL_only_gtt_to_pf_history),
                              ],
                             ["POAD", "POAD_synonym"],
                             task_space,
                             measuring_time="ep_len",
                             max_len=30000,
                             label=None, avg_window=10, title=None)

# ablation compositional
plot_crafter_score_evolution([(results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_sr, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                                (results_gtt_to_pf_ll_sr_update_256_HL_1e5_generalisation_compositional2, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                                (results_gtt_to_pf_ll_sr_update_256_HL_1e5_generalisation_compositional3, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                                (results_gtt_to_pf_ll_sr_update_256_HL_1e5_generalisation_compositional4, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history)],
                                 ["HERAKLES", "HERAKLES_compositional_2", "HERAKLES_compositional_3", "HERAKLES_compositional_4"],
                                    ["collect_wood","place_table","make_wood_pickaxe","collect_stone","collect_coal","place_furnace"],
                                    measuring_time="ep_len",
                                    max_len=30000,
                                    label=None, avg_window=10, title=None)


plot_crafter_score_evolution([(results_ablation_HL_only_gtt_to_pf_test_sr, results_ablation_HL_only_gtt_to_pf_history),
                                (results_ablation_HL_only_gtt_to_pf_generalisation_compositional2, results_ablation_HL_only_gtt_to_pf_history),
                                (results_ablation_HL_only_gtt_to_pf_generalisation_compositional3, results_ablation_HL_only_gtt_to_pf_history),
                                (results_ablation_HL_only_gtt_to_pf_generalisation_compositional4, results_ablation_HL_only_gtt_to_pf_history)],
                                    ["POAD", "POAD_compositional_2", "POAD_compositional_3", "POAD_compositional_4"],
                                    ["collect_wood","place_table","make_wood_pickaxe","collect_stone","collect_coal","place_furnace"],
                                    measuring_time="ep_len",
                                    max_len=30000,
                                    label=None, avg_window=10, title=None)



# track elementary action no need to put "move ... " already taken into account
task_space = ["collect_wood", "place_table"]
tracked_ea = ['chop_tree', 'build_table']
# tracked_sg = ['go_to_tree', 'collect_wood', 'place_table', 'go_to_table', 'make_wood_pickaxe']
tracked_sg = ['go_to_tree', 'collect_wood', 'place_table', 'go_to_table']
sampling_evolution((results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_HL_sampling, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                   task_space, track_ea=tracked_ea, track_sg=tracked_sg, measuring_time="ep_len")

tracked_ea = ['chop_tree', 'build_table', 'craft_wood_pickaxe']
tracked_sg = ['go_to_tree', 'collect_wood', 'place_table', 'go_to_table', 'make_wood_pickaxe']
sampling_evolution((results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_HL_sampling, results_gtt_to_pf_ll_sr_update_256_HL_1e5_history),
                   ["make_wood_pickaxe"], track_ea=tracked_ea, track_sg=tracked_sg, measuring_time="ep_len")





"""gene_task_space = ["collect_wood","collect_2_woods","collect_3_woods","collect_4_woods","make_wood_sword","acquire_wood","install_table","create_wood_pickaxe"]
results_generalisation([results_gtt_to_pf_ll_sr_update_256_HL_1e5_generalisation,
                        results_ablation_HL_only_gtt_to_pf_generalisation,
                        results_FuNs_generalisation],
       ["HERAKLES", "POAD", "FuN"],
                       gene_task_space,
                       10)

task_space=["go_to_tree","collect_wood","place_table","make_wood_pickaxe","collect_stone","place_furnace"]
results_generalisation([results_gtt_to_pf_ll_sr_update_256_HL_1e5_test_sr,
                        results_ablation_HL_only_gtt_to_pf_test_sr,
                        results_FuNs_test],
       ["HERAKLES", "POAD", "FuN"],
                       task_space,
                       10)"""
print()