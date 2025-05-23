import numpy as np

from paral_text_crafterenv import ParallelEnv
from crafter_text_gc import CrafterTextGC

class CrafterTextGoalCondition:
    def __init__(self, config_dict):
        self.n_parallel = config_dict["number_envs"]

        envs = []
        for i in range(config_dict["number_envs"]):
            if "description" in config_dict.keys():
                desc = config_dict["description"]
            else:
                desc = "long"
            if "conversion_name_dictionary" in config_dict.keys():
                conversion_name_dictionary = config_dict["conversion_name_dictionary"]
            else:
                conversion_name_dictionary = None
            if "reset_word_only_at_episode_termination" in config_dict.keys():
                reset_word_only_at_episode_termination = config_dict["reset_word_only_at_episode_termination"]
            else:
                reset_word_only_at_episode_termination = False
            if "length" in config_dict.keys():

                env = CrafterTextGC(train=True, seed=0, length=config_dict["length"], description=desc,
                                    reset_word_only_at_episode_termination=reset_word_only_at_episode_termination,
                                    conversion_name_dictionary=conversion_name_dictionary)
            else:
                env = CrafterTextGC(train=True, seed=0, description=desc,
                                    reset_word_only_at_episode_termination=reset_word_only_at_episode_termination,
                                    conversion_name_dictionary=conversion_name_dictionary)

            env.add_possible_actions(config_dict["action_space"])
            if "elementary_action_space" in config_dict.keys():
                env.add_possible_elementary_actions(config_dict["elementary_action_space"])
            else:
                env.add_possible_elementary_actions(config_dict["action_space"])
            env._seed=100 * config_dict["seed"] + i
            if "hl_traj_len_max" in config_dict.keys():
                env.hl_traj_len_max = config_dict["hl_traj_len_max"][i]
            envs.append(env)
        self._action_space = config_dict["action_space"]
        if "MAGELLAN_goal_sampler" in config_dict.keys():
            MAGELLAN_goal_sampler = config_dict["MAGELLAN_goal_sampler"]
        else:
            MAGELLAN_goal_sampler = False

        if "goal_sampler" in config_dict.keys():
            if "proba_subgoal_estimator" in config_dict.keys():
                self._env = ParallelEnv(envs, goal_sampler=config_dict["goal_sampler"],
                                        proba_subgoal_estimator=config_dict["proba_subgoal_estimator"],
                                        MAGELLAN_goal_sampler=MAGELLAN_goal_sampler)
            else:
                self._env = ParallelEnv(envs, goal_sampler=config_dict["goal_sampler"],
                                        MAGELLAN_goal_sampler=MAGELLAN_goal_sampler)
        else:
            if "proba_subgoal_estimator" in config_dict.keys():
                self._env = ParallelEnv(envs,
                                        proba_subgoal_estimator=config_dict["proba_subgoal_estimator"],
                                        )
            else:
                self._env = ParallelEnv(envs)

        self.goals = [None]*config_dict["number_envs"]

    def update_hl_traj_len_max(self, hl_traj_len_max=None):
        if hl_traj_len_max is not None:
            self._env.update_hl_traj_len_max(hl_traj_len_max)
        else:
            raise NotImplementedError("You need to give a hl_traj_len_max, here you put None")

    def update_action_space(self, list_actions):
        self._env.update_action_space(list_actions)

    def update_stop_mask(self, new_stop_mask):
        self._env.stop_mask = new_stop_mask

    def get_stop_mask(self):
        return self._env.stop_mask[:]

    def set_goals(self, list_goals:list):
        self._env.goals = list_goals
    def get_env(self):
        return self._env
    def get_stop_mask(self):
        return self._env.stop_mask[:]

    def reset(self, task_space=None, proba_subgoal=None, update_subgoal=None, history_sr_ll_estimator=None,
              history_sr_hl_estimator=None):
        obs, infos = self._env.reset(task_space, proba_subgoal, update_subgoal, history_sr_ll_estimator,
                                     history_sr_hl_estimator)
        return obs, infos

    def step(self, actions, steps_high_level_policy=None, hl_traj_len_max=None, sub_goal_instructions=None,
             len_low_level=None, ll_traj_len_max=None, task_space=None, proba_subgoal=None, update_subgoal=None,
             history_sr_ll_estimator=None, history_sr_hl_estimator=None):
        obs, rews, dones, infos = self._env.step(actions, steps_high_level_policy,
                                                 hl_traj_len_max, sub_goal_instructions,
                                                 len_low_level, ll_traj_len_max, task_space,
                                                 proba_subgoal, update_subgoal, history_sr_ll_estimator,
                                                 history_sr_hl_estimator)
        return obs, rews, dones, infos
