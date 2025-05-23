import gym
import torch
import numpy as np
from copy import deepcopy

from accelerate.commands.config.default import description

from crafter_text_gc.craftertextgc import describe_frame

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def update_action_space(proba_subgoal, action_space, env):
    for sg in proba_subgoal.keys():
        if proba_subgoal[sg] >= np.random.rand():
            parts = sg.split('_')
            name_sg = parts[0]
            for p in parts[1:]:
                name_sg += ' '+p
            action_space.append(name_sg)
    env.add_possible_actions(action_space)

def update_obs_infos_HL(env, obs, info, task_space, update_subgoal=None, history_sr_ll_estimator=None,
                        proba_subgoal=None, proba_subgoal_estimator=None, goal_sampler=None,
                        MAGELLAN_goal_sampler=False, history_sr_hl_estimator=None):


    # sample goal if there is a MAGELLAN goal sampler
    if (goal_sampler is not None) and MAGELLAN_goal_sampler:
        description_goal =  info["description_proba_goal_estimator"]
        list_goal_sampler_prompts = []
        for tsk in task_space:
            list_goal_sampler_prompts.append(description_goal.format(tsk.replace("_", " ")))
        goal = goal_sampler.sample(list_goal_sampler_prompts, history_sr_hl_estimator, description_goal)
        env.task_idx = env.task2id[goal]
        env.update_given_ach()
        env.task_progress = 0
        env.task_str = env.id2task[env.task_idx]
        info["goal"] = env.task_str
        info["description_goal_sampler"] = info["description_proba_goal_estimator"].format(info["goal"])

    # sample subgoals
    if proba_subgoal is not None:
        env.update_action_space(proba_subgoal)

    if proba_subgoal_estimator is not None:
        description = info["description_proba_subgoal_estimator"]
        list_proba_subgoal_estimator_prompts = []
        for tsk in task_space:
            list_proba_subgoal_estimator_prompts.append(description.format(tsk.replace("_", " ")))
        proba_subgoal_estimated = proba_subgoal_estimator(list_proba_subgoal_estimator_prompts,
                                                          update_subgoal, history_sr_ll_estimator)
        env.update_action_space(proba_subgoal_estimated)
    info["action_space"] = env.possible_actions
    info["possible_actions"] = env.possible_actions
    obs["textual_obs"] = describe_frame(info, env.description)

class ParallelEnv(gym.Env):
    """Parallel environment that holds a list of environments and can
       evaluate a low-level policy for use in reward shaping.
    """

    def __init__(self,
                 envs,  # List of environments
                 goal_sampler=None,
                 proba_subgoal_estimator=None,
                 MAGELLAN_goal_sampler=False):
        assert len(envs) >= 1, "No environment provided"
        self.envs = envs
        self.num_envs = len(self.envs)
        self.device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        # self.spec = deepcopy(self.envs[0].unwrapped.spec)
        # self.spec_id = f"ParallelShapedEnv<{self.spec.id}>"
        # self.env_name = self.envs[0].unwrapped.spec.id
        self.action_space = self.envs[0].action_space

        self.customize_max_steps = None

        # stop mask boolean variable to know if we execute a step in the env or not
        self.stop_mask = [False for _ in range(self.num_envs)]

        self.envs_per_proc = 64

        # Setup arrays to hold current observation and timestep
        # for each environment
        self.obss = []
        self.goals = []
        self.ts = np.array([0 for _ in range(self.num_envs)])

        # A goal sampler
        self.goal_sampler=goal_sampler
        self.MAGELLAN_goal_sampler = MAGELLAN_goal_sampler

        # proba_subgoal_estimator
        self.proba_subgoal_estimator = proba_subgoal_estimator

        # speed_memory
        self.none = np.array([None for _ in range(self.num_envs)])

    def __len__(self):
        return self.num_envs

    def gen_obs(self):
        return self.obss

    def render(self, mode="human", highlight=False):
        """Render a single environment"""
        return self.envs[0].render(mode, highlight)

    def reset(self,  task_space=None, proba_subgoal=None, update_subgoal=None, history_sr_ll_estimator=None,
              history_sr_hl_estimator=None):
        """Reset all environments"""
        self.obss = []
        infos = []
        goal_sampler = self.goal_sampler
        assert proba_subgoal is None or self.proba_subgoal_estimator is None
        if len(self.goals) == 0:
            self.goals = [None] * self.num_envs
        for env, g in zip(self.envs, self.goals):
            if g is None and not self.MAGELLAN_goal_sampler:
                g = goal_sampler.sample()
            obs, info = env.reset(g)
            update_obs_infos_HL(env, obs, info,task_space, update_subgoal, history_sr_ll_estimator,
                                proba_subgoal=proba_subgoal, proba_subgoal_estimator=self.proba_subgoal_estimator,
                                MAGELLAN_goal_sampler=self.MAGELLAN_goal_sampler, goal_sampler=goal_sampler,
                                history_sr_hl_estimator=history_sr_hl_estimator)
            self.obss += [obs]
            infos += [info]

        return [obs for obs in self.obss], infos

    def step(self, actions, steps_high_level_policy=None,  hl_traj_len_max=None, sub_goal_instructions=None,
             len_low_level=None, ll_traj_len_max=None, task_space=None, proba_subgoal=None, update_subgoal=None,
             history_sr_ll_estimator=None, history_sr_hl_estimator=None):
        """Complete a step and evaluate low-level policy / termination
           classifier as needed depending on reward shaping scheme.

           Returns:  obs: list of environment observations,
                     reward: np.array of extrinsic rewards,
                     done: np.array of booleans,
                     info: depends on self.reward_shaping. Output can be used
                           to shape the reward.
        """
        # Make sure input is numpy array
        if type(actions) != np.ndarray:
            if type(actions) == list or type(actions) == int:
                actions = np.array(actions)
            elif type(actions) == torch.Tensor:
                actions = actions.cpu().numpy()
            else:
                raise TypeError
        actions_to_take = actions.copy()

        if steps_high_level_policy is not None:
            if type(steps_high_level_policy) != np.ndarray:
                if type(steps_high_level_policy) == list or type(steps_high_level_policy) == int:
                    steps_high_level_policy = np.array(steps_high_level_policy)
                elif type(steps_high_level_policy) == torch.Tensor:
                    steps_high_level_policy = steps_high_level_policy.cpu().numpy()
                else:
                    raise TypeError
        else:
            steps_high_level_policy = self.none

        if hl_traj_len_max is None:
            hl_traj_len_max = [None for _ in range(self.num_envs)]
        else:
            if type(hl_traj_len_max) != np.ndarray:
                if type(hl_traj_len_max) == list or type(hl_traj_len_max) == int:
                    hl_traj_len_max = np.array(hl_traj_len_max)
                elif type(hl_traj_len_max) == torch.Tensor:
                    hl_traj_len_max = hl_traj_len_max.cpu().numpy()
                else:
                    raise TypeError
        if sub_goal_instructions is None:
            sub_goal_instructions = self.none
        else:
            if type(sub_goal_instructions) != np.ndarray:
                if type(sub_goal_instructions) == list or type(sub_goal_instructions) == int:
                    sub_goal_instructions = np.array(sub_goal_instructions)
                elif type(sub_goal_instructions) == torch.Tensor:
                    sub_goal_instructions = sub_goal_instructions.cpu().numpy()
                else:
                    raise TypeError
        if len_low_level is None:
            len_low_level = self.none
        else:
            if type(len_low_level) != np.ndarray:
                if type(len_low_level) == list or type(len_low_level) == int:
                    len_low_level = np.array(len_low_level)
                elif type(len_low_level) == torch.Tensor:
                    len_low_level = len_low_level.cpu().numpy()
                else:
                    raise TypeError
        if ll_traj_len_max is None:
            ll_traj_len_max = self.none
        else:
            if type(ll_traj_len_max) != np.ndarray:
                if type(ll_traj_len_max) == list or type(ll_traj_len_max) == int:
                    ll_traj_len_max = np.array(ll_traj_len_max)
                elif type(ll_traj_len_max) == torch.Tensor:
                    ll_traj_len_max = ll_traj_len_max.cpu().numpy()
                else:
                    raise TypeError

        # Make a step in the environment
        results = []
        goal_sampler = self.goal_sampler
        if len(self.goals)==0:
            goals = [None]*self.num_envs
            if self.goal_sampler is None:
                raise NotImplementedError("you need either a goal or a goal sampler.")
        else:
            goals=self.goals
        for idx_env, (env, a, stopped, len_traj_hl, hl_len_traj_max, verifier, len_traj_ll, ll_len_traj_max, goal) in enumerate(zip(self.envs,
                                                                                                                actions_to_take,
                                                                                                                self.stop_mask,
                                                                                                                steps_high_level_policy,
                                                                                                                hl_traj_len_max,
                                                                                                                sub_goal_instructions,
                                                                                                                len_low_level,
                                                                                                                ll_traj_len_max,
                                                                                                                goals)):
            if len_traj_hl is not None:
                env.nbr_steps_hl=int(len_traj_hl)

            # print("env {} stopped:{}".format(idx_env, stopped))
            if not stopped:
                if verifier is not None:
                    past_achievement = env._player.achievements[verifier]
                old_goal = env.task_str
                if len_traj_ll is not None and len_traj_ll>1:
                    # steps in env are counted from the HL agent point of view
                    env._step -= 1
                obs, reward, done, _, info = env.step(a)
                # print("env {} old goal: {} reward:{}".format(idx_env, old_goal, reward))

                if verifier is not None:
                    verif = ((env._player.achievements[verifier] - past_achievement)>0)
                    if verif:
                        reward_ll = 1 # (1 - 0.9 * (len_traj_ll / ll_len_traj_max))
                        done_ll = True
                    else:
                        reward_ll = 0.0
                        done_ll = False

                    if len_traj_ll >= ll_len_traj_max:
                        done_ll = True

                    if done_ll:
                        # end of trajectory for the low level
                        # the high level has to take a new decision thus we update its action space
                        update_obs_infos_HL(env, obs, info,task_space, update_subgoal, history_sr_ll_estimator,
                                            proba_subgoal=None, proba_subgoal_estimator=self.proba_subgoal_estimator)
                else:
                    # elementary action done by the high level
                    # the high level has to take a new decision thus we update its action space
                    verif = None
                    reward_ll = None
                    done_ll = None
                    update_obs_infos_HL(env, obs, info,task_space, update_subgoal, history_sr_ll_estimator,
                                        proba_subgoal=None, proba_subgoal_estimator=self.proba_subgoal_estimator)

                if len_traj_hl is not None:
                    manual_reset = len_traj_hl >= hl_len_traj_max
                else:
                    manual_reset = None

                if manual_reset is not None:
                    # we reset the environment
                    # if there is no sub-goal
                    # if the subgoal is done
                    # the last condition is there to ensure the low level terminate its trajectory
                    manual_reset = manual_reset and (verif is None or done_ll)
                    if done or manual_reset:
                        # update goal sampler if there is one
                        """if goal_sampler is not None:
                            goal_sampler.update(goals=[info['goal']], returns=[reward])"""
                        env.nbr_steps_hl=0
                        # sample a goal if there is a goals sampler
                        if goal_sampler is not None:
                            previous_goal = info['goal']
                            # print("A1 env {} previous goal {} old_goal {} with reward {}".format(idx_env, previous_goal, old_goal, reward))
                            if not self.MAGELLAN_goal_sampler:
                                g = goal_sampler.sample()
                            else:
                                g = None
                            if not env.reset_word_only_at_episode_termination:
                                # you have to manually reset the world in this case
                                env.reset_world = True
                            """action_space = deepcopy(env.possible_elementary_actions)
                            if proba_subgoal is not None:
                                # update the action space with LL task added to the action space
                                # in a non-deterministic way
                                update_action_space(proba_subgoal, action_space, env)"""

                            obs, info = env.reset(g)
                            # the high level has to take a new decision thus we update its action space
                            update_obs_infos_HL(env, obs, info,task_space, update_subgoal, history_sr_ll_estimator,
                                                proba_subgoal=proba_subgoal,
                                                proba_subgoal_estimator=self.proba_subgoal_estimator,
                                                MAGELLAN_goal_sampler=self.MAGELLAN_goal_sampler,
                                                goal_sampler=goal_sampler,
                                                history_sr_hl_estimator=history_sr_hl_estimator)
                            info["old_goals"] = previous_goal
                            info["old_goals_reward"] = reward
                        else:
                            obs, info = env.reset(goal)
                            # the high level has to take a new decision thus we update its action space
                            update_obs_infos_HL(env, obs, info,task_space, update_subgoal, history_sr_ll_estimator,
                                                proba_subgoal=proba_subgoal,
                                                proba_subgoal_estimator=self.proba_subgoal_estimator)
                        # print("max steps:{}\n".format(env.env.env.max_steps))
                        done = True
                else:
                    if done:
                            # update goal sampler if there is one
                        """if goal_sampler is not None:
                             goal_sampler.update(goals=[info['goal']], returns=[reward])"""
                        # sample a goal if there is a goals sampler
                        if goal_sampler is not None:
                            previous_goal = info['goal']
                            print("A2 previous goal {} with reward {}".format(previous_goal, reward))
                            if not self.MAGELLAN_goal_sampler:
                                g = goal_sampler.sample()
                            else:
                                g = None
                            if not env.reset_word_only_at_episode_termination:
                                # you have to manually reset the world in this case
                                env.reset_world = True

                            obs, info = env.reset(g)
                            # the high level has to take a new decision thus we update its action space
                            update_obs_infos_HL(env, obs, info,task_space, update_subgoal, history_sr_ll_estimator,
                                                proba_subgoal=proba_subgoal,
                                                proba_subgoal_estimator=self.proba_subgoal_estimator)
                            info["old_goals"] = previous_goal
                            info["old_goals_reward"] = reward
                        else:
                            obs, info = env.reset(goal)
                            # the high level has to take a new decision thus we update its action space
                            update_obs_infos_HL(env, obs, info,task_space, update_subgoal, history_sr_ll_estimator,
                                                proba_subgoal=proba_subgoal,
                                                proba_subgoal_estimator=self.proba_subgoal_estimator)
                        """if data[2] is not None:
                            env.env.env.max_steps = data[2]"""
                        # print("max steps:{}\n".format(env.env.env.max_steps))

                info["verifier"] = verif
                info["reward_ll"] = reward_ll
                info["done_ll"] = done_ll
                results.append((obs, reward, done, info))
                self.obss[idx_env] = obs
            else:
                results.append((None, 0, False, None))

        if self.goal_sampler is not None and not self.MAGELLAN_goal_sampler:
            self.goal_sampler.update(goals= [res[3]["old_goals"] for res in results if (res[3] is not None and "old_goals" in res[3].keys())],
                                     returns=[res[3]["old_goals_reward"] for res in results if (res[3] is not None and "old_goals" in res[3].keys())])

        obs, reward, done, info = zip(*results)
        reward = np.array(reward)
        done_mask = np.array(done)

        self.ts += 1
        self.ts[done_mask] *= 0

        return [obs for obs in self.obss], reward, done_mask, info

    def update_hl_traj_len_max(self, hl_traj_len_max=None):
        """Request all processes to modify hl_traj_len_max"""
        logger.info("requesting update_hl_traj_len_max")
        for env in self.envs:
            env.hl_traj_len_max = hl_traj_len_max

    def update_action_space(self, list_actions=None):
        """Request all processes to modify hl_traj_len_max"""
        logger.info("sending new action space")
        for env in self.envs:
            env.add_possible_actions(list_actions)
        logger.info("return update action space")
