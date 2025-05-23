import numpy as np
import pickle
import torch
import torch.nn.functional as F
from collections import deque

class GoalSampler:

    def __init__(self, goals):
        self.goals = goals

    def sample(self):
        pass

    def update(self, **kwargs):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass


class RandomGoalSampler(GoalSampler):

    def __init__(self, goals):
        super().__init__(goals)

    def sample(self):
        return self.goals[np.random.randint(0, len(self.goals))]

    def update(self, **kwargs):
        return None

    def load(self, path):
        pass

    def save(self, path):
        pass


class MALPGoalSampler(GoalSampler):

    def __init__(self, goals, malp_args):
        super().__init__(goals)
        self.epsilon_start = malp_args.epsilon_start
        self.epsilon_end = malp_args.epsilon_end
        self.epsilon_decay = malp_args.epsilon_decay
        self.epsilon = self.epsilon_start
        self.step = 0

        self.lp = {g: 0.0 for g in self.goals}
        self.goals_success = {g: deque(maxlen=malp_args.buffer_size) for g in self.goals}
        self.alpha = malp_args.alpha

    def sample(self):
        lp_values = np.array(list(self.lp.values()))
        sum_lp = np.sum(lp_values)

        if np.random.rand() < self.epsilon or sum_lp == 0:
            return self.goals[np.random.randint(0, len(self.goals))]
        else:
            p = lp_values / sum_lp
            return np.random.choice(self.goals, p=p)

    def update(self, **kwargs):

        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.step / self.epsilon_decay)
        self.step += 1

        for g, r in zip(kwargs['goals'], kwargs['returns']):
            if len(self.goals_success[g]) >= 1:
                alp = np.mean([np.abs(r - s) for s in self.goals_success[g]])
                self.lp[g] = self.alpha * alp + (1 - self.alpha) * self.lp[g]

            self.goals_success[g].append(r)



        return {'lp': self.lp}

    def load(self, path):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.epsilon = data['epsilon']
        self.step = data['step']

        if 'lp' in data:
            self.lp = data['lp']

        if 'goals_success' in data:
            self.goals_success = data['goals_success']

    def save(self, path):
        with open(path + "/goal_sampler.pkl", "wb") as f:
            pickle.dump({
                "epsilon": self.epsilon,
                "step": self.step,
                "lp": self.lp,
                "goals_success": self.goals_success,
            }, f)

class SRDiffGoalSampler(GoalSampler):

    def __init__(self, goals, srdiff_args):
        super().__init__(goals)
        self.epsilon_start = srdiff_args.epsilon_start
        self.epsilon_end = srdiff_args.epsilon_end
        self.epsilon_decay = srdiff_args.epsilon_decay
        self.epsilon = self.epsilon_start
        self.step = 0

        self.lp, self.sr, self.sr_delayed = {g: 0.0 for g in self.goals}, {g: 0.0 for g in self.goals}, {g: 0.0 for g in self.goals}
        self.goals_success = {g: deque(maxlen=srdiff_args.buffer_size) for g in self.goals}

    def sample(self):

        lp_values = np.array(list(self.lp.values()))
        sum_lp = np.sum(lp_values)

        if np.random.rand() < self.epsilon or sum_lp == 0:
            return self.goals[np.random.randint(0, len(self.goals))]
        else:
            p = lp_values / sum_lp
            return np.random.choice(self.goals, p=p)

    def update(self, **kwargs):

        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.step / self.epsilon_decay)
        self.step += 1

        for g, r in zip(kwargs['goals'], kwargs['returns']):
            self.goals_success[g].append(r)

        self.compute_lp()

        return {'sr': self.sr, 'sr_delayed': self.sr_delayed, 'lp': self.lp}

    def compute_lp(self):
        for g, buffer in self.goals_success.items():
            if len(buffer) < 2:
                self.sr[g] = 0
                self.sr_delayed[g] = 0
            else:
                buffer_array = np.array(buffer)
                midpoint = len(buffer_array) // 2
                self.sr[g] = np.mean(buffer_array[midpoint:])
                self.sr_delayed[g] = np.mean(buffer_array[:midpoint])

        self.lp = {g: np.abs(self.sr[g] - self.sr_delayed[g]) for g in self.lp.keys()}

    def load(self, path):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.epsilon = data['epsilon']
        self.step = data['step']

        if 'lp' in data:
            self.lp = data['lp']

        if 'sr' in data:
            self.sr = data['sr']

        if 'sr_delayed' in data:
            self.sr_delayed = data['sr_delayed']

        if 'goals_success' in data:
            self.goals_success = data['goals_success']

    def save(self, path):
        with open(path + "/goal_sampler.pkl", "wb") as f:
            pickle.dump({
                "epsilon": self.epsilon,
                "step": self.step,
                "lp": self.lp,
                "goals_success": self.goals_success,
                "sr": self.sr,
                "sr_delayed": self.sr_delayed
            }, f)

class MAGELLANGoalSampler(GoalSampler):

    def __init__(self, goals, agent, saving_path_high_level, config_args):
        super().__init__(goals)

        self.agent = agent
        self.epsilon_start = config_args.magellan_args.epsilon_start
        self.epsilon_end = config_args.magellan_args.epsilon_end
        self.epsilon_decay = config_args.magellan_args.epsilon_decay
        self.saving_path_high_level = saving_path_high_level

        self.epsilon = self.epsilon_start
        self.step = 0

        self.n_llm_processes = config_args.lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes
        self.recompute_freq = config_args.magellan_args.recompute_freq

    def sample(self, observations, history_sr_hl_estimator, base_prompt):

        if len(self.agent.update([""] * self.n_llm_processes, [[""]] * self.n_llm_processes,
                                 retrieve_weights_buffer=True)) > 0:
            sr, sr_delayed, lp = self.compute_lp(observations)
            lp_values = np.array(lp)
            history_sr_hl_estimator.append([base_prompt, lp, sr, sr_delayed])
            sum_lp = np.sum(lp_values)
        else:
            sum_lp = 0

        if np.random.rand() < self.epsilon or sum_lp == 0:
            return self.goals[np.random.randint(0, len(self.goals))]
        else:
            p = lp_values / sum_lp
            return np.random.choice(self.goals, p=p)

    def update(self, **kwargs):

        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.step / self.epsilon_decay)
        self.step += 1

        if self.step % self.recompute_freq == 0:
            self.agent.update([""] * self.n_llm_processes, [[""]] * self.n_llm_processes,
                              update_buffer_hl_sr_estimator=True,
                              saving_path_buffer_sr_hl_estimator_weights=kwargs['saving_path_buffer_sr_hl_estimator_weights'])


    def compute_lp(self, observations):

        # Compute delayed sr
        output = self.agent.custom_module_fns(['sr_hl_estimator_delayed'],
                                              contexts=observations,
                                              candidates=[[" "] for _ in range(len(observations))],
                                              require_grad=False, peft_adapter="sr_HL_adapters_delayed",
                                              pad_contexts=False)
        sr_delayed = F.sigmoid(torch.stack([_o['sr_hl_estimator_delayed']["sr_hl_estimated"] for _o in output]).squeeze()).numpy()

        # Compute current sr
        output = self.agent.custom_module_fns(["sr_hl_estimator"],
                                              contexts=observations,
                                              candidates=[[" "] for _ in range(len(observations))],
                                              require_grad=False, peft_adapter="sr_HL_adapters",
                                              pad_contexts=False)
        sr = F.sigmoid(torch.stack([_o["sr_hl_estimator"]["sr_hl_estimated"] for _o in output]).squeeze()).numpy()

        # Compute absolute lp
        lp = np.abs(sr - sr_delayed)

        # For numerical stability
        try:
            lp[lp < 0.01] = 0.0
        except:
            lp = np.array([lp])
            lp[lp < 0.01] = 0.0

        return sr, sr_delayed, lp

    def load(self, path):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.epsilon = data['epsilon']
        self.step = data['step']

    def save(self, path):
        with open(path + "/goal_sampler.pkl", "wb") as f:
            pickle.dump({
                "epsilon": self.epsilon,
                "step": self.step,
            }, f)