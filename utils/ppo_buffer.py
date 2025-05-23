'''
PPO implementation taken from https://github.com/openai/spinningup
'''
import copy

import numpy as np
import torch
from sympy.abc import delta

from . import discount_cumsum, combined_shape

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, gamma=0.99, lam=0.95):
        self.obs_buf = [None for _ in range(size)]
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, **kwargs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for k, v in kwargs.items():
            if not hasattr(self, k+"_buf"):
                setattr(self, k+"_buf", [None for _ in range(self.max_size)])
            getattr(self, k+"_buf")[self.ptr] = v
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        if hasattr(self, 'end_token_seq_buf'):
            end_token_seqs = np.append(self.end_token_seq_buf[path_slice], True)
            # gamma_power = np.cumsum(end_token_seqs)
            gamma_power = []
            for end_tk_seq_idx, _ in enumerate(end_token_seqs):
                if len(gamma_power) == 0:
                    gamma_power.append(0)
                else:
                    if end_token_seqs[end_tk_seq_idx-1]:
                        gamma_power.append(1+gamma_power[end_tk_seq_idx-1])
                    else:
                        gamma_power.append(gamma_power[end_tk_seq_idx-1])
            gamma_power = np.array(gamma_power)
            gamma_discounted = self.gamma ** gamma_power
            lamda_power = np.ones(len(end_token_seqs)-1)
            lamda_power[0] = 0
            lamda_power = np.cumsum(lamda_power)
            lamda_discounted = self.lam ** lamda_power
            # print("act: \n{} \nend token seqs: \n{} \ngamma power: \n{} \ngamma discounted: \n{} \nlamda power: \n{} \nlamda discounted: \n{} \nlambda_discounted*gamma_discounted: \n{} \n".format(self.act_buf[path_slice], end_token_seqs, gamma_power, gamma_discounted, lamda_power, lamda_discounted, lamda_discounted * gamma_discounted[:-1]))
            # token level
            # we do not discount inside an utterance
            deltas = rews[:-1] + [(end_token_seqs[:-1]*(self.gamma-1)) + 1] * vals[1:] - vals[:-1]
            discount_mat = np.matmul(deltas.reshape(deltas.shape[1], 1), np.expand_dims(lamda_discounted * gamma_discounted[:-1], axis=0))
            self.adv_buf[path_slice] = np.array([discount_mat.diagonal(-d).sum() for d in range(len(rews[:-1]))])
            # print("=======buffer adv calc===== \n last val: {} detas:{} adv_buf:{}".format(last_val, deltas, self.adv_buf[path_slice]))
            discount_mat = np.matmul(rews.reshape((len(rews), 1)), np.expand_dims(gamma_discounted, axis=0))

            self.ret_buf[path_slice] = np.array([discount_mat.diagonal(-d).sum() for d in range(len(rews[:-1]))])
        else:
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]


        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size  # buffer has to be full before you can get
        path_slice = slice(0, self.ptr)
        # the next two lines implement the advantage normalization trick
        """adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std"""
        data = {k[:-4]: getattr(self, k) for k in dir(self) if k.endswith('_buf')}
        """data = dict(obs=self.obs_buf, possible_act=self.possible_act_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)"""

        dict_return = dict()

        for k, v in data.items():
            if not isinstance(v, list):
                dict_return[k] = torch.as_tensor(v[path_slice].copy(), dtype=torch.float32)
            else:
                dict_return[k] = copy.deepcopy(v[path_slice])
        self.ptr, self.path_start_idx = 0, 0
        return dict_return
        """return {
            k: torch.as_tensor(v, dtype=torch.float32)
            if not isinstance(v, list) else copy.deepcopy(v)
            for k, v in data.items()
        }"""