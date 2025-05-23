'''
PPO implementation taken from https://github.com/openai/spinningup
'''
import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from babyai.rl.utils.dictlist import DictList
from collections import deque

def expq_x(x, q):
    return torch.pow(torch.clip(1 + (q-1)*x, min=0), 1/(q-1))

class AWRBufferMultiSkills:
    """
    A buffer for storing trajectories experienced by a AWR agent interacting
    with the environment on the long term with a fifo queue
    Each skill is stored in a different buffer, during update to avoid a skill to contribute more in the gradient
    by having more transitions than another skill, we sample the same number of transitions from each skill
    """

class AWRBuffer:
    """
    A buffer for storing trajectories experienced by a AWR agent interacting
    with the environment on the long terme with a fifo queue
    """
    def __init__(self, size, gamma=0.99, lam=0.95, device="cpu"):
        self.obs_buf = {"image": deque(maxlen=size), "instr": deque(maxlen=size),
                        "memory_actor": deque(maxlen=size), "memory_critic": deque(maxlen=size)}
        self.act_buf = deque(maxlen=size)
        self.rew_buf = deque(maxlen=size)
        self.end_traj_buf = deque(maxlen=size)

        self.ret_buf = deque(maxlen=size)
        self.val_buf = deque(maxlen=size)

        self.gamma, self.td_lam, self.max_size, self.len_buffer = gamma, lam, size, 0
        self.index_shuffle = np.arange(self.len_buffer)
        self.device = device

    def load(self, save_pah_low_level):
        """
        Load the buffer from a pickle file
        """
        with open(save_pah_low_level, 'rb') as f:
            data = pickle.load(f)
            self.obs_buf = data["obs"]
            self.act_buf = data["act"]
            self.rew_buf = data["rew"]
            self.end_traj_buf = data["end_traj"]
            self.len_buffer = len(self.act_buf)
            self.index_shuffle = np.arange(self.len_buffer)
        print("buffer size at loading: {}".format(self.len_buffer))
    def save(self, save_path):
        """
        Save the buffer to a pickle file
        """
        data = {"obs": self.obs_buf, "act": self.act_buf, "rew": self.rew_buf, "end_traj": self.end_traj_buf}
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print("buffer size at saving: {}".format(self.len_buffer))

    def store(self, trajectories, no_memory=False):
        for traj in trajectories:
            for k, _ in traj.items():
                if k == "obs":
                    for idx in range(len(traj[k])):
                        if not no_memory:
                            self.obs_buf["image"].append(traj[k][idx][0].image.to("cpu"))
                            if "instr" in traj[k][idx][0]:
                                self.obs_buf["instr"].append(traj[k][idx][0].instr.to("cpu"))

                            self.obs_buf["memory_actor"].append(traj[k][idx][1].to("cpu"))
                            self.obs_buf["memory_critic"].append(traj[k][idx][2].to("cpu"))
                        else:
                            self.obs_buf["image"].append(traj[k][idx].image.to("cpu"))
                            if "instr" in traj[k][idx]:
                                self.obs_buf["instr"].append(traj[k][idx].instr.to("cpu"))
                elif k == "act":
                    for idx in range(len(traj[k])):
                        self.act_buf.append(traj[k][idx])
                elif k == "rew":
                    for idx in range(len(traj[k])):
                        self.rew_buf.append(traj[k][idx])
                elif k == "end_traj":
                    for idx in range(len(traj[k])):
                        self.end_traj_buf.append(traj[k][idx])

        self.len_buffer = len(self.act_buf)
        self.index_shuffle = np.arange(self.len_buffer)
        print("buffer size after storing: {}".format(self.len_buffer))

    def compute_r_lambda(self, model, no_memory=False):
        """
        Compute the lambda return and the advantage for each trajectory
        """

        self.ret_buf = deque(maxlen=self.max_size)
        self.val_buf = deque(maxlen=self.max_size)

        preprocessed_obs = DictList()

        counter_start, counter_end = 0, 1
        for end_traj_idx in range(len(self.end_traj_buf)):

            # calculate the value of the state along the trajectory
            if self.end_traj_buf[end_traj_idx] > 0:
                preprocessed_obs.image = torch.cat([self.obs_buf["image"][idx].to(self.device).unsqueeze(0) for idx in range(counter_start, counter_end)])
                if len(self.obs_buf["instr"]) > 0:
                    preprocessed_obs.instr = torch.cat([self.obs_buf["instr"][idx].to(self.device).unsqueeze(0) for idx in range(counter_start, counter_end)])
                if not no_memory:
                    memory_low_level = torch.cat([self.obs_buf["memory_critic"][idx].to(self.device).unsqueeze(0) for idx in range(counter_start, counter_end)])

                with torch.no_grad():
                    if not no_memory:
                        model_results = model(preprocessed_obs, memory_low_level)
                    else:
                        model_results = model(preprocessed_obs)
                    self.val_buf.extend(model_results['value'])

                rewards = torch.cat([self.rew_buf[idx].to(self.device).unsqueeze(0) for idx in range(counter_start, counter_end)])
                if self.end_traj_buf[end_traj_idx] == 1:
                    # this is the true end of the trajectory no need to use a value function to bootstrap
                    values = torch.cat([model_results['value'][:-1], torch.tensor([0.0], device=self.device)])
                elif self.end_traj_buf[end_traj_idx] == 2:
                    # this is not the true end of the trajectory we need to bootstrap the value function
                    values = torch.cat([model_results['value'][:-1], self.val_buf[counter_end-1].to(self.device).unsqueeze(0)])
                else:
                    raise ValueError("The trajectory end flag is not correct")

                r_td_lambda = torch.zeros_like(rewards)
                for idx in reversed(range(len(rewards))):
                    if idx == len(rewards) - 1:
                        r_td_lambda[idx] = rewards[idx] + self.gamma * values[idx]
                    else:
                        r_td_lambda[idx] = rewards[idx] + self.gamma * ((1.0-self.td_lam) * values[idx+1] + self.td_lam * r_td_lambda[idx + 1])

                self.ret_buf.extend(r_td_lambda)

                counter_start = counter_end
            counter_end += 1


    def shuffle(self):
        """
        Shuffle the buffer
        """
        self.index_shuffle = np.random.permutation(self.len_buffer)

    def compute_critic_loss(self, model, critic_batch_idx, no_memory=False):
        """
        Compute the critic loss
        """
        preprocessed_obs = DictList()
        batch_index = [self.index_shuffle[idx] for idx in critic_batch_idx]
        preprocessed_obs.image = torch.cat([self.obs_buf["image"][idx].to(self.device).unsqueeze(0) for idx in batch_index])
        if len(self.obs_buf["instr"]) > 0:
            preprocessed_obs.instr = torch.cat([self.obs_buf["instr"][idx].to(self.device).unsqueeze(0) for idx in batch_index])
        if not no_memory:
            memory_low_level = torch.cat([self.obs_buf["memory_critic"][idx].to(self.device).unsqueeze(0) for idx in batch_index])
        returns = torch.cat([self.ret_buf[idx].to(self.device).unsqueeze(0) for idx in batch_index])
        if not no_memory:
            estimated_values = model(preprocessed_obs, memory_low_level)['value']
        else:
            estimated_values = model(preprocessed_obs)['value']
        critic_loss = torch.nn.functional.mse_loss(estimated_values, returns)

        return critic_loss

    def compute_policy_loss(self, model, policy_batch_idx, beta_awr, normalisation=False, tsallis_reg=False, q=None, no_memory=False):
        """
        Compute the policy loss
        """
        preprocessed_obs = DictList()
        batch_index = [self.index_shuffle[idx] for idx in policy_batch_idx]
        preprocessed_obs.image = torch.cat([self.obs_buf["image"][idx].to(self.device).unsqueeze(0) for idx in batch_index])
        if len(self.obs_buf["instr"]) > 0:
            preprocessed_obs.instr = torch.cat([self.obs_buf["instr"][idx].to(self.device).unsqueeze(0) for idx in batch_index])
        if not no_memory:
            memory_low_level = torch.cat([self.obs_buf["memory_actor"][idx].to(self.device).unsqueeze(0) for idx in batch_index])

        returns = torch.cat([self.ret_buf[idx].to(self.device).unsqueeze(0) for idx in batch_index])
        values = torch.cat([self.val_buf[idx].to(self.device).unsqueeze(0) for idx in batch_index])
        actions = torch.cat([self.act_buf[idx].to(self.device).unsqueeze(0) for idx in batch_index])

        advantages = returns - values
        # scaled_advantages = (advantages-advantages.mean()) / (advantages.std() + torch.tensor(1e-8)) # we add a small value to avoid division by zero

        if not no_memory:
            dist = model(preprocessed_obs, memory_low_level)['dist']
        else:
            dist = model(preprocessed_obs)['dist']
        entropy = dist.entropy()

        if normalisation:
            if tsallis_reg:
                normalized_weights = expq_x(beta_awr * advantages, q)
                normalized_weights = torch.clip(normalized_weights, 1e-5, 10000)
            else:
                normalized_weights = len(advantages) * F.softmax(beta_awr * advantages, dim=0)

            policy_loss = -dist.log_prob(actions) * normalized_weights
            return policy_loss.mean(), entropy.mean(), normalized_weights.max(), normalized_weights.min()

        else:
            exponential_weights = torch.exp(beta_awr * advantages)
            policy_loss = -dist.log_prob(actions) * torch.clamp(exponential_weights, max=20)
            return policy_loss.mean(), entropy.mean(), torch.clamp(exponential_weights, max=20).mean(), len(exponential_weights[exponential_weights > 20])

    def compute_policy_loss_simple(self, model, policy_batch_idx, beta_awr):
        """
        Compute the policy loss with no weights
        """
        preprocessed_obs = DictList()
        batch_index = [self.index_shuffle[idx] for idx in policy_batch_idx]
        preprocessed_obs.image = torch.cat([self.obs_buf["image"][idx].to(self.device).unsqueeze(0) for idx in batch_index])
        if len(self.obs_buf["instr"]) > 0:
            preprocessed_obs.instr = torch.cat([self.obs_buf["instr"][idx].to(self.device).unsqueeze(0) for idx in batch_index])
        memory_low_level = torch.cat([self.obs_buf["memory_actor"][idx].to(self.device).unsqueeze(0) for idx in batch_index])

        actions = torch.cat([self.act_buf[idx].to(self.device).unsqueeze(0) for idx in batch_index])

        dist = model(preprocessed_obs, memory_low_level)['dist']
        entropy = dist.entropy()
        policy_loss = -dist.log_prob(actions)

        return policy_loss.mean(), entropy.mean()


class AWRBuffer_store_trajectories:
    """
    A buffer for storing trajectories experienced by a AWR agent interacting
    with the environment during the collect phase
    """

    def __init__(self, size):
        self.size = size
        self.obs_buf = [None for _ in range(size)]
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.end_traj_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, **kwargs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        for k, v in kwargs.items():
            if not hasattr(self, k+"_buf"):
                setattr(self, k+"_buf", [None for _ in range(self.max_size)])
            getattr(self, k+"_buf")[self.ptr] = v
        self.ptr += 1

    def finish_path(self, bootstrap=False):

        # if the trajectory is not finished, we need to bootstrap the value of the last state
        if bootstrap:
            self.end_traj_buf[self.ptr-1] = 2
        else:
            self.end_traj_buf[self.ptr-1] = 1
        self.path_start_idx = self.ptr

    def finish_path_suppress_bad_traj(self, bootstrap=False, max_length=64):

        # if the trajectory is not finished, we need to bootstrap the value of the last state
        # if the trajectory reached the maximum length, we suppose that it can be due to an impossible goal
        # sample by the high level policy, we suppress the trajectory
        if bootstrap:
            self.end_traj_buf[self.ptr-1] = 2
        else:
            if (self.ptr - self.path_start_idx) >= max_length:
                for i in range(self.path_start_idx, self.ptr):
                    self.obs_buf[i] = None
                    self.act_buf[i] = 0
                    self.rew_buf[i] = 0
                    for k in dir(self):
                        if k.endswith('_buf') and k not in ['obs_buf', 'act_buf', 'rew_buf', 'end_traj_buf']:
                            getattr(self, k)[i] = None
                self.ptr = self.path_start_idx
            else:
                self.end_traj_buf[self.ptr-1] = 1
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size  # buffer has to be full before you can get
        path_slice = slice(0, self.ptr)
        data = {k[:-4]: getattr(self, k) for k in dir(self) if k.endswith('_buf')}

        dict_return = dict()

        for k, v in data.items():
            if not isinstance(v, list):
                dict_return[k] = torch.as_tensor(v[path_slice].copy(), dtype=torch.float32)
            else:
                dict_return[k] = copy.deepcopy(v[path_slice])
        self.ptr, self.path_start_idx = 0, 0
        self.end_traj_buf = np.zeros(self.size, dtype=np.float32)
        return dict_return
