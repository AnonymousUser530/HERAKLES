'''
PPO implementation taken from https://github.com/openai/spinningup
'''

import numpy as np
import scipy
import torch

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def format_goal(goal_str:str):
    parts = goal_str.split('_')
    name_g = parts[0].capitalize()
    for p in parts[1:]:
         name_g += ' '+p
    return name_g

def task_encoder(goal_str:str, enc_order):
    encoding = np.zeros(len(enc_order))
    task_words = goal_str.split('_')
    # bag of words encoding
    for i, word in enumerate(enc_order):
        if word in task_words:
            encoding[i] = 1
    return encoding

def push_to_tensor(tensor, x, device=None):

    if device is not None:
        if isinstance(x, torch.Tensor):
            return torch.cat((tensor[1:], x.to(device)))
        else:
            return torch.cat((tensor[1:], torch.tensor([[x]], device=device)))
    else:
        if isinstance(x, torch.Tensor):
            return torch.cat((tensor[1:], x))
        else:
            return torch.cat((tensor[1:], torch.tensor([[x]])))