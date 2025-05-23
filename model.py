# from https://github.com/jennyzzt/omni/blob/master/model.py#L120
import numpy as np
from copy import deepcopy
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from typing import Dict, Optional, Sequence, Tuple
from gym import spaces
import torch.nn.init as init

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)



class FanInInitReLULayer(nn.Module):
    def __init__(
        self,
        inchan: int,
        outchan: int,
        layer_type: str = "conv",
        init_scale: float = 1.0,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation: bool = True,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        # Layer
        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = layer(inchan, outchan, bias=self.norm is None, **layer_kwargs)
        self.use_activation = use_activation

        # Initialization
        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x: torch.Tensor):
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x

class CnnBasicBlock(nn.Module):
    def __init__(
        self,
        inchan: int,
        init_scale: float = 1.0,
        init_norm_kwargs: Dict = {},
    ):
        super().__init__()

        # Layers
        s = math.sqrt(init_scale)
        self.conv0 = FanInInitReLULayer(
            inchan,
            inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            **init_norm_kwargs,
        )
        self.conv1 = FanInInitReLULayer(
            inchan,
            inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            **init_norm_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv1(self.conv0(x))
        return x


class CnnDownStack(nn.Module):
    def __init__(
        self,
        inchan: int,
        nblock: int,
        outchan: int,
        init_scale: float = 1.0,
        pool: bool = True,
        post_pool_groups: Optional[int] = None,
        init_norm_kwargs: Dict = {},
        first_conv_norm: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Params
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool

        # Layers
        first_conv_init_kwargs = deepcopy(init_norm_kwargs)
        if not first_conv_norm:
            first_conv_init_kwargs["group_norm_groups"] = None
            first_conv_init_kwargs["batch_norm"] = False
        self.firstconv = FanInInitReLULayer(
            inchan,
            outchan,
            kernel_size=3,
            padding=1,
            **first_conv_init_kwargs,
        )
        self.post_pool_groups = post_pool_groups
        if post_pool_groups is not None:
            self.n = nn.GroupNorm(post_pool_groups, outchan)
        self.blocks = nn.ModuleList(
            [
                CnnBasicBlock(
                    outchan,
                    init_scale=init_scale / math.sqrt(nblock),
                    init_norm_kwargs=init_norm_kwargs,
                    **kwargs,
                )
                for _ in range(nblock)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            if self.post_pool_groups is not None:
                x = self.n(x)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape: Sequence[int]) -> Tuple[int, int, int]:
        c, h, w = inshape
        assert c == self.inchan
        if self.pool:
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)


class ImpalaCNN(nn.Module):
    def __init__(
        self,
        inshape: Sequence[int],
        chans: Sequence[int],
        outsize: int,
        nblock: int,
        init_norm_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        first_conv_norm: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Layers
        curshape = inshape
        self.stacks = nn.ModuleList()
        for i, outchan in enumerate(chans):
            stack = CnnDownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                init_scale=1.0 / math.sqrt(len(chans)),
                init_norm_kwargs=init_norm_kwargs,
                first_conv_norm=first_conv_norm if i == 0 else True,
                **kwargs,
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.dense = FanInInitReLULayer(
            math.prod(curshape),
            outsize,
            layer_type="linear",
            init_scale=1.4,
            **dense_init_norm_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stack in self.stacks:
            x = stack(x)
        x = x.reshape(x.size(0), -1)
        x = self.dense(x)
        return x


# Actor-critic model with shared vision encoder

class AC_model(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space_size: int,
        hidsize: int,
        memory=False,
        memory_size=64
    ):
        super().__init__()

        impala_kwargs={"chans": [64, 128, 128],
                       "outsize": 256,
                       "nblock": 2,
                       "post_pool_groups": 1}

        dense_init_norm_kwargs={"layer_norm": True}

        self.observation_space = observation_space
        self.action_space_size = action_space_size

        # Encoder
        obs_shape = self.observation_space["image"]
        self.enc = ImpalaCNN(
            obs_shape,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **impala_kwargs,
        )
        self.hidsize = hidsize

        self.memory = memory
        self.memory_size = memory_size
        if memory:
            self.self_attn = nn.MultiheadAttention(
            self.action_space_size,
            1,
            dropout=False,
            bias=True,
            batch_first=True
            )
            self.w1 = nn.Linear(self.action_space_size, 4*self.action_space_size, bias=False)
            self.w2 = nn.Linear(4*self.action_space_size, self.action_space_size, bias=False)
            self.w3 = nn.Linear(self.action_space_size, 4*self.action_space_size, bias=False)

            self.norm1 = nn.LayerNorm(self.action_space_size, eps=1e-5, bias=True)
            self.norm2 = nn.LayerNorm(self.action_space_size, eps=1e-5, bias=True)

            self.proj = nn.Linear(self.action_space_size*self.memory_size, self.memory_size, bias=True)


        outsize = impala_kwargs["outsize"]
        if self.memory:
            outsize += memory_size
        self.linear = FanInInitReLULayer(
            outsize,
            hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

        # Heads
        num_actions = self.action_space_size

        # actor
        # Layer
        self.linear_actor = nn.Linear(hidsize, num_actions)

        # Initialization
        init.orthogonal_(self.linear_actor.weight, gain=0.01)
        init.constant_(self.linear_actor.bias, val=0.0)

        # critic
        # Layer
        self.linear_critic = nn.Linear(hidsize, 1)

        # Initialization
        init.orthogonal_(self.linear_critic.weight, gain=0.1)
        init.constant_(self.linear_critic.bias, val=0.0)

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            need_weights=False,
            is_causal=False,
        )[0]
        return x

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        # SwiGLU feed forward network
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return x

    def forward(self, obs, memory=None) -> Dict[str, torch.Tensor]:
        # Pass through encoder
        x = self.enc(obs.image)
        if self.memory:
            # attention mechanism
            x_memory = self.norm1(
                memory
                + self._sa_block(memory)
            )
            x_memory = self.norm2(x_memory + self._ff_block(x_memory))
            x_memory = x_memory.view(-1, x_memory.shape[1]*x_memory.shape[2])
            x_memory = self.proj(x_memory)
            x = torch.cat([x, x_memory], dim=-1)
        latents = self.linear(x)

        # Pass through heads
        pi_latents = vf_latents = latents

        # actor
        pi_logits = Categorical(logits=F.log_softmax(self.linear_actor(pi_latents), dim=-1))

        # critic
        vpreds = self.linear_critic(vf_latents).squeeze(1)

        return {"dist": pi_logits, "value": vpreds}

# Actor-critic model with separated network for both
class Actor_model(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space_size: int,
        hidsize: int,
        memory=False,
        memory_size=64
    ):
        super().__init__()

        impala_kwargs={"chans": [64, 128, 128],
                       "outsize": 256,
                       "nblock": 2,
                       "post_pool_groups": 1}

        dense_init_norm_kwargs={"layer_norm": True}

        self.observation_space = observation_space
        self.action_space_size = action_space_size

        # Encoder
        obs_shape = self.observation_space["image"]
        self.enc = ImpalaCNN(
            obs_shape,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **impala_kwargs,
        )
        self.hidsize = hidsize

        self.memory = memory
        self.memory_size = memory_size
        if memory:
            self.self_attn = nn.MultiheadAttention(
            self.action_space_size,
            1,
            dropout=False,
            bias=True,
            batch_first=True
            )
            """self.linear1 = nn.Linear(1, 4)
            self.linear2 = nn.Linear(4, 1)"""
            self.w1 = nn.Linear(self.action_space_size, 4*self.action_space_size, bias=False)
            self.w2 = nn.Linear(4*self.action_space_size, self.action_space_size, bias=False)
            self.w3 = nn.Linear(self.action_space_size, 4*self.action_space_size, bias=False)

            self.norm1 = nn.LayerNorm(self.action_space_size, eps=1e-5, bias=True)
            self.norm2 = nn.LayerNorm(self.action_space_size, eps=1e-5, bias=True)

            self.proj = nn.Linear(self.action_space_size*self.memory_size, self.memory_size, bias=True)


        outsize = impala_kwargs["outsize"]
        if self.memory:
            outsize += memory_size
        self.linear = FanInInitReLULayer(
            outsize,
            hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

        # Heads
        num_actions = self.action_space_size

        # actor
        # Layer
        self.linear_actor = nn.Linear(hidsize, num_actions)

        # Initialization
        init.orthogonal_(self.linear_actor.weight, gain=0.01)
        init.constant_(self.linear_actor.bias, val=0.0)

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            need_weights=False,
            is_causal=False,
        )[0]
        return x

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        # SwiGLU feed forward network
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return x

    def forward(self, obs, memory=None) -> Dict[str, torch.Tensor]:
        # Pass through encoder
        x = self.enc(obs.image)
        if self.memory:
            # attention mechanism
            x_memory = self.norm1(
                memory
                + self._sa_block(memory)
            )
            x_memory = self.norm2(x_memory + self._ff_block(x_memory))
            x_memory = x_memory.view(-1, x_memory.shape[1]*x_memory.shape[2])
            x_memory = self.proj(x_memory)
            x = torch.cat([x, x_memory], dim=-1)
        latents = self.linear(x)

        # Pass through heads
        pi_latents = latents

        # actor
        pi_logits = Categorical(logits=F.log_softmax(self.linear_actor(pi_latents), dim=-1))

        return {"dist": pi_logits}

class Critic_model(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space_size: int,
        hidsize: int,
        memory=False,
        memory_size=64
    ):
        super().__init__()

        impala_kwargs={"chans": [64, 128, 128],
                       "outsize": 256,
                       "nblock": 2,
                       "post_pool_groups": 1}

        dense_init_norm_kwargs={"layer_norm": True}

        self.observation_space = observation_space
        self.action_space_size = action_space_size

        # Encoder
        obs_shape = self.observation_space["image"]
        self.enc = ImpalaCNN(
            obs_shape,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **impala_kwargs,
        )
        self.hidsize = hidsize

        self.memory = memory
        self.memory_size = memory_size
        if memory:
            self.self_attn = nn.MultiheadAttention(
            self.action_space_size,
            1,
            dropout=False,
            bias=True,
            batch_first=True
            )
            """self.linear1 = nn.Linear(1, 4)
            self.linear2 = nn.Linear(4, 1)"""
            self.w1 = nn.Linear(self.action_space_size, 4*self.action_space_size, bias=False)
            self.w2 = nn.Linear(4*self.action_space_size, self.action_space_size, bias=False)
            self.w3 = nn.Linear(self.action_space_size, 4*self.action_space_size, bias=False)

            self.norm1 = nn.LayerNorm(self.action_space_size, eps=1e-5, bias=True)
            self.norm2 = nn.LayerNorm(self.action_space_size, eps=1e-5, bias=True)

            self.proj = nn.Linear(self.action_space_size*self.memory_size, self.memory_size, bias=True)


        outsize = impala_kwargs["outsize"]
        if self.memory:
            outsize += memory_size
        self.linear = FanInInitReLULayer(
            outsize,
            hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

        # critic
        # Layer
        self.linear_critic = nn.Linear(hidsize, 1)

        # Initialization
        init.orthogonal_(self.linear_critic.weight, gain=0.1)
        init.constant_(self.linear_critic.bias, val=0.0)

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            need_weights=False,
            is_causal=False,
        )[0]
        return x

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        # SwiGLU feed forward network
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return x

    def forward(self, obs, memory=None) -> Dict[str, torch.Tensor]:
        # Pass through encoder
        x = self.enc(obs.image)
        if self.memory:
            # attention mechanism
            x_memory = self.norm1(
                memory
                + self._sa_block(memory)
            )
            x_memory = self.norm2(x_memory + self._ff_block(x_memory))
            x_memory = x_memory.view(-1, x_memory.shape[1]*x_memory.shape[2])
            x_memory = self.proj(x_memory)
            x = torch.cat([x, x_memory], dim=-1)
        latents = self.linear(x)

        # Pass through heads
        vf_latents = latents

        # critic
        vpreds = self.linear_critic(vf_latents).squeeze(1)

        return {"value": vpreds}