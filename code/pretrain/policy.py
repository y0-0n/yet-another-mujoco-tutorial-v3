import numpy as np
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from rl_games.algos_torch.running_mean_std import RunningMeanStd

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_activation: torch.nn.functional = F.relu,
        init_w: float = 3e-3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation

        # Fully connected 레이어 생성
        self.fc_layers = nn.ModuleList()
        self.hidden_layers = [hidden_dim] * 2
        in_layer = input_dim

        for i, hidden_layer in enumerate(self.hidden_layers):
            fc_layer = nn.Linear(in_layer, hidden_layer)
            in_layer = hidden_layer
            self.__setattr__("fc_layer{}".format(i), fc_layer)
            self.fc_layers.append(fc_layer)

        # 출력 레이어 생성
        self.last_fc_layer = nn.Linear(hidden_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))
        x = self.last_fc_layer(x)
        return x

class GaussianPolicy(MLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        is_deterministic: bool = False,
        init_w: float = 1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            init_w=init_w,
        )
        self.running_mean_std = RunningMeanStd(input_dim)
        self.action_mean_std = RunningMeanStd(output_dim)
        self.is_deterministic = is_deterministic
        self.last_fc_log_std = nn.Linear(hidden_dim, output_dim)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.running_mean_std(x)
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))
        mean = self.last_fc_layer(x)
        log_std = self.last_fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        if self.is_deterministic:
            action = mean
            log_prob = None
        else:
            normal = Normal(mean, std)
            action = normal.rsample()
            # action = self.action_mean_std(action)

            # 가우시안 분포에 대한 로그 확률값 계산
            log_prob = normal.log_prob(action)
            log_prob -= 2 * (np.log(2) - action - F.softplus(-2 * action))
            log_prob = log_prob.sum(-1, keepdim=True)

        return action, mean, std, log_prob
