"""DQN and quantile-regression DQN, with no RL framework hiding the update."""
from collections import deque
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.items = deque(maxlen=capacity)

    def add(self, observation, action, reward, next_observation, terminated):
        # Time-limit truncation is not an absorbing terminal state. We still
        # bootstrap there, using the final observation before resetting.
        self.items.append((observation.copy(), action, reward,
                           next_observation.copy(), float(terminated)))

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.items), batch_size, replace=False)
        rows = [self.items[int(index)] for index in indices]
        return tuple(torch.as_tensor(np.asarray(column))
                     for column in zip(*rows))

    def __len__(self):
        return len(self.items)


class QNetwork(nn.Module):
    def __init__(self, observation_dim, n_actions, n_quantiles=1):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions * n_quantiles),
        )

    def forward(self, observations):
        return self.network(observations).reshape(-1, self.n_actions, self.n_quantiles)


def action_scores(quantiles, alpha=1.0):
    """Mean of quantile outputs, or a discrete lower-tail approximation.

    QR outputs can cross. Sorting at decision time produces an ordered empirical
    distribution. Alpha is rounded up to a whole number of quantiles. This is a
    deployment heuristic, without a global CVaR optimality guarantee.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if alpha == 1:
        return quantiles.mean(dim=-1)
    count = max(1, math.ceil(alpha * quantiles.shape[-1]))
    return quantiles.sort(dim=-1).values[..., :count].mean(dim=-1)


def bellman_targets(rewards, terminated, next_values, gamma):
    return rewards + gamma * (1.0 - terminated) * next_values


def quantile_huber_loss(predicted, target, kappa=1.0):
    """Pairwise QR-DQN loss: delta[b,i,j] = target[b,j] - prediction[b,i].

    Tau_i=(i+0.5)/N. Average over target samples; sum over predicted quantiles,
    then average over the minibatch, following the QR-DQN objective.
    """
    delta = target[:, None, :] - predicted[:, :, None]
    absolute = delta.abs()
    huber = torch.where(absolute <= kappa, 0.5 * delta.square(),
                        kappa * (absolute - 0.5 * kappa))
    taus = (torch.arange(predicted.shape[1], dtype=predicted.dtype,
                         device=predicted.device) + 0.5) / predicted.shape[1]
    weights = (taus[None, :, None] - (delta.detach() < 0).float()).abs()
    return (weights * huber / kappa).mean(dim=2).sum(dim=1).mean()


class Agent:
    def __init__(self, observation_dim, n_actions, algorithm="dqn", n_quantiles=32,
                 learning_rate=5e-4, gamma=0.9):
        self.algorithm = algorithm
        self.gamma = gamma
        count = n_quantiles if algorithm == "qrdqn" else 1
        self.online = QNetwork(observation_dim, n_actions, count)
        self.target = QNetwork(observation_dim, n_actions, count)
        self.sync_target()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

    @torch.no_grad()
    def act(self, observation, alpha=1.0):
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        return int(action_scores(self.online(tensor), alpha).argmax(dim=1).item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def update(self, batch):
        observations, actions, rewards, next_observations, terminated = batch
        observations = observations.float()
        next_observations = next_observations.float()
        rewards, terminated = rewards.float(), terminated.float()
        batch_indices = torch.arange(observations.shape[0])
        predicted = self.online(observations)[batch_indices, actions.long()]
        with torch.no_grad():
            # Both learners use the same target-network greedy action rule.
            # QR-DQN is trained with mean-greedy control, even when later
            # evaluated with a lower-tail selector.
            next_distribution = self.target(next_observations)
            next_actions = action_scores(next_distribution).argmax(dim=1)
            selected = next_distribution[batch_indices, next_actions]
            targets = bellman_targets(rewards[:, None], terminated[:, None],
                                      selected, self.gamma)
        if self.algorithm == "dqn":
            loss = F.smooth_l1_loss(predicted, targets)
        else:
            loss = quantile_huber_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item()), float(grad_norm)
