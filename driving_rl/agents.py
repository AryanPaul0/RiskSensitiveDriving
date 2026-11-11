"""The learning machinery: remember driving steps, predict rewards, and improve predictions.

Start with ReplayBuffer, then QNetwork, then Agent.update.
An observation is the information the car can see. An action is one of five driving choices.
A reward is a score for one step. A return is the sum of future rewards, discounted
so rewards far in the future count less. A quantile is a predicted position in the
range of possible returns, similar to a percentile on a test. Arrays of numbers
are called tensors in PyTorch. We use tensors to do many calculations together
quickly; replacing them with Python loops would make training much slower.

Original implementation notes:
DQN and quantile-regression DQN, with no RL framework hiding the update.
"""

from collections import deque
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ReplayBuffer:
    """A limited-size memory of previous driving steps.
    Each stored step contains the observation, action, reward, next observation and
    whether the task truly ended. Old experiences are discarded as the buffer fills.
    """

    def __init__(self, capacity):
        """Set up this object once. Stored settings and weights are reused by later method calls."""
        # A deque with maxlen automatically removes the oldest step when memory is full.
        self.items = deque(maxlen=capacity)

    def add(self, observation, action, reward, next_observation, terminated):
        # Time-limit truncation is not an absorbing terminal state. We still
        # bootstrap there, using the final observation before resetting.
        """Remember a driving step. Copy observation arrays so later simulator changes
        cannot alter the stored experience. A time cutoff is not a true terminal event.
        """
        self.items.append(
            (
                observation.copy(),
                action,
                reward,
                next_observation.copy(),
                float(terminated),
            )
        )

    def sample(self, batch_size, rng):
        """Randomly pick a small batch of stored steps, without duplicates inside that
        batch. Group the observations, actions and other fields into tensors for PyTorch.
        """
        # Replay breaks up the order of consecutive driving steps and reuses expensive experience.
        indices = rng.choice(len(self.items), batch_size, replace=False)
        # Fetch the selected experiences; we keep their original field order.
        rows = [self.items[int(index)] for index in indices]
        return tuple(torch.as_tensor(np.asarray(column)) for column in zip(*rows))

    def __len__(self):
        """Return how many experiences are currently stored, which can be less than capacity."""
        return len(self.items)


class QNetwork(nn.Module):
    """A small neural network that maps observations to action-value predictions.
    DQN uses one output per action. QR-DQN uses 32 outputs per action by default. Both use the
    same two hidden layers; their output layers therefore have different sizes.
    """

    def __init__(self, observation_dim, n_actions, n_quantiles=1):
        """Set up this object once. Stored settings and weights are reused by later method calls."""
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        # Linear layers mix inputs; ReLU keeps positive values and replaces negative values with zero.
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions * n_quantiles),
        )

    def forward(self, observations):
        """Calculate predictions for a batch of observations.
        The result has three axes: batch items, actions and quantiles. The -1 in reshape
        asks PyTorch to infer the number of items in the batch.
        """
        return self.network(observations).reshape(-1, self.n_actions, self.n_quantiles)


def action_scores(quantiles, alpha=1.0):
    """Turn each action's return predictions into one score. With alpha=1, average all
    quantiles. With alpha=.25, average the lowest quarter instead. Higher scores win.
    Sorting handles predictions that are out of order. This cautious rule changes
    action selection, but it is not a proof of safety.

    Original implementation notes:
    Mean of quantile outputs, or a discrete lower-tail approximation.

    QR outputs can cross. Sorting at decision time produces an ordered empirical
    distribution. Alpha is rounded up to a whole number of quantiles. This is a
    deployment heuristic, without a global CVaR optimality guarantee.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    # alpha is the fraction of the predicted distribution used to score an action.
    if alpha == 1:
        return quantiles.mean(dim=-1)
    # For 32 outputs and alpha=.25, use the lowest 8 outputs.
    count = max(1, math.ceil(alpha * quantiles.shape[-1]))
    return quantiles.sort(dim=-1).values[..., :count].mean(dim=-1)


def bellman_targets(rewards, terminated, next_values, gamma):
    """Build the learning answer: reward now + discounted predicted value later.
    A true terminal event sets the later part to zero. A time-limit cutoff does not.
    """
    return rewards + gamma * (1.0 - terminated) * next_values


def quantile_huber_loss(predicted, target, kappa=1.0):
    """Measure how far the predicted quantiles are from the target quantiles.
    Compare every prediction with every target in the same replayed transition.
    Low quantiles penalize overestimates more; high quantiles penalize underestimates
    more. The Huber error is squared near zero and grows linearly for larger errors,
    which reduces the influence of very large mistakes. See the inline shape notes.

    Original implementation notes:
    Pairwise QR-DQN loss: delta[b,i,j] = target[b,j] - prediction[b,i].

    Tau_i=(i+0.5)/N. Average over target samples; sum over predicted quantiles,
    then average over the minibatch, following the QR-DQN objective.
    """
    # Shape: [batch, predicted quantile, target quantile]. None adds an axis.
    # Each transition gets all prediction/target pairs without slow Python loops.
    delta = target[:, None, :] - predicted[:, :, None]
    # delta is positive when the target is above the prediction: an underestimate.
    absolute = delta.abs()
    # Use the squared-error rule for small mistakes and a linear rule for large ones.
    huber = torch.where(
        absolute <= kappa, 0.5 * delta.square(), kappa * (absolute - 0.5 * kappa)
    )
    # Assigned ranks sit halfway between evenly spaced endpoints: (i + 0.5) / N.
    taus = (
        torch.arange(predicted.shape[1], dtype=predicted.dtype, device=predicted.device)
        + 0.5
    ) / predicted.shape[1]
    # The sign of each error determines which side of its assigned quantile we are on.
    weights = (taus[None, :, None] - (delta.detach() < 0).float()).abs()
    # Average over target samples, sum over predicted quantiles, then average the batch.
    # Keep this exact order: changing the sum to a mean rescales the training loss.
    return (weights * huber / kappa).mean(dim=2).sum(dim=1).mean()


class Agent:
    """Keep the trainable network, its slower-changing target copy and the optimizer
    together. act chooses actions; update learns from a batch of replayed steps.
    """

    def __init__(
        self,
        observation_dim,
        n_actions,
        algorithm="dqn",
        n_quantiles=32,
        learning_rate=5e-4,
        gamma=0.9,
    ):
        """Set up this object once. Stored settings and weights are reused by later method calls."""
        self.algorithm = algorithm
        self.gamma = gamma
        count = n_quantiles if algorithm == "qrdqn" else 1
        # The online network is the one the optimizer changes.
        self.online = QNetwork(observation_dim, n_actions, count)
        # The target network supplies more stable answers for the next-state calculation.
        self.target = QNetwork(observation_dim, n_actions, count)
        self.sync_target()
        # Do not train the target through backpropagation; copy weights into it instead.
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

    @torch.no_grad()
    def act(self, observation, alpha=1.0):
        """Choose the action with the highest score from the learned network.
        A one-observation batch keeps the input shape consistent with training.
        The no_grad decorator means this decision does not build a learning graph.
        """
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        return int(action_scores(self.online(tensor), alpha).argmax(dim=1).item())

    def sync_target(self):
        """Copy the current online weights to the target network. The online network
        changes at every optimizer update; the target changes only at scheduled copies.
        """
        self.target.load_state_dict(self.online.state_dict())

    def update(self, batch):
        """Perform one learning update using a small replay batch.
        Read current predictions, build Bellman targets with the target network,
        measure prediction error, and move the online weights to reduce that error.
        Target construction has no gradients; only the online network is trained.
        """
        # Unpack one mini-batch. Each field contains one entry per sampled transition.
        observations, actions, rewards, next_observations, terminated = batch
        observations = observations.float()
        next_observations = next_observations.float()
        rewards, terminated = rewards.float(), terminated.float()
        # Pair each observation with the action that was actually taken in replay.
        batch_indices = torch.arange(observations.shape[0])
        # Predict all actions, then keep only the recorded action for each batch item.
        predicted = self.online(observations)[batch_indices, actions.long()]
        # Targets are treated as fixed answers during this update, not as another trainable branch.
        with torch.no_grad():
            # Both learners use the same target-network greedy action rule.
            # QR-DQN is trained with mean-greedy control, even when later
            # evaluated with a lower-tail selector.
            # Training chooses the next action by mean return, even if evaluation later uses a cautious rule.
            next_distribution = self.target(next_observations)
            next_actions = action_scores(next_distribution).argmax(dim=1)
            selected = next_distribution[batch_indices, next_actions]
            targets = bellman_targets(
                rewards[:, None], terminated[:, None], selected, self.gamma
            )
        # DQN has one prediction per action; QR-DQN compares sets of quantile predictions.
        if self.algorithm == "dqn":
            loss = F.smooth_l1_loss(predicted, targets)
        else:
            loss = quantile_huber_loss(predicted, targets)
        # Clear gradients from the previous update before calculating new ones.
        self.optimizer.zero_grad(set_to_none=True)
        # Backpropagation calculates how each online weight contributed to the prediction error.
        loss.backward()
        # Limit the combined gradient size to prevent one update from changing weights too sharply.
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        # Adam uses those gradients to adjust the online weights.
        self.optimizer.step()
        return float(loss.item()), float(grad_norm)
