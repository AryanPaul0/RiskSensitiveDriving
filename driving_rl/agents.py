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

from collections import deque  # Use bounded replay memory that automatically forgets its oldest transition when full.
import math  # Use ceiling to turn a tail fraction into a whole number of quantile heads.

import numpy as np  # Batch stored numerical observations before converting them to tensors.
import torch  # Use PyTorch tensors, automatic gradients, and neural-network training.
from torch import nn  # Use trainable layers and gradient-clipping tools.
from torch.nn import functional as F  # Use the scalar DQN prediction-error loss.


class ReplayBuffer:  # Group the state and operations for ReplayBuffer.
    """A limited-size memory of previous driving steps.
    Each stored step contains the observation, action, reward, next observation and
    whether the task truly ended. Old experiences are discarded as the buffer fills.
    """

    def __init__(self, capacity):  # Keep at most capacity transitions; discard the oldest when memory fills.
        """Keep at most capacity transitions; discard the oldest when memory fills."""  # Describe this operation for readers and help().
        self.items = deque(maxlen=capacity)  # A deque with maxlen automatically removes the oldest step when memory is full.

    def add(self, observation, action, reward, next_observation, terminated):  # Remember a driving step. Time-limit truncation is not an absorbing terminal state. We still bootstrap there, using the final observation before resetting.
        """Remember a driving step. Copy observation arrays so later simulator changes
        cannot alter the stored experience. A time cutoff is not a true terminal event.
        """
        self.items.append(  # Save one observation-action-reward-next-observation-terminal transition for later replay.
            (  # Include an independent copy of the currently observed traffic features, action, reward, an independent copy of the traffic features after the action, the true-terminal flags as a floating-point value in the surrounding operation.
                observation.copy(),  # Keep the old observation independent of simulator arrays that may change after this step.
                action,  # Remember the action actually taken, including random exploration actions.
                reward,  # Remember the immediate reward produced by that action.
                next_observation.copy(),  # Keep the actual next observation, including the final observation at a time cutoff.
                float(terminated),  # Store true ending as 1.0 and continuing/cutoff as 0.0 for the Bellman continuation mask.
            )  # Finish assembling the grouped values above.
        )  # Finish assembling this learner's items.append inputs.

    def sample(self, batch_size, rng):  # Randomly pick a small batch of stored steps, without duplicates inside that batch.
        """Randomly pick a small batch of stored steps, without duplicates inside that
        batch. Group the observations, actions and other fields into tensors for PyTorch.
        """
        indices = rng.choice(len(self.items), batch_size, replace=False)  # Replay breaks up the order of consecutive driving steps and reuses expensive experience.
        rows = [self.items[int(index)] for index in indices]  # Fetch the selected experiences; we keep their original field order.
        return tuple(torch.as_tensor(np.asarray(column)) for column in zip(*rows))  # Return five batched tensors; zip(*rows) groups all observations together, all actions together, etc., rather than leaving one tuple per transition.

    def __len__(self):  # Return how many experiences are currently stored, which can be less than capacity.
        """Return how many experiences are currently stored, which can be less than capacity."""  # Describe this operation for readers and help().
        return len(self.items)  # Return the number of entries in this learner's items.


class QNetwork(nn.Module):  # Group the state and operations for QNetwork.
    """A small neural network that maps observations to action-value predictions.
    DQN uses one output per action. QR-DQN uses 32 outputs per action by default. Both use the
    same two hidden layers; their output layers therefore have different sizes.
    """

    def __init__(self, observation_dim, n_actions, n_quantiles=1):  # Build the observation-to-return predictor used by either learner.
        """Build the observation-to-return predictor used by either learner."""  # Describe this operation for readers and help().
        super().__init__()  # Initialize nn.Module bookkeeping so PyTorch tracks the network layers and trainable weights.
        self.n_actions = n_actions  # Remember how many maneuver scores each observation needs: five in this highway setup.
        self.n_quantiles = n_quantiles  # Remember the heads per action: one expected value for DQN, or 32 quantiles for QR-DQN.
        self.network = nn.Sequential(  # Pass traffic inputs through two hidden layers and then a return-prediction layer.
            nn.Linear(observation_dim, 128),  # Learn 128 combinations of the 40 traffic features to build the first hidden representation.
            nn.ReLU(),  # Keep positive hidden activations and replace negative ones with zero, allowing nonlinear traffic-to-return relationships.
            nn.Linear(128, 128),  # Learn a second 128-feature representation from the first hidden layer.
            nn.ReLU(),  # Keep positive hidden activations and replace negative ones with zero, allowing nonlinear traffic-to-return relationships.
            nn.Linear(128, n_actions * n_quantiles),  # Produce 5 scalar values or 5*32 quantiles; no final ReLU, because predicted returns can be negative.
        )  # Finish assembling nn.Sequential inputs.

    def forward(self, observations):  # Calculate predictions for a batch of observations.
        """Calculate predictions for a batch of observations.
        The result has three axes: batch items, actions and quantiles. The -1 in reshape
        asks PyTorch to infer the number of items in the batch.
        """
        return self.network(observations).reshape(-1, self.n_actions, self.n_quantiles)  # Return [batch,action,quantile] predictions; -1 infers batch size, with 1 head per action for DQN or 32 for QR-DQN.


def action_scores(quantiles, alpha=1.0):  # Turn each action's return predictions into one score.
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
    if not 0 < alpha <= 1:  # Reject a tail fraction outside (0,1]; an action score needs at least some of the predicted distribution.
        raise ValueError("alpha must be in (0, 1]")  # Reject this invalid case instead of continuing with misleading experiment results.
    if alpha == 1:  # Use the ordinary mean-return decision rule when the entire predicted distribution is included.
        return quantiles.mean(dim=-1)  # Return one expected-return score per action by averaging the quantile axis: [B,5,N] -> [B,5]; DQN has N=1.
    count = max(1, math.ceil(alpha * quantiles.shape[-1]))  # For 32 outputs and alpha=.25, use the lowest 8 outputs.
    return quantiles.sort(dim=-1).values[..., :count].mean(dim=-1)  # Sort each action's return predictions, keep its lowest count entries, and return their average as a cautious score; ... preserves earlier axes.


def bellman_targets(rewards, terminated, next_values, gamma):  # Build the learning answer: reward now + discounted predicted value later.
    """Build the learning answer: reward now + discounted predicted value later.
    A true terminal event sets the later part to zero. A time-limit cutoff does not.
    """
    return rewards + gamma * (1.0 - terminated) * next_values  # Return reward now plus discounted next-action value/quantiles; terminal=1 removes continuation, while a time cutoff keeps it.


def quantile_huber_loss(predicted, target, kappa=1.0):  # Measure how far the predicted quantiles are from the target quantiles.
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
    delta = target[:, None, :] - predicted[:, :, None]  # Compare every target with every predicted quantile in its own transition: [B,1,N] - [B,N,1] broadcasts to [B,N,N]; None inserts size-1 axes.
    absolute = delta.abs()  # Measure error size without its sign to choose the quadratic or linear Huber penalty.
    huber = torch.where(  # Build an elementwise error penalty that is quadratic for small mistakes and linear for large mistakes.
        absolute <= kappa, 0.5 * delta.square(), kappa * (absolute - 0.5 * kappa)  # For errors within kappa use half the squared error; otherwise use the linear branch to reduce outlier influence.
    )  # Finish assembling torch.where inputs.
    taus = (  # Assign each predicted head its quantile rank, halfway through one of N equal probability bins.
        torch.arange(predicted.shape[1], dtype=predicted.dtype, device=predicted.device)  # Make ranks 0..N-1 using the prediction tensor's device and number format.
        + 0.5  # Move each rank to the middle of its bin so the first and last heads do not target probability 0 or 1.
    ) / predicted.shape[1]  # Divide midpoint ranks by N, giving fractions such as .5/32 through 31.5/32.
    weights = (taus[None, :, None] - (delta.detach() < 0).float()).abs()  # Weight errors by quantile rank: tau=.25 penalizes overestimates .75 and underestimates .25; detach excludes the sign test from gradients.
    return (weights * huber / kappa).mean(dim=2).sum(dim=1).mean()  # Return QR loss: average over target samples, sum over predicted quantiles, then average transitions; changing this order changes loss scaling.


class Agent:  # Group the state and operations for Agent.
    """Keep the trainable network, its slower-changing target copy and the optimizer
    together. act chooses actions; update learns from a batch of replayed steps.
    """

    def __init__(  # Create a learner plus a separate, periodically refreshed target copy.
        self,  # These networks, settings, and updates belong to this particular learner object.
        observation_dim,  # Receive the observation feature count used to size the input layer.
        n_actions,  # Receive the number of discrete maneuvers the learner must score.
        algorithm="dqn",  # Choose scalar DQN unless the caller requests the quantile-based learner.
        n_quantiles=32,  # Use 32 return quantiles per action for QR-DQN by default.
        learning_rate=5e-4,  # Start with an Adam learning-rate setting of .0005.
        gamma=0.9,  # Weight a reward one decision later by .9; further rewards receive successive powers of .9.
    ):  # Begin the init procedure with the inputs listed above.
        """Create a learner plus a separate, periodically refreshed target copy."""  # Describe this operation for readers and help().
        self.algorithm = algorithm  # Remember which output interpretation and loss this learner uses.
        self.gamma = gamma  # Remember the discount factor used to construct Bellman targets.
        count = n_quantiles if algorithm == "qrdqn" else 1  # Give QR-DQN multiple return heads per action and DQN a single expected-return head.
        self.online = QNetwork(observation_dim, n_actions, count)  # The online network is the one the optimizer changes.
        self.target = QNetwork(observation_dim, n_actions, count)  # The target network supplies more stable answers for the next-state calculation.
        self.sync_target()  # Start the target with the same values as the online network rather than an unrelated random guess.
        self.target.requires_grad_(False)  # Do not learn target weights by gradient descent; refresh them only by copying online weights.
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)  # Let Adam adjust only online-network weights using the configured learning rate.

    @torch.no_grad()  # Make action selection inference-only: do not build a gradient graph when this method runs.
    def act(self, observation, alpha=1.0):  # Choose the action with the highest score from the learned network.
        """Choose the action with the highest score from the learned network.
        A one-observation batch keeps the input shape consistent with training.
        The no_grad decorator means this decision does not build a learning graph.
        """
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)  # Make one observation a [1,40] float32 batch; unsqueeze(0) adds the batch axis expected by the network.
        return int(action_scores(self.online(tensor), alpha).argmax(dim=1).item())  # Return the action with the highest mean/tail score for this observation; dim=1 selects over five actions, item() extracts its single index.

    def sync_target(self):  # Copy the current online weights to the target network.
        """Copy the current online weights to the target network. The online network
        changes at every optimizer update; the target changes only at scheduled copies.
        """
        self.target.load_state_dict(self.online.state_dict())  # Copy online weight values into the separate target network; the target then stays fixed until the next scheduled sync.

    def update(self, batch):  # Perform one learning update using a small replay batch.
        """Perform one learning update using a small replay batch.
        Read current predictions, build Bellman targets with the target network,
        measure prediction error, and move the online weights to reduce that error.
        Target construction has no gradients; only the online network is trained.
        """
        observations, actions, rewards, next_observations, terminated = batch  # Unpack the five replay columns; row i in every column describes the same recorded transition.
        observations = observations.float()  # Convert current-observation features to float32 to match the network's weights.
        next_observations = next_observations.float()  # Convert next-observation features to the same number format before target prediction.
        rewards, terminated = rewards.float(), terminated.float()  # Make rewards and terminal flags suitable for arithmetic; actions remain integer indices.
        batch_indices = torch.arange(observations.shape[0])  # Number minibatch transitions 0..B-1 so each one can be paired with its own action index.
        predicted = self.online(observations)[batch_indices, actions.long()]  # Predict [B,5,N], then select paired (batch row, recorded action) entries -> [B,N]; e.g. rows [0,1] and actions [2,4] take [0,2,:] and [1,4,:]. long() supplies integer indices.
        with torch.no_grad():  # Treat predictions in this block as fixed learning targets or inference results; do not backpropagate through them.
            next_distribution = self.target(next_observations)  # Predict all five next actions using the fixed target network: output shape [B,5,N].
            next_actions = action_scores(next_distribution).argmax(dim=1)  # Choose one next action per transition using mean return; both this choice and its value use the target network, so this is not Double DQN.
            selected = next_distribution[batch_indices, next_actions]  # Keep each transition's chosen next action quantiles: paired row/action indexing changes [B,5,N] into [B,N].
            targets = bellman_targets(  # Construct the reward-plus-discounted-continuation learning answers for the recorded transitions.
                rewards[:, None], terminated[:, None], selected, self.gamma  # Supply [B,1] rewards and terminal flags so they broadcast across [B,N] target quantiles without mixing transitions.
            )  # Finish assembling bellman targets inputs.
        if self.algorithm == "dqn":  # Use scalar value fitting for DQN; QR-DQN instead fits quantile positions with asymmetric loss.
            loss = F.smooth_l1_loss(predicted, targets)  # Measure scalar DQN prediction error with smooth L1, which softens the effect of very large errors.
        else:  # Handle the alternative when the earlier condition (this learner's algorithm equals 'dqn') is false.
            loss = quantile_huber_loss(predicted, targets)  # Compare every predicted and target quantile with rank-dependent weighting, so heads learn different parts of the return distribution.
        self.optimizer.zero_grad(set_to_none=True)  # Clear the previous update's gradients so the new minibatch does not accidentally accumulate them.
        loss.backward()  # Calculate how each online-network weight affects prediction error; these gradients guide learning.
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)  # Limit the combined gradient norm to 10 before the optimizer step; save the pre-clipping norm as a diagnostic.
        self.optimizer.step()  # Use Adam and the current gradients to adjust online-network weights toward lower prediction error.
        return float(loss.item()), float(grad_norm)  # Return loss and the gradient norm before clipping as plain numbers for training diagnostics.
