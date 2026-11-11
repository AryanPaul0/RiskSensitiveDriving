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

# from collections import brings the named tools into this file; parentheses allow several lines.
from collections import deque
# import makes these modules available; "as" gives a module a shorter local name.
import math

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import torch
# from torch import brings the named tools into this file; parentheses allow several lines.
from torch import nn
# from torch.nn import brings the named tools into this file; parentheses allow several lines.
from torch.nn import functional as F


# class defines the ReplayBuffer object type; names in parentheses are parent classes to inherit from.
class ReplayBuffer:
    """A limited-size memory of previous driving steps.
    Each stored step contains the observation, action, reward, next observation and
    whether the task truly ended. Old experiences are discarded as the buffer fills.
    """

    # def defines __init__; parentheses list its inputs, and the colon starts its indented body.
    def __init__(self, capacity):
        # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
        """Set up this object once. Stored settings and weights are reused by later method calls."""
        # A deque with maxlen automatically removes the oldest step when memory is full.
        # = stores a result in self.items; call deque using parentheses to supply its inputs.
        self.items = deque(maxlen=capacity)

    # def defines add; parentheses list its inputs, and the colon starts its indented body.
    def add(self, observation, action, reward, next_observation, terminated):
        # Time-limit truncation is not an absorbing terminal state. We still
        # bootstrap there, using the final observation before resetting.
        """Remember a driving step. Copy observation arrays so later simulator changes
        cannot alter the stored experience. A time cutoff is not a true terminal event.
        """
        # Call self.items.append using parentheses to supply its inputs; this line does not store a separate return value.
        self.items.append(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            (
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                observation.copy(),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                action,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                reward,
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                next_observation.copy(),
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                float(terminated),
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )

    # def defines sample; parentheses list its inputs, and the colon starts its indented body.
    def sample(self, batch_size, rng):
        """Randomly pick a small batch of stored steps, without duplicates inside that
        batch. Group the observations, actions and other fields into tensors for PyTorch.
        """
        # Replay breaks up the order of consecutive driving steps and reuses expensive experience.
        # = stores a result in indices; call rng.choice using parentheses to supply its inputs.
        indices = rng.choice(len(self.items), batch_size, replace=False)
        # Fetch the selected experiences; we keep their original field order.
        # = stores a result in rows; collect or generate one result per item; the "for" part names the item being processed.
        rows = [self.items[int(index)] for index in indices]
        # return ends this function and sends its result back to the code that called it.
        return tuple(torch.as_tensor(np.asarray(column)) for column in zip(*rows))

    # def defines __len__; parentheses list its inputs, and the colon starts its indented body.
    def __len__(self):
        # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
        """Return how many experiences are currently stored, which can be less than capacity."""
        # return ends this function and sends its result back to the code that called it.
        return len(self.items)


# class defines the QNetwork object type; names in parentheses are parent classes to inherit from.
class QNetwork(nn.Module):
    """A small neural network that maps observations to action-value predictions.
    DQN uses one output per action. QR-DQN uses 32 outputs per action by default. Both use the
    same two hidden layers; their output layers therefore have different sizes.
    """

    # def defines __init__; parentheses list its inputs, and the colon starts its indented body.
    def __init__(self, observation_dim, n_actions, n_quantiles=1):
        # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
        """Set up this object once. Stored settings and weights are reused by later method calls."""
        # super() accesses the parent class; __init__() sets up the PyTorch machinery this network inherits.
        super().__init__()
        # = stores a result in self.n_actions; evaluate the expression on the right before using its result.
        self.n_actions = n_actions
        # = stores a result in self.n_quantiles; evaluate the expression on the right before using its result.
        self.n_quantiles = n_quantiles
        # Linear layers mix inputs; ReLU keeps positive values and replaces negative values with zero.
        # = stores a result in self.network; call nn.Sequential using parentheses to supply its inputs.
        self.network = nn.Sequential(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            nn.Linear(observation_dim, 128),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            nn.ReLU(),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            nn.Linear(128, 128),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            nn.ReLU(),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            nn.Linear(128, n_actions * n_quantiles),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )

    # def defines forward; parentheses list its inputs, and the colon starts its indented body.
    def forward(self, observations):
        """Calculate predictions for a batch of observations.
        The result has three axes: batch items, actions and quantiles. The -1 in reshape
        asks PyTorch to infer the number of items in the batch.
        """
        # return ends this function and sends its result back to the code that called it.
        return self.network(observations).reshape(-1, self.n_actions, self.n_quantiles)


# def defines action_scores; parentheses list its inputs, and the colon starts its indented body.
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
    # if checks not 0 < alpha <= 1; only a true condition runs the indented block.
    if not 0 < alpha <= 1:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("alpha must be in (0, 1]")
    # alpha is the fraction of the predicted distribution used to score an action.
    # if checks alpha == 1; only a true condition runs the indented block.
    if alpha == 1:
        # return ends this function and sends its result back to the code that called it.
        return quantiles.mean(dim=-1)
    # For 32 outputs and alpha=.25, use the lowest 8 outputs.
    # = stores a result in count; call max using parentheses to supply its inputs.
    count = max(1, math.ceil(alpha * quantiles.shape[-1]))
    # return ends this function and sends its result back to the code that called it.
    return quantiles.sort(dim=-1).values[..., :count].mean(dim=-1)


# def defines bellman_targets; parentheses list its inputs, and the colon starts its indented body.
def bellman_targets(rewards, terminated, next_values, gamma):
    """Build the learning answer: reward now + discounted predicted value later.
    A true terminal event sets the later part to zero. A time-limit cutoff does not.
    """
    # * multiplies before + adds; a terminal flag of 1 makes the future-value term zero.
    return rewards + gamma * (1.0 - terminated) * next_values


# def defines quantile_huber_loss; parentheses list its inputs, and the colon starts its indented body.
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
    # The colon means all entries; None inserts an axis, so subtraction compares every predicted quantile with every target.
    delta = target[:, None, :] - predicted[:, :, None]
    # delta is positive when the target is above the prediction: an underestimate.
    # The dot calls the tensor's abs method; parentheses run it, giving error sizes without their signs.
    absolute = delta.abs()
    # Use the squared-error rule for small mistakes and a linear rule for large ones.
    # = stores a result in huber; call torch.where using parentheses to supply its inputs.
    huber = torch.where(
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        absolute <= kappa, 0.5 * delta.square(), kappa * (absolute - 0.5 * kappa)
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Assigned ranks sit halfway between evenly spaced endpoints: (i + 0.5) / N.
    # = stores a result in taus; apply / to these operands; numbers are divided, while Path objects join folder names.
    taus = (
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        torch.arange(predicted.shape[1], dtype=predicted.dtype, device=predicted.device)
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + 0.5
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ) / predicted.shape[1]
    # The sign of each error determines which side of its assigned quantile we are on.
    # detach prevents gradients through the sign test; < makes booleans, float changes them to 0/1, and abs gives nonnegative quantile weights.
    weights = (taus[None, :, None] - (delta.detach() < 0).float()).abs()
    # Average over target samples, sum over predicted quantiles, then average the batch.
    # Keep this exact order: changing the sum to a mean rescales the training loss.
    # Chained dot calls run left to right: average target samples, sum predicted quantiles, then average transitions; this order sets the loss scale.
    return (weights * huber / kappa).mean(dim=2).sum(dim=1).mean()


# class defines the Agent object type; names in parentheses are parent classes to inherit from.
class Agent:
    """Keep the trainable network, its slower-changing target copy and the optimizer
    together. act chooses actions; update learns from a batch of replayed steps.
    """

    # def defines __init__; parentheses list its inputs, and the colon starts its indented body.
    def __init__(
        # self is the particular object using this method; Python supplies it automatically when we call object.method().
        self,
        # This parameter receives the input size; the comma separates it from the next parameter.
        observation_dim,
        # This parameter receives how many actions the network must score; the comma separates parameters.
        n_actions,
        # In a function header, = gives a default: use "dqn" if the caller does not supply algorithm.
        algorithm="dqn",
        # This parameter defaults to 32 quantile outputs per action unless the caller supplies a different value.
        n_quantiles=32,
        # The default 5e-4 means 0.0005; scientific notation makes this small learning-rate value shorter to write.
        learning_rate=5e-4,
        # This parameter defaults to a discount of 0.9, so a reward one step later is weighted by 0.9.
        gamma=0.9,
    # ) ends the parameter list; the colon starts the function body, which must be indented.
    ):
        # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
        """Set up this object once. Stored settings and weights are reused by later method calls."""
        # = stores a result in self.algorithm; evaluate the expression on the right before using its result.
        self.algorithm = algorithm
        # = stores a result in self.gamma; evaluate the expression on the right before using its result.
        self.gamma = gamma
        # = stores a result in count; choose the value before "if" when its condition is true, otherwise the value after "else".
        count = n_quantiles if algorithm == "qrdqn" else 1
        # The online network is the one the optimizer changes.
        # = stores a result in self.online; call QNetwork using parentheses to supply its inputs.
        self.online = QNetwork(observation_dim, n_actions, count)
        # The target network supplies more stable answers for the next-state calculation.
        # = stores a result in self.target; call QNetwork using parentheses to supply its inputs.
        self.target = QNetwork(observation_dim, n_actions, count)
        # Call self.sync_target using parentheses to supply its inputs; this line does not store a separate return value.
        self.sync_target()
        # Do not train the target through backpropagation; copy weights into it instead.
        # Call self.target.requires_grad_ using parentheses to supply its inputs; this line does not store a separate return value.
        self.target.requires_grad_(False)
        # = stores a result in self.optimizer; call torch.optim.Adam using parentheses to supply its inputs.
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

    # @ attaches a decorator to the next function; no_grad disables gradient tracking while that function runs.
    @torch.no_grad()
    # def defines act; parentheses list its inputs, and the colon starts its indented body.
    def act(self, observation, alpha=1.0):
        """Choose the action with the highest score from the learned network.
        A one-observation batch keeps the input shape consistent with training.
        The no_grad decorator means this decision does not build a learning graph.
        """
        # Make a float32 tensor, then unsqueeze(0) adds a batch axis; the network expects a batch even for one observation.
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        # Read inside out: predict, score actions, find the best index along the action axis, extract its scalar, and return a Python integer.
        return int(action_scores(self.online(tensor), alpha).argmax(dim=1).item())

    # def defines sync_target; parentheses list its inputs, and the colon starts its indented body.
    def sync_target(self):
        """Copy the current online weights to the target network. The online network
        changes at every optimizer update; the target changes only at scheduled copies.
        """
        # The inner call gets online weights; the outer call copies them into the target. Dots access methods on each network.
        self.target.load_state_dict(self.online.state_dict())

    # def defines update; parentheses list its inputs, and the colon starts its indented body.
    def update(self, batch):
        """Perform one learning update using a small replay batch.
        Read current predictions, build Bellman targets with the target network,
        measure prediction error, and move the online weights to reduce that error.
        Target construction has no gradients; only the online network is trained.
        """
        # Unpack one mini-batch. Each field contains one entry per sampled transition.
        # = unpacks the right-hand result into (observations, actions, rewards, next_observations, terminated), matching values to names in order.
        observations, actions, rewards, next_observations, terminated = batch
        # = stores a result in observations; call observations.float using parentheses to supply its inputs.
        observations = observations.float()
        # = stores a result in next_observations; call next_observations.float using parentheses to supply its inputs.
        next_observations = next_observations.float()
        # = unpacks the right-hand result into (rewards, terminated), matching values to names in order.
        rewards, terminated = rewards.float(), terminated.float()
        # Pair each observation with the action that was actually taken in replay.
        # = stores a result in batch_indices; call torch.arange using parentheses to supply its inputs.
        batch_indices = torch.arange(observations.shape[0])
        # Predict all actions, then keep only the recorded action for each batch item.
        # = stores a result in predicted; use square brackets to select part of self.online(observations).
        predicted = self.online(observations)[batch_indices, actions.long()]
        # Targets are treated as fixed answers during this update, not as another trainable branch.
        # with temporarily disables gradient tracking for this block; these predictions are fixed answers, not a second training path.
        with torch.no_grad():
            # Both learners use the same target-network greedy action rule.
            # QR-DQN is trained with mean-greedy control, even when later
            # evaluated with a lower-tail selector.
            # Training chooses the next action by mean return, even if evaluation later uses a cautious rule.
            # = stores a result in next_distribution; call self.target using parentheses to supply its inputs.
            next_distribution = self.target(next_observations)
            # = stores a result in next_actions; call action_scores(next_distribution).argmax using parentheses to supply its inputs.
            next_actions = action_scores(next_distribution).argmax(dim=1)
            # = stores a result in selected; use square brackets to select part of next_distribution.
            selected = next_distribution[batch_indices, next_actions]
            # = stores a result in targets; call bellman_targets using parentheses to supply its inputs.
            targets = bellman_targets(
                # Commas separate four inputs; [:, None] turns each batch vector into a column so it broadcasts across the selected quantiles.
                rewards[:, None], terminated[:, None], selected, self.gamma
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # DQN has one prediction per action; QR-DQN compares sets of quantile predictions.
        # if checks self.algorithm == 'dqn'; only a true condition runs the indented block.
        if self.algorithm == "dqn":
            # = stores a result in loss; call F.smooth_l1_loss using parentheses to supply its inputs.
            loss = F.smooth_l1_loss(predicted, targets)
        # else provides the alternative branch; after a loop it instead means the loop finished without break.
        else:
            # = stores a result in loss; call quantile_huber_loss using parentheses to supply its inputs.
            loss = quantile_huber_loss(predicted, targets)
        # Clear gradients from the previous update before calculating new ones.
        # Clear the previous gradients before this update; set_to_none=True is a named option, not a comparison.
        self.optimizer.zero_grad(set_to_none=True)
        # Backpropagation calculates how each online weight contributed to the prediction error.
        # backward() calculates gradients: how changing each trainable weight would change this loss.
        loss.backward()
        # Limit the combined gradient size to prevent one update from changing weights too sharply.
        # = stores a result in grad_norm; call nn.utils.clip_grad_norm_ using parentheses to supply its inputs.
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        # Adam uses those gradients to adjust the online weights.
        # step() uses the gradients to update the online network weights; self identifies this particular agent.
        self.optimizer.step()
        # return ends this function and sends its result back to the code that called it.
        return float(loss.item()), float(grad_norm)
