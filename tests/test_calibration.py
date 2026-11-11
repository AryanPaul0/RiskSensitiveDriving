import numpy as np
import pytest

from driving_rl.calibration import discounted_return_bounds, quantile_compatibility


def test_terminal_return_is_exact_but_cutoff_retains_continuation():
    low, high = discounted_return_bounds([.4, -2.], .9, True)
    assert low == high == pytest.approx(-1.4)
    low, high = discounted_return_bounds([.4], .9, False)
    assert low == pytest.approx(.4 - .9 * 2)
    assert high == pytest.approx(.4 + .9 * 4)


def test_time_limit_bounds_contain_concrete_safe_and_crash_continuations():
    rewards = [.12] * 30
    low, high = discounted_return_bounds(rewards, .9, False)
    prefix = sum(.9**i * value for i, value in enumerate(rewards))
    crash = prefix + .9**30 * -2
    safe = prefix + .9**30 * 4
    assert low <= crash <= high
    assert low <= safe <= high
    assert high-low == pytest.approx(6 * .9**30)


def test_point_mass_ties_are_not_falsely_declared_miscalibrated():
    result = quantile_compatibility(np.full((20,4), 2.), [2.]*20, [2.]*20)
    assert result["coverage_lower"] == [0.]*4
    assert result["coverage_upper"] == [1.]*4
    assert result["minimum_mean_absolute_compatibility_gap"] == 0


def test_censoring_widens_compatibility_without_imputing_success_returns():
    result = quantile_compatibility([[0.,2.,4.]], [1.], [3.])
    assert result["coverage_lower"] == [0.,0.,1.]
    assert result["coverage_upper"] == [0.,1.,1.]
    assert result["minimum_mean_absolute_compatibility_gap"] > 0


def test_invalid_return_intervals_are_rejected():
    with pytest.raises(ValueError):
        quantile_compatibility([[1.,2.]], [3.], [1.])
    with pytest.raises(ValueError):
        quantile_compatibility([[1.,2.]], [float("nan")], [3.])
    with pytest.raises(ValueError):
        quantile_compatibility([[1.,2.]], [[0.]], [[3.]])


def test_calibration_collector_retains_survivors_and_provenance(tmp_path, monkeypatch):
    # Force IDLE in a traffic-free real simulator to exercise censoring without
    # depending on a successful trained policy or random collision outcomes.
    from driving_rl.agents import Agent
    from driving_rl.experiment import save_checkpoint
    from driving_rl.calibration import main
    import json
    import torch
    agent = Agent(40,5,"qrdqn")
    for parameter in agent.online.parameters():
        with torch.no_grad():
            parameter.zero_()
    with torch.no_grad():
        agent.online.network[-1].bias[32:64] = 1.
    from driving_rl.envs import make_env as real_make_env
    def traffic_free_env(*args, **kwargs):
        env = real_make_env(*args, **kwargs)
        env.unwrapped.config["vehicles_count"] = 0
        return env
    monkeypatch.setattr("driving_rl.calibration.make_env", traffic_free_env)
    config = {"observation_dim":40,"n_actions":5,"algorithm":"qrdqn","quantiles":32,
              "learning_rate":.0005,"gamma":.9,"seed":0}
    checkpoint = tmp_path / "test.pt"
    save_checkpoint(checkpoint,agent,config,20000)
    output = tmp_path / "calibration.json"
    monkeypatch.setattr("sys.argv",["calibration","--checkpoint",str(checkpoint),
                        "--episodes","2","--densities","1","--out",str(output)])
    main()
    result = json.loads(output.read_text())
    assert len(result["episodes"]) == 2
    assert result["provenance"]["seed_start"] == 12000
    for row in result["episodes"]:
        assert row["return_lower"] <= row["return_upper"]
        assert len(row["initial_quantiles"]) == 32
        assert row["length"] == len(row["rewards"])
        assert row["censored"] and row["length"] == 30
        # IDLE at 25 m/s yields reward .2; the continuing return is 2.
        assert row["return_lower"] <= 2. <= row["return_upper"]
