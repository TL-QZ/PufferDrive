import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pufferlib import pufferl
from pufferlib import pufferl_sac


class _Policy(nn.Module):
    action_count = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))


class _VecEnv:
    def __init__(self):
        self.driver_env = SimpleNamespace(capture_final_observations=True)
        self.single_observation_space = SimpleNamespace(shape=(3,))
        self.single_action_space = SimpleNamespace(n=2)
        self.agents_per_batch = 2
        self.num_agents = 2

    def close(self):
        self.closed = True


class _Logger:
    def __init__(self):
        self.close_args = None

    def close(self, model_path, early_stop):
        self.close_args = (model_path, early_stop)


def test_sac_config_keeps_evaluation_metadata_out_of_learner_keys():
    args = pufferl_sac.load_config(overrides=[])

    assert args["train"]["evaluation_interval_epochs"] is None
    assert args["train"]["evaluation_benchmarks"] == "carla_fast"
    assert not (pufferl_sac.SAC_EVALUATION_KEYS & pufferl_sac.SAC_TRAIN_KEYS)


@pytest.mark.parametrize("render_scenarios", [False, True])
def test_standalone_eval_bootstraps_from_benchmark_env_and_passes_actor(monkeypatch, render_scenarios):
    args = {
        "load_model_path": "actor.pt",
        "load_id": None,
        "train": {"device": "cpu", "seed": 42, "data_dir": "unused"},
        "policy": {"action_type": "discrete", "shared_network": False},
        "env": {
            "map_dir": "training/missing",
            "num_agents": 16,
            "min_agents_per_env": 4,
            "max_agents_per_env": 16,
            "num_maps": 4,
        },
        "vec": {"backend": "Multiprocessing", "num_envs": 8, "num_workers": 8, "batch_size": 8, "seed": 123},
        "eval": {
            "render_scenarios": render_scenarios,
            "render_filter": None,
            "failure_replay_csv": None,
            "capture_observations": False,
            "benchmark_config": "benchmarks.yaml",
            "benchmarks": "carla_fast",
        },
    }
    actor = _Policy()
    vecenv = _VecEnv()
    captured = {}

    monkeypatch.setattr(pufferl_sac, "_restore_checkpoint_architecture", lambda *args: None)
    monkeypatch.setattr(pufferl_sac, "load_policy", lambda actor_args, actor_vecenv, env_name: actor)

    def build_benchmark_args(source_args, benchmark, environment_config):
        actor_args = copy.deepcopy(source_args)
        actor_args["env"].update(environment_config)
        actor_args["train"]["seed"] = 66
        actor_args["vec"]["seed"] = 456
        return actor_args

    monkeypatch.setattr(
        pufferl.drive_benchmark,
        "load_benchmark_config",
        lambda config_path, selected: ({"map_dir": "benchmark/maps"}, [{"name": "carla_fast"}]),
    )
    monkeypatch.setattr(pufferl.drive_benchmark, "build_benchmark_args", build_benchmark_args)

    def load_env(env_name, actor_args, seed):
        captured["actor_env"] = actor_args["env"].copy()
        captured["actor_vec"] = actor_args["vec"].copy()
        captured["seed"] = seed
        captured["actor_train_seed"] = actor_args["train"]["seed"]
        return vecenv

    def run_eval(**kwargs):
        captured["eval"] = kwargs
        return {"carla_fast": {"summary": None, "episodes": []}}

    monkeypatch.setattr(pufferl, "load_env", load_env)
    if render_scenarios:
        from pufferlib import sac_evaluation
        monkeypatch.setattr(sac_evaluation, "eval", run_eval)
    else:
        monkeypatch.setattr(pufferl, "eval", run_eval)

    result = pufferl_sac.eval("puffer_drive", args=args, benchmark_names="carla_fast")

    assert result["carla_fast"]["summary"] is None
    assert captured["actor_env"]["num_agents"] == 1
    assert captured["actor_env"]["min_agents_per_env"] == 1
    assert captured["actor_env"]["max_agents_per_env"] == 1
    assert captured["actor_env"]["num_maps"] == 1
    assert captured["actor_env"]["map_dir"] == "benchmark/maps"
    assert captured["actor_vec"]["backend"] == "Serial"
    assert captured["actor_vec"]["num_envs"] == 1
    assert captured["seed"] == 456
    assert captured["actor_train_seed"] == 66
    assert vecenv.closed
    assert captured["eval"]["policy"] is actor
    assert captured["eval"]["args"]["load_model_path"] == "actor.pt"
    assert captured["eval"]["benchmark_names"] == "carla_fast"
    assert args["env"]["num_agents"] == 16


@pytest.mark.parametrize(("global_rank", "expected_epochs"), [(0, [2, 3]), (1, [])])
def test_training_evaluation_runs_on_rank_zero_at_interval_and_final(
    tmp_path, monkeypatch, global_rank, expected_epochs,
):
    args = pufferl_sac.load_config(overrides=[])
    args["train"].update(
        total_timesteps=3,
        evaluation_interval_epochs=2,
        data_dir=str(tmp_path),
        fresh_run=True,
    )
    vecenv = _VecEnv()
    actor = _Policy()
    critic = _Policy()
    logger = _Logger()
    captured_evaluations = []

    class _FakeSAC:
        def __init__(self, config, vecenv, policy, critic, replay, logger):
            self.config = config
            self.vecenv = vecenv
            self.uncompiled_policy = policy
            self.agent_steps = 0
            self.epoch = 0
            self.rank = global_rank
            self.distributed = False
            self.logger = logger or _Logger()
            self.closed = False

        def evaluate(self):
            pass

        def train(self):
            self.epoch += 1
            self.agent_steps += 1
            return {"epoch": self.epoch}

        def close(self):
            self.closed = True
            return "final_model.pt"

    monkeypatch.setattr(pufferl_sac, "PuffeSAC", _FakeSAC)
    monkeypatch.setattr(pufferl_sac, "_initialize_process_group", lambda *args: (False, "cpu", 2, global_rank))
    monkeypatch.setattr(pufferl.drive_benchmark, "validate_training_evaluation_config", lambda args: True)
    monkeypatch.setattr(pufferl, "derive_rank_seeds", lambda *args: (1, 2))
    monkeypatch.setattr(pufferl, "torch_device", lambda device: "cpu")
    monkeypatch.setattr(pufferl, "_get_git_metadata", lambda: {})
    monkeypatch.setattr(pufferl, "_save_experiment_config", lambda *args: None)
    monkeypatch.setattr(
        pufferl,
        "run_training_evaluation",
        lambda **kwargs: captured_evaluations.append(kwargs),
    )

    pufferl_sac.train(
        env_name="puffer_drive",
        args=args,
        vecenv=vecenv,
        policy=actor,
        critic=critic,
        replay=object(),
        logger=logger,
    )

    assert [item["epoch"] for item in captured_evaluations] == expected_epochs
    assert all(item["policy"] is actor for item in captured_evaluations)
    assert all(item["global_step"] == item["epoch"] for item in captured_evaluations)
    if global_rank == 0:
        assert logger.close_args == ("final_model.pt", False)
    else:
        assert logger.close_args is None
