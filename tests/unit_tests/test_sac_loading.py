import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pufferlib import pufferl, pufferl_sac
from pufferlib.sac_replay import TensorReplay


class _TinyActor(nn.Module):
    action_count = 3

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)


class _TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Linear(3, 3)
        self.q2 = nn.Linear(3, 3)


class _FakeVecEnv:
    def __init__(self):
        self.agents_per_batch = 4
        self.num_agents = 4
        self.single_observation_space = SimpleNamespace(shape=(3,))
        self.single_action_space = SimpleNamespace(n=3)
        self.driver_env = SimpleNamespace(capture_final_observations=True)

    def async_reset(self, seed):
        self.seed = seed

    def close(self):
        pass


def test_source_architecture_respects_ppo_env_whitelist_and_actor_warmstart(tmp_path):
    startup_args = {
        "load_model_path": None,
        "policy": {"actor_hidden_size": 8, "target_only": True},
        "rnn": {"hidden_size": 8, "target_only": True},
        "policy_name": "Drive",
        "rnn_name": "Recurrent",
        "env": {
            "simulation_mode": "replay",
            "map_dir": "target/maps",
            "reward_goal": 9.0,
            "dynamics_model": "classic",
            "goal_source": "target",
            "obs_slots_lane_n": 12,
        },
    }

    def load_architecture(args):
        assert args["load_model_path"] == "source_actor.pt"
        source_args = copy.deepcopy(args)
        source_args["policy"].update(actor_hidden_size=64)
        source_args["rnn"].update(hidden_size=512)
        source_args["policy_name"] = "DriveSAC"
        source_args["rnn_name"] = None
        source_args["env"].update(
            simulation_mode="gigaflow",
            map_dir="source/maps",
            reward_goal=1.0,
            dynamics_model="jerk",
            goal_source="map",
            obs_slots_lane_n=70,
        )
        return source_args, "source/config.yaml"

    fake_ppo = SimpleNamespace(
        drive_benchmark=SimpleNamespace(load_checkpoint_architecture=load_architecture),
    )
    pufferl_sac._restore_checkpoint_architecture(startup_args, "source_actor.pt", fake_ppo)

    assert startup_args["policy"]["actor_hidden_size"] == 64
    assert startup_args["policy"]["target_only"] is True
    assert startup_args["rnn"]["hidden_size"] == 512
    assert startup_args["rnn"]["target_only"] is True
    assert startup_args["policy_name"] == "DriveSAC"
    assert startup_args["rnn_name"] is None
    assert startup_args["env"]["dynamics_model"] == "jerk"
    assert startup_args["env"]["goal_source"] == "map"
    assert startup_args["env"]["obs_slots_lane_n"] == 70
    assert startup_args["env"]["simulation_mode"] == "replay"
    assert startup_args["env"]["map_dir"] == "target/maps"
    assert startup_args["env"]["reward_goal"] == 9.0

    actor = _TinyActor()
    critic = _TinyCritic()
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
    actor_checkpoint = _TinyActor().state_dict()
    expected_actor = {key: value.clone() for key, value in actor_checkpoint.items()}
    wrapped_actor_checkpoint = {
        f"module._orig_mod.{key}": value for key, value in actor_checkpoint.items()
    }
    checkpoint_path = tmp_path / "actor.pt"
    torch.save(wrapped_actor_checkpoint, checkpoint_path)

    pufferl_sac._load_actor_weights(actor, checkpoint_path, torch.device("cpu"), pufferl)

    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, expected_actor[key])
    for before, after in zip(critic_before, critic.parameters()):
        torch.testing.assert_close(before, after)


def test_autodiscovery_explicit_resume_and_fresh_run_precedence(tmp_path):
    data_dir = tmp_path / "run"
    data_dir.mkdir()
    discovered_path = data_dir / "trainer_state.pt"
    discovered_path.touch()

    config = {"data_dir": str(data_dir), "resume_state_path": None, "fresh_run": False}
    assert pufferl_sac._resolve_resume_state_path(config) == (str(discovered_path), True)
    assert pufferl_sac._resolve_resume_state_path(
        {**config, "resume_state_path": "explicit.pt"}
    ) == ("explicit.pt", False)
    assert pufferl_sac._resolve_resume_state_path(
        config, load_model_path="actor.pt"
    ) == (None, False)
    assert pufferl_sac._resolve_resume_state_path(
        config, load_id="remote-run"
    ) == (None, False)
    with pytest.raises(ValueError, match="explicit local load_model_path"):
        pufferl_sac._resolve_resume_state_path(config, load_model_path="latest")
    assert pufferl_sac._resolve_resume_state_path(
        {**config, "fresh_run": True}
    ) == (None, False)
    assert pufferl_sac._resolve_resume_state_path(
        {**config, "resume_state_path": "explicit.pt", "fresh_run": True}
    ) == ("explicit.pt", False)
    with pytest.raises(ValueError, match="Cannot combine"):
        pufferl_sac._resolve_resume_state_path(
            {**config, "resume_state_path": "explicit.pt"}, load_model_path="actor.pt"
        )
    with pytest.raises(ValueError, match="Cannot combine"):
        pufferl_sac._resolve_resume_state_path(
            {**config, "resume_state_path": "explicit.pt"}, load_id="remote-run"
        )

    mismatch_path = tmp_path / "mismatched_run.pt"
    torch.save({
        "state_version": pufferl_sac.STATE_VERSION,
        "algorithm": "sac",
        "scope": "learner_only",
        "config": {"run_name": "another-run"},
    }, mismatch_path)
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.replay = TensorReplay(2, (1,))
    learner.global_step = learner.epoch = learner.valid_inserts = 0
    learner.learner_updates = learner.sampled_transitions = 0
    learner.device = torch.device("cpu")
    learner.config = {**pufferl_sac.DEFAULT_CONFIG, "run_name": "target-run"}
    with pytest.raises(ValueError, match="run_name does not match"):
        learner.load_training_state(mismatch_path, same_run=True)


def test_remote_download_broadcasts_actor_path_and_failures(monkeypatch):
    from pufferlib import sac_logging

    current_rank = [0]
    broadcast_message = [None]
    download_calls = []

    def broadcast_object_list(payload, src):
        if current_rank[0] == src:
            broadcast_message[0] = copy.deepcopy(payload[0])
        else:
            payload[0] = copy.deepcopy(broadcast_message[0])

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(
        sac_logging, "download_model",
        lambda args: download_calls.append(args["load_id"]) or "/shared/artifacts/actor.pt",
    )

    args = {"load_id": "remote-run"}
    assert pufferl_sac._download_remote_model(args, global_rank=0, world_size=2) == "/shared/artifacts/actor.pt"
    current_rank[0] = 1
    assert pufferl_sac._download_remote_model(args, global_rank=1, world_size=2) == "/shared/artifacts/actor.pt"
    assert download_calls == ["remote-run"]

    def fail_download(_args):
        raise ValueError("artifact is unavailable")

    monkeypatch.setattr(sac_logging, "download_model", fail_download)
    current_rank[0] = 0
    with pytest.raises(ValueError, match="artifact is unavailable"):
        pufferl_sac._download_remote_model(args, global_rank=0, world_size=2)
    current_rank[0] = 1
    with pytest.raises(RuntimeError, match="artifact is unavailable"):
        pufferl_sac._download_remote_model(args, global_rank=1, world_size=2)


def test_injected_remote_warmstart_runs_after_process_group_and_strips_provider_config(
        monkeypatch, tmp_path):
    events = []
    actor_path = str(tmp_path / "artifacts" / "actor.pt")
    args = {
        "data_dir": str(tmp_path),
        "device": "cpu",
        "seed": 5,
        "run_name": "injected-remote",
        "total_timesteps": 1,
        "batch_size": 8,
        "minibatch_size": 4,
        "max_minibatch_size": 4,
        "learning_starts": 2,
        "fresh_run": False,
        "resume_state_path": None,
        "load_id": "remote-run",
        "wandb": True,
        "wandb_project": "drive-sac",
        "wandb_group": "experiments",
        "neptune": False,
        "tag": "warmstart",
    }

    class _DummyLogger:
        def close(self, *_args):
            pass

    class _CompletedLearner:
        def __init__(self, config, _vecenv, _policy, _critic, _replay, logger):
            remote_keys = {
                "load_model_path", "load_id", "wandb", "wandb_project", "wandb_group",
                "neptune", "neptune_name", "neptune_project", "no_model_upload", "tag",
            }
            assert not (set(config) & remote_keys)
            assert config["resume_state_path"] is None
            assert config["run_name"] == "injected-remote"
            self.config = {**pufferl_sac.DEFAULT_CONFIG, **config}
            self.agent_steps = self.config["total_timesteps"]
            self.logger = logger
            self.closed = False
            events.append("learner")

        def close(self):
            self.closed = True
            return None

    def initialize_process_group(device, _ppo):
        events.append("process_group")
        return False, device, 1, 0

    def download_remote_model(remote_args, global_rank, world_size):
        assert remote_args is args
        assert global_rank == 0 and world_size == 1
        events.append("download")
        return actor_path

    def load_actor_weights(_policy, path, device, _ppo):
        assert path == actor_path
        assert torch.device(device) == torch.device("cpu")
        events.append("actor_weights")

    monkeypatch.setattr(pufferl_sac, "_initialize_process_group", initialize_process_group)
    monkeypatch.setattr(pufferl_sac, "_download_remote_model", download_remote_model)
    monkeypatch.setattr(pufferl_sac, "_load_actor_weights", load_actor_weights)
    monkeypatch.setattr(pufferl_sac, "PuffeSAC", _CompletedLearner)

    pufferl_sac.train(
        args=args,
        policy=_TinyActor(),
        critic=_TinyCritic(),
        replay=TensorReplay(8, (3,)),
        logger=_DummyLogger(),
    )

    assert events == ["process_group", "download", "actor_weights", "learner"]


def test_failed_resume_does_not_overwrite_existing_experiment_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    original_config = b"preserve this existing config\n"
    config_path.write_bytes(original_config)
    startup_args = {
        "train": {
            "seed": 5,
            "device": "cpu",
            "data_dir": str(tmp_path),
            "resume_state_path": "existing/trainer_state.pt",
            "fresh_run": False,
            "total_timesteps": 10,
            "batch_size": 8,
            "minibatch_size": 4,
            "max_minibatch_size": 4,
            "learning_starts": 2,
        },
        "policy": {"action_type": "discrete", "shared_network": False},
        "policy_name": "DriveSAC",
        "rnn_name": None,
        "rnn": {},
        "env": {},
        "vec": {"seed": 5},
        "run_name": "same-run",
        "load_model_path": None,
        "load_id": None,
        "wandb": False,
        "neptune": False,
        "tb": False,
    }

    class _DummyUtilization:
        def stop(self):
            pass

        def join(self):
            pass

    class _DummyLogger:
        def close(self, *_args):
            pass

    class _FailingLearner:
        def __init__(self, config, vecenv, _policy, _critic, _replay, logger):
            self.config = {**pufferl_sac.DEFAULT_CONFIG, **config}
            self.vecenv = vecenv
            self.logger = logger
            self.utilization = _DummyUtilization()
            self.closed = False

        def load_training_state(self, _path, same_run=False):
            assert same_run is False
            raise ValueError("invalid test checkpoint")

    monkeypatch.setattr(pufferl_sac, "load_config", lambda _env_name: startup_args)
    monkeypatch.setattr(pufferl_sac, "_restore_checkpoint_architecture", lambda *_args: None)
    monkeypatch.setattr(pufferl_sac, "_initialize_process_group", lambda device, _ppo: (False, device, 1, 0))
    monkeypatch.setattr(pufferl_sac, "PuffeSAC", _FailingLearner)
    monkeypatch.setattr(pufferl, "_get_git_metadata", lambda: {})
    actor = _TinyActor()
    replay = TensorReplay(8, (3,))

    with pytest.raises(ValueError, match="invalid test checkpoint"):
        pufferl_sac.train(
            vecenv=_FakeVecEnv(),
            policy=actor,
            critic=_TinyCritic(),
            replay=replay,
            logger=_DummyLogger(),
        )

    assert config_path.read_bytes() == original_config
