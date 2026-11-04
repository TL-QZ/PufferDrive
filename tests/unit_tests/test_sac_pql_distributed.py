"""CPU/Gloo integration check for distributed PQL learner accounting."""

from collections import defaultdict
from datetime import timedelta
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from pufferlib import pufferl_sac
from pufferlib.sac_sequence_replay import SequenceReplay


_WORLD_SIZE = 2
_OBSERVATION_SHAPE = (3,)


class _TinyActor(nn.Module):
    action_count = 2
    is_continuous = False

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, observations):
        return self.linear(observations)


class _TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Linear(3, 2)
        self.q2 = nn.Linear(3, 2)

    def forward(self, observations):
        return self.q1(observations), self.q2(observations)


class _DummyUtilization:
    def stop(self):
        pass

    def join(self):
        pass


class _VecEnv:
    num_agents = 2
    agents_per_batch = 2
    single_observation_space = SimpleNamespace(shape=_OBSERVATION_SHAPE)
    single_action_space = SimpleNamespace(n=2)

    def async_reset(self, seed):
        self.seed = seed

    def close(self):
        self.closed = True


def _config(data_dir):
    return {
        "device": "cpu",
        "seed": 103,
        "run_name": "pql-gloo-test",
        "data_dir": str(data_dir),
        "batch_size": 4,
        "minibatch_size": 4,
        "max_minibatch_size": 2,
        "update_epochs": 1,
        "total_timesteps": 500,
        "gamma": 0.97,
        "replay_update_ratio": 0.5,
        "learning_rate": 0.001,
        "critic_learning_rate": 0.001,
        "alpha_learning_rate": 0.001,
        "initial_alpha": 0.2,
        "target_entropy": 0.5,
        "target_entropy_schedule": "adaptive",
        "target_entropy_ema_decay": 0.5,
        "target_entropy_mean_threshold": 1e-12,
        "target_entropy_std_threshold": 0.0,
        "target_entropy_decay_factor": 0.5,
        "target_entropy_required_count": 100,
        "target_entropy_force_decay_steps": 100,
        "tau": 0.2,
        "checkpoint_interval": None,
        "action_count": 2,
        "return_estimator": "pql",
        "n_steps": 1,
        "pql_horizon": 4,
        "pql_lambda": 0.6,
        "compile": False,
    }


def _build_rank_replay(rank, replay=None):
    segment_lengths = (3, 1) if rank == 0 else (2, 2)
    if replay is None:
        replay = SequenceReplay(4, _OBSERVATION_SHAPE, num_agents=2)
    for time_idx in range(max(segment_lengths)):
        agent_ids = [
            agent_id for agent_id, length in enumerate(segment_lengths)
            if time_idx < length
        ]
        observations = torch.tensor([
            [float(2 * rank + agent_id + 1), float(time_idx + 1), float(rank + time_idx)]
            for agent_id in agent_ids
        ])
        replay.add_sequence(
            torch.tensor(agent_ids),
            observations,
            torch.tensor([(rank + agent_id + time_idx) % 2 for agent_id in agent_ids]),
            torch.tensor([
                0.2 * rank + 0.1 * agent_id + 0.05 * time_idx
                for agent_id in agent_ids
            ]),
            observations + 0.4,
            torch.full((len(agent_ids),), 0.97),
            torch.zeros(len(agent_ids), dtype=torch.bool),
            torch.zeros(len(agent_ids), dtype=torch.bool),
        )
    return replay


def _state(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _assert_rank_agreement(tensors):
    for tensor in tensors:
        tensor = tensor.detach().contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(_WORLD_SIZE)]
        dist.all_gather(gathered, tensor)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0], rtol=1e-6, atol=1e-7)


def _prepare_process(rank, init_method):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=120),
    )
    from pufferlib import pufferl

    pufferl.Utilization = _DummyUtilization
    pufferl.PuffeRL.print_dashboard = lambda *_args, **_kwargs: None


def _pql_unequal_segments_worker(rank, init_method, result_dir, data_dir):
    _prepare_process(rank, init_method)
    try:
        torch.manual_seed(31)
        replay = SequenceReplay(4, _OBSERVATION_SHAPE, num_agents=2)
        learner = pufferl_sac.PuffeSAC(
            _config(Path(data_dir) / f"rank-{rank}"),
            _VecEnv(),
            _TinyActor(),
            _TinyCritic(),
            replay,
            reset_env=False,
        )
        _build_rank_replay(rank, learner.replay)
        learner.agent_steps = 100
        raw_alpha_gradients = []
        learner.log_alpha.register_hook(
            lambda gradient: raw_alpha_gradients.append(gradient.detach().clone()) or gradient
        )
        target_before = _state(learner.uncompiled_target_critic)
        captured = {}
        original_prepare = learner._prepare_pql_targets

        def capture_targets(indices, offsets):
            captured["indices"] = indices.detach().cpu().clone()
            captured["offsets"] = offsets.detach().cpu().clone()
            captured["targets"] = original_prepare(indices, offsets).detach().cpu().clone()
            return captured["targets"].to(learner.device)

        learner._prepare_pql_targets = capture_targets
        losses = defaultdict(float)
        updates = learner._train_sac_transition(
            losses, lambda *_args, **_kwargs: None, 0,
        )

        assert updates == 1
        assert learner.learner_updates == 1
        assert learner.sampled_transitions == 8
        assert len(raw_alpha_gradients) == 1
        assert captured["indices"].numel() == 4
        assert captured["indices"].unique().numel() == 4
        segment_lengths = torch.diff(captured["offsets"]).tolist()
        assert sorted(segment_lengths) == ([1, 3] if rank == 0 else [2, 2])
        controller_state = learner.target_entropy_controller.state_dict()
        assert controller_state["decay_count"] == 1
        assert controller_state["forced_decay_count"] == 1
        assert controller_state["last_decay_agent_steps"] == 100
        local_alpha_gradient = raw_alpha_gradients[0]
        gathered_alpha_gradients = [torch.empty_like(local_alpha_gradient) for _ in range(_WORLD_SIZE)]
        dist.all_gather(gathered_alpha_gradients, local_alpha_gradient)
        torch.testing.assert_close(
            learner.log_alpha.grad,
            torch.stack(gathered_alpha_gradients).mean(),
            rtol=1e-5,
            atol=1e-7,
        )
        gathered_entropies = [torch.zeros((), dtype=torch.float64) for _ in range(_WORLD_SIZE)]
        dist.all_gather(
            gathered_entropies,
            torch.tensor(losses["entropy"], dtype=torch.float64),
        )
        global_entropy = torch.stack(gathered_entropies).mean().item()
        assert controller_state["mean_entropy"] == pytest.approx(
            0.5 * (0.5 + global_entropy),
        )
        assert controller_state["variance"] == pytest.approx(
            0.25 * (global_entropy - 0.5) ** 2,
        )

        expected_target = {
            key: target_before[key].lerp(
                _state(learner.uncompiled_critic)[key], learner.config["tau"],
            )
            for key in target_before
        }
        for key, value in learner.uncompiled_target_critic.state_dict().items():
            torch.testing.assert_close(value.detach().cpu(), expected_target[key])
        _assert_rank_agreement([
            *learner.uncompiled_policy.parameters(),
            *learner.uncompiled_critic.parameters(),
            *learner.uncompiled_target_critic.parameters(),
            learner.log_alpha,
        ])
        gathered_controller_states = [None] * _WORLD_SIZE
        dist.all_gather_object(gathered_controller_states, controller_state)
        assert gathered_controller_states[0] == gathered_controller_states[1]
        dist.barrier()
        torch.save({
            "indices": captured["indices"],
            "offsets": captured["offsets"],
            "targets": captured["targets"],
            "learner_updates": learner.learner_updates,
            "sampled_transitions": learner.sampled_transitions,
            "alpha": learner.alpha.detach().cpu(),
            "raw_alpha_gradient": local_alpha_gradient.detach().cpu(),
            "entropy": losses["entropy"],
            "controller_state": controller_state,
            "policy": _state(learner.uncompiled_policy),
            "critic": _state(learner.uncompiled_critic),
            "target_critic": _state(learner.uncompiled_target_critic),
            "target_entropy": losses["entropy"],
        }, Path(result_dir) / f"pql-rank-{rank}.pt")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_with_timeout(worker, args, timeout_seconds=120):
    process_context = mp.spawn(worker, args=args, nprocs=_WORLD_SIZE, join=False)
    for _ in range(timeout_seconds):
        if process_context.join(timeout=1):
            return
    for process in process_context.processes:
        if process.is_alive():
            process.terminate()
    for process in process_context.processes:
        process.join(timeout=5)
    pytest.fail(f"two-rank Gloo worker exceeded {timeout_seconds}s timeout")


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="PyTorch Gloo distributed support is unavailable",
)
def test_gloo_pql_handles_different_segment_shapes_with_equal_update_accounting(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    init_path = tmp_path / "pql-gloo-init"
    data_dir = tmp_path / "pql-gloo-data"
    _spawn_with_timeout(
        _pql_unequal_segments_worker,
        (f"file://{init_path}", str(result_dir), str(data_dir)),
    )
    results = [
        torch.load(result_dir / f"pql-rank-{rank}.pt", map_location="cpu", weights_only=False)
        for rank in range(_WORLD_SIZE)
    ]

    assert [result["learner_updates"] for result in results] == [1, 1]
    assert [result["sampled_transitions"] for result in results] == [8, 8]
    assert [sorted(torch.diff(result["offsets"]).tolist()) for result in results] == [
        [1, 3], [2, 2],
    ]
    assert not torch.allclose(results[0]["targets"], results[1]["targets"])
    assert results[0]["controller_state"] == results[1]["controller_state"]
    assert results[0]["controller_state"]["decay_count"] == 1
    assert results[0]["controller_state"]["forced_decay_count"] == 1
    assert results[0]["raw_alpha_gradient"].item() != pytest.approx(
        results[1]["raw_alpha_gradient"].item(), abs=1e-7,
    )
    expected_mean_entropy = sum(result["entropy"] for result in results) / _WORLD_SIZE
    assert results[0]["controller_state"]["mean_entropy"] == pytest.approx(
        0.5 * (0.5 + expected_mean_entropy),
    )
    assert results[0]["alpha"] == results[1]["alpha"]
    for key in ("policy", "critic", "target_critic"):
        assert results[0][key].keys() == results[1][key].keys()
        for name in results[0][key]:
            torch.testing.assert_close(
                results[0][key][name], results[1][key][name], rtol=1e-6, atol=1e-7,
            )
