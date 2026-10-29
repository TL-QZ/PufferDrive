"""Small CPU/Gloo integration checks for distributed discrete SAC."""

from collections import defaultdict
from datetime import timedelta
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from pufferlib import pufferl_sac
from pufferlib.sac_replay import TensorReplay


WORLD_SIZE = 2
_OBSERVATION_SHAPE = (3,)


class _TinyActor(nn.Module):
    action_count = 2
    is_continuous = False

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
            self.linear.bias.zero_()

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


class _RecordingLogger:
    run_id = "sac-gloo-test"

    def __init__(self):
        self.records = []
        self.close_args = None

    def log(self, values, step):
        self.records.append((dict(values), step))

    def close(self, model_path, early_stop):
        self.close_args = (model_path, early_stop)


class _RankVecEnv:
    """Same-step toy vecenv with different row counts and mask rates per rank."""

    def __init__(self, rank, single_row=False, receive_delay=0.0):
        self.rank = rank
        self.single_row = single_row
        self.receive_delay = receive_delay
        self.num_agents = 1 if single_row else 2
        self.agents_per_batch = self.num_agents
        self.single_observation_space = SimpleNamespace(shape=_OBSERVATION_SHAPE)
        self.single_action_space = SimpleNamespace(n=2)
        self.receive_count = 0
        self.reset_seed = None
        self.closed = False

    def async_reset(self, seed):
        self.reset_seed = seed

    def recv(self):
        if self.receive_delay:
            time.sleep(self.receive_delay)
        self.receive_count += 1
        if self.single_row or self.rank == 1:
            agent_ids = np.asarray([0], dtype=np.int64)
        else:
            agent_ids = np.asarray([0, 1], dtype=np.int64)

        observations = np.asarray([
            [2.0 + 0.05 * self.receive_count, float(agent_id), 1.0]
            if self.rank == 1 else
            [0.05 + 0.01 * self.receive_count, float(agent_id), 1.0]
            for agent_id in agent_ids
        ], dtype=np.float32)
        rewards = np.asarray(
            [0.1 * self.rank + 0.01 * int(agent_id) + 0.02 * self.receive_count
             for agent_id in agent_ids],
            dtype=np.float32,
        )
        masks = np.ones(agent_ids.size, dtype=np.bool_)
        if not self.single_row and self.rank == 0:
            masks[1] = False
        return (
            observations,
            rewards,
            np.zeros(agent_ids.size, dtype=np.bool_),
            np.zeros(agent_ids.size, dtype=np.bool_),
            [],
            agent_ids,
            masks,
        )

    def send(self, actions):
        self.last_actions = np.asarray(actions)

    def close(self):
        self.closed = True


def _config(data_dir, **overrides):
    config = {
        "env": "puffer_drive",
        "device": "cpu",
        "seed": 2**32 + 17,
        "run_name": "sac-gloo-test",
        "data_dir": str(data_dir),
        "batch_size": 2,
        "minibatch_size": 2,
        "max_minibatch_size": 2,
        "update_epochs": 1,
        "total_timesteps": 100,
        "gamma": 0.97,
        "replay_update_ratio": 0.5,
        "learning_rate": 0.001,
        "critic_learning_rate": 0.001,
        "alpha_learning_rate": 0.001,
        "initial_alpha": 0.2,
        "target_entropy": 0.4,
        "tau": 0.2,
        "checkpoint_interval": 50,
        "action_count": 2,
    }
    config.update(overrides)
    return config


def _prepare_process(rank, init_method):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/pufferdrive-sac-ddp-inductor")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=WORLD_SIZE,
        timeout=timedelta(seconds=120),
    )
    from pufferlib import pufferl

    pufferl.Utilization = _DummyUtilization
    pufferl.PuffeRL.print_dashboard = lambda *_args, **_kwargs: None


def _state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _assert_rank_agreement(tensors):
    for tensor in tensors:
        tensor = tensor.detach().contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, tensor)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0], rtol=1e-6, atol=1e-7)


def _full_replay_restore_worker(rank, init_method, result_dir, data_dir):
    _prepare_process(rank, init_method)
    original_ddp_forward = torch.nn.parallel.DistributedDataParallel.forward
    try:
        torch.manual_seed(13)
        vecenv = _RankVecEnv(rank)
        actor = _TinyActor()
        critic = _TinyCritic()
        replay = TensorReplay(2, _OBSERVATION_SHAPE)
        logger = _RecordingLogger()
        learner = pufferl_sac.PuffeSAC(
            _config(data_dir, compile=True, max_minibatch_size=1),
            vecenv, actor, critic, replay, logger,
        )

        def record_ddp_forward(module, *args, **kwargs):
            raise AssertionError("no-grad collection must not call a DDP forward")

        torch.nn.parallel.DistributedDataParallel.forward = record_ddp_forward
        learner.evaluate()
        torch.nn.parallel.DistributedDataParallel.forward = original_ddp_forward
        expected_local_step = 6 if rank == 0 else 3
        assert learner.global_step == expected_local_step
        assert learner.agent_steps == 9
        assert learner.valid_inserts == 4
        assert len(replay) == 2

        replay_batch = replay.get_batch(torch.arange(2))
        target_entropy = 0.4

        raw_alpha_gradients = []
        learner.log_alpha.register_hook(
            lambda gradient: raw_alpha_gradients.append(gradient.detach().clone()) or gradient
        )
        initial_actor_state = _state(learner.uncompiled_policy)
        initial_critic_state = _state(learner.uncompiled_critic)
        first_replay_batch = tuple(value.detach().cpu().clone() for value in replay_batch)
        learner.last_log_time = time.time() - 1.0
        learner.train()

        assert learner.learner_updates == 1
        assert learner.sampled_transitions == 4
        assert learner.agent_steps == 9
        assert learner.valid_inserts == 4
        assert len(raw_alpha_gradients) == 1
        raw_alpha_gradient = raw_alpha_gradients[0]
        gathered_alpha_gradients = [torch.empty_like(raw_alpha_gradient) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered_alpha_gradients, raw_alpha_gradient)
        expected_alpha_gradient = torch.stack(gathered_alpha_gradients).mean()
        torch.testing.assert_close(
            learner.log_alpha.grad, expected_alpha_gradient, rtol=1e-5, atol=1e-7,
        )
        assert gathered_alpha_gradients[0].item() * gathered_alpha_gradients[1].item() < 0
        _assert_rank_agreement([
            *learner.uncompiled_policy.parameters(),
            *learner.uncompiled_critic.parameters(),
            *learner.uncompiled_target_critic.parameters(),
            learner.log_alpha,
        ])

        first_actor_state = _state(learner.uncompiled_policy)
        first_critic_state = _state(learner.uncompiled_critic)
        first_target_state = _state(learner.uncompiled_target_critic)
        first_log_alpha = learner.log_alpha.detach().cpu().clone()
        final_model_path = learner.close()
        dist.barrier()
        checkpoint_path = Path(data_dir) / "trainer_state.pt"
        assert checkpoint_path.is_file()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert state["agent_steps"] == 9
        assert state["global_step"] == 6
        assert (final_model_path is not None) == (rank == 0)

        fresh_vecenv = _RankVecEnv(rank)
        fresh = pufferl_sac.PuffeSAC(
            _config(
                Path(data_dir) / "fresh",
                target_entropy=target_entropy,
                max_minibatch_size=1,
                compile=False,
            ),
            fresh_vecenv,
            _TinyActor(),
            _TinyCritic(),
            TensorReplay(2, _OBSERVATION_SHAPE),
            _RecordingLogger(),
        )
        fresh.load_training_state(str(checkpoint_path))
        assert fresh.agent_steps == 9
        assert fresh.global_step == (5 if rank == 0 else 4)
        assert len(fresh.replay) == 0
        fresh.evaluate()
        assert fresh.agent_steps == 18
        assert fresh.global_step == (11 if rank == 0 else 7)
        assert len(fresh.replay) == 2
        assert fresh.valid_inserts == 8
        fresh.last_log_time = time.time() - 1.0
        fresh.train()
        assert fresh.learner_updates == 2
        assert fresh.sampled_transitions == 8
        assert fresh.valid_inserts == 8
        assert fresh.agent_steps == 18
        _assert_rank_agreement([
            *fresh.uncompiled_policy.parameters(),
            *fresh.uncompiled_critic.parameters(),
            *fresh.uncompiled_target_critic.parameters(),
            fresh.log_alpha,
        ])
        dist.barrier()
        fresh.close()

        torch.save({
            "initial_actor": initial_actor_state,
            "initial_critic": initial_critic_state,
            "replay_batch": first_replay_batch,
            "target_entropy": target_entropy,
            "actor": first_actor_state,
            "critic": first_critic_state,
            "target_critic": first_target_state,
            "log_alpha": first_log_alpha,
            "local_step": expected_local_step,
            "reset_seed": vecenv.reset_seed,
            "logger_records": len(logger.records),
            "raw_alpha_gradient": raw_alpha_gradient.detach().cpu(),
        }, Path(result_dir) / f"rank-{rank}.pt")
    finally:
        torch.nn.parallel.DistributedDataParallel.forward = original_ddp_forward
        if dist.is_initialized():
            dist.destroy_process_group()


def _small_budget_train_worker(rank, init_method, result_dir, data_dir):
    _prepare_process(rank, init_method)
    try:
        torch.manual_seed(19)
        vecenv = _RankVecEnv(rank, single_row=True)
        policy = _TinyActor()
        critic = _TinyCritic()
        replay = TensorReplay(8, _OBSERVATION_SHAPE)
        logger = _RecordingLogger()
        callback_ranks = []

        def early_stop(_logs):
            callback_ranks.append(rank)
            return True

        config = _config(
            data_dir,
            batch_size=8,
            minibatch_size=2,
            max_minibatch_size=2,
            total_timesteps=4,
        )
        pufferl_sac.train(
            args=config,
            vecenv=vecenv,
            policy=policy,
            critic=critic,
            replay=replay,
            logger=logger,
            early_stop_fn=early_stop,
        )
        dist.barrier()
        checkpoint_path = Path(data_dir) / "trainer_state.pt"
        assert checkpoint_path.is_file()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert state["agent_steps"] == 4
        assert state["global_step"] == 2
        assert state["valid_inserts"] == 2
        assert state["sampled_transitions"] == 0
        assert state["learner_updates"] == 0
        assert len(list((Path(data_dir) / "models").glob("model_*.pt"))) == 1
        assert callback_ranks == ([0] if rank == 0 else [])
        assert (len(logger.records) > 0) == (rank == 0)

        early_data_dir = Path(data_dir) / "early-stop-before-budget"
        early_vecenv = _RankVecEnv(rank, single_row=True, receive_delay=0.1)
        early_logger = _RecordingLogger()
        early_callback_ranks = []
        early_config = _config(
            early_data_dir,
            batch_size=2,
            minibatch_size=2,
            max_minibatch_size=2,
            total_timesteps=100,
        )
        pufferl_sac.train(
            args=early_config,
            vecenv=early_vecenv,
            policy=_TinyActor(),
            critic=_TinyCritic(),
            replay=TensorReplay(2, _OBSERVATION_SHAPE),
            logger=early_logger,
            early_stop_fn=lambda _logs: early_callback_ranks.append(rank) or True,
        )
        dist.barrier()
        early_state = torch.load(
            early_data_dir / "trainer_state.pt", map_location="cpu", weights_only=False,
        )
        assert early_state["agent_steps"] == 6
        assert early_state["agent_steps"] < early_state["config"]["total_timesteps"]
        assert early_state["learner_updates"] == 1
        assert early_callback_ranks == ([0] if rank == 0 else [])
        assert (len(early_logger.records) > 0) == (rank == 0)
        torch.save({
            "callback_ranks": callback_ranks,
            "logger_records": len(logger.records),
            "replay_size": len(replay),
            "env_closed": vecenv.closed,
            "checkpoint_model_count": len(list((Path(data_dir) / "models").glob("model_*.pt"))),
            "early_callback_ranks": early_callback_ranks,
            "early_logger_records": len(early_logger.records),
            "early_env_closed": early_vecenv.closed,
        }, Path(result_dir) / f"budget-rank-{rank}.pt")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _adaptive_target_entropy_worker(rank, init_method, result_dir, data_dir):
    _prepare_process(rank, init_method)
    try:
        torch.manual_seed(31)
        observations = torch.tensor(
            [[0.0, 0.0, 1.0]] if rank == 0 else [[2.0, 0.0, 1.0], [4.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        local_count = observations.shape[0]
        replay = TensorReplay(local_count, _OBSERVATION_SHAPE)
        replay.add(
            observations,
            torch.zeros(local_count, dtype=torch.int64),
            torch.zeros(local_count),
            observations.clone(),
            torch.zeros(local_count),
        )
        learner = pufferl_sac.PuffeSAC(
            _config(
                Path(data_dir) / f"rank-{rank}",
                batch_size=local_count,
                minibatch_size=local_count,
                max_minibatch_size=local_count,
                target_entropy=0.5,
                target_entropy_schedule="adaptive",
                target_entropy_ema_decay=0.5,
                target_entropy_mean_threshold=1e-12,
                target_entropy_std_threshold=0.0,
                target_entropy_decay_factor=0.5,
                target_entropy_required_count=100,
                target_entropy_force_decay_steps=100,
            ),
            _RankVecEnv(rank),
            _TinyActor(),
            _TinyCritic(),
            replay,
            _RecordingLogger(),
        )
        # Each rank receives the same global count, independent of its local rows.
        learner.agent_steps = 100

        losses = defaultdict(float)
        updates = learner._train_sac_transition(losses, lambda *_args, **_kwargs: None, 0)
        assert updates == 1
        local_entropy = losses["entropy"]
        controller_state = learner.target_entropy_controller.state_dict()
        gathered_states = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_states, controller_state)
        assert gathered_states[0] == gathered_states[1]
        torch.save({
            "count": local_count,
            "entropy": local_entropy,
            "controller_state": controller_state,
        }, Path(result_dir) / f"adaptive-rank-{rank}.pt")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_with_timeout(worker, args, timeout_seconds=60):
    process_context = mp.spawn(worker, args=args, nprocs=WORLD_SIZE, join=False)
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
def test_gloo_full_replay_ddp_updates_checkpoint_and_fresh_env_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor-cache"))
    init_path = tmp_path / "full-replay-gloo-init"
    data_dir = tmp_path / "full-replay-run"
    result_dir = tmp_path / "full-replay-results"
    result_dir.mkdir()
    _spawn_with_timeout(
        _full_replay_restore_worker,
        (f"file://{init_path}", str(result_dir), str(data_dir)),
        timeout_seconds=180,
    )

    rank_results = [
        torch.load(result_dir / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
        for rank in range(WORLD_SIZE)
    ]
    assert [result["local_step"] for result in rank_results] == [6, 3]
    assert rank_results[0]["reset_seed"] != rank_results[1]["reset_seed"]
    assert rank_results[1]["logger_records"] == 0

    baseline_config = _config(
        tmp_path / "global-batch-baseline",
        batch_size=4,
        minibatch_size=4,
        max_minibatch_size=4,
        target_entropy=rank_results[0]["target_entropy"],
    )
    baseline_replay = TensorReplay(4, _OBSERVATION_SHAPE)
    baseline_replay.add(*(
        torch.cat([result["replay_batch"][field] for result in rank_results], dim=0)
        for field in range(5)
    ))
    from pufferlib import pufferl

    monkeypatch.setattr(pufferl, "Utilization", _DummyUtilization)
    monkeypatch.setattr(pufferl.PuffeRL, "print_dashboard", lambda *_args, **_kwargs: None)
    baseline_actor = _TinyActor()
    baseline_actor.load_state_dict(rank_results[0]["initial_actor"])
    baseline_critic = _TinyCritic()
    baseline_critic.load_state_dict(rank_results[0]["initial_critic"])
    baseline = pufferl_sac.PuffeSAC(
        baseline_config,
        _RankVecEnv(0),
        baseline_actor,
        baseline_critic,
        baseline_replay,
    )
    baseline._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
    for name, baseline_module in (
        ("actor", baseline.uncompiled_policy),
        ("critic", baseline.uncompiled_critic),
        ("target_critic", baseline.uncompiled_target_critic),
    ):
        for rank_result in rank_results:
            for key, value in baseline_module.state_dict().items():
                torch.testing.assert_close(
                    rank_result[name][key], value.detach().cpu(), rtol=5e-5, atol=5e-6,
                )
    assert rank_results[0]["raw_alpha_gradient"].item() * rank_results[1]["raw_alpha_gradient"].item() < 0
    for rank_result in rank_results:
        torch.testing.assert_close(
            rank_result["log_alpha"], baseline.log_alpha.detach().cpu(), rtol=5e-5, atol=5e-6,
        )
    baseline.close()


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="PyTorch Gloo distributed support is unavailable",
)
def test_gloo_small_budget_skips_incomplete_replay_and_broadcasts_outer_early_stop(tmp_path):
    init_path = tmp_path / "small-budget-gloo-init"
    data_dir = tmp_path / "small-budget-run"
    result_dir = tmp_path / "small-budget-results"
    result_dir.mkdir()
    _spawn_with_timeout(
        _small_budget_train_worker,
        (f"file://{init_path}", str(result_dir), str(data_dir)),
        timeout_seconds=60,
    )
    results = [
        torch.load(result_dir / f"budget-rank-{rank}.pt", map_location="cpu", weights_only=False)
        for rank in range(WORLD_SIZE)
    ]
    assert results[0]["callback_ranks"] == [0]
    assert results[1]["callback_ranks"] == []
    assert results[0]["logger_records"] > 0
    assert results[1]["logger_records"] == 0
    assert results[0]["replay_size"] == results[1]["replay_size"] == 1
    assert results[0]["env_closed"] and results[1]["env_closed"]
    assert results[0]["checkpoint_model_count"] == results[1]["checkpoint_model_count"] == 1
    assert results[0]["early_callback_ranks"] == [0]
    assert results[1]["early_callback_ranks"] == []
    assert results[0]["early_logger_records"] > 0
    assert results[1]["early_logger_records"] == 0
    assert results[0]["early_env_closed"] and results[1]["early_env_closed"]


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="PyTorch Gloo distributed support is unavailable",
)
def test_gloo_adaptive_target_entropy_uses_weighted_global_entropy(tmp_path):
    init_path = tmp_path / "adaptive-entropy-gloo-init"
    result_dir = tmp_path / "adaptive-entropy-results"
    data_dir = tmp_path / "adaptive-entropy-run"
    result_dir.mkdir()
    _spawn_with_timeout(
        _adaptive_target_entropy_worker,
        (f"file://{init_path}", str(result_dir), str(data_dir)),
        timeout_seconds=90,
    )

    results = [
        torch.load(result_dir / f"adaptive-rank-{rank}.pt", map_location="cpu", weights_only=False)
        for rank in range(WORLD_SIZE)
    ]
    weighted_entropy = sum(result["entropy"] * result["count"] for result in results) / sum(
        result["count"] for result in results
    )
    unweighted_rank_mean = sum(result["entropy"] for result in results) / WORLD_SIZE
    assert weighted_entropy != pytest.approx(unweighted_rank_mean, abs=1e-3)

    controller_states = [result["controller_state"] for result in results]
    assert controller_states[0] == controller_states[1]
    state = controller_states[0]
    assert state["mean_entropy"] == pytest.approx(0.5 + 0.5 * (weighted_entropy - 0.5))
    assert state["variance"] == pytest.approx(0.25 * (weighted_entropy - 0.5) ** 2)
    assert state["target_entropy"] == pytest.approx(0.25)
    assert state["decay_count"] == 1
    assert state["forced_decay_count"] == 1
    assert state["last_decay_agent_steps"] == 100
