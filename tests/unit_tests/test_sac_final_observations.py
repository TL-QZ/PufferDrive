from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
import gymnasium

from pufferlib import PufferEnv
from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl_sac import PuffeSAC
from pufferlib.vector import FINAL_OBSERVATIONS_INFO_KEY, Serial, _remap_final_observation_infos


def test_map_resample_keeps_captured_scenario_final_and_copies_other_slots():
    env = object.__new__(Drive)
    env.agent_offsets = np.array([0, 3])
    env.final_observation_valid = np.array([1, 0, 0, 0], dtype=np.uint8)
    env.final_observations = np.full((4, 2), -1, dtype=np.float32)
    env.final_observations[0] = [8, 9]
    env.observations = np.array([[80, 90], [2, 3], [4, 5], [6, 7]], dtype=np.float32)
    env.capture_final_observations = True

    env._capture_map_resample_final_observations()
    info = env._final_observation_info()

    assert env.final_observation_valid.tolist() == [1, 1, 1, 0]
    np.testing.assert_array_equal(env.final_observations[:3], [[8, 9], [2, 3], [4, 5]])
    np.testing.assert_array_equal(info["_pufferlib_final_observations"]["env_ids"], [0, 1, 2])
    np.testing.assert_array_equal(info["_pufferlib_final_observations"]["observations"], env.final_observations[:3])


def test_final_observation_vector_remap_supports_noncontiguous_worker_ids():
    infos = [{
        FINAL_OBSERVATIONS_INFO_KEY: {
            "env_ids": np.array([1, 3], dtype=np.int64),
            "observations": np.array([[10], [30]], dtype=np.float32),
        }
    }]

    _remap_final_observation_infos(infos, np.array([101, 407, 903, 1205], dtype=np.int64))

    np.testing.assert_array_equal(infos[0][FINAL_OBSERVATIONS_INFO_KEY]["env_ids"], [407, 1205])


class _FinalInfoEnv(PufferEnv):
    def __init__(self, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.num_agents = 2
        super().__init__(buf=buf)

    def reset(self, seed=None):
        self.observations[:] = 0
        return self.observations, []

    def step(self, actions):
        self.observations[:] = 1
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            [{FINAL_OBSERVATIONS_INFO_KEY: {
                "env_ids": np.array([1], dtype=np.int64),
                "observations": np.array([[5, 6]], dtype=np.float32),
            }}],
        )

    def close(self):
        return None


def _make_final_info_env(**kwargs):
    return _FinalInfoEnv(**kwargs)


def test_serial_offsets_final_observation_slots_across_envs():
    vecenv = Serial([_make_final_info_env] * 2, [[], []], [{}, {}], num_envs=2)
    try:
        vecenv.reset(seed=1)
        _, _, _, _, infos = vecenv.step(np.zeros(vecenv.action_space.shape, dtype=np.int32))
    finally:
        vecenv.close()

    final_ids = [
        info[FINAL_OBSERVATIONS_INFO_KEY]["env_ids"]
        for info in infos if FINAL_OBSERVATIONS_INFO_KEY in info
    ]
    np.testing.assert_array_equal(np.concatenate(final_ids), [1, 3])


class _Profile:
    def __call__(self, *args, **kwargs):
        return None

    def end(self):
        return None


class _Replay:
    def __init__(self):
        self.transitions = None

    def __len__(self):
        return 0

    def add(self, *transitions):
        self.transitions = tuple(value.clone() for value in transitions)


class _VecEnv:
    single_observation_space = SimpleNamespace(shape=(2,))

    def __init__(self, infos):
        self.infos = infos
        self.sent_actions = None

    def recv(self):
        return (
            np.array([[100, 101], [200, 201]], dtype=np.float32),
            np.array([1, 2], dtype=np.float32),
            np.array([False, True]),
            np.array([True, True]),
            self.infos,
            np.array([42, 7], dtype=np.int64),
            np.array([True, True]),
        )

    def send(self, actions):
        self.sent_actions = actions


def _make_evaluate_only_sac(vecenv):
    trainer = object.__new__(PuffeSAC)
    trainer.profile = _Profile()
    trainer.epoch = 0
    trainer.replay = _Replay()
    trainer.batch_size = 2
    trainer.refresh_count = 2
    trainer.config = {"total_timesteps": 2, "gamma": 0.9}
    trainer.global_step = 0
    trainer.valid_inserts = 0
    trainer.device = torch.device("cpu")
    trainer.vecenv = vecenv
    trainer.pending = torch.zeros(50, dtype=torch.bool)
    trainer.pending[[42, 7]] = True
    trainer.observations = torch.zeros((50, 2), dtype=torch.float32)
    trainer.observations[42] = torch.tensor([4, 5])
    trainer.observations[7] = torch.tensor([6, 7])
    trainer.actions = torch.zeros(50, dtype=torch.int64)
    trainer.stats = defaultdict(list)
    trainer.env_continuous = False
    trainer.policy = lambda observations: torch.zeros((observations.shape[0], 2))
    trainer.uncompiled_policy = trainer.policy
    trainer.amp_context = nullcontext()
    trainer.pytorch = SimpleNamespace(
        sample_logits=lambda logits, **kwargs: (torch.zeros(logits.shape[0], dtype=torch.long), None, None, None)
    )
    trainer.helpers = SimpleNamespace(
        pufferlib=SimpleNamespace(
            unroll_nested_dict=lambda info: [],
            utils=SimpleNamespace(reduce_environment_metrics=lambda stats: {}),
        )
    )
    return trainer


def test_sac_evaluate_bootstraps_nonterminal_truncation_from_final_obs():
    vecenv = _VecEnv([{
        FINAL_OBSERVATIONS_INFO_KEY: {
            "env_ids": np.array([42], dtype=np.int64),
            "observations": np.array([[9, 10]], dtype=np.float32),
        }
    }])
    trainer = _make_evaluate_only_sac(vecenv)

    trainer.evaluate()

    _, _, _, next_observations, discounts = trainer.replay.transitions
    torch.testing.assert_close(next_observations, torch.tensor([[9, 10], [6, 7]], dtype=torch.float32))
    torch.testing.assert_close(discounts, torch.tensor([0.9, 0.0]))
    assert vecenv.sent_actions.shape == (2,)
