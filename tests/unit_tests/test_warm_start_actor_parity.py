from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pufferlib import pufferl, pufferl_sac
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import compute_effective_road_obs_count
from pufferlib.ocean.torch import Drive
from pufferlib.ocean.torch_sac import DriveSAC, DriveSACCritic
from pufferlib.pufferl_sac_ppo_warm_start import (
    IsolatedRNG,
    WarmStartCoordinator,
    transfer_actor_and_verify,
)
from pufferlib.sac_replay import TensorReplay
from pufferlib.spaces import Box
from project.soft_actor_critic_warm_start.configuration import resolve_config, smoke_config


def _mock_env(env_config):
    dynamics_model = env_config["dynamics_model"]
    dynamics_model_flag = {
        "classic": binding.DYNAMICS_MODEL_CLASSIC,
        "jerk": binding.DYNAMICS_MODEL_JERK,
    }[dynamics_model]
    obs_slots_lane_kept = compute_effective_road_obs_count(
        env_config["obs_slots_lane_n"], env_config["obs_dropout_lane"],
    )
    obs_slots_boundary_kept = compute_effective_road_obs_count(
        env_config["obs_slots_boundary_n"], env_config["obs_dropout_boundary"],
    )
    num_reward_coefs = binding.NUM_REWARD_COEFS if env_config["reward_conditioning"] else 0
    goal_dim = binding.GOAL_FEATURES * env_config["num_goals"]
    return SimpleNamespace(
        dynamics_model=dynamics_model,
        dynamics_model_flag=dynamics_model_flag,
        ego_features=binding.EGO_FEATURES,
        num_reward_coefs=num_reward_coefs,
        goal_dim=goal_dim,
        obs_slots_partners_n=env_config["obs_slots_partners_n"],
        partner_features=binding.PARTNER_FEATURES,
        obs_slots_lane_kept=obs_slots_lane_kept,
        lane_features=binding.LANE_FEATURES,
        obs_slots_boundary_kept=obs_slots_boundary_kept,
        boundary_features=binding.BOUNDARY_FEATURES,
        obs_slots_traffic_controls_n=env_config["obs_slots_traffic_controls_n"],
        traffic_control_features=binding.TRAFFIC_CONTROL_FEATURES,
        single_action_space=SimpleNamespace(shape=(2,)),
    )


def _synthetic_observations(env, batch_size):
    context_dim = env.num_reward_coefs + env.goal_dim
    lane_dim = env.obs_slots_lane_kept * env.lane_features
    boundary_dim = env.obs_slots_boundary_kept * env.boundary_features
    partner_dim = env.obs_slots_partners_n * env.partner_features
    traffic_dim = env.obs_slots_traffic_controls_n * env.traffic_control_features
    observation_size = (
        env.ego_features + context_dim + partner_dim + lane_dim + boundary_dim + traffic_dim
        + binding.OBS_VALID_COUNT_FEATURES
    )
    observations = torch.rand(batch_size, observation_size, dtype=torch.float32) * 2.0 - 1.0

    traffic_start = env.ego_features + context_dim + partner_dim + lane_dim + boundary_dim
    traffic_continuous_features = env.traffic_control_features - 2
    for slot_idx in range(env.obs_slots_traffic_controls_n):
        type_idx = traffic_start + slot_idx * env.traffic_control_features + traffic_continuous_features
        state_idx = type_idx + 1
        observations[:, type_idx] = 0
        observations[:, state_idx] = 0

    count_start = observation_size - binding.OBS_VALID_COUNT_FEATURES
    observations[:, count_start:] = 1
    return observations


class _NoopUtilization:
    def stop(self):
        pass

    def join(self):
        pass


class _DeterministicVecEnv:
    def __init__(self, observations, num_agents):
        self.observations = observations.numpy().copy()
        self.num_agents = num_agents
        self.agents_per_batch = num_agents
        self.single_observation_space = Box(
            -1.0, 1.0, shape=(self.observations.shape[1],), dtype=np.float32,
        )
        self.single_action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = self.single_action_space
        self.sent_actions = []
        self.step = 0

    def async_reset(self, seed):
        self.reset_seed = seed
        self.step = 0

    def recv(self):
        observations = self.observations.copy()
        observations[:, 0] += self.step * 0.01
        rewards = (np.arange(self.num_agents, dtype=np.float32) + self.step) * 0.01
        terminals = np.zeros(self.num_agents, dtype=np.float32)
        truncations = np.zeros(self.num_agents, dtype=np.float32)
        infos = [{} for _ in range(self.num_agents)]
        env_ids = np.arange(self.num_agents, dtype=np.int64)
        masks = np.ones(self.num_agents, dtype=np.bool_)
        self.step += 1
        return observations, rewards, terminals, truncations, infos, env_ids, masks

    def send(self, actions):
        self.sent_actions.append(np.asarray(actions).copy())


def _make_ppo_learner(args, env, vecenv, run_name):
    train_config = deepcopy(args["train"])
    train_config["checkpoint_interval"] = 1000
    train_config.update(env="puffer_drive", run_name=run_name)
    policy = Drive(env, **args["policy"])
    learner = pufferl.PuffeRL(
        train_config, vecenv, policy, logger=pufferl.NoLogger(train_config),
    )
    return learner


def _make_hybrid_sac(args, env, vecenv, run_name, sac_rng):
    sac_config = {
        key: value for key, value in args["train"].items()
        if key in pufferl_sac.SAC_TRAIN_KEYS
    }
    sac_config.update(data_dir=f"/tmp/{run_name}", env="puffer_drive", run_name=run_name)
    with sac_rng.activate():
        policy = DriveSAC(env, **args["policy"])
        critic = DriveSACCritic(env, **args["policy"])
        sac_config["action_count"] = policy.action_count
        replay = TensorReplay(
            sac_config["batch_size"], vecenv.single_observation_space.shape,
        )
        learner = pufferl_sac.PuffeSAC(
            sac_config,
            vecenv,
            policy,
            critic,
            replay,
            logger=pufferl.NoLogger(sac_config),
            reset_env=False,
        )
    return learner


def test_baseline_drive_actor_transfers_to_sac_with_fp32_probability_and_gradient_parity(tmp_path):
    previous_thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(23)
        resolved = resolve_config(tmp_path / "warm_start_parity")
        ppo_config = resolved["ppo"]
        sac_config = resolved["sac"]
        assert ppo_config["policy"] == sac_config["policy"]
        assert ppo_config["env"] == sac_config["env"]

        env = _mock_env(ppo_config["env"])
        ppo_policy = Drive(env, **ppo_config["policy"]).eval()
        sac_policy = DriveSAC(env, **sac_config["policy"]).eval()
        observations = _synthetic_observations(env, batch_size=4)

        # The coordinator helper performs the production handoff copy and its parity check.
        transfer_actor_and_verify(ppo_policy, sac_policy, observations)

        ppo_action_logits, _ = ppo_policy.forward_eval(observations)
        ppo_logits = ppo_action_logits[0]
        sac_logits = sac_policy(observations)
        ppo_probabilities = torch.softmax(ppo_logits.float(), dim=-1)
        sac_probabilities = torch.softmax(sac_logits.float(), dim=-1)
        torch.testing.assert_close(ppo_probabilities, sac_probabilities, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(ppo_policy.action_table, sac_policy.action_table)
        torch.testing.assert_close(ppo_policy.action_table_physical, sac_policy.action_table_physical)

        for module_name in ("actor_backbone", "actor_head"):
            ppo_module = getattr(ppo_policy, module_name)
            sac_module = getattr(sac_policy, module_name)
            ppo_parameters = dict(ppo_module.named_parameters())
            sac_parameters = dict(sac_module.named_parameters())
            assert ppo_parameters.keys() == sac_parameters.keys()
            for parameter_name in ppo_parameters:
                ppo_parameter = ppo_parameters[parameter_name]
                sac_parameter = sac_parameters[parameter_name]
                assert ppo_parameter.data_ptr() != sac_parameter.data_ptr()
                torch.testing.assert_close(ppo_parameter, sac_parameter, rtol=0, atol=0)

        selected_actions = torch.arange(observations.shape[0], dtype=torch.long) % sac_policy.action_count
        ppo_loss = F.cross_entropy(ppo_logits.float(), selected_actions)
        sac_loss = F.cross_entropy(sac_logits.float(), selected_actions)
        ppo_loss.backward()
        sac_loss.backward()
        for module_name in ("actor_backbone", "actor_head"):
            ppo_parameters = dict(getattr(ppo_policy, module_name).named_parameters())
            sac_parameters = dict(getattr(sac_policy, module_name).named_parameters())
            for parameter_name in ppo_parameters:
                ppo_gradient = ppo_parameters[parameter_name].grad
                sac_gradient = sac_parameters[parameter_name].grad
                assert ppo_gradient is not None
                assert sac_gradient is not None
                assert torch.isfinite(ppo_gradient).all()
                torch.testing.assert_close(ppo_gradient, sac_gradient, rtol=1e-5, atol=1e-7)
    finally:
        torch.set_num_threads(previous_thread_count)


def test_hybrid_critic_cycles_preserve_paired_native_ppo_weights_and_actions(tmp_path, monkeypatch):
    previous_thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(pufferl, "Utilization", _NoopUtilization)
    monkeypatch.setattr(pufferl.PuffeRL, "print_dashboard", lambda self, clear=False: None)
    monkeypatch.setattr(pufferl_sac.PuffeSAC, "print_dashboard", lambda self, clear=False: None)
    try:
        config = smoke_config(resolve_config(tmp_path / "paired_config"))
        ppo_args = config["ppo"]
        sac_args = config["sac"]
        env = _mock_env(ppo_args["env"])
        observations = _synthetic_observations(env, batch_size=ppo_args["env"]["num_agents"])
        native_vecenv = _DeterministicVecEnv(observations, ppo_args["env"]["num_agents"])
        hybrid_vecenv = _DeterministicVecEnv(observations, ppo_args["env"]["num_agents"])

        torch.manual_seed(0)
        native_ppo = _make_ppo_learner(ppo_args, env, native_vecenv, "paired_native")
        shared_training_rng = pufferl.capture_rng_state()

        torch.manual_seed(0)
        hybrid_ppo = _make_ppo_learner(ppo_args, env, hybrid_vecenv, "paired_hybrid")
        sac_rng = IsolatedRNG(pufferl, seed=61)
        hybrid_sac = _make_hybrid_sac(sac_args, env, hybrid_vecenv, "paired_hybrid", sac_rng)
        coordinator = WarmStartCoordinator(
            hybrid_ppo,
            hybrid_sac,
            handoff_steps=config["warm_start"]["handoff_steps"],
            sac_steps=config["warm_start"]["sac_steps"],
            ppo_args=ppo_args,
            sac_args=sac_args,
            sac_rng=sac_rng,
            replay_generator=torch.Generator(device="cpu").manual_seed(73),
        )
        hybrid_ppo.transition_observer = coordinator.observe_transition

        pufferl.restore_rng_state({"rng_state": shared_training_rng})
        for _ in range(2):
            native_ppo.evaluate()
            native_ppo.train()
        native_state = {
            key: value.detach().clone()
            for key, value in native_ppo.uncompiled_policy.state_dict().items()
        }
        native_actions = native_vecenv.sent_actions

        pufferl.restore_rng_state({"rng_state": shared_training_rng})
        cycle_results = [coordinator.run_ppo_cycle() for _ in range(2)]
        handoff_result = coordinator.handoff(observations)

        assert [result["cumulative_agent_steps"] for result in cycle_results] == [64, 128]
        assert all(result["critic_updates"] > 0 for result in cycle_results)
        assert handoff_result["boundary_agent_steps"] == 128
        assert hybrid_ppo.epoch == native_ppo.epoch == 2
        assert hybrid_ppo.global_step == native_ppo.global_step == 128
        assert len(hybrid_vecenv.sent_actions) == len(native_actions) == 16
        for native_action, hybrid_action in zip(native_actions, hybrid_vecenv.sent_actions):
            np.testing.assert_array_equal(native_action, hybrid_action)

        hybrid_state = hybrid_ppo.uncompiled_policy.state_dict()
        assert native_state.keys() == hybrid_state.keys()
        for key, native_tensor in native_state.items():
            torch.testing.assert_close(native_tensor, hybrid_state[key], rtol=0, atol=0)

        with torch.no_grad():
            native_logits, _ = native_ppo.uncompiled_policy.forward_eval(observations)
            hybrid_logits, _ = hybrid_ppo.uncompiled_policy.forward_eval(observations)
        torch.testing.assert_close(native_logits[0], hybrid_logits[0], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_thread_count)
