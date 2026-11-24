"""Frozen PPO policy and value inference for Q-calibration rollouts."""

from contextlib import nullcontext
import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _clean_state_dict(state_dict):
    cleaned = {}
    for name, value in state_dict.items():
        while name.startswith(("module.", "_orig_mod.")):
            name = name.split(".", 1)[1]
        if name in cleaned:
            raise ValueError(f"Checkpoint has duplicate policy key after prefix cleanup: {name}")
        cleaned[name] = value
    return cleaned


class FrozenPPO:
    """Load a feed-forward discrete Drive PPO policy and freeze its parameters."""

    def __init__(self, checkpoint_path, model_env, device="cpu"):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"PPO checkpoint not found: {self.checkpoint_path}")
        self.config_path = self.checkpoint_path.with_name("config.yaml")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"PPO config not found beside checkpoint: {self.config_path}")

        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(self.config, dict):
            raise ValueError("PPO config must contain a mapping")
        if self.config.get("package") != "ocean" or self.config.get("policy_name") != "Drive":
            raise ValueError("PPO checkpoint must use pufferlib.ocean.torch.Drive")
        if self.config.get("rnn_name") is not None or self.config.get("train", {}).get("use_rnn", False):
            raise ValueError("PPO probing currently supports feed-forward checkpoints only")
        policy_config = self.config.get("policy")
        if not isinstance(policy_config, dict) or policy_config.get("action_type") != "discrete":
            raise ValueError("PPO policy must use the discrete jerk-action head")
        if getattr(model_env, "dynamics_model", None) != self.config.get("env", {}).get("dynamics_model"):
            raise ValueError("PPO and probe environments use different dynamics models")
        if getattr(model_env, "dynamics_model", None) != "jerk":
            raise ValueError("PPO probing currently supports jerk actions only")
        expected_obs_size = getattr(model_env, "num_obs", None)
        if expected_obs_size is None or int(expected_obs_size) <= 0:
            raise ValueError("model_env must expose its positive observation size")

        self.train_config = dict(self.config.get("train") or {})
        self.gamma = float(self.train_config["gamma"])
        if not np.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("PPO gamma must be finite and in [0, 1]")
        self.device = torch.device(device)
        self.amp = (
            self.device.type == "cuda"
            and bool(self.train_config.get("amp", False))
            and self.train_config.get("precision") == "bfloat16"
        )
        if self.amp and not torch.cuda.is_bf16_supported():
            raise RuntimeError("PPO checkpoint requests bfloat16 inference on an unsupported CUDA device")

        from pufferlib.ocean.torch import Drive

        self.policy_model = Drive(model_env, **policy_config).to(self.device).eval()
        state_dict = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("PPO final model must be a non-empty policy state dictionary")
        self.policy_model.load_state_dict(_clean_state_dict(state_dict), strict=True)
        self.policy_model.requires_grad_(False)

        self.action_table = self.policy_model.action_table.detach().cpu().numpy().copy()
        self.action_table_physical = self.policy_model.action_table_physical.detach().cpu().numpy().copy()
        if self.action_table.shape != (12, 2) or self.action_table_physical.shape != (12, 2):
            raise ValueError("Expected twelve discrete jerk actions")
        self.observation_size = int(expected_obs_size)
        self.initial_hashes = {"policy": _state_digest(self.policy_model)}
        self.audit = {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "config": str(self.config_path),
            "config_sha256": sha256_file(self.config_path),
            "gamma": self.gamma,
            "gae_lambda": float(self.train_config.get("gae_lambda", 0.95)),
            "amp_bfloat16": self.amp,
            "device": str(self.device),
            "model_hashes": dict(self.initial_hashes),
            "action_table_physical": self.action_table_physical.tolist(),
        }

    def _tensor(self, observations):
        values = np.asarray(observations)
        if values.ndim != 2 or values.shape[1] != self.observation_size:
            raise ValueError(
                f"Expected observations with shape [batch, {self.observation_size}], got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("PPO observations must be finite")
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def context(self):
        if self.amp:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def hashes(self):
        return {"policy": _state_digest(self.policy_model)}

    def verify_frozen(self):
        if self.hashes() != self.initial_hashes:
            raise AssertionError("Frozen PPO policy parameters changed")

    @torch.inference_mode()
    def _forward(self, observations):
        tensor = self._tensor(observations)
        with self.context():
            actions, values = self.policy_model(tensor)
        if not isinstance(actions, (tuple, list)) or len(actions) != 1:
            raise ValueError("PPO Drive policy returned an unsupported action head")
        logits = actions[0].float()
        if logits.ndim != 2 or logits.shape != (tensor.shape[0], 12):
            raise ValueError(f"PPO policy logits have unexpected shape: {tuple(logits.shape)}")
        if values.shape != (tensor.shape[0], 1):
            raise ValueError(f"PPO value output has unexpected shape: {tuple(values.shape)}")
        return logits, values[:, 0].float()

    @torch.inference_mode()
    def policy(self, observations):
        logits, _ = self._forward(observations)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.exp().cpu().numpy(), log_probs.cpu().numpy()

    @torch.inference_mode()
    def values(self, observations):
        _, values = self._forward(observations)
        return values.cpu().numpy()
