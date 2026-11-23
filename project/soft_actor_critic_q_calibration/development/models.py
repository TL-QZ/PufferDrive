"""Read-only loading and inference for a coherent frozen learner snapshot."""

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


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenModels:
    def __init__(self, checkpoint_path, env, device="cpu", compile_actor=False):
        from pufferlib.ocean.torch_sac import DriveSAC, DriveSACCritic

        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.config = yaml.safe_load(self.checkpoint_path.with_name("config.yaml").read_text())
        # These are trusted local training artifacts, including RNG metadata.
        snapshot = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        self.train_config = dict(snapshot["config"])
        self.device = torch.device(device)
        self.alpha = float(snapshot["log_alpha"].exp())
        self.gamma = float(self.train_config["gamma"])
        self.return_estimator = self.train_config.get("return_estimator", "n_step")
        if self.return_estimator not in {"n_step", "pql"}:
            raise ValueError("Unsupported checkpoint return estimator")
        if self.return_estimator == "pql" and self.train_config.get(
            "pql_entropy_convention", "soft_each_return_endpoint"
        ) != "soft_each_return_endpoint":
            raise ValueError("Unsupported PQL entropy convention")
        if self.train_config.get("n_steps", 1) != 1:
            raise ValueError("This probe implementation supports one-step SAC and PQL, not fixed n-step")
        if not self.train_config.get("critic_entropy_in_target", True):
            raise ValueError("Selected checkpoint does not use the soft bootstrap required by this experiment")
        if not np.isfinite(self.alpha) or self.alpha < 0 or not 0 <= self.gamma <= 1:
            raise ValueError("Invalid checkpoint alpha/gamma")
        self.actor = DriveSAC(env, **self.config["policy"]).to(self.device).eval()
        self.critic = DriveSACCritic(env, **self.config["policy"]).to(self.device).eval()
        self.target = DriveSACCritic(env, **self.config["policy"]).to(self.device).eval()
        self.actor.load_state_dict(snapshot["policy_state_dict"], strict=True)
        self.critic.load_state_dict(snapshot["critic_state_dict"], strict=True)
        self.target.load_state_dict(snapshot["target_critic_state_dict"], strict=True)
        for model in (self.actor, self.critic, self.target):
            model.requires_grad_(False)
        self.amp = bool(self.config["train"].get("amp", False) and self.device.type == "cuda")
        self.actor_call = torch.compile(self.actor, mode="default", dynamic=True) if compile_actor else self.actor
        self.action_table = self.actor.action_table.detach().cpu().numpy().copy()
        self.action_table_physical = self.actor.action_table_physical.detach().cpu().numpy().copy()
        if self.action_table.shape != (12, 2):
            raise ValueError("Expected twelve jerk actions")
        self.initial_hashes = self.hashes()
        self.audit = {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "config_sha256": sha256_file(self.checkpoint_path.with_name("config.yaml")),
            "epoch": int(snapshot["epoch"]), "agent_steps": int(snapshot["agent_steps"]),
            "alpha": self.alpha, "gamma": self.gamma,
            "return_estimator": self.return_estimator,
            "pql_horizon": int(self.train_config.get("pql_horizon", 128)),
            "pql_lambda": float(self.train_config.get("pql_lambda", .95)),
            "amp_bfloat16": self.amp, "device": str(self.device),
            "model_hashes": self.initial_hashes,
            "action_table_physical": self.action_table_physical.tolist(),
        }

    def context(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()

    def hashes(self):
        return {name: model_digest(getattr(self, name)) for name in ("actor", "critic", "target")}

    def verify_frozen(self):
        if self.hashes() != self.initial_hashes:
            raise AssertionError("Frozen checkpoint parameters changed")

    @torch.inference_mode()
    def policy(self, observations):
        tensor = torch.as_tensor(observations, device=self.device)
        with self.context():
            logits = self.actor_call(tensor)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return log_probs.exp().cpu().numpy(), log_probs.cpu().numpy()

    @torch.inference_mode()
    def q_values(self, observations):
        with self.context():
            q1, q2 = self.critic(torch.as_tensor(observations, device=self.device))
        return q1.float().cpu().numpy(), q2.float().cpu().numpy()

    @torch.inference_mode()
    def next_values(self, observations):
        tensor = torch.as_tensor(observations, device=self.device)
        with self.context():
            logits = self.actor(tensor)
            q1, q2 = self.target(tensor)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        values = (log_probs.exp() * (torch.minimum(q1.float(), q2.float()) - self.alpha * log_probs)).sum(-1)
        return values.cpu().numpy()
