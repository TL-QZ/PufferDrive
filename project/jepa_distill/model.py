"""Condition B student model, objectives, and explicit EMA target updates."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Mapping, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelOutputs(NamedTuple):
    """Context, four-slot categorical logits, and endpoint representations."""

    context_latents: torch.Tensor
    chunk_logits: torch.Tensor
    predicted_endpoint: torch.Tensor
    target_endpoint: torch.Tensor


class LossTerms(NamedTuple):
    """Scalar total and component losses."""

    total: torch.Tensor
    distillation: torch.Tensor
    jepa: torch.Tensor
    variance: torch.Tensor


def _mapping_copy(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return deepcopy(dict(value))


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _finite_float(value: Any, *, name: str, strictly_positive: bool = False) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not torch.isfinite(torch.tensor(numeric)) or (strictly_positive and numeric <= 0):
        qualifier = "finite and positive" if strictly_positive else "finite"
        raise ValueError(f"{name} must be {qualifier}, got {value!r}")
    return numeric


def _validate_tensor(
    value: torch.Tensor,
    *,
    name: str,
    ndim: int,
    trailing_shape: tuple[Optional[int], ...] = (),
    finite: bool = True,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {tuple(value.shape)}")
    if trailing_shape and tuple(value.shape[-len(trailing_shape) :]) != tuple(
        expected if expected is not None else actual
        for actual, expected in zip(value.shape[-len(trailing_shape) :], trailing_shape)
    ):
        actual_shape = tuple(value.shape[-len(trailing_shape) :])
        for actual, expected in zip(actual_shape, trailing_shape):
            if expected is not None and actual != expected:
                raise ValueError(f"{name} has incompatible shape {tuple(value.shape)}")
    if finite and not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _activation(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported model activation {name!r}") from exc


def _make_mlp(
    input_dim: int,
    hidden_sizes: list[int],
    output_dim: int,
    activation_name: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    activation_cls = _activation(activation_name)
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation_cls())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class ConditionBModel(nn.Module):
    """Drive-backbone student with chunk decoder, JEPA predictor, and EMA target.

    The constructor accepts a full Condition B config.  Offline reconstruction
    needs ``teacher_config`` (resolved saved PufferDrive args) and
    ``observation_layout``; neither path constructs a live environment.
    """

    def __init__(self, config: Mapping[str, Any], *, teacher: Optional[nn.Module] = None) -> None:
        super().__init__()
        self._config = _mapping_copy(config, name="config")
        model_config = self._config.get("model", self._config)
        if not isinstance(model_config, Mapping):
            raise ValueError("config.model must be a mapping")
        self.chunk_length = _positive_int(model_config.get("chunk_length", 4), name="model.chunk_length")
        self.execution_horizon = _positive_int(
            model_config.get("execution_horizon", 1), name="model.execution_horizon"
        )
        if self.execution_horizon != 1:
            raise ValueError("Condition B currently requires model.execution_horizon=1")
        self.num_action_classes = _positive_int(
            model_config.get("num_action_classes", 12), name="model.num_action_classes"
        )

        decoder_hidden_sizes = model_config.get("decoder_hidden_sizes", [256])
        predictor_hidden_sizes = model_config.get("predictor_hidden_sizes", [1024, 1024])
        if not isinstance(decoder_hidden_sizes, (list, tuple)):
            raise ValueError("model.decoder_hidden_sizes must be a list")
        if not isinstance(predictor_hidden_sizes, (list, tuple)):
            raise ValueError("model.predictor_hidden_sizes must be a list")
        self.decoder_hidden_sizes = [
            _positive_int(value, name="model.decoder_hidden_sizes entry") for value in decoder_hidden_sizes
        ]
        self.predictor_hidden_sizes = [
            _positive_int(value, name="model.predictor_hidden_sizes entry") for value in predictor_hidden_sizes
        ]
        self.activation = str(model_config.get("activation", "relu")).lower()
        _activation(self.activation)

        teacher_config = self._config.get("teacher_config")
        if teacher_config is None and teacher is not None:
            teacher_config = getattr(teacher, "condition_b_resolved_config", None)
            if teacher_config is None:
                teacher_config = getattr(teacher, "resolved_config", None)
            if teacher_config is not None:
                self._config["teacher_config"] = deepcopy(teacher_config)
        observation_layout = self._config.get("observation_layout")
        if observation_layout is None and teacher is not None:
            observation_layout = getattr(teacher, "condition_b_observation_layout", None)
        if observation_layout is None and teacher is not None:
            observation_layout = getattr(teacher, "observation_layout", None)
        if observation_layout is None and teacher_config is not None:
            from .teacher import observation_layout_from_config

            observation_layout = observation_layout_from_config(teacher_config)
        if not isinstance(observation_layout, Mapping):
            raise ValueError("config.observation_layout is required for Condition B model construction")
        self.observation_layout = _mapping_copy(observation_layout, name="observation_layout")
        required_layout_keys = (
            "ego_features",
            "partner_features",
            "lane_features",
            "boundary_features",
            "traffic_control_features",
            "num_reward_coefs",
            "goal_dim",
            "obs_slots_partners_n",
            "obs_slots_lane_kept",
            "obs_slots_boundary_kept",
            "obs_slots_traffic_controls_n",
        )
        missing_layout = [key for key in required_layout_keys if key not in self.observation_layout]
        if missing_layout and teacher is None:
            raise ValueError(f"observation_layout is missing keys: {', '.join(missing_layout)}")
        self.observation_dim = _positive_int(
            self.observation_layout.get("observation_dim"), name="observation_layout.observation_dim"
        )

        latent_dim = _positive_int(model_config.get("latent_dim", 1024), name="model.latent_dim")
        self.latent_dim = latent_dim

        context_encoder = self._build_context_encoder(
            config=self._config,
            observation_layout=self.observation_layout,
            teacher=teacher,
        )
        if int(getattr(context_encoder, "out_dim", -1)) != self.latent_dim:
            raise ValueError(
                "model.latent_dim must equal the Drive actor backbone output width "
                f"({getattr(context_encoder, 'out_dim', None)})"
            )
        self.context_encoder = context_encoder
        for parameter in self.context_encoder.parameters():
            parameter.requires_grad_(True)

        normalized_actions, physical_actions = self._build_action_tables(
            config=self._config,
            teacher=teacher,
            num_action_classes=self.num_action_classes,
        )
        self.register_buffer("action_table", normalized_actions, persistent=False)
        self.register_buffer("action_table_physical", physical_actions, persistent=False)

        # The target is a distinct copy so EMA updates never alias online
        # parameters.  It remains frozen and in eval mode for every student
        # training mode.
        self.target_encoder = deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()

        self.decoder = _make_mlp(
            self.latent_dim,
            self.decoder_hidden_sizes,
            self.chunk_length * self.num_action_classes,
            self.activation,
        )
        self.predictor = _make_mlp(
            self.latent_dim + self.chunk_length * 2,
            self.predictor_hidden_sizes,
            self.latent_dim,
            self.activation,
        )

        configured_loss = self._config.get("loss", {})
        self.loss_config = _mapping_copy(configured_loss, name="config.loss") if configured_loss else {}
        configured_ema = self._config.get("ema", {})
        self.ema_config = _mapping_copy(configured_ema, name="config.ema") if configured_ema else {}

    @staticmethod
    def _build_action_tables(
        *,
        config: Mapping[str, Any],
        teacher: Optional[nn.Module],
        num_action_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy Drive action metadata or derive the selected dynamics table."""

        for module in (teacher,):
            if module is None:
                continue
            normalized = getattr(module, "action_table", None)
            physical = getattr(module, "action_table_physical", None)
            if isinstance(normalized, torch.Tensor) and isinstance(physical, torch.Tensor):
                if tuple(normalized.shape) != (num_action_classes, 2) or tuple(physical.shape) != tuple(
                    normalized.shape
                ):
                    raise ValueError("teacher action tables do not match model.num_action_classes")
                if not torch.isfinite(normalized).all() or not torch.isfinite(physical).all():
                    raise ValueError("teacher action tables must contain only finite values")
                return normalized.detach().float().cpu().clone(), physical.detach().float().cpu().clone()

        teacher_config = config.get("teacher_config")
        if not isinstance(teacher_config, Mapping):
            # A metadata-only test teacher may not carry action semantics. The
            # default Condition B table is still explicit and serializable.
            dynamics_model = "jerk"
        else:
            environment = teacher_config.get("env")
            if not isinstance(environment, Mapping):
                raise ValueError("teacher_config.env must be a mapping")
            dynamics_model = environment.get("dynamics_model", "jerk")

        from pufferlib.ocean.drive import binding

        if dynamics_model == "jerk":
            long_values = torch.tensor(binding.JERK_LONG, dtype=torch.float32)
            lateral_values = torch.tensor(binding.JERK_LAT, dtype=torch.float32)
        elif dynamics_model == "classic":
            long_values = torch.tensor(binding.ACCELERATION_VALUES, dtype=torch.float32)
            lateral_values = torch.tensor(binding.STEERING_VALUES, dtype=torch.float32)
        else:
            raise ValueError(f"unsupported teacher dynamics model {dynamics_model!r}")
        long_normalized = torch.where(
            long_values < 0.0,
            long_values / -long_values[0],
            long_values / long_values[-1],
        )
        lateral_normalized = lateral_values / lateral_values[-1]
        class_indices = torch.arange(long_values.numel() * lateral_values.numel())
        normalized = torch.stack(
            [
                long_normalized[class_indices // lateral_values.numel()],
                lateral_normalized[class_indices % lateral_values.numel()],
            ],
            dim=-1,
        )
        physical = torch.stack(
            [
                long_values[class_indices // lateral_values.numel()],
                lateral_values[class_indices % lateral_values.numel()],
            ],
            dim=-1,
        )
        if tuple(normalized.shape) != (num_action_classes, 2):
            raise ValueError(
                f"resolved {dynamics_model} action table has {normalized.shape[0]} classes; "
                f"expected {num_action_classes}"
            )
        return normalized, physical

    @staticmethod
    def _build_context_encoder(
        *,
        config: Mapping[str, Any],
        observation_layout: Mapping[str, Any],
        teacher: Optional[nn.Module],
    ) -> nn.Module:
        if teacher is not None:
            actor_backbone = getattr(teacher, "actor_backbone", teacher)
            if not isinstance(actor_backbone, nn.Module) or not hasattr(actor_backbone, "out_dim"):
                raise TypeError("teacher must expose a Drive actor_backbone module")
            return deepcopy(actor_backbone)

        teacher_config = config.get("teacher_config")
        if not isinstance(teacher_config, Mapping):
            raise ValueError("config.teacher_config is required when teacher is omitted")
        policy = teacher_config.get("policy")
        if not isinstance(policy, Mapping):
            raise ValueError("config.teacher_config.policy must be a mapping")

        from pufferlib.ocean.torch import DriveBackbone

        required_policy_keys = (
            "ego_input_size",
            "partner_input_size",
            "lane_input_size",
            "boundary_input_size",
            "traffic_control_input_size",
            "context_input_size",
            "backbone_hidden_size",
            "backbone_num_layers",
            "encoder_activation",
            "encoder_layer_norm",
            "backbone_activation",
            "backbone_layer_norm",
            "mask_padded_features",
        )
        missing = [key for key in required_policy_keys if key not in policy]
        if missing:
            raise ValueError(f"teacher_config.policy is missing keys: {', '.join(missing)}")

        layout_env = SimpleNamespace(**dict(observation_layout))
        backbone_args = {key: deepcopy(policy[key]) for key in required_policy_keys}
        backbone_args.update(
            {
                "env": layout_env,
                "ego_dim": int(observation_layout["ego_features"]),
            }
        )
        return DriveBackbone(**backbone_args)

    def _validate_observations(self, observations: torch.Tensor, *, name: str = "observations") -> None:
        _validate_tensor(
            observations,
            name=name,
            ndim=2,
            trailing_shape=(self.observation_dim,),
        )

    def _validate_context(self, context_latents: torch.Tensor) -> None:
        _validate_tensor(
            context_latents,
            name="context_latents",
            ndim=2,
            trailing_shape=(self.latent_dim,),
        )

    def _validate_controls(self, executed_controls: torch.Tensor) -> None:
        _validate_tensor(
            executed_controls,
            name="executed_controls",
            ndim=3,
            trailing_shape=(self.chunk_length, 2),
        )

    def _encode_context_unchecked(self, observations: torch.Tensor) -> torch.Tensor:
        if hasattr(self.context_encoder, "ego_dim"):
            return self.context_encoder(observations, self.context_encoder.ego_dim)
        return self.context_encoder(observations)

    def encode_context(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode current observations ``[B,D_o]`` to ``[B,D_z]``."""

        self._validate_observations(observations)
        return self._encode_context_unchecked(observations)

    def _decode_chunk_unchecked(self, context_latents: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(context_latents)
        return logits.view(-1, self.chunk_length, self.num_action_classes)

    def decode_chunk(self, context_latents: torch.Tensor) -> torch.Tensor:
        """Decode context ``[B,D_z]`` to ``[B,K,C]`` categorical logits."""

        self._validate_context(context_latents)
        return self._decode_chunk_unchecked(context_latents)

    def _predict_endpoint_unchecked(
        self,
        context_latents: torch.Tensor,
        executed_controls: torch.Tensor,
    ) -> torch.Tensor:
        controls_flat = executed_controls.reshape(executed_controls.shape[0], -1)
        return self.predictor(torch.cat((context_latents, controls_flat), dim=1))

    def predict_endpoint(
        self,
        context_latents: torch.Tensor,
        executed_controls: torch.Tensor,
    ) -> torch.Tensor:
        """Predict endpoint ``[B,D_z]`` from latent and logged controls ``[B,K,2]``."""

        self._validate_context(context_latents)
        self._validate_controls(executed_controls)
        if context_latents.shape[0] != executed_controls.shape[0]:
            raise ValueError("context_latents and executed_controls batch sizes must match")
        return self._predict_endpoint_unchecked(context_latents, executed_controls)

    def _encode_target_unchecked(self, target_observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.target_encoder, "ego_dim"):
                return self.target_encoder(target_observations, self.target_encoder.ego_dim)
            return self.target_encoder(target_observations)

    def encode_target(self, target_observations: torch.Tensor) -> torch.Tensor:
        """Encode endpoint observations with the frozen EMA encoder."""

        self._validate_observations(target_observations, name="target_observations")
        return self._encode_target_unchecked(target_observations)

    def forward(
        self,
        observations: torch.Tensor,
        executed_controls: torch.Tensor,
    ) -> ModelOutputs:
        """Run a window batch with observations ``[B,K+1,D_o]`` and controls ``[B,K,2]``."""

        _validate_tensor(observations, name="observations", ndim=3, finite=True)
        if observations.shape[1] != self.chunk_length + 1 or observations.shape[2] != self.observation_dim:
            raise ValueError(
                "observations must have shape "
                f"[B,{self.chunk_length + 1},{self.observation_dim}], got {tuple(observations.shape)}"
            )
        self._validate_controls(executed_controls)
        if observations.shape[0] != executed_controls.shape[0]:
            raise ValueError("observations and executed_controls batch sizes must match")

        context_latents = self._encode_context_unchecked(observations[:, 0])
        chunk_logits = self._decode_chunk_unchecked(context_latents)
        predicted_endpoint = self._predict_endpoint_unchecked(context_latents, executed_controls)
        target_endpoint = self._encode_target_unchecked(observations[:, -1])
        return ModelOutputs(context_latents, chunk_logits, predicted_endpoint, target_endpoint)

    @staticmethod
    def _validate_loss_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> None:
        _validate_tensor(student_logits, name="student_logits", ndim=3)
        _validate_tensor(teacher_logits, name="teacher_logits", ndim=3)
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "student_logits and teacher_logits must have the same shape, "
                f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
            )

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute ``T²`` soft cross-entropy, reduced over batch and slots."""

        self._validate_loss_logits(student_logits, teacher_logits)
        if tuple(student_logits.shape[1:]) != (self.chunk_length, self.num_action_classes):
            raise ValueError(
                "logits must have shape "
                f"[B,{self.chunk_length},{self.num_action_classes}], got {tuple(student_logits.shape)}"
            )
        temperature_value = _finite_float(temperature, name="temperature", strictly_positive=True)
        teacher_probabilities = F.softmax(teacher_logits.detach() / temperature_value, dim=-1)
        student_log_probabilities = F.log_softmax(student_logits / temperature_value, dim=-1)
        return (
            -torch.sum(teacher_probabilities * student_log_probabilities, dim=-1).mean()
            * temperature_value
            * temperature_value
        )

    def jepa_loss(
        self,
        predicted_endpoint: torch.Tensor,
        target_endpoint: torch.Tensor,
        *,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """Compare unit-normalized endpoint vectors with squared L2 distance."""

        self._validate_context(predicted_endpoint)
        self._validate_context(target_endpoint)
        if predicted_endpoint.shape != target_endpoint.shape:
            raise ValueError("predicted_endpoint and target_endpoint must have the same shape")
        epsilon_value = _finite_float(epsilon, name="epsilon", strictly_positive=True)
        predicted_norm = predicted_endpoint / predicted_endpoint.norm(dim=-1, keepdim=True).clamp_min(epsilon_value)
        target_norm = target_endpoint.detach() / target_endpoint.norm(dim=-1, keepdim=True).clamp_min(epsilon_value)
        return (predicted_norm - target_norm).square().sum(dim=-1).mean()

    def variance_loss(
        self,
        context_latents: torch.Tensor,
        *,
        gamma: float = 1.0,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """Penalize low population standard deviation per latent dimension."""

        self._validate_context(context_latents)
        if context_latents.shape[0] < 2:
            raise ValueError("variance_loss requires at least two samples")
        gamma_value = _finite_float(gamma, name="gamma")
        epsilon_value = _finite_float(epsilon, name="epsilon", strictly_positive=True)
        population_variance = context_latents.var(dim=0, correction=0)
        population_std = torch.sqrt(population_variance + epsilon_value)
        return F.relu(gamma_value - population_std).mean()

    @staticmethod
    def _loss_section(config: Optional[Mapping[str, Any]], fallback: Mapping[str, Any]) -> Mapping[str, Any]:
        if config is None:
            return fallback
        if not isinstance(config, Mapping):
            raise ValueError("loss config must be a mapping")
        nested = config.get("loss")
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise ValueError("config.loss must be a mapping")
            return nested
        return config

    @staticmethod
    def _loss_value(section: Mapping[str, Any], names: tuple[str, ...], default: float) -> float:
        for name in names:
            if name in section:
                return _finite_float(section[name], name=name)
        return default

    def compute_losses(
        self,
        outputs: ModelOutputs,
        teacher_logits: torch.Tensor,
        *,
        config: Optional[Mapping[str, Any]] = None,
    ) -> LossTerms:
        """Compute weighted distillation, JEPA, and variance scalar losses.

        ``config`` may be the full Condition B mapping or its ``loss``
        subsection.  Both the YAML names and ``lambda_D/J/V`` aliases are
        accepted for callers holding an explicit loss mapping.
        """

        if not isinstance(outputs, ModelOutputs):
            raise TypeError("outputs must be a ModelOutputs instance")
        self._validate_loss_logits(outputs.chunk_logits, teacher_logits)
        section = self._loss_section(config, self.loss_config)
        temperature = self._loss_value(section, ("temperature",), 1.0)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        epsilon = self._loss_value(section, ("epsilon",), 1e-6)
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        gamma = self._loss_value(section, ("variance_floor", "gamma"), 1.0)
        if gamma < 0:
            raise ValueError("variance_floor must be non-negative")
        distillation_weight = self._loss_value(
            section,
            ("distillation_weight", "lambda_D", "lambda_d"),
            1.0,
        )
        jepa_weight = self._loss_value(section, ("jepa_weight", "lambda_J", "lambda_j"), 1.0)
        variance_weight = self._loss_value(
            section,
            ("variance_weight", "lambda_V", "lambda_v"),
            0.1,
        )
        for weight_name, weight in (
            ("distillation_weight", distillation_weight),
            ("jepa_weight", jepa_weight),
            ("variance_weight", variance_weight),
        ):
            if weight < 0:
                raise ValueError(f"{weight_name} must be non-negative")

        distillation = self.distillation_loss(
            outputs.chunk_logits,
            teacher_logits,
            temperature=temperature,
        )
        jepa = self.jepa_loss(outputs.predicted_endpoint, outputs.target_endpoint, epsilon=epsilon)
        variance = self.variance_loss(outputs.context_latents, gamma=gamma, epsilon=epsilon)
        total = distillation_weight * distillation + jepa_weight * jepa + variance_weight * variance
        return LossTerms(total, distillation, jepa, variance)

    def update_target_encoder(self, *, tau: float = 0.99) -> None:
        """Apply one explicit ``target = tau*target + (1-tau)*online`` update."""

        tau_value = _finite_float(tau, name="tau")
        if not 0.0 <= tau_value <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau!r}")
        with torch.no_grad():
            online_parameters = dict(self.context_encoder.named_parameters())
            for name, target_parameter in self.target_encoder.named_parameters():
                target_parameter.mul_(tau_value).add_(online_parameters[name], alpha=1.0 - tau_value)
            online_buffers = dict(self.context_encoder.named_buffers())
            for name, target_buffer in self.target_encoder.named_buffers():
                target_buffer.copy_(online_buffers[name])
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> "ConditionBModel":
        """Keep the EMA target frozen/eval while toggling the online model."""

        super().train(mode)
        self.target_encoder.eval()
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        return self

    def export_metadata(self) -> dict[str, Any]:
        """Return a self-contained config suitable for offline reconstruction."""

        metadata: dict[str, Any] = {
            "teacher_config": deepcopy(self._config.get("teacher_config")),
            "observation_layout": deepcopy(self.observation_layout),
            "model": {
                "chunk_length": self.chunk_length,
                "execution_horizon": self.execution_horizon,
                "latent_dim": self.latent_dim,
                "num_action_classes": self.num_action_classes,
                "decoder_hidden_sizes": deepcopy(self.decoder_hidden_sizes),
                "predictor_hidden_sizes": deepcopy(self.predictor_hidden_sizes),
                "activation": self.activation,
            },
            "loss": deepcopy(self.loss_config),
            "ema": deepcopy(self.ema_config),
            "action_table": self.action_table.detach().cpu().tolist(),
            "action_table_physical": self.action_table_physical.detach().cpu().tolist(),
        }
        if metadata["teacher_config"] is None:
            metadata.pop("teacher_config")
        return metadata
