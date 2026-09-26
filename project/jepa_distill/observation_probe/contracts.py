"""Probe interfaces only; no tensors, models or runtime resources are created here.

Shapes use B=batch windows, K=action horizon, D=observation width, Z=latent
width, S=object slots, F=continuous features and C=category classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class WindowSource:
    """Identify an existing valid window without assuming a scenario identity."""

    manifest_path: str
    window_index: int


@dataclass(frozen=True)
class ProbeBatch:
    """Current/future observations [B,D], logged controls [B,K,2], B sources.

    Future means the endpoint t+K of the same valid agent window. Data remains
    in the saved observation normalization; teacher logits are not targets.
    """

    current_observations: Tensor
    future_observations: Tensor
    executed_controls: Tensor
    source_ids: tuple[WindowSource, ...]


@dataclass(frozen=True)
class DecoderOutput:
    """Unpacked predictions; names and dimensions derive from the layout audit.

    Continuous fields: [B,F] for fixed fields or [B,S,F] for object groups.
    Categorical logits: [B,C] or [B,S,C] per categorical field.
    Presence logits: [B,S] for partners, lanes, boundaries and traffic controls.
    Predicted counts will be derived from presence, not decoded independently.
    Slots are exchangeable only within an object group; fixed fields retain
    their named meanings. Outputs must not be rounded/masked before losses.
    """

    continuous: Mapping[str, Tensor]
    categorical_logits: Mapping[str, Tensor]
    presence_logits: Mapping[str, Tensor]


@dataclass(frozen=True)
class ObjectAssignment:
    """Matched [N,3] integer triples: batch, predicted slot, true slot.

    N counts valid matches in one object group. Unmatched predicted slots
    remain subject to presence loss; an empty true set is valid.
    """

    indices: Tensor


@dataclass(frozen=True)
class ReconstructionLoss:
    """Differentiable scalar total/components and valid-element counts by group."""

    total: Tensor
    components: Mapping[str, Tensor]
    valid_counts: Mapping[str, int]


@dataclass(frozen=True)
class ProbeProgress:
    """Zero-based collection/epoch/next-batch cursor; optimizer_step counts updates."""

    collection_index: int
    epoch_index: int
    next_batch_index: int
    optimizer_step: int


@dataclass(frozen=True)
class ProbeCheckpoint:
    """Future resume payload, separate from Condition B's checkpoint format.

    Record the frozen JEPA checkpoint hash, ordered manifest identities and
    resolved probe configuration in identity. RNG includes shuffle/sampler
    state. These contracts do not yet implement save/load or DDP behavior.
    """

    decoder_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    progress: ProbeProgress
    rng_state: Mapping[str, Any]
    identity: Mapping[str, Any]
