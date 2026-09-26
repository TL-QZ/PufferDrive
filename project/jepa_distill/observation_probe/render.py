"""Render decoded observation geometry with the existing ego-view plotter."""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Mapping

from .contracts import DecoderOutput

if TYPE_CHECKING:
    import numpy as np


def decoded_observation(
    prediction: DecoderOutput,
    *,
    observation_layout: Mapping[str, int],
    presence_threshold: float,
) -> np.ndarray:
    """Pack predicted heads as [B,D], using predicted presence, never true masks."""
    import numpy as np
    from .schema import ObservationSchema

    packed = ObservationSchema(observation_layout).pack(
        prediction, presence_threshold=presence_threshold
    )
    values = packed.detach().cpu().numpy().copy()
    if not np.isfinite(values).all():
        raise ValueError("Decoded observations contain non-finite values")
    return values


def render_comparison(
    actual_future: np.ndarray,
    reconstructed_future: np.ndarray,
    predicted_future: np.ndarray,
    *,
    environment_config: Mapping[str, Any],
) -> np.ndarray:
    """Return labeled RGB panels in the same future ego-local coordinate frame.

    Copies protect memmaps from plotting mutations. Decoder geometry is shown
    as predicted, including errors; no true masks or geometric repair is used.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from pufferlib.viz import plot_observation

    observations = (actual_future, reconstructed_future, predicted_future)
    if any(values.ndim != 1 or not np.isfinite(values).all() for values in observations):
        raise ValueError("Rendering requires three finite flat observations")
    if len({values.shape for values in observations}) != 1:
        raise ValueError("Comparison observations must have the same layout width")
    kwargs = {
        name: environment_config[name]
        for name in inspect.signature(plot_observation).parameters
        if name in environment_config and name not in {"obs", "agent_idx"}
    }
    # C observations store segment half-length; plot_observation divides its
    # interpreted length by two. Correct only this adapter, not stored data.
    kwargs['obs_norm_road_seg_length_m'] = 2 * environment_config['obs_norm_road_seg_length_m']
    titles = ("Actual future observation", "Decoded actual future latent", "Decoded predicted future latent")
    figure = Figure(figsize=(18, 6), dpi=120, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    for axis, observation, title in zip(figure.subplots(1, 3), observations, titles):
        axis.imshow(plot_observation(observation[None].copy(), **kwargs))
        axis.set_title(title)
        axis.axis("off")
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()
