"""Compact W&B metrics for the probe; full diagnostics remain in metrics.jsonl."""
from __future__ import annotations

from typing import Any, Mapping

from ..monitoring import WandbMonitor


# Preserve existing metric names so curves remain comparable across runs.
KEY_METRICS = frozenset({
    "train/loss_total",
    "probe/reconstruction/loss_total",
    "probe/prediction/loss_total",
    "probe/reconstruction/partners/position_mae_m",
    "probe/prediction/partners/position_mae_m",
    "probe/persistence/partners/position_mae_m",
    "probe/training_mean/partners/position_mae_m",
    "probe/prediction/lanes/position_mae_m",
    "probe/prediction/boundaries/position_mae_m",
    "probe/prediction/partners/presence_precision",
    "probe/prediction/partners/presence_recall",
    "probe/prediction/traffic_controls.state/accuracy",
})
PROGRESS_KEYS = frozenset({
    "event_index",
    "progress/optimizer_step",
    "progress/collection_round_idx",
    "progress/update_epoch_idx",
})


class ProbeMonitor(WandbMonitor):
    """Filter only remote payloads after the base logger saves the full event.

    Optional final test evaluation uses the same selection under ``test/``.
    Missing metrics (e.g. no traffic controls) stay absent, never become zeros.
    Checkpoint identity, event cursors and local histories use the base logger.
    """

    def _remote_log(self, payload: Mapping[str, Any], event_index: int) -> None:
        selected = {
            key: value for key, value in payload.items()
            if key in PROGRESS_KEYS or key.removeprefix("test/") in KEY_METRICS
        }
        super()._remote_log(selected, event_index)
