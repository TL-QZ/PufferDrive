"""Read-only native state diagnostics for the Q-calibration analysis.

Call :func:`install_private_binding` before creating Drive environments. The
function builds and installs a private binding only in the current Python
process; all generated C, build outputs, and compiler intermediates stay under
``native/`` next to this file.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np


NATIVE_DIR = Path(__file__).resolve().parent / "native"
REWARD_COEFFICIENT_NAMES = (
    "goal_radius",
    "goal_speed",
    "collision",
    "offroad",
    "comfort",
    "lane_align",
    "vel_align",
    "lane_center",
    "center_bias",
    "velocity",
    "reverse",
    "stop_line",
    "timestep",
    "overspeed",
    "throttle",
    "steer",
    "acc",
)
GRAPH_DISTANCE_STATUS = {
    0: "no_active_goal",
    1: "missing_current_or_goal_lane",
    2: "lane_graph_unavailable",
    3: "current_or_goal_lane_absent_from_graph",
    4: "unreachable_or_invalid_graph_entry",
    5: "reachable",
    6: "inactive_or_padded_slot",
}

_private_binding: ModuleType | None = None


def _load_builder() -> ModuleType:
    builder_path = NATIVE_DIR / "build_private_binding.py"
    spec = importlib.util.spec_from_file_location("_q_calibration_binding_builder", builder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load private binding builder at {builder_path}")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def install_private_binding(force_rebuild: bool = False) -> ModuleType:
    """Build and install the private binding in this process's Drive module."""
    global _private_binding

    if force_rebuild and "diagnostic_binding" in sys.modules:
        raise RuntimeError("Cannot force-rebuild after diagnostic_binding has been loaded in this process")
    builder = _load_builder()
    extension_path = builder.build_private_binding(force_rebuild=force_rebuild)

    module = sys.modules.get("diagnostic_binding")
    if module is None:
        spec = importlib.util.spec_from_file_location("diagnostic_binding", extension_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load private Drive binding at {extension_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["diagnostic_binding"] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop("diagnostic_binding", None)
            raise
    if not hasattr(module, "vec_get_diagnostic"):
        raise RuntimeError("Loaded private Drive binding does not expose vec_get_diagnostic")

    drive_module = importlib.import_module("pufferlib.ocean.drive.drive")
    drive_module.binding = module
    _private_binding = module
    return module


def get_diagnostics(env: Any) -> dict[str, np.ndarray]:
    """Return read-only native metadata in the same row order as ``env.obs``.

    Query before policy inference/action submission to capture that action's
    state. Results contain one row per observation slot, including padded
    inactive slots. The raw graph distance is from current lane to active goal
    lane in the lane-graph matrix; status codes distinguish missing data from
    an unreachable or invalid matrix entry.
    """
    binding = _private_binding
    if binding is None or not hasattr(binding, "vec_get_diagnostic"):
        raise RuntimeError("Call install_private_binding() before querying diagnostics")
    if getattr(env, "c_envs", None) is None:
        raise RuntimeError("Cannot query diagnostics from a closed or uninitialized Drive environment")

    result = binding.vec_get_diagnostic(env.c_envs)
    expected_rows = int(env.num_agents)
    row_count = result["slot_idx"].shape[0]
    if row_count != expected_rows:
        raise RuntimeError(
            f"Native diagnostic row count {row_count} does not match Drive observation rows {expected_rows}"
        )
    for name, values in result.items():
        if values.shape[0] != row_count:
            raise RuntimeError(f"Native diagnostic field {name!r} has inconsistent row count {values.shape[0]}")
    return result
