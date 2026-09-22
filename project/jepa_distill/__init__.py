"""Condition B training components, loaded only when explicitly requested.

Keeping CLI modules lazy avoids executing them once during package import and
again under ``python -m``. Importing this package creates no model or simulator.
"""
from importlib import import_module

_EXPORTS = {
    'collect': ('collect_dataset',),
    'dataset': ('TrainingBatch', 'TrainingSample', 'TrajectoryDataset', 'WindowReference', 'build_window_index', 'validate_manifest'),
    'evaluate': ('StudentPolicyAdapter', 'evaluate_student'),
    'model': ('ConditionBModel', 'LossTerms', 'ModelOutputs'),
    'monitoring': ('MetricProgress', 'WandbMonitor'),
    'teacher': ('load_teacher', 'resolve_teacher_config'),
    'train': ('load_checkpoint', 'save_checkpoint', 'train', 'train_step', 'validate'),
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    for module_name, names in _EXPORTS.items():
        if name in names:
            return getattr(import_module(f'.{module_name}', __name__), name)
    raise AttributeError(f'{__name__!r} has no attribute {name!r}')
