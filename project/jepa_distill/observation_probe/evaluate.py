"""Held-out reconstruction and predicted-future evaluation of a frozen JEPA probe."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .contracts import ProbeBatch

if TYPE_CHECKING:
    from ..model import ConditionBModel
    from .model import ObservationDecoder


def fit_training_mean(dataset, config, schema):
    """Fit a bounded constant baseline before deleting the first training round."""
    import torch

    count = min(len(dataset), config['evaluation']['training_mean_windows'])
    if count < 1:
        raise ValueError('Cannot fit a training mean from an empty collection')
    indices = torch.linspace(0, len(dataset) - 1, count).long().tolist()
    total = torch.zeros(schema.observation_dim, dtype=torch.float64)
    batch_size = config['evaluation']['batch_size']
    for start in range(0, count, batch_size):
        batch = dataset.get_batch(indices[start:start + batch_size])
        total += batch.future_observations.double().sum(0)
    mean = (total / count).float()
    traffic = schema.offsets['traffic_controls']
    if schema.capacities['traffic_controls']:
        objects = mean[traffic].view(schema.capacities['traffic_controls'], 7)
        objects[:, 5:7] = objects[:, 5:7].round()
    mean[schema.offsets['counts']] = mean[schema.offsets['counts']].round()
    if not torch.isfinite(mean).all():
        raise ValueError('Training mean contains non-finite observations')
    return mean


def _training_mean(config, schema):
    """Use persisted baseline statistics; cached mode can still fit them on demand."""
    import torch

    saved_mean = config.get('_training_mean_observation')
    if saved_mean is not None:
        mean = torch.as_tensor(saved_mean, dtype=torch.float32, device='cpu')
        if mean.shape != (schema.observation_dim,) or not torch.isfinite(mean).all():
            raise ValueError('Saved training mean must be a finite observation-sized vector')
        return mean
    if config['data'].get('mode') == 'online':
        raise ValueError('Online probe evaluation requires its saved training mean')
    from ..runtime import resolve_path
    from .data import ProbeCollectionDataset

    data = config['data']
    root = resolve_path(data['collection_root'])
    first_round = data['collection_rounds'][0]
    manifests = [root / f'rank_{rank:03d}' / 'train' / f'round_{first_round:04d}' / 'manifest.json'
                 for rank in data['source_ranks']]
    dataset = ProbeCollectionDataset(manifests)
    try:
        return fit_training_mean(dataset, config, schema)
    finally:
        dataset.close()


def _accumulate_metrics(prediction, observations, schema, assignments, prefix, totals, counts, xy_scale, threshold):
    """Accumulate element numerators/counts; do not average batch means for MAE."""
    import torch
    targets = schema.unpack(observations)
    for name, values in prediction.continuous.items():
        if name in schema.groups:
            matched = assignments[name].indices
            batch_idx, predicted_idx, actual_idx = matched.unbind(1)
            predicted = values[batch_idx, predicted_idx]
            actual = targets.continuous[name][batch_idx, actual_idx]
            valid = targets.continuous_valid[name][batch_idx, actual_idx]
            error = (predicted - actual).abs()
            if name in ('partners', 'lanes', 'boundaries', 'traffic_controls'):
                key = f'{prefix}/{name}/position_mae_m'
                position_width = 4 if name == 'traffic_controls' else 2
                totals[key] += float(error[:, :position_width].sum()) * xy_scale
                counts[key] += error[:, :position_width].numel()
            expected_presence = torch.zeros_like(prediction.presence_logits[name], dtype=torch.bool)
            expected_presence[batch_idx, predicted_idx] = True
            active = prediction.presence_logits[name].sigmoid() >= threshold
            true_positive = int((active & expected_presence).sum())
            precision = f'{prefix}/{name}/presence_precision'
            recall = f'{prefix}/{name}/presence_recall'
            totals[precision] += true_positive
            counts[precision] += int(active.sum())
            totals[recall] += true_positive
            counts[recall] += int(expected_presence.sum())
            count_error = (active.sum(1) - expected_presence.sum(1)).abs()
            key = f'{prefix}/{name}/count_mae'
            totals[key] += float(count_error.sum())
            counts[key] += count_error.numel()
        else:
            error = (values - targets.continuous[name]).abs()
            valid = targets.continuous_valid[name]
        key = f'{prefix}/{name}/mae'
        totals[key] += float(error.masked_select(valid).sum())
        counts[key] += int(valid.sum())
    for name, logits in prediction.categorical_logits.items():
        group = name.split('.')[0]
        batch_idx, predicted_idx, actual_idx = assignments[group].indices.unbind(1)
        correct = logits[batch_idx, predicted_idx].argmax(-1) == targets.categorical[name][batch_idx, actual_idx]
        valid = targets.categorical_valid[name][batch_idx, actual_idx]
        key = f'{prefix}/{name}/accuracy'
        totals[key] += int((correct & valid).sum())
        counts[key] += int(valid.sum())


def evaluate_probe(
    jepa: ConditionBModel,
    decoder: ObservationDecoder,
    batches: Iterable[ProbeBatch],
    *,
    config: Mapping[str, Any],
) -> Mapping[str, float]:
    """Evaluate four paths on identical held-out windows; no gradients or updates.

    Actual target-latent reconstruction selects decoder checkpoints. Predicted
    latent decoding, current-observation persistence and training-mean baselines
    are diagnostic. Scenario isolation is a provenance question, not inferred
    from distinct collection seeds. Training mode is restored on return.
    """
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from .schema import ObservationSchema
    from .losses import match_objects, reconstruction_loss
    from .render import decoded_observation, render_comparison

    schema = ObservationSchema(jepa.observation_layout)
    device = next(decoder.parameters()).device
    environment = jepa.export_metadata()['teacher_config']['env']
    baseline = _training_mean(config, schema).to(device)
    totals, counts = defaultdict(float), defaultdict(int)
    decoder_was_training = decoder.training
    jepa.eval()
    decoder.eval()
    processed = 0
    rendered = 0
    render_dir = config.get('_render_dir')
    threshold = config['rendering']['presence_threshold']
    try:
        with torch.no_grad():
            for batch in batches:
                remaining = config['evaluation']['max_windows'] - processed
                if remaining <= 0:
                    break
                actual = batch.future_observations[:remaining].to(device)
                current = batch.current_observations[:remaining].to(device)
                controls = batch.executed_controls[:remaining].to(device)
                real = decoder(functional.normalize(jepa.encode_target(actual), dim=-1, eps=1e-6))
                predicted = decoder(functional.normalize(jepa.predict_endpoint(jepa.encode_context(current), controls), dim=-1, eps=1e-6))
                paths = {'probe/reconstruction': real, 'probe/prediction': predicted,
                         'probe/persistence': schema.as_prediction(current),
                         'probe/training_mean': schema.as_prediction(baseline[None].expand(actual.shape[0], -1))}
                for prefix, output in paths.items():
                    assignments = match_objects(output, actual, observation_layout=jepa.observation_layout, matching_config=config['matching'])
                    loss = reconstruction_loss(output, actual, assignments, observation_layout=jepa.observation_layout, loss_config=config['loss'])
                    key = f'{prefix}/loss_total'
                    totals[key] += float(loss.total) * actual.shape[0]
                    counts[key] += actual.shape[0]
                    _accumulate_metrics(output, actual, schema, assignments, prefix, totals, counts,
                                        environment['obs_norm_xy_offset_m'], threshold)
                if render_dir and rendered < config['evaluation']['render_samples']:
                    actual_values = actual.cpu().numpy()
                    real_values = decoded_observation(real, observation_layout=jepa.observation_layout, presence_threshold=threshold)
                    predicted_values = decoded_observation(predicted, observation_layout=jepa.observation_layout, presence_threshold=threshold)
                    image_count = min(actual.shape[0], config['evaluation']['render_samples'] - rendered)
                    Path(render_dir).mkdir(parents=True, exist_ok=True)
                    for sample_idx in range(image_count):
                        image = render_comparison(actual_values[sample_idx], real_values[sample_idx], predicted_values[sample_idx], environment_config=environment)
                        Image.fromarray(image).save(Path(render_dir) / f'comparison_{rendered:03d}.png')
                        rendered += 1
                processed += actual.shape[0]
    finally:
        decoder.train(decoder_was_training)
    if processed == 0:
        raise ValueError('Evaluation contains no eligible windows')
    metrics = {key: totals[key] / count for key, count in counts.items() if count > 0}
    metrics.update({f'{key}/elements': float(count) for key, count in counts.items()})
    metrics['probe/windows'] = float(processed)
    return metrics
