"""Plot verified native full-evaluation distance metrics without rescaling goals."""
import csv
import os
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT / 'mplconfig' / 'distance_plot'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    with (OUTPUT / 'full_carla_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if any(row['status'] != 'finished' for row in rows):
        raise RuntimeError('All native evaluations must finish before plotting.')
    rows.sort(key=lambda row: row['case'] != 'ppo')
    labels = {
        'tes': 'TES round 2', 'nstep4_native10': 'n-step 4',
        'nstep2_native10': 'n-step 2 TES', 'pql_round4_1_native10': 'PQL round 4.1',
        'pql_official1_native10': 'PQL official 1', 'ppo': 'PPO BASELINE',
        'pql_official2': 'PQL official 2',
    }
    names = [f"{labels[row['case']]} ({float(row['goal_spacing_m']):g} m goals)" for row in rows]
    colors = ['#111111' if row['case'] == 'ppo' else '#3769a5' if float(row['goal_spacing_m']) == 30 else '#d17b18' for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, metric, title in zip(axes, ['total_distance_travelled_km', 'avg_distance_per_infraction'],
                                  ['Total distance traveled (km)', 'Pooled distance per infraction (m)']):
        values = [float(row[metric]) for row in rows]
        bars = axis.barh(names, values, color=colors)
        axis.invert_yaxis()
        axis.get_yticklabels()[0].set_fontweight('bold')
        axis.set_xlabel(title)
        axis.set_xlim(0, max(values) * 1.23)
        axis.bar_label(bars, labels=[f'{value:,.0f}' for value in values], padding=4)
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Full CARLA: native goal spacing · 1,000 scenarios per checkpoint')
    figure.text(.5, .015, 'Distance is summed across agents and episodes; infraction denominator counts affected agent episodes.',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .045, 1, .96))
    for extension in ['png', 'pdf']:
        figure.savefig(OUTPUT / f'full_carla_distances.{extension}', dpi=200, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    main()
