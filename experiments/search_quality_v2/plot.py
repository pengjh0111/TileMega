"""Regenerate SQ1–SQ4 from the compact result files in one case directory."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#555555']


def rows(directory, name):
    with (directory / name).open() as handle:
        return list(csv.DictReader(handle))


def save(fig, directory, name):
    fig.tight_layout(pad=.7)
    for extension in ('pdf', 'png'):
        fig.savefig(directory / f'fig_{name}.{extension}', dpi=300)
    plt.close(fig)


def funnel(directory, n, population):
    fig, (ax, lower) = plt.subplots(2, 1, figsize=(3.3, 4.1), gridspec_kw={'height_ratios': [1, 1]})
    xs = np.arange(5)
    ax.bar(xs, n['funnel_counts'], color=COLORS, width=.62)
    ax.set_yscale('log')
    ax.set_ylim(.8, n['funnel_counts'][0] * 22)
    ax.set_xticks(xs, ['$\\mathcal{Q}(p)$', 'E', 'Q', 'S', 'Select'])
    ax.set_ylabel('Candidate count')
    for x, y, elapsed in zip(xs, n['funnel_counts'], n['funnel_seconds'], strict=True):
        tag = f'{y:.2e}' if x == 0 else str(int(y))
        cost = f'{elapsed:.1f}s' if elapsed is not None else ''
        ax.text(x, y * 1.4, f'{tag}\n{cost}', ha='center', va='bottom', fontsize=7)
    rng = np.random.default_rng(20261124)
    for i, label in enumerate(('U', 'E', 'Q', 'S')):
        values = [float(r['tpot_ms']) for r in population if r[label] == 'True']
        if not values:
            continue
        lower.scatter(i + rng.uniform(-.15, .15, len(values)), values, s=5, alpha=.56,
                      color=COLORS[i], linewidths=0)
        lower.boxplot([values], positions=[i], showfliers=False, widths=.37,
                      medianprops={'color': 'black'})
    lower.axhline(n['oracle_ms'], color='black', linestyle='--', linewidth=.7)
    lower.set_xticks(range(4), ['U', 'E', 'Q', 'S'])
    lower.set_ylabel('TPOT (ms)')
    save(fig, directory, 'SQ1')


def regret(directory, n):
    curve = rows(directory, 'regret_curves.csv')
    fig, ax = plt.subplots(figsize=(3.3, 2.55))
    for i, method in enumerate(('TileMega replay order', 'Pooled order', 'Operator-sum order', 'Random')):
        part = [r for r in curve if r['method'] == method]
        ks = [int(r['k']) for r in part]
        gaps = [float(r['gap_pct']) for r in part]
        ax.plot(ks, gaps, label=method, color=COLORS[i], linewidth=1.1,
                linestyle='--' if method == 'Random' else '-')
        if method == 'Random':
            ax.fill_between(ks, [float(r['p05']) for r in part], [float(r['p95']) for r in part],
                            color=COLORS[i], alpha=.14)
    for item, marker, label in [(n['selection'][1], 'o', 'Predicted'), (n['selection'][2], 'x', 'Measured')]:
        if item['gap_pct'] is not None:
            ax.scatter(item['compiled_candidates'], item['gap_pct'], marker=marker, s=34,
                       color='black', label=label, zorder=8)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
    ax.set_xlabel('Compiled candidates, $k$')
    ax.set_ylabel('Regret (%)')
    ax.legend(fontsize=6.2, frameon=False, ncol=2)
    save(fig, directory, 'SQ2')


def scatter(directory, n, population):
    fig, ax = plt.subplots(figsize=(3.3, 2.55))
    priority = ('S', 'Q', 'U', 'E')
    styles = {'U': ('o', COLORS[0]), 'E': ('.', COLORS[1]), 'Q': ('s', COLORS[2]),
              'S': ('D', COLORS[3]), 'Baseline': ('x', COLORS[4])}
    grouped = {k: [] for k in styles}
    for row in population:
        label = next((k for k in priority if row[k] == 'True'), 'Baseline')
        grouped[label].append(row)
    for label, (marker, color) in styles.items():
        part = grouped[label]
        if part:
            ax.scatter([float(r['pool_ns']) / 1e6 for r in part], [float(r['tpot_ms']) for r in part],
                       s=12, alpha=.7, marker=marker, color=color, label=label)
    ax.text(.04, .92, f'Spearman $\\rho$ = {n["spearman"]:.3f}', transform=ax.transAxes)
    ax.set_xlabel('Pooled prediction (ms)')
    ax.set_ylabel('Measured TPOT (ms)')
    ax.legend(frameon=False, fontsize=6.5, loc='lower right')
    save(fig, directory, 'SQ3')


def local(directory, n):
    data = n['local_matched_attention']['pair']
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.05))
    if data:
        keys = ['isolated_ms', 'program_ms']
        for axis, key, title in zip(axes, keys, ('Attention (ms)', 'Full TPOT (ms)'), strict=True):
            axis.bar(range(len(data)), [row[key] for row in data], color=COLORS[:len(data)])
            axis.set_xticks(range(len(data)), [f'{r["chunks"]}/{r["merge"]}' for r in data], fontsize=7)
            axis.set_ylabel(title)
    else:
        axes[0].text(.5, .5, 'No matched pair\nin measured P', ha='center', va='center')
        for axis in axes:
            axis.axis('off')
    save(fig, directory, 'SQ4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    d = parser.parse_args().directory
    matplotlib.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    n = json.loads((d / 'numbers.json').read_text())
    population = rows(d, 'analysis_population.csv')
    funnel(d, n, population)
    regret(d, n)
    scatter(d, n, population)
    local(d, n)


if __name__ == '__main__':
    main()
