#!/usr/bin/env python3
"""Reproduce diagnostic figures from completed, unchanged study outputs.

The displayed analysis is the manuscript's seven selected main settings. No
candidate models are refitted and no threshold sweep or Pareto frontier is
implied. Source score tables were copied read-only from the original HPC run.
"""
from pathlib import Path
import collections
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fmvmm-manuscript-diagnostics-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/figures'
SOURCE = ROOT / 'results/baseline.json.gz'
from common import read_json
OUT.mkdir(parents=True, exist_ok=True)
SETTINGS = [
    ('gaussian_control', 'G3', 3), ('ordinary_heterogeneous', 'H3', 3),
    ('skew_robust', 'S3', 3), ('gh_heterogeneous', 'GH3', 3),
    ('four_skew', 'S4', 4), ('four_mixed', 'H4', 4), ('five_mixed', 'H5', 5),
]
FAMILIES = ['mvn', 'mvt', 'mvsn', 'mvst', 'msnc', 'msl', 'mssl', 'mvsl',
            'mghp', 'mgst', 'mvhb', 'mnig', 'mvvg']
COLORS = ['#4477AA', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#66CCEE', '#332288']
raw = read_json('baseline.json.gz')
all_rows = raw['tasks']
main = [r for r in all_rows if r['task']['kind'] == 'simulation'
        and r['task']['block'] == 'main'
        and r['task']['scenario'] in {s[0] for s in SETTINGS}]
assert len(main) == 350
assert len({r['task']['task_id'] for r in main}) == 350
assert all(r['table_status'] == 'qualified' and not r['missing_indices'] for r in main)
assert sum(r['reference'] == 'exhaustive' for r in main) == 206
assert sum(r['reference'] == 'challengers' for r in main) == 144
assert all(r['qualified_attained_bic_gap'] >= -1e-6 for r in main)
assert all(r['screen_shared_accounted_seconds'] > 0 and
           r['reference_shared_accounted_seconds'] > 0 for r in main)
groups = {scenario: [r for r in main if r['task']['scenario'] == scenario]
          for scenario, code, k in SETTINGS}
assert all(len(rows) == 50 for rows in groups.values())
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 220,
})


def gap(r):
    value = r['qualified_attained_bic_gap']
    return 0.0 if abs(value) <= 1e-6 else value


def decorate(ax):
    ax.set_axisbelow(True)
    ax.grid(color='0.9', linewidth=.45)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig)


data = {
    'summary_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'selection': 'Seven manuscript main settings; 50 replications per setting.',
    'cost_definition': 'Saved pilot/profile time plus saved candidate-operation times; '
                       'reference denominators contain requested fits only. '
                       'These are accounted operation seconds, not separately '
                       'measured end-to-end runtime or CPU-time measurements.',
    'figures': {},
    'observations': [dict(
        task_id=r['task']['task_id'], scenario=r['task']['scenario'],
        k=r['task']['fit_k'], n=r['task']['n'], p=r['task']['p'],
        reference=r['reference'], gap=gap(r), retained=r['retained_structures'],
        catalogue=r['total_canonical_structures'],
        table_seconds=r['table_seconds'],
        screened_operation_seconds=r['screen_shared_accounted_seconds'],
        reference_operation_seconds=r['reference_shared_accounted_seconds'],
        cost_ratio=r['screen_shared_accounted_seconds']/r['reference_shared_accounted_seconds'],
    ) for r in main],
}

# Observed performance pairs are not a threshold-tuned Pareto frontier: different
# replications/settings have different data, so no cross-case domination claim.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), sharey=True)
for ax, scope, title in zip(axes, ('exhaustive', 'challengers'),
                           ('(a) Full comparison searches', '(b) Partial comparison searches')):
    for (scenario, code, k), color in zip(SETTINGS, COLORS):
        rr = [r for r in groups[scenario] if r['reference'] == scope]
        ax.scatter([r['screen_shared_accounted_seconds']/r['reference_shared_accounted_seconds'] for r in rr],
                   [gap(r) for r in rr], s=16, facecolor=color, edgecolor='white',
                   linewidth=.35, alpha=.75, label=code, zorder=3)
    ax.set_title(title)
    ax.set_xlabel('Screened / comparison accounted work')
    ax.set_xscale('log')
    ax.set_xlim(.0025, 1.03)
    ax.set_xticks([.003, .01, .03, .1, .3, 1], ['0.003', '0.01', '0.03', '0.1', '0.3', '1'])
    ax.set_yscale('symlog', linthresh=.1)
    ax.set_ylim(-.025, 350)
    ax.set_yticks([0, .1, 1, 10, 100], ['0', '0.1', '1', '10', '100'])
    decorate(ax)
axes[0].set_ylabel('BIC loss')
handles = [Line2D([0], [0], linestyle='', marker='o', markersize=4,
                  color=c, label=s[1]) for s, c in zip(SETTINGS, COLORS)]
fig.legend(handles=handles, ncol=7, loc='lower center', bbox_to_anchor=(.52, -.015), frameon=False,
           columnspacing=1.8, handletextpad=.35)
fig.tight_layout(rect=(0, .08, 1, 1), w_pad=1.5)
save(fig, 'bic_cost_tradeoff')
data['figures']['bic_cost_tradeoff'] = {
    'points': 350, 'exhaustive': 206, 'challengers': 144,
    'interpretation': 'Observed cost-loss pairs for one fixed screening allowance; '
                      'no optimized Pareto frontier is estimated. Lower is preferable on both axes.',
}

# Include the complete atom at zero and every large-loss observation. ECDFs are
# conditional only on the seven displayed settings, not on successful outcomes.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharex=True, sharey=True)
ecdf_stats = []
for i, ((scenario, code, k), color) in enumerate(zip(SETTINGS, COLORS)):
    values = np.sort([gap(r) for r in groups[scenario]])
    unique, counts = np.unique(values, return_counts=True)
    x = np.r_[-.02, unique, 300]
    y = np.r_[0, np.cumsum(counts)/len(values), 1]
    axis = axes[0 if k == 3 else 1]
    axis.step(x, y, where='post', lw=1.7, color=color,
              linestyle=('-' if i % 2 == 0 else '--'),
              label=f'{code} ({sum(values == 0)}/50 at zero)')
    axis.scatter([0], [sum(values == 0)/50], marker='o', s=12, color=color, zorder=4)
    ecdf_stats.append({'setting': code, 'n': 50, 'agreement': int(sum(values == 0)),
                       'maximum': float(values.max()),
                       'unique_x': unique.tolist(), 'cdf_y': (np.cumsum(counts)/len(values)).tolist()})
for axis, title in zip(axes, ('(a) Three components', '(b) Four and five components')):
    axis.set_title(title)
    axis.set_xscale('symlog', linthresh=.1)
    axis.set_xlim(-.02, 300)
    axis.set_ylim(0, 1.035)
    axis.set_xticks([0, .1, 1, 10, 100], ['0', '0.1', '1', '10', '100'])
    axis.set_yticks([0, .25, .5, .75, 1])
    axis.set_xlabel('BIC loss')
    axis.legend(loc='lower right', frameon=False, handlelength=2.6)
    decorate(axis)
axes[0].set_ylabel('Empirical cumulative probability')
fig.tight_layout(w_pad=1.5)
save(fig, 'bic_loss_ecdf')
data['figures']['bic_loss_ecdf'] = ecdf_stats

# Accounted work grouped by K: n, p, families and comparison protocol differ,
# hence the display does not estimate a scaling law in K.
fig, ax = plt.subplots(figsize=(7, 3.45))
cost_stats = []
for i, (scenario, code, k) in enumerate(SETTINGS):
    rr = groups[scenario]
    current = {'setting': code, 'k': k, 'n': rr[0]['task']['n'], 'p': rr[0]['task']['p']}
    for scope, offset, color, marker in (
        ('screened', -.2, '#4477AA', 'o'),
        ('exhaustive', 0, '#CC6677', 's'),
        ('challengers', .2, '#228833', '^'),
    ):
        chosen = rr if scope == 'screened' else [r for r in rr if r['reference'] == scope]
        field = 'screen_shared_accounted_seconds' if scope == 'screened' else 'reference_shared_accounted_seconds'
        values = np.asarray([r[field]/3600 for r in chosen])
        if not len(values):
            continue
        q25, median, q75 = np.quantile(values, [.25, .5, .75])
        current[scope] = {'count': len(values), 'q25_hours': float(q25),
                          'median_hours': float(median), 'q75_hours': float(q75)}
        if len(values) > 2:
            ax.errorbar(i+offset, median, yerr=[[median-q25], [q75-median]], fmt=marker,
                        ms=4, capsize=2, color=color, lw=1.1)
        else:
            ax.scatter([i+offset-.025, i+offset+.025], values,
                       marker=marker, s=17, facecolors='none', edgecolors=color, linewidths=1)
    cost_stats.append(current)
ax.set_yscale('log')
ax.set_ylabel('Accounted work (hours)')
ax.set_xticks(range(7), [code for _, code, _ in SETTINGS])
ax.set_xlim(-.55, 6.55)
ax.set_xlabel('Simulation setting')
for boundary in (3.5, 5.5):
    ax.axvline(boundary, color='0.7', lw=.6, ls=':')
for x, text in ((1.5, '$K=3$'), (4.5, '$K=4$'), (6, '$K=5$')):
    ax.text(x, 1.025, text, transform=ax.get_xaxis_transform(), ha='center', fontsize=10)
handles = [Line2D([0], [0], marker=m, color=c, lw=1, label=t, markersize=4)
           for c, m, t in [('#4477AA', 'o', 'Screened'), ('#CC6677', 's', 'Full comparison'),
                           ('#228833', '^', 'Partial comparison')]]
ax.legend(handles=handles, frameon=False, loc='upper left', ncol=1)
decorate(ax)
fig.tight_layout()
save(fig, 'accounted_cost_by_k')
data['figures']['accounted_cost_by_k'] = {
    'groups': cost_stats,
    'interpretation': 'Medians and interquartile ranges; hollow squares show each '
                      'of the two exhaustive audits in S4, H4 and H5. '
                      'Different designs prevent a causal scaling inference in K.',
}

# Score-deficit illustrations use genuine saved tables, in the manuscript's
# family order. The positive-loss case is explicitly selected as a failure
# illustration and must not be called a representative replication.
table_specs = [
    ('sim_ordinary_heterogeneous_main_000', '(a) H3: replication 000', 'Replication 000'),
    ('sim_five_mixed_main_036', '(b) H5: replication 036', 'Replication 036'),
]
heatmaps = []
for task_id, title, label in table_specs:
    path = ROOT / 'results/score_tables' / f'{task_id}.json.gz'
    table = read_json('score_tables/'+task_id+'.json.gz')
    row = next(r for r in main if r['task']['task_id'] == task_id)
    assert table['task'] == row['task'] and table['status'] == 'qualified'
    assert table['identity']['task_index'] == row['task_index']
    assert all(table['identity'].get(key) == value for key, value in raw['identity'].items())
    assert set(table['families']) == set(FAMILIES)
    scores = np.asarray(table['scores'])[:, [table['families'].index(f) for f in FAMILIES]]
    assert scores.shape == (row['task']['fit_k'], 13) and np.isfinite(scores).all()
    # Frozen local_pilot.py stores the full negative-BIC component contribution
    # 2*weighted_loglik - dimension*log(n); ranked_catalogue uses 2*log(n).
    allowance = 2*np.log(row['task']['n'])
    raw_deficits = scores.max(axis=1, keepdims=True)-scores
    deficits = raw_deficits/allowance
    # Independent assignment-solver audit against every saved catalogue entry.
    # Repeated family columns are legitimate assignment slots. Preserve the
    # documented all-Gaussian fallback even when its deficit exceeds allowance.
    retained_indices = []
    for catalog_index, candidate in enumerate(table['catalogue']):
        costs = raw_deficits[:, [FAMILIES.index(f) for f in candidate['structure']]]
        ri, ci = linear_sum_assignment(costs)
        optimum = float(costs[ri, ci].sum())
        assert np.isclose(optimum, candidate['deficit'], rtol=1e-12, atol=1e-9)
        saved_assignment_cost = sum(raw_deficits[j, FAMILIES.index(f)]
                                    for j, f in enumerate(candidate['assignment']))
        assert np.isclose(optimum, saved_assignment_cost, rtol=1e-12, atol=1e-9)
        fallback = candidate['structure'] == ['mvn']*row['task']['fit_k']
        retained = bool(optimum <= allowance+1e-10 or fallback)
        assert retained == candidate['retained']
        if retained:
            retained_indices.append(catalog_index)
    assert retained_indices == table['retained_indices']
    assert len(retained_indices) == row['retained_structures']
    heatmaps.append(dict(task_id=task_id, title=title, label=label, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         n=row['task']['n'], gap=gap(row), scores=scores.tolist(), deficits=deficits.tolist(),
                         families=FAMILIES, pilot_labels_preserved=True,
                         allowance=allowance, catalogue_entries_validated=len(table['catalogue']),
                         retained_indices_validated=retained_indices))
assert heatmaps[1]['gap'] == max(gap(r) for r in groups['five_mixed'])
fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8),
                         gridspec_kw={'height_ratios': [3, 5], 'hspace': .73})
norm = SymLogNorm(linthresh=.1, vmin=0, vmax=100, base=10)
for ax, entry in zip(axes, heatmaps):
    values = np.asarray(entry['deficits'])
    im = ax.imshow(values, norm=norm, cmap='YlGnBu', aspect='auto')
    for (row, col), value in np.ndenumerate(values):
        shown = ('0' if value < 5e-7 else f'{value:.2f}' if value < 1 else f'{value:.1f}')
        ax.text(col, row, shown, ha='center', va='center', fontsize=7.4,
                color=('white' if norm(value) > .59 else 'black'))
        if value <= 1 + 1e-10:
            ax.add_patch(Rectangle((col-.49, row-.49), .98, .98, fill=False,
                                   lw=1.1, edgecolor='#A92335'))
    ax.set_xticks(range(13), [f.upper() for f in FAMILIES], rotation=40, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(len(values)), [f'{j+1}' for j in range(len(values))])
    ax.set_ylabel('Initial component')
    ax.set_title(entry['title'], loc='left', pad=7)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.subplots_adjust(left=.08, right=.89, top=.94, bottom=.14)
cax = fig.add_axes([.92, .22, .02, .61])
cbar = fig.colorbar(im, cax=cax, ticks=[0, .1, 1, 10, 100])
cbar.ax.set_yticklabels(['0', '0.1', '1', '10', '100'])
cbar.set_label(r'$(\max_h \widehat S_{jh}-\widehat S_{j\ell})/(2\log n)$', labelpad=7)
save(fig, 'family_score_heatmaps')
data['figures']['family_score_heatmaps'] = {
    'tables': heatmaps,
    'interpretation': 'Unrounded full-scale negative-BIC score deficits are divided by the shared allowance 2*log(n). '
                      'Red outlines mark deficits at most one; this is only individual eligibility, '
                      'whereas joint retention requires the sum over components to be at most one. '
                      'Pilot labels are arbitrary and do not assert correspondence to generating labels.',
}
(ROOT / 'build/tables/manuscript_diagnostics_data.json').write_text(json.dumps(data, indent=2)+'\n')
print(json.dumps({'main_replications': len(main),
                  'agreements': sum(r['qualified_attained_bic_agreement'] for r in main),
                  'scope_counts': dict(collections.Counter(r['reference'] for r in main)),
                  'max_gap': max(gap(r) for r in main),
                  'figures': list(data['figures'])}, indent=2))
