#!/usr/bin/env python3
"""Reproduce manuscript comparison figures and table rows, without refitting.

Run with scripts/reproduce.py from a portable repository checkout.
Every random-search statistic is first averaged over 30 repetitions within a
single simulated dataset. Fixed-scenario Monte Carlo standard errors use
within-scenario variation. Signed BIC differences are never truncated at zero.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
import statistics as st

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fmvmm-paper-comparisons-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results'
OUT = ROOT / 'build/figures'
TABLES = ROOT / 'build/tables/comparison_statistics.json'
from common import read_json
OUT.mkdir(parents=True, exist_ok=True)
TABLES.parent.mkdir(parents=True, exist_ok=True)
CUTS = [0, 1, 2, 4, 8]
SETTINGS = [
    ('gaussian_control', 'G3'), ('ordinary_heterogeneous', 'H3'),
    ('skew_robust', 'S3'), ('gh_heterogeneous', 'GH3'),
    ('four_skew', 'S4'), ('four_mixed', 'H4'), ('five_mixed', 'H5'),
]
METHODS = [
    ('cutoff_2', r'Screening ($2\log n$)'),
    ('top_r_2', 'Top-2 per component'),
    ('top_r_3', 'Top-3 per component'),
    ('top_m_10', 'Top-10 combinations'),
    ('top_m_25', 'Top-25 combinations'),
    ('random_c2', 'Random, matched budget'),
    ('beam_w3_bmatched_cutoff_2', 'Beam 3, matched budget'),
    ('beam_w5_bmatched_cutoff_2', 'Beam 5, matched budget'),
    ('beam_w3_b50', 'Beam 3, 50 fits'),
    ('greedy', 'Greedy updating'), ('fab', 'Constrained FAB'),
]
ALL_METHODS = [f'cutoff_{a}' for a in CUTS] + [f'top_r_{r}' for r in (1,2,3,4)] + [f'top_m_{m}' for m in (5,10,25,50,100)] + ['random_c2'] + [f'beam_w{w}_b{b}' for w in (3,5) for b in ('matched_cutoff_2','25','50')] + ['greedy','fab']
TOL = 1e-6
summary = read_json('comparisons.json.gz')
protocol = read_json('protocol.json.gz')
old = read_json('baseline.json.gz')
real_audit = json.loads((ROOT/'build/tables/real_audit.json').read_text())
prior_audit = read_json('comparison_audit_expected.json.gz')
cases = summary['cases']
assert summary['complete'] and len(cases) == 406 and not summary['missing_indices']
assert summary['protocol_sha256'] == read_json('manifest.json')['files']['protocol.json.gz']['original_file_sha256']
for c in cases:
    assert c['status'] == 'complete' and c['task'] == protocol['cases'][c['index']]['task']
main = [c for c in cases if c['task'].get('block') == 'main' and c['task']['fit_k'] == 3]
assert len(main) == 200 and all(c['reference_scope'] == 'exhaustive' for c in main)
assert Counter(c['task']['scenario'] for c in main) == {s:50 for s,_ in SETTINGS[:4]}


def runs(case, method):
    if method == 'random_c2':
        selected = [m for name,m in case['methods'].items() if name.startswith('random_c2_r')]
        assert len(selected) == 30
        return selected
    return [case['methods'][method]]


def mcse(values, strata):
    """Standard error of an average over fixed simulation settings."""
    grouped = defaultdict(list)
    for x,s in zip(values,strata):
        grouped[s].append(x)
    n = len(values)
    if not n or any(len(v)<2 for v in grouped.values()):
        return None
    return math.sqrt(sum((len(v)/n)**2 * st.variance(v)/len(v) for v in grouped.values()))


def describe(rows, method):
    fields = defaultdict(list)
    eligible_strata = []
    statuses = Counter()
    eligibility = []
    for c in rows:
        selected = runs(c,method)
        eligible = [m for m in selected if m.get('best')]
        eligibility.append(len(eligible)/len(selected))
        for m in selected:
            assert m['complete']
            statuses[m.get('status','fixed_search_complete')] += 1
        for key in ('accounted_seconds','fitting_seconds','requested_fits','incremental_fit_attempts','incremental_profile_attempts'):
            if all(key in m for m in selected):
                fields[key].append(st.mean(m[key] for m in selected))
        if not eligible:
            continue
        gaps = [m['bic_gap_original_reference'] for m in eligible]
        changes = [m['bic_change_from_cutoff_2'] for m in eligible]
        fields['bic_difference'].append(st.mean(gaps))
        fields['bic_change_from_cutoff_2'].append(st.mean(changes))
        fields['probability_difference_over_2'].append(st.mean(g>2 for g in gaps))
        fields['probability_exact_agreement'].append(st.mean(abs(g)<=TOL for g in gaps))
        fields['probability_no_worse_than_reference'].append(st.mean(g<=TOL for g in gaps))
        fields['minimum_individual_difference'].append(min(gaps))
        fields['maximum_individual_difference'].append(max(gaps))
        for name in ('test_log_density','test_ari','test_posterior_tv'):
            base_value=c['methods']['cutoff_2']['best'].get('quality',{}).get(name)
            good=[m['best'].get('quality',{}).get(name) for m in eligible]
            if base_value is not None and all(v is not None for v in good):
                fields[name+'_change'].append(st.mean(good)-base_value)
        eligible_strata.append((c['task'].get('scenario'),c['task'].get('block')))
    result = {
        'datasets':len(rows), 'eligible_dataset_equivalents':sum(eligibility),
        'statuses':dict(statuses),
        'means':{k:st.mean(v) for k,v in fields.items()},
        'accounted_ratio_to_cutoff2':sum(fields['accounted_seconds'])/sum(c['methods']['cutoff_2']['accounted_seconds'] for c in rows),
        'fitting_ratio_to_cutoff2':sum(fields['fitting_seconds'])/sum(c['methods']['cutoff_2']['fitting_seconds'] for c in rows),
    }
    result['monte_carlo_se'] = {k:mcse(fields[k],eligible_strata) for k in ('bic_difference','bic_change_from_cutoff_2','probability_difference_over_2')}
    result['minimum_difference'] = min(fields['minimum_individual_difference'])
    result['maximum_difference'] = max(fields['maximum_individual_difference'])
    return result

main_stats={m:describe(main,m) for m in ALL_METHODS}
for m,s in main_stats.items():
    a=prior_audit['main_K3'][m]
    assert math.isclose(s['means']['bic_difference'],a['bic_gap']['mean'],abs_tol=1e-10)
    assert math.isclose(s['accounted_ratio_to_cutoff2'],a['accounted_ratio_to_cutoff_2'],rel_tol=1e-12)
reference_cost=sum(next(r for r in old['tasks'] if r['task']['task_id']==c['task']['task_id'])['reference_shared_accounted_seconds'] for c in main)
for s in main_stats.values():
    s['accounted_fraction_of_full_comparison'] = s['means']['accounted_seconds']*200/reference_cost

by_setting={}
for scenario,code in SETTINGS:
    rows=[c for c in cases if c['task'].get('scenario')==scenario and c['task'].get('block')=='main']
    assert len(rows)==50
    by_setting[code]={'scenario':scenario,'reference_scopes':dict(Counter(c['reference_scope'] for c in rows)),
                      'methods':{m:describe(rows,m) for m in ALL_METHODS if m in rows[0]['methods'] or (m=='random_c2' and 'random_c2_r00' in rows[0]['methods'])}}
stress={}
for block in ('overlap','unequal'):
    rows=[c for c in cases if c['task'].get('block')==block]
    if rows:
        stress[block]={m:describe(rows,m) for m in ALL_METHODS}


def fmt(x,digits=3):
    if x is None:return '--'
    return f'{x:.{digits}f}'


def main_row(method,label):
    s=main_stats[method]; means=s['means']
    return ' & '.join([label,fmt(means['bic_difference']),fmt(100*means['probability_difference_over_2'],1),
                       fmt(means.get('requested_fits'),2),fmt(s['accounted_ratio_to_cutoff2'],2),fmt(s['fitting_ratio_to_cutoff2'],2)])+r' \\'

main_latex=[main_row(m,label) for m,label in METHODS]
cutoff_latex=[]
for a in CUTS:
    s=main_stats[f'cutoff_{a}'];m=s['means']
    label='0' if a==0 else (r'$\log n$' if a==1 else fr'${a}\log n$')
    cutoff_latex.append(' & '.join([label,f"{round(200*m['probability_exact_agreement'])}/200",fmt(m['bic_difference'],4),fmt(100*m['probability_difference_over_2'],1),fmt(m['requested_fits'],2),fmt(s['accounted_ratio_to_cutoff2'],2),fmt(s['fitting_ratio_to_cutoff2'],2)])+r' \\')
scenario_latex=[]
for code,s in by_setting.items():
    for method,label in METHODS:
        if method not in s['methods']:continue
        z=s['methods'][method]
        scenario_latex.append(' & '.join([code,label,fmt(z['means']['bic_difference']),fmt(100*z['means']['probability_difference_over_2'],1),fmt(z['accounted_ratio_to_cutoff2'],2),fmt(z['fitting_ratio_to_cutoff2'],2)])+r' \\')
cutoff_setting_latex=[]
for code,s in by_setting.items():
    for a in CUTS:
        z=s['methods'][f'cutoff_{a}'];m=z['means']
        cutoff_setting_latex.append(' & '.join([code,str(a),fmt(m['bic_change_from_cutoff_2']),fmt(m['requested_fits'],2),fmt(z['accounted_ratio_to_cutoff2'],2),fmt(z['fitting_ratio_to_cutoff2'],2)])+r' \\')

real_stats={}
real_latex=[]
for method,label in METHODS:
    rr=[x['methods'][method] for x in real_audit['joint_results']]
    ss=real_audit['method_summary'][method]
    rs=dict(ss)
    rs['mean_signed_difference_from_full_comparison']=st.mean(r['bic_gap_original_reference'] for r in rr)
    # This is exact agreement per randomized search, averaged within each split.
    # Do not replace it by agreement of the mean random-search BIC.
    matched={split:[] for split in ('full','holdout')}
    for g in real_audit['joint_results']:
        original=g['original_reference']['bic']
        cc=[c for c in cases if c['task'].get('dataset_id')==g['dataset'] and c['task'].get('split')==g['split'] and c['task'].get('repeat')==g['repeat']]
        by_rep=[]
        for rep in range(30) if method=='random_c2' else [None]:
            name=f'random_c2_r{rep:02d}' if rep is not None else method
            mm=[c['methods'][name] for c in cc if c['methods'][name].get('best')]
            if method=='fab':
                # The existing audited joint-K report uses G, never BIC, for FAB.
                selected_bic=g['methods'][method]['mean_bic']
            else:
                selected_bic=min(m['best']['bic'] for m in mm)
            by_rep.append(abs(selected_bic-original)<=TOL)
        matched[g['split']].append(st.mean(by_rep))
    rs['agreement_with_full_comparison_dataset_equivalents']={k:sum(v) for k,v in matched.items()}
    real_stats[method]=rs
    real_latex.append(' & '.join([label,fmt(rs['agreement_with_full_comparison_dataset_equivalents']['full'],2)+'/3',fmt(rs['agreement_with_full_comparison_dataset_equivalents']['holdout'],2)+'/9',fmt(rs['mean_signed_difference_from_full_comparison']),fmt(rs['aggregate_accounted_ratio'],2),fmt(rs['aggregate_fitting_ratio'],2),str(rs['fully_available_joint_searches'])+'/12'])+r' \\')

plt.rcParams.update({
    'font.family':'serif','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
    'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':220,
})
COLORS=['#4477AA','#228833','#CCBB44','#EE6677','#AA3377','#66CCEE','#332288']


def grid(ax,axis='both'):
    ax.set_axisbelow(True);ax.grid(axis=axis,color='.90',lw=.45)


def save(fig,name):
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'comparison_{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

# Directly aligned rows make unlike search costs and objective differences easy
# to compare without pretending there is a universal Pareto frontier.
fig,axes=plt.subplots(1,2,figsize=(7.25,4.3),sharey=True,gridspec_kw={'width_ratios':[1,1.1],'wspace':.2})
y=np.arange(len(METHODS))
axes[0].set_yticks(y,[label for _,label in METHODS])
axes[0].invert_yaxis()
for i,(method,label) in enumerate(METHODS):
    z=main_stats[method];color='#225ea8' if method=='cutoff_2' else '.35'
    mean=z['means']['bic_difference']
    axes[0].scatter(mean,i,color=color,s=22,zorder=3)
    axes[0].annotate(fmt(mean),(mean,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=7.5,color=color)
    axes[1].scatter(z['accounted_ratio_to_cutoff2'],i,color=color,s=22,zorder=3)
    axes[1].scatter(z['fitting_ratio_to_cutoff2'],i,edgecolors=color,facecolors='white',marker='s',s=21,zorder=4)
axes[0].set_xscale('symlog',linthresh=.025)
axes[0].set_xlim(-.005,145)
axes[0].set_xticks([0,.1,1,10,100],['0','0.1','1','10','100'])
axes[0].set_xlabel('Mean signed BIC difference\n(nonlinear scale)')
axes[0].set_title('(a) Difference from full comparison')
axes[1].axvline(1,color='.7',lw=.8,ls='--')
axes[1].set_xlim(.75,3.3)
axes[1].set_xticks([1,1.5,2,2.5,3])
axes[1].set_xlabel(r'Work relative to screening at $2\log n$')
axes[1].set_title('(b) Computational work')
for ax in axes:grid(ax,'x')
legend=[Line2D([0],[0],marker='o',linestyle='',color='.35',markersize=4,label='Accounted work'),Line2D([0],[0],marker='s',linestyle='',markeredgecolor='.35',markerfacecolor='white',markersize=4,label='Fitting work')]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.73,.015),ncol=2,frameon=False,columnspacing=1.0)
fig.subplots_adjust(left=.30,right=.985,bottom=.20,top=.93)
save(fig,'methods')

# An ECDF is a pointwise average over equally weighted datasets. Thirty random
# repetitions give each dataset total weight one; no independent-SE claim.
ecdf_methods=[('cutoff_2',r'Screening, $2\log n$','#225ea8','-',2),('top_m_10','Top-10','#228833','--',1.2),('top_m_25','Top-25','#555555',':',1.8),('top_r_3','Top-3 per component','#CCBB44','-',1.2),('random_c2','Random, matched','#EE6677','--',1.4),('beam_w3_bmatched_cutoff_2','Beam 3, matched','#AA3377',':',1.6),('greedy','Greedy','#66AACC','-.',1.3),('fab','Constrained FAB','#332288','-',1.2)]
fig,ax=plt.subplots(figsize=(7.1,3.7))
for method,label,color,style,width in ecdf_methods:
    values=[]
    for c in main:values.extend(m['bic_gap_original_reference'] for m in runs(c,method) if m.get('best'))
    assert len(values)==(6000 if method=='random_c2' else 200)
    v=np.sort(values)
    ax.step(np.r_[-10,v,500],np.r_[0,np.arange(1,len(v)+1)/len(v),1],where='post',label=label,color=color,ls=style,lw=width)
ax.axvline(0,color='.7',lw=.6,zorder=0)
ax.set_xscale('symlog',linthresh=1)
ax.set_xlim(-10,500);ax.set_ylim(0,1.025)
ax.set_xticks([-10,-1,0,1,10,100],['-10','-1','0','1','10','100'])
ax.set_xlabel('Signed BIC difference from full comparison (nonlinear scale)')
ax.set_ylabel('Fraction at or below this difference')
grid(ax)
ax.legend(loc='lower right',ncol=2,frameon=True,edgecolor='white',facecolor='white',fontsize=7.6)
fig.tight_layout(pad=.7)
save(fig,'bic_ecdf')

# Sensitivity: full-comparison accuracy is evaluated only on the 200 K=3
# datasets. The second panel reports costs, which remain meaningful with the
# stated partial comparisons at K=4 and K=5.
fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),gridspec_kw={'wspace':.30})
xx=[main_stats[f'cutoff_{a}']['accounted_ratio_to_cutoff2'] for a in CUTS]
yy=[100*main_stats[f'cutoff_{a}']['means']['probability_exact_agreement'] for a in CUTS]
axes[0].plot(xx,yy,'o-',color='#225ea8',lw=1.2,markersize=4)
offsets=[(5,-11),(5,-11),(6,-12),(-20,-15),(-30,-15)]
for a,x,y,offset in zip(CUTS,xx,yy,offsets):
    label='0' if a==0 else (r'$\log n$' if a==1 else fr'${a}\log n$')
    axes[0].annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=8)
for method,label,offset in [('top_m_10','Top-10',(6,4)),('top_m_25','Top-25',(-18,5))]:
    z=main_stats[method];x=z['accounted_ratio_to_cutoff2'];y=100*z['means']['probability_exact_agreement']
    axes[0].scatter(x,y,marker='D',color='#228833',s=20,zorder=4)
    axes[0].annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=7.6,color='#166326')
axes[0].set_xscale('log');axes[0].set_xlim(.6,13);axes[0].set_ylim(55,107)
axes[0].set_xticks([1,2,5,10],['1','2','5','10'])
axes[0].set_yticks([60,70,80,90,100])
axes[0].set_xlabel(r'Accounted work relative to $2\log n$')
axes[0].set_ylabel('Exact BIC agreement (%)')
axes[0].set_title('(a) Three-component settings')
for (scenario,code),color in zip(SETTINGS,COLORS):
    values=[by_setting[code]['methods'][f'cutoff_{a}']['accounted_ratio_to_cutoff2'] for a in CUTS]
    axes[1].plot(range(5),values,color=color,marker='o',markersize=3,lw=1,label=code)
axes[1].set_xticks(range(5),['0',r'$\log n$',r'$2\log n$',r'$4\log n$',r'$8\log n$'])
axes[1].set_yscale('log');axes[1].set_yticks([.1,1,10,100],['0.1','1','10','100']);axes[1].set_ylim(.1,75)
axes[1].set_xlabel('Screening cutoff')
axes[1].set_ylabel(r'Accounted work relative to $2\log n$')
axes[1].set_title('(b) Cost across all seven settings')
axes[1].legend(loc='upper left',ncol=2,frameon=False,fontsize=7,columnspacing=.7,handlelength=1.3)
for ax in axes:grid(ax)
fig.subplots_adjust(left=.08,right=.98,bottom=.20,top=.92)
save(fig,'cutoffs')

out={
    'schema':1,
    'source_sha256':{name: entry['sha256'] for name,entry in read_json('manifest.json')['files'].items()},
    'definitions':{
        'signed_bic_difference':'Selected BIC minus the original full comparison for K=3. Negative differences from adaptive starts are retained.',
        'cost':'Ratios of sums across matched datasets, including shared preparation and failed operations. Accounted work includes saved operation time; fitting work uses measured fitting operations. Neither is an independently measured end-to-end speedup.',
        'random':'Thirty independent searches within each dataset, averaged before dataset-level summaries.',
        'mcse':'Fixed scenario weights with variance estimated within each scenario.',
        'requested_fits':'Unique complete-mixture fits selected after score-table construction; common initial mixtures are additional shared work. Top-M may exceed M because the all-MVN combination is mandatory. Greedy and FAB counts are omitted from the common fixed-search fit-count column.',
        'real':'Twelve joint-K searches: three full datasets and nine held-out splits, each comparing K=1,2,3. FAB selects by its own objective. Its nine capped K=3 fits make nine joint searches incomplete across K; their results are descriptive only.',
    },
    'main_K3':main_stats,
    'by_setting':by_setting,
    'stress':stress,
    'real_joint_K':real_stats,
    'latex_rows':{
        'main_methods_columns':['Method','Mean signed BIC difference','BIC difference > 2 (%)','Mean selected complete fits','Accounted ratio','Fitting ratio'],
        'main_methods':main_latex,
        'cutoffs_columns':['Cutoff','Exact agreement','Mean signed BIC difference','BIC difference > 2 (%)','Mean selected complete fits','Accounted ratio','Fitting ratio'],
        'cutoffs':cutoff_latex,
        'scenario_methods_columns':['Setting','Method','Mean signed BIC difference','BIC difference > 2 (%)','Accounted ratio','Fitting ratio'],
        'scenario_methods':scenario_latex,
        'cutoff_setting_columns':['Setting','Cutoff multiplier','Mean BIC change from cutoff 2','Mean selected complete fits','Accounted ratio','Fitting ratio'],
        'cutoff_setting':cutoff_setting_latex,
        'real_methods_columns':['Method','Full-data agreement with full comparison','Holdout agreement with full comparison','Mean signed BIC difference','Accounted ratio','Fitting ratio','All K available'],
        'real_methods':real_latex,
    },
    'figures':{
        'comparison_methods':'200 main K=3 datasets; aligned mean signed BIC differences and accounted/fitting work ratios. Mean differences can hide individual negative values, shown separately in the ECDF.',
        'comparison_bic_ecdf':'200 main K=3 datasets; all signed values preserved. Random curve averages 30 repetitions within each equally weighted dataset.',
        'comparison_cutoffs':'Panel a: 200 main K=3 full comparisons. Panel b: accounted-work ratios for seven settings, each normalized by its own cutoff-2 cost.',
    },
}
TABLES.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print('Wrote',TABLES)
print('Regenerated three comparison figures and numerical summaries.')
