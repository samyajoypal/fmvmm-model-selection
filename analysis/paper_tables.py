"""Generate all 19 paper tables as unrounded CSV and verify key findings.

The family names/parameter formulas and original feature dimensions are design
metadata. All performance columns are computed from saved per-dataset results.
"""
from collections import defaultdict, Counter
import json
import math
import statistics as st
from pathlib import Path
from common import ROOT, TABLES, read_json, write_json, write_csv
summary=read_json('comparisons.json.gz');cases=summary['cases']
base=read_json('baseline.json.gz')['tasks']
comparisons=json.loads((TABLES/'comparison_statistics.json').read_text())
real=json.loads((TABLES/'real_audit.json').read_text())
SETTINGS=[('gaussian_control','G3'),('ordinary_heterogeneous','H3'),('skew_robust','S3'),('gh_heterogeneous','GH3'),('four_skew','S4'),('four_mixed','H4'),('five_mixed','H5')]
main=[c for c in cases if c['task'].get('block')=='main' and c['task']['fit_k']==3]
CUTS=[0,1,2,4,8]
TOL=1e-6
inventory=[]
def export(name,label,rows,source,notes=''):
    write_csv(name,rows);inventory.append(dict(file=name,paper_label=label,rows=len(rows),source=source,notes=notes))
def mean(v):return st.mean(v)
def se(v):return st.stdev(v)/math.sqrt(len(v))
def wilson(k,n):
    z=1.959963984540054;d=1+z*z/n;c=(k/n+z*z/(2*n))/d;w=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/d
    return [100*(c-w),100*(c+w)]
def methods(c,m):
    return [v for k,v in c['methods'].items() if k.startswith(m+'_r')] if m.startswith('random_c') else [c['methods'][m]]
def stats(rows,m):
    eligible=[];gaps=[];changes=[];prob_agree=[];prob_le2=[];counts=[];accounted=[];fitting=[];cost0=[];fit0=[]
    for c in rows:
        mm=methods(c,m);assert mm
        good=[v for v in mm if v.get('best')]
        if good:
            gg=[v['bic_gap_original_reference'] for v in good];ch=[v['bic_change_from_cutoff_2'] for v in good]
            gaps.append(mean(gg));changes.append(mean(ch));prob_agree.append(mean(abs(v)<=TOL for v in gg));prob_le2.append(mean(v<=2 for v in gg));eligible.append(c)
        if all('requested_fits' in v for v in mm):counts.append(mean(v['requested_fits'] for v in mm))
        accounted.append(mean(v['accounted_seconds'] for v in mm));fitting.append(mean(v['fitting_seconds'] for v in mm))
        cost0.append(c['methods']['cutoff_2']['accounted_seconds']);fit0.append(c['methods']['cutoff_2']['fitting_seconds'])
    return dict(method=m,datasets=len(rows),datasets_with_accepted_output=len(eligible),mean_bic_difference=mean(gaps),mean_bic_change_from_cutoff2=mean(changes),agreement_dataset_equivalents=sum(prob_agree),agreement_percent=100*mean(prob_agree),bic_difference_at_most_2_percent=100*mean(prob_le2),bic_difference_over_2_percent=100*(1-mean(prob_le2)),mean_requested_fits=mean(counts) if counts else None,accounted_work_ratio=sum(accounted)/sum(cost0),fitting_work_ratio=sum(fitting)/sum(fit0))
# Main Table 1: analytic design, not estimated outcomes.
FAMILY_NAMES=[('MVN','Multivariate normal','a_p'),('MVT','Multivariate Student t','a_p+1'),('MVSN','Multivariate skew-normal','a_p+p'),('MVST','Multivariate skew-t (SMSN)','a_p+p+1'),('MSNC','Multivariate skew-contaminated normal','a_p+p+2'),('MSL','Multivariate slash','a_p+1'),('MSSL','Multivariate skew-slash','a_p+p+1'),('MVSL','Multivariate skew-Laplace','a_p+p'),('MGHP','Multivariate generalized hyperbolic','a_p+p+2'),('MGST','Multivariate generalized skew-t','a_p+p+1'),('MVHB','Multivariate hyperbolic','a_p+p+1'),('MNIG','Multivariate normal-inverse Gaussian','a_p+p+1'),('MVVG','Multivariate variance-gamma','a_p+p+1')]
export('main_01_family_library.csv','tab:families',[dict(family=a,name=b,component_parameters=c,a_p='p+p*(p+1)/2') for a,b,c in FAMILY_NAMES],'Analytic model definitions; no fitted quantities.')
design=[];coverage=[];selection=[];prediction=[]
for scenario,code in SETTINGS:
    rows=[r for r in base if r['task'].get('scenario')==scenario and r['task'].get('block')=='main'];assert len(rows)==50
    t=rows[0]['task'];count=sum(r['qualified_attained_bic_agreement'] for r in rows)
    design.append(dict(setting=code,scenario=scenario,n=t['n'],p=t['p'],K=t['fit_k'],generating_families=t['truth_families'],replications=len(rows),test_n=t['test_n'],separation=t['separation'],weights=t['weights']))
    scopes=Counter(r['reference'] for r in rows)
    coverage.append(dict(setting=code,K=t['fit_k'],available_combinations=math.comb(13+t['fit_k']-1,t['fit_k']),full_replications=scopes['exhaustive'],partial_replications=scopes['challengers']))
    lo,hi=wilson(count,len(rows)) if scopes['exhaustive']==len(rows) else (None,None)
    selection.append(dict(setting=code,agreement=count,replications=len(rows),wilson_lower_percent=lo,wilson_upper_percent=hi,median_bic_loss=st.median(r['qualified_attained_bic_gap'] for r in rows),maximum_bic_loss=max(r['qualified_attained_bic_gap'] for r in rows),median_retained=st.median(r['retained_structures'] for r in rows),median_reduction_percent=100*st.median(r['candidate_count_reduction'] for r in rows),accounted_work_ratio=sum(r['screen_shared_accounted_seconds'] for r in rows)/sum(r['reference_shared_accounted_seconds'] for r in rows)))
    out=dict(setting=code,replications=len(rows))
    for field,label in [('test_log_density','log_density'),('test_ari','ari'),('test_posterior_tv','tv')]:
        diff=[r['screening_best_qualified']['quality'][field]-r['reference_best_qualified']['quality'][field] for r in rows]
        out['mean_'+label+'_difference']=mean(diff);out['mcse_'+label+'_difference']=se(diff)
    out['mean_log_density_difference_from_initial']=mean(r['screening_best_qualified']['quality']['test_log_density']-r['best_qualified_pilot']['quality']['test_log_density'] for r in rows)
    prediction.append(out)
export('main_02_simulation_design.csv','tab:simulation',design,'baseline tasks; static generating design')
export('main_03_comparison_coverage.csv','tab:comparison-coverage',coverage,'baseline tasks and search types')
export('main_04_selection.csv','tab:selection',selection,'baseline per-replication outcomes','Wilson intervals only for homogeneous full-search protocols; no pooled interval for mixed full/partial groups.')
MAIN_METHODS=['cutoff_2','top_r_2','top_r_3','top_m_10','top_m_25','random_c2','beam_w3_bmatched_cutoff_2','beam_w5_bmatched_cutoff_2','beam_w3_b50','greedy','fab']
export('main_05_method_comparison.csv','tab:method-comparison',[stats(main,m) for m in MAIN_METHODS],'comparison per-dataset outcomes','Random repeats are averaged within each dataset. Empty requested-fit counts for iterative methods are intentional.')
export('main_06_cutoffs.csv','tab:cutoff-comparison',[dict(cutoff_multiplier=a,**stats(main,f'cutoff_{a}')) for a in CUTS],'comparison per-dataset outcomes')
export('main_07_prediction.csv','tab:prediction',prediction,'baseline paired selected-model quality','MCSE is sample SD of paired differences / sqrt(50).')
export('main_08_real_design.csv','tab:real-design',[dict(dataset=d,observations=n,original_dimension=p,fitted_dimension=k,test_size=test) for d,n,p,k,test in [('faithful',272,2,2,55),('seeds',210,7,3,42),('wdbc',569,30,3,114)]],'Public raw dataset dimensions and fixed preprocessing protocol; static design')
# Baseline selection over K: all costs and fit counts are summed across K.
realbase=defaultdict(list)
for r in base:
    t=r['task']
    if t['kind']=='real_data':realbase[(t['dataset_id'],t['split'],t['repeat'])].append(r)
full=[];holdout=[]
for (dataset,split,repeat),rows in sorted(realbase.items()):
    s=min(rows,key=lambda r:r['screening_best_qualified']['bic']);c=min(rows,key=lambda r:r['reference_best_qualified']['bic'])
    bs=s['screening_best_qualified'];bc=c['reference_best_qualified'];nret=sum(r['retained_structures'] for r in rows)
    row=dict(dataset=dataset,split=split,repeat=repeat,selected_K=s['task']['fit_k'],comparison_K=c['task']['fit_k'],families=bs['structure'],bic=bs['bic'],bic_loss=bs['bic']-bc['bic'],retained=nret,available_combinations=559,reduction_percent=100*(1-nret/559),accounted_work_ratio=sum(r['screen_shared_accounted_seconds'] for r in rows)/sum(r['reference_shared_accounted_seconds'] for r in rows))
    if split=='full':full.append(row)
    else:
        row['test_log_density_difference']=bs['quality']['test_log_density']-bc['quality']['test_log_density'];row['screened_test_ari']=bs['quality'].get('test_ari');row['comparison_test_ari']=bc['quality'].get('test_ari');holdout.append(row)
export('main_09_real_full.csv','tab:real-full',full,'baseline real-data selection across K=1,2,3')
export('main_10_real_holdout.csv','tab:real-holdout',holdout,'baseline real-data selection across K=1,2,3')
REAL_METHODS=['cutoff_2','cutoff_4','cutoff_8','top_r_3','top_m_10','top_m_25','random_c2','beam_w3_bmatched_cutoff_2','beam_w3_b50','greedy','fab']
realrows=[]
for m in REAL_METHODS:
    v=real['method_summary'][m]
    realrows.append(dict(method=m,mean_bic_change=v['mean_bic_change'] if m!='fab' else None,largest_bic_change=v['max_bic_change'] if m!='fab' else None,accounted_work_ratio=v['aggregate_accounted_ratio'],fitting_work_ratio=v['aggregate_fitting_ratio'],all_K_available=v['fully_available_joint_searches'],analyses=12))
export('main_11_real_comparison.csv','tab:real-comparisons',realrows,'recomputed real-data selections','FAB BIC aggregates deliberately absent because 9/12 searches lack converged K=3 outputs; costs include those runs.')
# Supplement numerical tables, ordered as in manuscript.
groups=[('main_K3',main),('overlap',[c for c in cases if c['task'].get('block')=='overlap']),('unequal',[c for c in cases if c['task'].get('block')=='unequal']),('real_data',[c for c in cases if c['task']['kind']=='real_data'])]
caps=[]
for name,rows in groups:
    caps.append(dict(experiment=name,runs_per_method=len(rows),greedy_limit=sum(c['methods']['greedy']['status']=='round_cap' for c in rows),fab_limit=sum(c['methods']['fab']['status']=='iteration_cap' for c in rows),greedy_status_counts=dict(Counter(c['methods']['greedy']['status'] for c in rows)),fab_status_counts=dict(Counter(c['methods']['fab']['status'] for c in rows))))
export('supp_01_iteration_limits.csv','supp:tab:caps',caps,'comparison status fields')
FIXED=[f'cutoff_{a}' for a in CUTS]+[f'top_r_{r}' for r in [1,2,3,4]]+[f'top_m_{m}' for m in [5,10,25,50,100]]
SEARCH=[f'random_c{a}' for a in CUTS]+[f'beam_w{w}_b{b}' for w in [3,5] for b in ['matched_cutoff_2',25,50]]+['greedy','fab']
export('supp_02_fixed_budgets.csv','supp:tab:fixed-budgets',[stats(main,m) for m in FIXED],'comparison per-dataset outcomes')
export('supp_03_search_budgets.csv','supp:tab:search-budgets',[stats(main,m) for m in SEARCH],'comparison per-dataset outcomes')
scenario=[]
for sc,code in SETTINGS[:4]+[('overlap','H3 overlap'),('unequal','H3 unequal')]:
    rows=[c for c in cases if (c['task'].get('scenario')==sc and c['task'].get('block')=='main') or c['task'].get('block')==sc]
    for m in ['cutoff_2','top_m_10','top_m_25','top_r_3','random_c2','beam_w3_bmatched_cutoff_2','greedy','fab']:scenario.append(dict(setting=code,**stats(rows,m)))
export('supp_04_setting_comparisons.csv','supp:tab:scenario-methods',scenario,'comparison per-dataset outcomes','H3 overlap FAB mean conditional on 8 converged outputs; 2 caps retained separately.')
large=[]
for sc,code in SETTINGS[4:]:
    rows=[c for c in cases if c['task'].get('scenario')==sc and c['task'].get('block')=='main']
    for a in [0,1,4,8]:large.append(dict(setting=code,cutoff_multiplier=a,improved=sum(c['methods'][f'cutoff_{a}']['bic_change_from_cutoff_2'] < -TOL for c in rows),**stats(rows,f'cutoff_{a}')))
export('supp_05_large_cutoffs.csv','supp:tab:large-cutoffs',large,'comparison per-dataset outcomes','Changes relative to cutoff 2, not claims about an unobserved exhaustive optimum.')
export('supp_06_real_comparisons.csv','supp:tab:real-comparisons',[r for r in realrows if r['method'] not in ['fab','beam_w3_b50']],'recomputed joint-K real-data outcomes')
fab=[];percase=[]
for g in real['joint_results']:
    f=g['methods']['fab'];b=g['methods']['cutoff_2']
    fab.append(dict(dataset=g['dataset'],split=g['split'],repeat=g['repeat'],available_K=f['available_k'],selected_K=list(f['selected_k_counts'])[0],test_log_density_change=None if f['mean_test_log_density'] is None else f['mean_test_log_density']-b['mean_test_log_density'],test_ari_change=None if f['mean_test_ari'] is None else f['mean_test_ari']-b['mean_test_ari']))
    for m in ['cutoff_2','cutoff_8','top_m_10','greedy','random_c2','beam_w3_bmatched_cutoff_2']:
        percase.append(dict(dataset=g['dataset'],split=g['split'],repeat=g['repeat'],method=m,bic_difference=g['methods'][m]['bic_gap_original_reference']))
export('supp_07_fab_real_diagnostics.csv','supp:tab:fab-real-diagnostics',fab,'recomputed joint-K real-data outcomes','FAB maximizes its own objective over available converged K; incomplete searches are explicitly retained.')
export('supp_08_real_case_comparisons.csv','supp:tab:real-case-methods',percase,'recomputed joint-K real-data outcomes')
# Independent numerical checks against published totals; no summary-of-summary
# inputs enter these checks.
checks={}
def check(name,condition,value):
    assert condition,(name,value);checks[name]=dict(passed=True,value=value)
check('all_cases_complete',len(cases)==406 and summary['complete'] and not summary['missing_indices'],len(cases))
check('all_method_tasks_complete',all(m['complete'] for c in cases for m in c['methods'].values()),True)
accepted_records=0
for c in cases:
    for name,m in c['methods'].items():
        if not m.get('best'):continue
        assert math.isclose(m['bic_gap_original_reference'],m['best']['bic']-c['original_reference']['bic'],abs_tol=1e-8),(c['index'],name,'gap')
        assert math.isclose(m['bic_change_from_cutoff_2'],m['best']['bic']-c['methods']['cutoff_2']['best']['bic'],abs_tol=1e-8),(c['index'],name,'change')
        accepted_records+=1
check('record_level_BIC_arithmetic',accepted_records>0,accepted_records)
check('main_baseline_agreement',sum(r['agreement'] for r in selection)==312,sum(r['agreement'] for r in selection))
expected=[125,169,187,199,200]
for a,num in zip(CUTS,expected):
    actual=sum(abs(c['methods'][f'cutoff_{a}']['bic_gap_original_reference'])<=TOL for c in main)
    check(f'cutoff_{a}_main_agreement',actual==num,actual)
check('top25_main_agreement',sum(abs(c['methods']['top_m_25']['bic_gap_original_reference'])<=TOL for c in main)==200,200)
check('greedy_iteration_limits',sum(r['greedy_limit'] for r in caps)==14,sum(r['greedy_limit'] for r in caps))
check('fab_iteration_limits',sum(r['fab_limit'] for r in caps)==11,sum(r['fab_limit'] for r in caps))
check('real_retained_total',sum(r['retained'] for r in full+holdout)==655,sum(r['retained'] for r in full+holdout))
check('fab_complete_joint_K',real['method_summary']['fab']['fully_available_joint_searches']==3,real['method_summary']['fab']['fully_available_joint_searches'])
check('H5_maximum_loss',abs(max(r['maximum_bic_loss'] for r in selection)-221.2838993963087)<1e-6,max(r['maximum_bic_loss'] for r in selection))
status=read_json('additional_fit_audit.json.gz')
check('new_fit_outcomes',sum(status['status_counts'].values())==167695,status['status_counts'])
# Raw method outputs are retained without clipping negative BIC differences.
write_csv('case_method_outcomes.csv',[dict(task_id=c['task']['task_id'],case_index=c['index'],kind=c['task']['kind'],setting=c['task'].get('scenario',c['task'].get('dataset_id')),block=c['task'].get('block',c['task'].get('split')),K=c['task']['fit_k'],reference_scope=c['reference_scope'],method=name,status=m.get('status','fixed_search_complete'),has_accepted_output=bool(m.get('best')),bic_difference=m.get('bic_gap_original_reference'),bic_change_from_cutoff2=m.get('bic_change_from_cutoff_2'),accounted_seconds=m['accounted_seconds'],fitting_seconds=m['fitting_seconds'],requested_fits=m.get('requested_fits'),status_counts=m.get('status_counts')) for c in cases for name,m in c['methods'].items()])
write_json('paper_table_inventory.json',inventory);write_json('verification.json',checks)
print(f'Generated {len(inventory)} paper tables and independently verified {len(checks)} numerical checks.')
