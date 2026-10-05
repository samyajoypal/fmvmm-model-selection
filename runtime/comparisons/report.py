"""Paired summaries; random-search repeats are not independent datasets."""
import math
import re
import statistics
from collections import defaultdict


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def stats(values):
    values=[float(x) for x in values if finite(x)]
    if not values:return dict(n=0)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values),
                mcse=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None)


def real_joint(cases):
    groups=defaultdict(list)
    for case in cases:
        task=case['task']
        if task['kind']=='real_data':
            groups[(task['dataset_id'],task['split'],task['repeat'])].append(case)
    output=[]
    for (dataset,split,repeat),rows in sorted(groups.items()):
        names=sorted(set(key for row in rows for key in row.get('methods',{})))
        combined={}
        for name in names:
            by_k=[(row['task']['fit_k'],row.get('methods',{}).get(name,{})) for row in rows]
            options=[(k,m) for k,m in by_k if m.get('best')]
            if name=='fab':
                options=[(k,m) for k,m in options if finite(m.get('native_objective'))]
                chosen=min(options,key=lambda km:(-km[1]['native_objective'],km[0])) if options else None
            else:
                chosen=min(options,key=lambda km:(km[1]['best']['bic'],km[0])) if options else None
            combined[name]=dict(selected_k=chosen[0] if chosen else None,
                best=chosen[1]['best'] if chosen else None,
                native_objective=chosen[1].get('native_objective') if chosen else None,
                all_k_available=len(by_k)==3 and {k for k,m in by_k}=={1,2,3} and
                    all(m.get('complete') and m.get('best') for k,m in by_k),
                accounted_seconds=sum(m.get('accounted_seconds',0) for k,m in by_k),
                fitting_seconds=sum(m.get('fitting_seconds',0) for k,m in by_k),
                selection_criterion='FAB lower bound' if name=='fab' else 'BIC')
        output.append(dict(dataset=dataset,split=split,repeat=repeat,methods=combined))
    return output


def build_report(cases):
    grouped=defaultdict(list)
    for case in cases:
        task=case['task']
        if task['kind']!='simulation':continue
        methods=case.get('methods',{})
        collapsed=defaultdict(list)
        for name,m in methods.items():
            label=re.sub(r'_r\d+$','',name) if name.startswith('random_') else name
            collapsed[label].append(m)
        base=methods.get('cutoff_2',{})
        for name,runs in collapsed.items():
            eligible=[m for m in runs if m.get('complete') and m.get('best')]
            attempted=[m for m in runs if m.get('complete')]
            vals=lambda key:[m.get(key) for m in eligible if finite(m.get(key))]
            mean=lambda v:statistics.mean(v) if v else None
            gap=vals('bic_gap_original_reference')
            common=vals('bic_gap_common_best')
            paired=vals('bic_change_from_cutoff_2')
            item=dict(task_id=task['task_id'],method=name,search_repeats=len(runs),
                eligible_search_repeats=len(eligible),
                no_qualified_winner_rate=sum(not m.get('best') for m in runs)/len(runs),
                incomplete_search_rate=sum(not m.get('complete') for m in runs)/len(runs),
                mean_bic_gap_original_reference=mean(gap),
                mean_bic_gap_common_best=mean(common),
                mean_bic_change_from_cutoff_2=mean(paired),
                agreement_rate=mean([abs(v)<=1e-6 for v in gap]),
                loss_over_2_rate=mean([v>2 for v in gap]),
                loss_over_10_rate=mean([v>10 for v in gap]),
                maximum_observed_bic_gap=max(gap) if gap else None,
                mean_test_log_density_change=mean(vals('test_log_density_change_from_cutoff_2')),
                mean_accounted_seconds=mean([m['accounted_seconds'] for m in attempted if finite(m.get('accounted_seconds'))]),
                baseline_accounted_seconds=base.get('accounted_seconds'),
                all_repeats_complete=all(m.get('complete') for m in runs))
            grouped[(task['scenario'],task['block'],task['fit_k'],case.get('reference_scope'),name)].append(item)
    summaries=[]
    for (scenario,block,k,scope,method),rows in sorted(grouped.items()):
        summary=dict(scenario=scenario,block=block,K=k,reference_scope=scope,method=method,
                     scheduled_datasets=len(rows),
                     datasets_with_selected_model=sum(row['eligible_search_repeats']>0 for row in rows),
                     datasets_with_all_searches_complete=sum(row['all_repeats_complete'] for row in rows))
        for field in ('mean_bic_gap_original_reference','mean_bic_gap_common_best','mean_bic_change_from_cutoff_2',
                      'agreement_rate','loss_over_2_rate','loss_over_10_rate','mean_test_log_density_change',
                      'no_qualified_winner_rate','incomplete_search_rate'):
            summary[field]=stats([row[field] for row in rows])
        paired=[row for row in rows if finite(row['mean_accounted_seconds']) and finite(row['baseline_accounted_seconds'])]
        denom=sum(row['baseline_accounted_seconds'] for row in paired)
        summary['accounted_time_ratio_to_cutoff_2']=sum(row['mean_accounted_seconds'] for row in paired)/denom if denom else None
        gaps=[row['maximum_observed_bic_gap'] for row in rows if finite(row['maximum_observed_bic_gap'])]
        summary['largest_observed_bic_gap']=max(gaps) if gaps else None
        summaries.append(summary)
    return dict(simulation=summaries,real_data_joint_K=real_joint(cases),
                interpretation='Random repeats averaged within each dataset before means/MCSE. MCSE describes independent simulation datasets, not independent search repetitions. BIC/agreement/loss summaries are conditional on a qualified selected model; no-qualified-winner and incomplete rates retain every scheduled search. Timing includes every completed attempt, even a capped or unresolved method without a selected model. Real-data splits overlap; no inferential intervals assigned. Signed BIC gaps preserved; original numerical full/partial search is not a global optimum.',
                simulation_dataset_details={str(key):value for key,value in grouped.items()})
