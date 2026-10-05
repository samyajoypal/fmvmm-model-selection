"""Recompute joint-K real-data selections from per-K numerical outputs.

BIC methods minimize BIC. FAB maximizes native_objective. Capped FAB outputs
are unavailable for selection, but their work remains in all cost totals.
"""
from collections import defaultdict, Counter
import statistics as st
import math
from common import read_json, write_json
summary=read_json('comparisons.json.gz')
real=[c for c in summary['cases'] if c['task']['kind']=='real_data']
groups=defaultdict(list)
for c in real:groups[(c['task']['dataset_id'],c['task']['split'],c['task']['repeat'])].append(c)
output={'cases':len(real),'joint_results':[],'method_summary':{}}
methods=[m for m in real[0]['methods'] if not m.startswith('random')]+[f'random_c{i}' for i in [0,1,2,4,8]]
for key,rows in sorted(groups.items()):
    assert sorted(c['task']['fit_k'] for c in rows)==[1,2,3]
    original_row=min(rows,key=lambda c:c['original_reference']['bic'])
    original=original_row['original_reference']
    record=dict(dataset=key[0],split=key[1],repeat=key[2],original_reference={**original,'selected_k':original_row['task']['fit_k']},methods={})
    baseline=min(c['methods']['cutoff_2']['best']['bic'] for c in rows)
    for mname in methods:
        names=[f'{mname}_r{i:02d}' for i in range(30)] if mname.startswith('random') else [mname]
        selected=[]
        for name in names:
            available=[c for c in rows if c['methods'][name].get('best')]
            if not available:raise ValueError('No accepted joint-K model')
            win=(max(available,key=lambda c:c['methods'][name]['native_objective']) if name=='fab'
                 else min(available,key=lambda c:c['methods'][name]['best']['bic']))
            m=win['methods'][name]
            selected.append(dict(best=m['best'],selected_k=win['task']['fit_k'],
               available_k=sorted(c['task']['fit_k'] for c in available),all_k_available=len(available)==3,
               fitting_seconds=sum(c['methods'][name]['fitting_seconds'] for c in rows),
               accounted_seconds=sum(c['methods'][name]['accounted_seconds'] for c in rows)))
        bic=[v['best']['bic'] for v in selected]
        ld=[v['best']['quality'].get('test_log_density') for v in selected if v['best']['quality'].get('test_log_density') is not None]
        ari=[v['best']['quality'].get('test_ari') for v in selected if v['best']['quality'].get('test_ari') is not None]
        x=dict(search_repeats=len(selected),eligible=len(selected),all_k_available=sum(v['all_k_available'] for v in selected),
           mean_bic=st.mean(bic),bic_change_from_cutoff_2=st.mean(bic)-baseline,bic_gap_original_reference=st.mean(bic)-original['bic'],
           agrees_with_cutoff2=sum(abs(b-baseline)<=1e-6 for b in bic)/len(selected),
           better_than_cutoff2=sum(b<baseline-1e-6 for b in bic)/len(selected),worse_than_cutoff2=sum(b>baseline+1e-6 for b in bic)/len(selected),
           fitting_seconds=st.mean(v['fitting_seconds'] for v in selected),accounted_seconds=st.mean(v['accounted_seconds'] for v in selected),
           selected_k_counts=dict(Counter(v['selected_k'] for v in selected)),
           mean_test_log_density=st.mean(ld) if ld else None,mean_test_ari=st.mean(ari) if ari else None)
        if len(selected)==1:x.update(selected_structure=selected[0]['best']['structure'],available_k=selected[0]['available_k'])
        record['methods'][mname]=x
    output['joint_results'].append(record)
for m in methods:
    rs=[g['methods'][m] for g in output['joint_results']];bs=[g['methods']['cutoff_2'] for g in output['joint_results']]
    output['method_summary'][m]=dict(joint_searches=12,fully_available_joint_searches=sum(r['all_k_available']==r['search_repeats'] for r in rs),
        sum_within_dataset_agreement=sum(r['agrees_with_cutoff2'] for r in rs),sum_within_dataset_better=sum(r['better_than_cutoff2'] for r in rs),
        sum_within_dataset_worse=sum(r['worse_than_cutoff2'] for r in rs),mean_bic_change=st.mean(r['bic_change_from_cutoff_2'] for r in rs),
        max_bic_change=max(r['bic_change_from_cutoff_2'] for r in rs),aggregate_fitting_ratio=sum(r['fitting_seconds'] for r in rs)/sum(b['fitting_seconds'] for b in bs),
        aggregate_accounted_ratio=sum(r['accounted_seconds'] for r in rs)/sum(b['accounted_seconds'] for b in bs))
expected=read_json('real_audit_expected.json.gz')
for method,actual in output['method_summary'].items():
    for k,v in actual.items():assert math.isclose(v,expected['method_summary'][method][k],abs_tol=1e-9,rel_tol=1e-12),(method,k,v)
write_json('real_audit.json',output)
print('Verified real-data selections against the independent audited summary (12 joint-K analyses).')
