"""Check aggregation and failure rules using small independent fixtures."""
from pathlib import Path
import tempfile
import json
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import worker
from contextlib import nullcontext
from test_fixed_selection import example


def record(bic, status='qualified', seconds=3.):
    row=dict(bic=bic,names=['mvn'],structure=['mvn'],qualified=status=='qualified',
             diagnostics=dict(converged=True,final_stationarity_acceptable=True,hit_cap=False))
    return dict(row=row,status=status,qualified=status=='qualified',seconds=seconds,fit_seconds=seconds-1,
                quality=dict(test_log_density=-bic/100))


def best(pool):
    eligible={k:v for k,v in pool.items() if v.get('status','qualified')=='qualified'
              and v.get('qualified',True) and v.get('row',{}).get('qualified',False)}
    if not eligible:return None
    key=min(eligible,key=lambda k:(eligible[k]['row']['bic'],k))
    return dict(key=key,bic=eligible[key]['row']['bic'],structure=eligible[key]['row']['structure'],
                quality=eligible[key].get('quality',{}))


class WorkerTests(unittest.TestCase):
    def test_plan_to_summary_replays_all_selectors_and_beam(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';out=root/'new';src.mkdir()
            protocol_path=root/'protocol.json';protocol_path.write_text('{}')
            task=dict(task_id='fixture_only',seed=7,reference='exhaustive',fit_k=2,kind='simulation')
            table=json.loads(json.dumps(example()))
            table.update(status='qualified',data={'Xtrain':[[0,0]]*10},fit_indices=list(range(6)),
                         seconds=20,incumbents={})
            backend=SimpleNamespace(loads=json.loads)
            def qualified(row):return bool(row.get('qualified'))
            runner=SimpleNamespace(lock=lambda p:nullcontext(),best=best,qualified=qualified,
                fit_path=lambda r,t,i:r/f'{i}.json',task_identity=lambda c,t,i:{'source_index':i},
                TERMINAL={'qualified','unqualified','deadline'})
            for i,row in enumerate(table['catalogue']):
                val=record(20-i);val['row']['names']=row['assignment'];val['row']['structure']=row['structure']
                val['identity']=dict(source_index=0,table_sha256='hash',candidate_index=i,assignment=row['assignment'])
                (src/f'{i}.json').write_text(json.dumps(val))
            protocol=dict(cases=[dict(competitors=True)],random_seed_offset=10,
                selections=dict(cutoff_multipliers=[0,1,2,4,8],top_r=[1,2,3],top_m=[1,3],random_repeats=2),
                beam=dict(widths=[3],budgets=['matched_cutoff_2',3],random_seed_offset=4),enabled_adaptive_methods=[])
            args=SimpleNamespace(base_output=src,output=out,protocol=protocol_path,index=0)
            ctx=dict(backend=backend,runner=runner)
            def write(p,v,c):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
            with patch.object(worker,'prepare_context',return_value=(protocol,runner,ctx)),\
                 patch.object(worker,'source_table',return_value=(table,'hash',task,0)),\
                 patch.object(worker,'write',side_effect=write):
                plan=worker.plan(args)
                result=worker.summarize(args)
            self.assertEqual(plan['source_index'],0)
            self.assertEqual(plan['extra_fit_indices'],[])
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['original_reference']['bic'],15)
            self.assertIn('beam_w3_b3',result['methods'])
            self.assertTrue(all(v['accounted_seconds']>=20 for v in result['methods'].values()))

    def test_failure_cost_and_incumbent_protection(self):
        fits={'0':record(10), '1':record(1,'unqualified',5), '2':dict(status='deadline',seconds=7)}
        runner=SimpleNamespace(best=best,TERMINAL={'qualified','unqualified','deadline'})
        out=worker.selection_summary([0,1,2],fits,{'initial':record(9)},20,runner)
        self.assertEqual(out['best']['bic'],9)
        self.assertTrue(out['complete'])
        self.assertFalse(out['all_fits_qualified'])
        self.assertEqual(out['accounted_seconds'],35)
        self.assertEqual(out['requested_fits'],3)

    def test_missing_record_never_becomes_success(self):
        runner=SimpleNamespace(best=best,TERMINAL={'qualified'})
        out=worker.selection_summary([0,1],{'0':record(10)},{},2,runner)
        self.assertFalse(out['complete'])
        self.assertEqual(out['missing_indices'],[1])

    def test_compact_preserves_acceptance_and_metrics(self):
        x=record(10)
        x['row']['alpha']=[list(range(100))]
        x['row']['trace']=[1,2,3]
        y=worker.compact(x)
        self.assertNotIn('alpha',y['row'])
        self.assertEqual(y['row']['diagnostics'],x['row']['diagnostics'])
        self.assertEqual(y['quality'],x['quality'])

    def test_merge_reports_every_missing_case(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); protocol=out/'protocol.json';protocol.write_text('{}')
            args=SimpleNamespace(output=out,protocol=protocol)
            p=dict(schema='test',cases=[dict(task={'task_id':'a'}),dict(task={'task_id':'b'})])
            ctx={};written={}
            def write(path,value,_):written[path.name]=value
            with patch.object(worker,'prepare_context',return_value=(p,None,ctx)),patch.object(worker,'write',side_effect=write):
                with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                    worker.merge(args)
            self.assertEqual(written['summary.json']['missing_indices'],[0,1])
            self.assertFalse(written['summary.json']['complete'])


if __name__=='__main__':unittest.main()
