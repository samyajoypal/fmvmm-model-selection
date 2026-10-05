import unittest
from report import build_report,real_joint


class ReportTests(unittest.TestCase):
    def test_random_repeats_are_collapsed_within_dataset(self):
        cases=[]
        for i,gaps in enumerate([[0,2],[2,4]]):
            methods={f'random_c2_r{r:02d}':dict(complete=True,best={'bic':g},bic_gap_original_reference=g,
                     accounted_seconds=10) for r,g in enumerate(gaps)}
            cases.append(dict(task=dict(task_id=str(i),kind='simulation',scenario='test',block='main',fit_k=3),
                              reference_scope='exhaustive',methods=methods))
        result=build_report(cases)['simulation'][0]
        self.assertEqual(result['mean_bic_gap_original_reference']['n'],2)
        self.assertEqual(result['mean_bic_gap_original_reference']['mean'],2)
        self.assertAlmostEqual(result['mean_bic_gap_original_reference']['mcse'],1)

    def test_fab_joint_K_uses_own_objective_not_BIC(self):
        cases=[]
        for k,bic,g in [(1,50,5),(2,20,3),(3,30,4)]:
            value=dict(complete=True,best={'bic':bic},native_objective=g,accounted_seconds=2)
            cases.append(dict(task=dict(kind='real_data',dataset_id='test',split='full',repeat=0,fit_k=k),
                              methods={'fab':value,'cutoff_2':value}))
        r=real_joint(cases)[0]['methods']
        self.assertEqual(r['fab']['selected_k'],1)
        self.assertEqual(r['cutoff_2']['selected_k'],2)
        self.assertEqual(r['fab']['accounted_seconds'],6)
        self.assertTrue(r['fab']['all_k_available'])

    def test_failed_method_still_charged_for_computation(self):
        methods={'fab':dict(complete=True,best=None,accounted_seconds=100),
                 'cutoff_2':dict(complete=True,best={'bic':10},accounted_seconds=20)}
        cases=[dict(task=dict(task_id='a',kind='simulation',scenario='test',block='main',fit_k=3),
                    reference_scope='exhaustive',methods=methods)]
        row=next(r for r in build_report(cases)['simulation'] if r['method']=='fab')
        self.assertEqual(row['accounted_time_ratio_to_cutoff_2'],5)
        self.assertEqual(row['no_qualified_winner_rate']['mean'],1)
        self.assertEqual(row['mean_bic_gap_original_reference']['n'],0)


if __name__=='__main__':unittest.main()
