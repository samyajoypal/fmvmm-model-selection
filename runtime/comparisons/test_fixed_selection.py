import itertools
import math
import unittest
from fixed_selection import cutoff, top_m, top_r, random_subset, all_selections


def example(scores=None):
    scores = scores or [[10., 7., 0.], [9., 8., 0.]]
    families = ['mvn', 'mvt', 'mvsn']
    rows = []
    for combination in itertools.combinations_with_replacement(sorted(families), len(scores)):
        gap, assignment = min((sum(max(scores[j]) - scores[j][families.index(a)] for j,a in enumerate(perm)), perm)
                               for perm in set(itertools.permutations(combination)))
        rows.append(dict(structure=combination, assignment=assignment, deficit=gap))
    return dict(catalogue=rows, scores=scores, families=families)


class FixedSelections(unittest.TestCase):
    def test_cutoffs_are_nested_and_zero_allows_ties(self):
        table=example([[10.,10.,0.],[9.,8.,0.]])
        sets=[set(cutoff(table,b,100)) for b in [0,1,2,4,8]]
        self.assertTrue(all(a <= b for a,b in zip(sets,sets[1:])))
        structures={tuple(table['catalogue'][i]['structure']) for i in sets[0]}
        self.assertEqual(structures,{('mvn','mvn'),('mvn','mvt')})

    def test_top_m_equivalence_exact_gap_and_forced_mvn(self):
        table=example([[0.,10.,9.],[0.,10.,9.]])
        for b in [0,1,2,4,8]:
            admitted=[r for r in table['catalogue'] if r['deficit'] <= b*math.log(100)+1e-10]
            self.assertEqual(cutoff(table,b,100),top_m(table,len(admitted)))

    def test_top_r_uses_distinct_combinations(self):
        table=example()
        choices=top_r(table,2)
        self.assertEqual(len(choices),3) # four ordered choices, three combinations
        self.assertEqual(len(top_r(table,3)),len(table['catalogue']))

    def test_random_exact_budget_uniform_objects(self):
        table=example()
        self.assertEqual(len(random_subset(table,3,1)),3)
        mandatory=next(i for i,r in enumerate(table['catalogue']) if r['structure']==('mvn','mvn'))
        counts={i:0 for i in range(len(table['catalogue'])) if i != mandatory}
        for seed in range(5000):
            chosen=random_subset(table,2,seed)
            self.assertIn(mandatory,chosen)
            counts[next(i for i in chosen if i!=mandatory)]+=1
        self.assertLess(max(counts.values())-min(counts.values()),200)

    def test_random_repeats_nested_across_matched_budgets(self):
        table=example()
        spec=dict(cutoff_multipliers=[0,1,2,4,8],top_r=[1,2],top_m=[1,3],random_repeats=3)
        selections=all_selections(table,10,spec,42,True)
        for rep in range(3):
            earlier=set()
            for b in spec['cutoff_multipliers']:
                chosen=set(selections[f'random_c{b}_r{rep:02d}']['indices'])
                self.assertLessEqual(earlier,chosen)
                self.assertEqual(len(chosen),len(selections[f'cutoff_{b}']['indices']))
                earlier=chosen

    def test_all_mvn_is_shared_safeguard(self):
        table=example([[0.,10.,9.],[0.,10.,9.]])
        mandatory=next(i for i,r in enumerate(table['catalogue']) if r['structure']==('mvn','mvn'))
        self.assertIn(mandatory,top_m(table,1))
        self.assertIn(mandatory,top_r(table,1))
        self.assertEqual(random_subset(table,1,4),[mandatory])


if __name__=='__main__': unittest.main()
