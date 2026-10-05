"""Prespecified selections from a fixed family-score table (no model fitting)."""
import itertools
import math
import random


def validate(table):
    rows, scores, families = table['catalogue'], table['scores'], table['families']
    if not rows or not scores or len(families) != len(set(families)):
        raise ValueError('Empty or duplicate score-table inputs')
    if any(len(row) != len(families) or not all(math.isfinite(float(x)) for x in row)
           for row in scores):
        raise ValueError('Every score must be finite')
    structures = [tuple(row['structure']) for row in rows]
    if len(structures) != len(set(structures)):
        raise ValueError('Duplicate unordered combinations')
    k = len(scores)
    mandatory = structures.index(('mvn',) * k)
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['deficit'], structures[i]))
    return mandatory, order


def cutoff(table, multiplier, n):
    if multiplier < 0 or n < 1:
        raise ValueError('Invalid cutoff')
    mandatory, _ = validate(table)
    # The tolerance is inherited from the published 2 log n implementation.
    threshold = float(multiplier) * math.log(n) + 1e-10
    return sorted({i for i, row in enumerate(table['catalogue'])
                   if row['deficit'] <= threshold} | {mandatory})


def top_m(table, m):
    mandatory, order = validate(table)
    if not isinstance(m, int) or m < 1:
        raise ValueError('M must be a positive integer')
    return sorted(set(order[:m]) | {mandatory})


def top_r(table, r):
    mandatory, _ = validate(table)
    if not isinstance(r, int) or not 1 <= r <= len(table['families']):
        raise ValueError('Invalid per-component rank')
    families = table['families']
    choices = [[families[h] for h in sorted(range(len(families)),
               key=lambda h: (-float(scores[h]), families[h]))[:r]]
               for scores in table['scores']]
    combinations = {tuple(sorted(a)) for a in itertools.product(*choices)}
    # All methods use the same best-scoring assignment and fit initialization
    # for an unordered combination, even if another assignment selected it.
    return sorted({i for i, row in enumerate(table['catalogue'])
                   if tuple(row['structure']) in combinations} | {mandatory})


def random_subset(table, budget, seed):
    mandatory, _ = validate(table)
    if not isinstance(budget, int) or not 1 <= budget <= len(table['catalogue']):
        raise ValueError('Budget is the total number of unique complete-fit requests')
    pool = [i for i in range(len(table['catalogue'])) if i != mandatory]
    return sorted([mandatory] + random.Random(int(seed)).sample(pool, budget - 1))


def all_selections(table, n, spec, seed, competitors):
    result = {}
    mandatory, order = validate(table)
    for multiplier in spec['cutoff_multipliers']:
        indices = cutoff(table, multiplier, n)
        result[f'cutoff_{multiplier}'] = dict(kind='cutoff', multiplier=multiplier,
                                            indices=indices)
        # Audit the equivalence without counting it as an independent method.
        unforced = [i for i in order if table['catalogue'][i]['deficit'] <=
                    multiplier * math.log(n) + 1e-10]
        assert set(indices) == set(order[:len(unforced)]) | {mandatory}
    if competitors:
        for r in spec['top_r']:
            result[f'top_r_{r}'] = dict(kind='top_r', r=r, indices=top_r(table, r))
        for m in spec['top_m']:
            result[f'top_m_{m}'] = dict(kind='top_m', m=m, indices=top_m(table, m))
        for multiplier in spec['cutoff_multipliers']:
            budget = len(result[f'cutoff_{multiplier}']['indices'])
            for rep in range(spec['random_repeats']):
                # Reuse a seed across budgets: equivalent to prefixes of one
                # shuffled catalogue. Python sample is not prefix-stable, so
                # construct the permutation explicitly.
                rng = random.Random(int(seed) + 104729 * rep)
                pool = [i for i in range(len(table['catalogue'])) if i != mandatory]
                rng.shuffle(pool)
                key = f'random_c{multiplier}_r{rep:02d}'
                result[key] = dict(kind='random', matched_cutoff=multiplier,
                                   repeat=rep, seed=int(seed) + 104729 * rep,
                                   indices=sorted([mandatory] + pool[:budget - 1]))
    return result
