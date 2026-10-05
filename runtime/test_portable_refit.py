"""Tests of publication orchestration, without numerical model fitting."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/refit.py'
spec = importlib.util.spec_from_file_location('refit_under_test', SOURCE)
refit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refit)


class PortableRefitTests(unittest.TestCase):
    def test_scopes_and_source_indices(self):
        _, all_cases = refit.select_cases('all')
        _, sim = refit.select_cases('simulation')
        _, real = refit.select_cases('real_data')
        self.assertEqual((len(all_cases), len(sim), len(real)), (406, 370, 36))
        self.assertEqual(sum(c['competitors'] for c in all_cases), 256)
        self.assertEqual(set(c['task']['dataset_id'] for c in real), {'faithful', 'seeds', 'wdbc'})
        manifest = refit.read(refit.BASE / 'manifest.json')
        for case in all_cases:
            self.assertEqual(case['task'], manifest['tasks'][case['source_index']])

    def test_single_case_preserves_registered_identity(self):
        original, cases = refit.select_cases('all', 'sim_gaussian_control_main_000')
        self.assertEqual(len(cases), 1)
        protocol = refit.make_protocol(original, cases, {'inventory_sha256': 'new-public-inventory'})
        self.assertEqual(protocol['cases'][0], original['cases'][0])
        self.assertEqual(len(original['cases']), 406)
        self.assertEqual(protocol['selections'], original['selections'])
        self.assertEqual(protocol['greedy'], original['greedy'])
        self.assertEqual(protocol['fab'], original['fab'])

    def test_nonmatching_case_is_not_silently_replaced(self):
        with self.assertRaises(ValueError):
            refit.select_cases('real_data', 'sim_gaussian_control_main_000')

    def test_resume_requires_identical_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            with refit.run_directory(output, {'case': 1}, False):
                pass
            with self.assertRaises(FileExistsError):
                with refit.run_directory(output, {'case': 1}, False):
                    pass
            with refit.run_directory(output, {'case': 1}, True):
                pass
            with self.assertRaises(ValueError):
                with refit.run_directory(output, {'case': 2}, True):
                    pass

    def test_published_inputs_cannot_be_output_directories(self):
        for name in ('runtime', 'data', 'results', 'analysis', 'scripts'):
            with self.assertRaises(ValueError):
                with refit.run_directory(refit.ROOT / name / 'new', {}, False):
                    pass

    def test_bounded_executor_does_not_drop_any_operation(self):
        items = [('test', i, None) for i in range(137)]
        with patch.object(refit, 'execute', lambda x: {'index': x[1]}), redirect_stdout(io.StringIO()):
            with ThreadPoolExecutor(max_workers=3) as pool:
                rows = refit.run_stage(pool, 'test', iter(items), 3, True)
        self.assertEqual(sorted(x['index'] for x in rows), list(range(137)))

    def test_refit_uses_new_summary_before_comparison_planning(self):
        original, cases = refit.select_cases('all', 'sim_gaussian_control_main_000')
        events = []
        fresh = {'tasks': [{'new_run_benchmark': 12345}]}
        class Pool:
            def __init__(self, max_workers, **kwargs):
                self.pool = ThreadPoolExecutor(max_workers=max_workers)
            def __enter__(self):
                return self.pool
            def __exit__(self, *args):
                self.pool.shutdown(wait=True)
        class Runner:
            def summarize(self, bundle, output):
                events.append('fresh_summary')
                return fresh
            def atomic_write(self, path, value, backend):
                refit.write(path, value)
        class Worker:
            def merge(self, args):
                return {'cases': [{}], 'complete': True}
        def initialize(config):
            refit._CONFIG = config
            refit._RUNNER = Runner()
            refit._WORKER = Worker()
            refit._CONTEXT = {'backend': object()}
        def execute(item):
            stage, index, detail = item
            events.append(stage)
            if stage == 'plan':
                summary = Path(refit._CONFIG['protocol']).parent / 'base_summary.json'
                self.assertEqual(refit.read(summary), fresh)
                return {'index': index, 'status': 'complete', 'extra_fit_indices': [4]}
            if stage == 'table':
                return {'index': index, 'status': 'qualified', 'fit_indices': [1, 2]}
            return {'index': index, 'status': 'complete'}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(scope='all', case=cases[0]['task_id'], jobs=2,
                                   output=Path(directory) / 'refit', resume=False)
            with patch.object(refit, 'verify_sources', return_value={'inventory_sha256': 'test'}), \
                 patch.object(refit, 'verify_runtime', return_value={'test': True}), \
                 patch.object(refit, 'initialize', initialize), \
                 patch.object(refit, 'execute', execute), \
                 patch.object(refit, 'ProcessPoolExecutor', Pool), redirect_stdout(io.StringIO()):
                refit.refit(args)
            self.assertLess(events.index('fresh_summary'), events.index('plan'))
            self.assertEqual(events.count('base_fit'), 2)
            self.assertEqual(events.count('extra_fit'), 1)
            self.assertEqual(events.count('adaptive'), 2)
            self.assertEqual(refit.read(args.output / 'REFIT_STATUS.json')['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
