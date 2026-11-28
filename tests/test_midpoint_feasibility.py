"""SYNTHETIC fixtures only; no optical-device validity is implied."""
import hashlib
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from vlac.cache import SQLiteCache
from vlac.midpoint_feasibility import RawFallback, candidates, refinement_widths, summary


def record(width, wave, kappa=10+0j, delta=0., valid=True):
    return {'setting':'production', 'width_m':width, 'wavelength_m':wave, 'valid':valid,
        'coefficients':{'kappa':{'real':kappa.real, 'imag':kappa.imag},
                        'delta_beta':{'real':delta, 'imag':0.}}}


class MidpointFeasibilityTests(unittest.TestCase):
    def test_ratio_magnitude_alone_cannot_hide_imaginary_coupling(self):
        rows = candidates([record(1., 1., 10j), record(1., 2., 10j)], [1., 2.], 20.)
        self.assertEqual(rows[0]['D_over_abs_kappa'], [2., 2.])
        self.assertAlmostEqual(rows[0]['coupling_error'], 2**.5)

    def test_a_common_width_must_work_at_both_wavelengths(self):
        rows = [record(1., 1., 10+0j), record(1., 2., 2+0j),
                record(2., 1., 2+0j), record(2., 2., 10+0j)]
        result = summary(rows, [1., 2.], 20.)
        self.assertAlmostEqual(result['best_coupling_fit']['coupling_error'], .8)

    def test_detuning_and_coupling_have_separate_best_widths(self):
        rows = [record(1., w, 10+0j, 40.) for w in (1., 2.)]
        rows += [record(2., w, 8+0j, 0.) for w in (1., 2.)]
        result = summary(rows, [1., 2.], 20.)
        self.assertEqual(result['best_coupling_fit']['width_m'], 1.)
        self.assertEqual(result['best_detuning_fit']['width_m'], 2.)
        self.assertEqual(result['best_full_midpoint_fit']['width_m'], 2.)

    def test_rejected_and_missing_samples_cannot_supply_a_solution(self):
        rows = [record(1., 1.), record(1., 2., valid=False), record(2., 1.)]
        self.assertEqual(candidates(rows, [1., 2.], 20.), [])

    def test_refinement_proposals_include_each_zero_and_common_crossing(self):
        rows = [{'width_m':1., 'delta_beta_real_rad_per_m':[-2., -4.]},
                {'width_m':3., 'delta_beta_real_rad_per_m':[6., 4.]}]
        self.assertEqual(refinement_widths(rows), [1.5, 1.75, 2.])

    def test_source_cache_is_read_only_and_checksum_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = SQLiteCache(Path(temp)/'source.sqlite')
            primary = SQLiteCache(Path(temp)/'new.sqlite')
            source.put('key', {'real_raw_fixture':False, 'synthetic':True}, kind='native_raw_modes')
            before = hashlib.sha256(source.path.read_bytes()).hexdigest()
            cache = RawFallback(primary, source.path)
            self.assertIsNone(cache.get('key', kind='native_coefficients'))
            self.assertTrue(cache.get('key', kind='native_raw_modes')['synthetic'])
            self.assertEqual(cache.hits, 1)
            self.assertEqual(before, hashlib.sha256(source.path.read_bytes()).hexdigest())
            with closing(sqlite3.connect(source.path)) as db, db:
                db.execute("UPDATE cache_v2 SET sha256='bad'")
            other = RawFallback(SQLiteCache(Path(temp)/'other.sqlite'), source.path)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                other.get('key', kind='native_raw_modes')
