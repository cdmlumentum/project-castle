"""Synthetic coefficient fixtures; these tests do not model a device."""
from dataclasses import replace
from pathlib import Path
import unittest

from vlac.config import load_config
from vlac.coefficient_stability import compare_coefficients
from vlac.modes import extract_coupling

ROOT=Path(__file__).resolve().parents[1]


class CoefficientStabilityTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'config/default.json')
        self.base=extract_coupling(4.,2.,5.,1.)

    def test_full_complex_kappa_difference_is_checked(self):
        actual=replace(self.base,kappa=1j*self.base.kappa)
        result=compare_coefficients(self.config,actual,self.base,1.26e-6)
        self.assertFalse(result['passed'])
        self.assertGreater(result['kappa_error_rad_per_m'],2.)

    def test_same_square_root_gauge_is_compared_without_discarding_phase(self):
        result=compare_coefficients(self.config,replace(self.base,kappa=-self.base.kappa),self.base,1.26e-6)
        self.assertTrue(result['passed'])
        self.assertTrue(result['kappa_gauge_flipped_for_comparison'])
        self.assertEqual(result['kappa_error_rad_per_m'],0.)

    def test_flagged_reference_is_never_accepted_as_converged(self):
        flagged=replace(self.base,extraction_flag='negative_kappa_squared')
        self.assertFalse(compare_coefficients(self.config,self.base,flagged,1.26e-6)['passed'])
        self.assertFalse(compare_coefficients(self.config,flagged,self.base,1.26e-6)['passed'])

if __name__=='__main__': unittest.main()
