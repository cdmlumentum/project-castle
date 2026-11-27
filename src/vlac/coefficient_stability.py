"""Targeted coefficient checks using the existing Block 0 acceptance metrics."""
from .block0 import compare_saved_coefficients


def compare_coefficients(config, actual, reference, wavelength):
    import numpy as np
    factor=2*np.pi/wavelength
    saved={name:getattr(actual,name) for name in ('beta_upper','beta_lower','beta_plus','beta_minus','delta_beta','kappa')}
    saved['flag']=actual.extraction_flag
    ref={'wavelength_m':wavelength,'neff':[getattr(reference,name)/factor
         for name in ('beta_upper','beta_lower','beta_plus','beta_minus')],
         'delta_beta':reference.delta_beta,'kappa':reference.kappa}
    result=compare_saved_coefficients(config,saved,ref)
    result['reference_extraction_flag']=reference.extraction_flag
    result['actual_extraction_flag']=actual.extraction_flag
    result['passed'] &= reference.extraction_flag=='ok'
    result['raw_complex_kappa_difference_rad_per_m']=abs(actual.kappa-reference.kappa)
    result['kappa_gauge_flipped_for_comparison']=abs(-actual.kappa-reference.kappa)<abs(actual.kappa-reference.kappa)
    return result
