"""Direct broadband characterization of frozen geometry, without optimization."""
from dataclasses import replace
import numpy as np
from scipy.interpolate import PchipInterpolator

from .block_io import coefficient_data, decode_coefficients, digest, geometry_data, validate_grid
from .checkpoints import Checkpoint
from .profiles import evaluate_design_profiles
from .provenance import checkpoint_identity
from .runtime import checkpoint_directory


def refine(axis):
    return np.sort(np.r_[axis, (axis[:-1]+axis[1:])/2])


def interpolation_error(base, actual, z, wavelengths, zz, ww, scale, length, kappa_scale=None):
    error = 0.
    for name, normalization in (('delta_beta', scale), ('kappa', scale/2 if kappa_scale is None else kappa_scale)):
        values = np.array([[getattr(c, name) for c in row] for row in base], complex)
        expected = np.array([[getattr(c, name) for c in row] for row in actual], complex)
        predicted = []
        for component in (values.real, values.imag):
            along_z = PchipInterpolator(z, component, axis=1)(zz)
            predicted.append(PchipInterpolator(wavelengths, along_z, axis=0)(ww))
        error = max(error, float(np.max(abs(expected-predicted[0]-1j*predicted[1])))/max(normalization, 1e-30))
    # Common imaginary beta controls absolute guided power even when detuning
    # and coupling are constant. Bound its first-order log-power error as well.
    mean = np.array([[(c.beta_upper+c.beta_lower).imag/2 for c in row] for row in base])
    actual_mean = np.array([[(c.beta_upper+c.beta_lower).imag/2 for c in row] for row in actual])
    predicted_mean = PchipInterpolator(wavelengths, PchipInterpolator(z,mean,axis=1)(zz),axis=0)(ww)
    error = max(error, float(2*length*np.max(abs(actual_mean-predicted_mean))))
    return error


def characterize_profile(backend, config, bank, endpoint, solution, *, progress=None, endpoint_wavelengths_only=False):
    method = 'direct_nested_z_band_edges_v1' if endpoint_wavelengths_only else 'direct_nested_tensor_pchip_v1'
    geometry = geometry_data(config, endpoint, solution)
    identity = checkpoint_identity(config, 3, {'geometry': geometry['geometry_id'],
        'sampling': {'initial_wavelengths': config.device.wavelength_samples_initial,
                     'max_z': config.device.z_samples_max, 'max_wavelengths': config.device.wavelength_samples_max},
        'method': method, 'initial_z': solution.z_m, 'initial_wavelengths': solution.wavelengths_m})
    checkpoint = Checkpoint(checkpoint_directory(config), 'characterization', identity)
    saved = checkpoint.load() if config.execution.resume else None
    if saved and saved.get('complete'):
        validate_characterization(saved['result'], config, endpoint, solution)
        return saved['result']
    z = np.asarray(saved['z'] if saved else solution.z_m)
    initial_wavelengths = ([config.device.wavelength_min_m,config.device.wavelength_max_m] if endpoint_wavelengths_only
        else np.unique(np.r_[solution.wavelengths_m,np.linspace(config.device.wavelength_min_m,
                         config.device.wavelength_max_m,config.device.wavelength_samples_initial)]))
    wavelengths = np.asarray(saved['wavelengths'] if saved else initial_wavelengths)
    if endpoint_wavelengths_only and not np.array_equal(wavelengths,initial_wavelengths):
        raise ValueError('Endpoint-only characterization cannot resume an interior-wavelength grid.')
    history = saved['history'] if saved else []
    grids = saved['grids'] if saved else None
    def evaluate(zz, ww):
        return {state: evaluate_design_profiles(backend, config, bank, endpoint,
                solution.w1_control_m, solution.d_c2c_control_m, zz, ww, state=state, progress=progress)
                for state in ('cross', 'bar')}
    if grids is None:
        grids = evaluate(z, wavelengths)
    passed = False
    for iteration in range(len(history), config.characterization.max_refinements+1):
        zz, ww = refine(z), wavelengths if endpoint_wavelengths_only else refine(wavelengths)
        if len(zz)>config.device.z_samples_max or len(ww)>config.device.wavelength_samples_max:
            history.append({'iteration': iteration, 'passed': False, 'reason': 'axis_limit_before_independent_check'})
            break
        checked = evaluate(zz, ww)
        from .block_io import solution_kappa_max
        errors = {state: interpolation_error(grids[state], checked[state], z, wavelengths, zz, ww,
                                             endpoint.delta_beta_max_rad_per_m, config.device.length_m,
                                             solution_kappa_max(solution,endpoint,config)) for state in grids}
        quality = {state: validate_grid(checked[state], replace(solution, z_m=zz, wavelengths_m=ww), config)
                   for state in grids}
        passed = all(v <= config.device.grid_convergence_rtol for v in errors.values()) and all(v['passed'] for v in quality.values())
        history.append({'iteration': iteration, 'base_shape': [len(wavelengths), len(z)],
                        'retained_shape': [len(ww), len(zz)], 'errors': errors, 'quality': quality, 'passed': passed})
        z, wavelengths, grids = zz, ww, checked
        checkpoint.save({'z': z, 'wavelengths': wavelengths, 'grids': grids, 'history': history})
        if passed:
            break
    result = {'schema_version': 1, 'method': method,
              'geometry_id': geometry['geometry_id'], 'z_m': z.tolist(), 'wavelengths_m': wavelengths.tolist(),
              'cross': coefficient_data(grids['cross']), 'bar': coefficient_data(grids['bar']),
              'validation': {'status': 'passed' if passed else 'review', 'sampling_converged': passed,
                             'tolerance': config.device.grid_convergence_rtol, 'history': history,
                             'error_metric': 'max normalized complex detuning/coupling error and 2*length*common-attenuation error',
                             'profile_optimized': False, 'physical_device_validation': False,
                             'wavelength_validation':('endpoint wavelengths only; no interior checks' if endpoint_wavelengths_only else 'joint wavelength midpoint checks'),
                             'scope': ('Direct cross/bar z interpolation checks at endpoint wavelengths only.' if endpoint_wavelengths_only else
                                       'Direct cross/bar coefficient interpolation checks in z and wavelength, including joint midpoints.')}}
    result['sha256'] = digest(result)
    validate_characterization(result, config, endpoint, solution)
    checkpoint.save({'complete': passed, 'result': result, 'z': z, 'wavelengths': wavelengths,
                     'grids': grids, 'history': history})
    return result


def validate_characterization(value, config, endpoint, solution):
    raw = dict(value)
    expected = raw.pop('sha256', None)
    if value.get('schema_version') != 1 or expected != digest(raw):
        raise ValueError('Unsupported or corrupt characterization dataset')
    if value['geometry_id'] != geometry_data(config, endpoint, solution)['geometry_id']:
        raise ValueError('Characterization belongs to different frozen geometry')
    z, wavelengths = np.asarray(value['z_m']), np.asarray(value['wavelengths_m'])
    endpoint_only = value.get('method')=='direct_nested_z_band_edges_v1'
    if endpoint_only and not np.array_equal(wavelengths,[config.device.wavelength_min_m,config.device.wavelength_max_m]):
        raise ValueError('Endpoint-only characterization contains other wavelengths.')
    for axis, bounds, minimum in ((z, [0, config.device.length_m],3),
                         (wavelengths, [config.device.wavelength_min_m, config.device.wavelength_max_m],2 if endpoint_only else 3)):
        if len(axis)<minimum or not np.all(np.isfinite(axis)) or np.any(np.diff(axis)<=0) or not np.allclose(axis[[0,-1]], bounds, rtol=0, atol=1e-15):
            raise ValueError('Invalid characterization grid')
    for state in ('cross', 'bar'):
        quality = validate_grid(decode_coefficients(value[state]), replace(solution, z_m=z, wavelengths_m=wavelengths), config)
        if value['validation']['status'] == 'passed' and not quality['passed']:
            raise ValueError('Passed characterization contains rejected coefficients')
    return True
