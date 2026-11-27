"""Explicit, evidence-gated zero compatibility; never unconditional clipping.

Evidence is an independently reviewed bound, not a tolerance inferred from a
small relative beta difference. Diagnostic sensitivity is not a certified bound.
"""
from dataclasses import asdict, replace
import json
import numpy as np


def case_id(geometry, wavelength):
    from .block_io import digest
    return digest({'geometry':asdict(geometry),'wavelength_m':float(wavelength)})


def evidence_identity(config):
    from .block_io import physics_signature, digest
    from .provenance import optical_identity
    physics = physics_signature(config,1)
    physics.pop('coupling_zero_evidence_sha256',None)
    return digest({'physics':physics,'optical':optical_identity(config)})


def apply_zero_compatibility(coefficient, config, geometry, wavelength):
    path = config.resolved_path(config.solver.coupling_zero_evidence)
    if coefficient.extraction_flag != 'negative_kappa_squared' or path is None:
        return coefficient
    from .block_io import digest
    evidence = json.loads(path.read_text(encoding='utf-8'))
    if (evidence.get('schema_version') != 1 or evidence.get('status') != 'reviewed_bound'
            or evidence.get('identity') != evidence_identity(config)):
        raise ValueError('Zero-compatibility evidence is unreviewed or incompatible.')
    record = evidence.get('cases',{}).get(case_id(geometry,wavelength))
    if record is None:
        return coefficient  # Exact geometry only; never extrapolate an error bound.
    if not record.get('matched_background_checked') or not record.get('mesh_domain_checked') or not record.get('review_rationale'):
        raise ValueError('Zero bound needs matched-background, mesh/domain evidence and a review rationale.')
    q = coefficient.delta_pm**2-coefficient.delta_12**2
    center = complex(record['raw_q_real'],record['raw_q_imag'])
    uncertainty = float(record['q_absolute_error_bound'])
    if (not np.isfinite(uncertainty) or uncertainty <= 0 or abs(q-center)>max(1e-2,abs(center)*1e-8)
            or abs(q)>uncertainty or abs(geometry.d_c2c_m)<config.solver.large_separation_check_m):
        return coefficient
    # Complex disk bound: |kappa| <= sqrt(|q|+u_q). Use the whole device
    # length conservatively; no unmeasured localization interval is assumed.
    upper = float(np.sqrt(abs(q)+uncertainty))
    if upper*config.device.length_m > config.convergence.phase_error_rad:
        return coefficient
    audit = {'policy':'zero_compatible_within_uncertainty_v1',
        'raw_q_real':q.real,'raw_q_imag':q.imag,'q_absolute_error_bound':uncertainty,
        'kappa_absolute_upper_bound_rad_per_m':upper,'evidence_sha256':digest(evidence),
        'case_id':case_id(geometry,wavelength),'evidence':record,
        'geometry':asdict(geometry),'wavelength_m':float(wavelength),
        'identity':evidence['identity'],'whole_length_phase_bound_rad':upper*config.device.length_m}
    return replace(coefficient,kappa=0j,extraction_flag='zero_compatible_within_uncertainty',zero_compatibility=audit)


def validate_zero_compatibility(coefficient, config):
    """Recheck the saved correction as well as its bound on every handoff."""
    audit = coefficient.zero_compatibility
    if not isinstance(audit,dict) or audit.get('policy') != 'zero_compatible_within_uncertainty_v1':
        raise ValueError('Missing zero-compatibility audit.')
    q = coefficient.delta_pm**2-coefficient.delta_12**2
    from .block_io import digest
    path = config.resolved_path(config.solver.coupling_zero_evidence)
    if path is None: raise ValueError('Saved zero correction requires its independent evidence file.')
    evidence = json.loads(path.read_text(encoding='utf-8'))
    record = evidence.get('cases',{}).get(audit['case_id'])
    expected_case = digest({'geometry':audit['geometry'],'wavelength_m':audit['wavelength_m']})
    if (evidence.get('schema_version') != 1 or evidence.get('status') != 'reviewed_bound'
            or evidence.get('identity') != audit['identity'] or digest(evidence)!=audit['evidence_sha256']
            or record != audit['evidence'] or expected_case != audit['case_id']
            or record is None or not record.get('matched_background_checked')
            or not record.get('mesh_domain_checked') or not record.get('review_rationale')
            or abs(audit['geometry']['d_c2c_m'])<config.solver.large_separation_check_m
            or record['q_absolute_error_bound'] != audit['q_absolute_error_bound']
            or abs(q-complex(record['raw_q_real'],record['raw_q_imag']))>max(1e-2,abs(q)*1e-8)):
        raise ValueError('Saved zero correction disagrees with its independent evidence.')
    u = float(audit['q_absolute_error_bound'])
    upper = float(np.sqrt(abs(q)+u))
    if (coefficient.kappa != 0j or not np.isfinite(u) or u <= 0 or q.real>=0 or abs(q)>u
            or abs(q-complex(audit['raw_q_real'],audit['raw_q_imag']))>max(1e-2,abs(q)*1e-8)
            or not np.isclose(upper,audit['kappa_absolute_upper_bound_rad_per_m'])
            or upper*config.device.length_m>config.convergence.phase_error_rad
            or audit['identity'] != evidence_identity(config)):
        raise ValueError('Saved zero-compatibility bound is inconsistent or too large.')
    return True
