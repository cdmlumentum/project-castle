"""Guarded translation-aware alternative; tracking overlap is not power overlap."""
from dataclasses import replace
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .modes import InvalidModeError, ModeReference, field_overlap, select_mode


METHOD = 'guarded_translation_v1'


class CandidateRejected(InvalidModeError):
    """Optical extraction cannot be used as an objective or future reference."""


class EndpointConsistencyError(CandidateRejected):
    """Fixed endpoints disagree with the independently verified handoff."""


def translated_reference(reference, candidate):
    """Translate an isolated-guide field, zero-padding outside its saved grid.

    Only isolated modes are translated. A coupled field cannot be shifted as
    a whole when just the upper ridge moves.
    """
    if reference.x_m is None or candidate.x_m is None:
        raise CandidateRejected('Translation-aware tracking requires SI field coordinates.')
    x, y = np.asarray(reference.x_m).reshape(-1), np.asarray(reference.y_m).reshape(-1)
    qx, qy = np.meshgrid(np.asarray(candidate.x_m).reshape(-1)-candidate.center_x_m+reference.center_x_m,
                         np.asarray(candidate.y_m).reshape(-1), indexing='ij')
    queries = np.column_stack([qx.ravel(), qy.ravel()])
    fields = [RegularGridInterpolator((x,y), f, bounds_error=False, fill_value=0)(queries).reshape(qx.shape)
              for f in (reference.ex, reference.ey)]
    return replace(reference, ex=fields[0], ey=fields[1], x_m=candidate.x_m,
                   y_m=candidate.y_m, center_x_m=candidate.center_x_m)


def select_isolated(candidates, references, guide, config):
    if any(not ref.valid for ref in references):
        raise CandidateRejected('A rejected mode reference cannot be reused.')
    eligible = [m for m in candidates if m.te_fraction >= config.solver.minimum_te_fraction
                and np.isfinite(m.neff) and np.all(np.isfinite(m.ex)) and np.all(np.isfinite(m.ey))
                and np.sum(abs(m.ex)**2+abs(m.ey)**2)>0]
    if not eligible:
        raise CandidateRejected('No finite TE-eligible isolated '+guide+' mode.')
    if not references:
        return select_mode(eligible, guide=guide, minimum_te_fraction=config.solver.minimum_te_fraction)
    scored = []
    for mode in eligible:
        overlap = max(field_overlap(mode, translated_reference(getattr(ref,guide),mode)) for ref in references)
        if overlap >= config.solver.minimum_mode_overlap:
            scored.append((.7*overlap+.2*mode.core_fraction+.1*mode.te_fraction,mode,overlap))
    if not scored:
        raise CandidateRejected('No isolated '+guide+' mode passes translated tracking overlap.')
    _, mode, overlap = max(scored,key=lambda value:(value[0],value[1].neff.real))
    return mode, overlap


def require_candidate(coefficient, config):
    values = [getattr(coefficient,name) for name in ('beta_upper','beta_lower','beta_plus','beta_minus','delta_beta','kappa')]
    if (not np.all(np.isfinite(values)) or not np.isfinite(coefficient.min_mode_overlap)
            or coefficient.min_mode_overlap < config.solver.minimum_mode_overlap
            or coefficient.extraction_flag != 'ok'):
        raise CandidateRejected('Rejected optical extraction: '+coefficient.extraction_flag+
                                f'; minimum overlap={coefficient.min_mode_overlap:.6g}')


def endpoint_consistency(grid, wavelengths, endpoint, config, *, state='cross'):
    """Compare actual isolated betas with Block 1, not just saved geometry.

    The existing domain neff tolerance is unchanged. Check each guide as well
    as detuning sign, so common-mode beta errors cannot conceal a branch swap.
    """
    if len(grid)!=len(wavelengths) or not len(grid) or any(len(row)<2 for row in grid):
        raise EndpointConsistencyError('Missing endpoint coefficient samples.')
    checks = []
    for wavelength, row in zip(wavelengths,grid):
        for end, coefficient in (('input',row[0]),('output',row[-1])):
            matches = [r for r in endpoint.endpoint_records if r['state']==state and r['end']==end
                       and abs(r['wavelength_m']-wavelength)<1e-15]
            if len(matches)!=1:
                raise EndpointConsistencyError('Missing unique verified Block 1 endpoint record.')
            record = matches[0]
            errors = {}
            for guide in ('upper','lower'):
                expected = complex(record[f'beta_{guide}_real_rad_per_m'],record[f'beta_{guide}_imag_rad_per_m'])
                errors[guide] = float(abs(getattr(coefficient,'beta_'+guide)-expected)*wavelength/(2*np.pi))
            same_sign = np.sign(coefficient.delta_beta.real)==np.sign(record['delta_beta_real_rad_per_m'])
            checks.append({'wavelength_m':float(wavelength),'end':end,'neff_errors':errors,
                           'same_detuning_sign':bool(same_sign),
                           'passed':bool(same_sign and max(errors.values())<=config.solver.domain_convergence_neff_tolerance)})
    return {'passed':all(c['passed'] for c in checks),'checks':checks,
            'detuning_signs_passed':all(c['same_detuning_sign'] for c in checks),
            'neff_tolerance':config.solver.domain_convergence_neff_tolerance}


def enforce_endpoint_report(report, config):
    """Only the absolute domain-index comparison may be a pilot warning."""
    if report['passed']:
        return
    if (not report.get('detuning_signs_passed',False)
            or config.solver.endpoint_domain_policy != 'warn_exploratory'):
        raise EndpointConsistencyError('Endpoint consistency failed at unchanged tolerance: '+str(report))


def require_anchor_match(mode, anchor, config):
    overlap = field_overlap(mode,translated_reference(anchor,mode))
    if not np.isfinite(overlap) or overlap<config.solver.minimum_mode_overlap:
        raise EndpointConsistencyError(f'Independent endpoint field identity failed: overlap={overlap:.6g}')
    return overlap
