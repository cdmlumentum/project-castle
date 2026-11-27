"""Block 2 geometry and Blackman comparisons derived from saved FDE samples."""
from pathlib import Path
import numpy as np
from .block_io import write_json
from .export import _coefficient_rows, _write_csv, plt
from .geometry import ClampedCubicProfile

FILES = ('block2_geometry.json', 'block2_geometry.csv', 'block2_target_comparison.json',
         'block2_target_comparison.csv', 'block2_target_comparison.png', 'block2_target_comparison.pdf')


def comparison_summary(config, endpoint, solution):
    scale = endpoint.delta_beta_max_rad_per_m
    rows = []
    for wavelength, coefficients in zip(solution.wavelengths_m, solution.cross_coefficients):
        delta = np.asarray([c.delta_beta for c in coefficients])
        kappa = np.asarray([c.kappa for c in coefficients])
        delta_error = np.abs(delta.real-solution.delta_target_rad_per_m)
        kappa_error = np.abs(kappa-solution.kappa_target_rad_per_m)
        di, ki = int(np.argmax(delta_error)), int(np.argmax(kappa_error))
        rows.append({'wavelength_m':float(wavelength),
            'delta_beta_max_abs_error_rad_per_m':float(delta_error[di]),
            'delta_beta_rms_error_rad_per_m':float(np.sqrt(np.mean(delta_error**2))),
            'delta_beta_max_normalized_error':float(delta_error[di]/scale),
            'delta_beta_worst_z_m':float(solution.z_m[di]),
            'kappa_max_abs_complex_error_rad_per_m':float(kappa_error[ki]),
            'kappa_rms_complex_error_rad_per_m':float(np.sqrt(np.mean(kappa_error**2))),
            'kappa_max_normalized_error':float(kappa_error[ki]/(.5*scale)),
            'kappa_worst_z_m':float(solution.z_m[ki]),
            'delta_beta_max_abs_imag_rad_per_m':float(np.max(np.abs(delta.imag))),
            'kappa_max_abs_imag_rad_per_m':float(np.max(np.abs(kappa.imag)))})
    objective = max(max(r['delta_beta_max_normalized_error'],r['kappa_max_normalized_error']) for r in rows)
    if not np.isclose(objective,solution.objective,rtol=1e-9,atol=1e-12):
        raise ValueError('Block 2 comparison disagrees with saved minimax objective.')
    return {'schema_version':1, 'scope':'cross-state FDE samples at saved wavelengths and z; no full-device validation',
        'beta_definition':'delta_beta = beta_upper - beta_lower; no absolute-beta Blackman target is imposed',
        'theta_definition':'integrated Blackman angle, theta(0)=0 and theta(L)=pi',
        'delta_beta_target':'-D*cos(theta)', 'kappa_target':'(D/2)*sin(theta)',
        'D_rad_per_m':scale, 'objective_definition':'max(abs(Re(delta_beta)-target)/D, abs(complex(kappa)-target)/(D/2))',
        'objective_achieved':objective, 'optimization_converged':bool(solution.optimizer_history and solution.optimizer_history[-1]['success']),
        'grid_converged':bool(solution.grid_converged), 'by_wavelength':rows}


def export_profile_diagnostics(output, config, endpoint, solution):
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    summary = comparison_summary(config,endpoint,solution)
    w = ClampedCubicProfile(config.device.length_m,solution.w1_control_m)
    dc = ClampedCubicProfile(config.device.length_m,solution.d_c2c_control_m)
    z = np.linspace(0,config.device.length_m,501)
    geometry = {'schema_version':1,'units':'SI: metres', 'profile_definition':'open-clamped cubic B-splines; coefficients are not sampled widths',
        'length_m':config.device.length_m,'lower_width_m':config.geometry.w1_lower_m,
        'waveguide_height_m':config.geometry.h_wg_m,'rib_height_m':endpoint.h_rib_m,
        'vertical_gap_m':endpoint.h_gap_m,'box_thickness_m':config.geometry.box_thickness_m,
        'upper_width_controls_m':solution.w1_control_m.tolist(),
        'lateral_offset_controls_m':solution.d_c2c_control_m.tolist(),
        'fixed_upper_endpoint_widths_m':[endpoint.w1_input_m,endpoint.w1_output_m],
        'fixed_lateral_endpoint_offsets_m':[config.geometry.d_c2c_input_m,config.geometry.d_c2c_output_m],
        'sampled_upper_width_range_m':[float(np.min(w(z))),float(np.max(w(z)))],
        'sampled_lateral_offset_range_m':[float(np.min(dc(z))),float(np.max(dc(z)))],
        'range_sampling_points':len(z)}
    write_json(out/FILES[0],geometry)
    rows = [{'z_m':float(zi),'upper_width_m':float(w(zi)),'lower_width_m':config.geometry.w1_lower_m,
             'lateral_offset_m':float(dc(zi)),'vertical_gap_m':endpoint.h_gap_m,
             'waveguide_height_m':config.geometry.h_wg_m,'rib_height_m':endpoint.h_rib_m} for zi in z]
    _write_csv(out/FILES[1],list(rows[0]),rows)
    write_json(out/FILES[2],summary)
    rows = _coefficient_rows(solution,None,endpoint,config)
    for row in rows:
        row['delta_beta_abs_normalized_error'] = abs(row['delta_beta_residual_rad_per_m'])/endpoint.delta_beta_max_rad_per_m
        row['kappa_abs_complex_error_rad_per_m'] = abs(complex(row['kappa_residual_real_rad_per_m'],row['kappa_residual_imag_rad_per_m']))
        row['kappa_abs_normalized_error'] = row['kappa_abs_complex_error_rad_per_m']/(.5*endpoint.delta_beta_max_rad_per_m)
    _write_csv(out/FILES[3],list(rows[0]),rows)
    fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x = solution.z_m*1e6
    for iw,wavelength in enumerate(solution.wavelengths_m):
        c = solution.cross_coefficients[iw]
        delta = np.asarray([v.delta_beta for v in c]); kappa = np.asarray([v.kappa for v in c])
        label = f'{wavelength*1e9:.0f} nm'
        axes[0,0].plot(x,delta.real,'o-',label=label)
        axes[0,1].plot(x,kappa.real,'o-',label='Re '+label)
        axes[0,1].plot(x,kappa.imag,':',label='Im '+label)
        axes[1,0].plot(x,np.abs(delta.real-solution.delta_target_rad_per_m)/endpoint.delta_beta_max_rad_per_m,'o-',label=label)
        axes[1,1].plot(x,np.abs(kappa-solution.kappa_target_rad_per_m)/(.5*endpoint.delta_beta_max_rad_per_m),'o-',label=label)
    # Dense target curves are analytic only; FDE curves join the saved samples.
    from .profiles import target_profiles
    targets = target_profiles(z,config.device.length_m,endpoint.delta_beta_max_rad_per_m)
    for ax,target in zip(axes[0],targets): ax.plot(z*1e6,target,'k--',label='Blackman target')
    axes[0,0].set(title='Upper-minus-lower detuning',ylabel='Re(Δβ) (rad/m)')
    axes[0,1].set(title='Complex coupling',ylabel='κ (rad/m)')
    axes[1,0].set(title='Normalized detuning error',ylabel='|Re(Δβ) − target| / D')
    axes[1,1].set(title='Normalized complex coupling error',ylabel='|κ − target| / (D/2)')
    for ax in axes.flat:
        ax.set_xlabel('z (µm)'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle(f'Block 2 cross state; J={solution.objective:.5g}; sampled FDE coefficients')
    fig.savefig(out/FILES[4],dpi=170); fig.savefig(out/FILES[5]); plt.close(fig)
    return {'geometry':FILES[0],'target_comparison':FILES[2],'files':list(FILES)}
