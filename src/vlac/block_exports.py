"""Human-readable exports available immediately after each individual block."""
from pathlib import Path

from .block_io import write_json
from .export import _coefficient_rows, _plot_profiles, _write_csv
from .geometry import ClampedCubicProfile, blackman_theta


def export_profiles(output, config, endpoint, solution, bar=None):
    out = Path(output)
    w = ClampedCubicProfile(config.device.length_m,solution.w1_control_m)
    d = ClampedCubicProfile(config.device.length_m,solution.d_c2c_control_m)
    theta = blackman_theta(solution.z_m,config.device.length_m)
    rows = [{'z_m':z,'z_um':z*1e6,'theta_rad':theta[i],'w1_m':float(w(z)),'w1_um':float(w(z)*1e6),
             'd_c2c_m':float(d(z)),'d_c2c_um':float(d(z)*1e6),'h_gap_m':endpoint.h_gap_m,'h_rib_m':endpoint.h_rib_m,
             'length_m':config.device.length_m,'delta_beta_target_rad_per_m':solution.delta_target_rad_per_m[i],
             'kappa_target_rad_per_m':solution.kappa_target_rad_per_m[i]} for i,z in enumerate(solution.z_m)]
    _write_csv(out/'optimized_profiles.csv',list(rows[0]),rows)
    coefficients = _coefficient_rows(solution,bar,endpoint,config)
    _write_csv(out/'coefficients.csv',list(coefficients[0]),coefficients)
    for state in ('cross','bar'):
        selected = [r for r in coefficients if r['state']==state]
        if selected: _write_csv(out/f'{state}_coefficients.csv',list(selected[0]),selected)
    _plot_profiles(out,solution,bar,endpoint,config)
    write_json(out/'optimizer_history.json',{'objective_achieved':solution.objective,'grid_converged':solution.grid_converged,
        'grid_history':solution.grid_convergence_history,'optimizer_history':solution.optimizer_history,
        'endpoint_search_converged':endpoint.converged,
        'objective_definition':'max(abs(Re(delta)-target)/D, abs(kappa-target)/(D/2)) across z and wavelength'})
