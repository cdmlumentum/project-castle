"""Inspection artifacts for Block 1; never inputs to the endpoint optimizer."""
from dataclasses import asdict
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .block_io import clean, write_json


FILES = ("endpoint_betas_linear.png", "endpoint_betas_linear.pdf",
         "endpoint_mode_profiles.npz", "endpoint_mode_profiles.png",
         "endpoint_inspection.lms", "endpoint_diagnostics.json")


def profile_packet(backend, mode, geometry, wavelength, guide, end):
    """Retain the selected, normalized transverse fields already used in verification."""
    config = backend.config
    compact = config.solver.endpoint_compact_x
    domain = backend.endpoint_isolated_domain_info(compact_x=compact)
    x = np.linspace(-domain["x_span_m"]/2, domain["x_span_m"]/2, config.solver.tracking_grid_nx)
    y = np.linspace(domain["y_min_m"], domain["y_max_m"], config.solver.tracking_grid_ny)
    if config.solver.calibration:
        from .block0 import certified_profile
        x, y = certified_profile(config, "isolated" if compact else "coupled").grid()
    return {"mode": asdict(mode), "x_m": x, "y_m": y, "guide": guide, "end": end,
            "state": geometry.state, "wavelength_m": float(wavelength),
            "geometry": asdict(geometry), "coordinate_frame": domain["coordinate_frame"]}


def plot_endpoint_betas(output, config, endpoint):
    """Plot only calculated endpoints; connecting lines are linear illustrations."""
    length = config.device.length_m * 1e6
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    fig.subplots_adjust(left=.075, right=.91, bottom=.26, top=.75, wspace=.7)
    colors = ("#172f6d", "#23879a", "#c0392b")
    for iw, (ax, wavelength) in enumerate(zip(axes, endpoint.wavelengths_m)):
        rows = {(r["state"], r["end"]): r for r in endpoint.endpoint_records
                if np.isclose(r["wavelength_m"], wavelength, rtol=0, atol=1e-15)}
        for state, color, style in zip(("cross", "bar"), colors, ("-", "--")):
            upper = [rows[state, end]["beta_upper_real_rad_per_m"]*1e-6 for end in ("input", "output")]
            ax.plot([0, length], upper, style, color=color, lw=2.4, marker="o", ms=4,
                    label=f"Upper guide - {state} state")
        lower = [rows["cross", end]["beta_lower_real_rad_per_m"]*1e-6 for end in ("input", "output")]
        lower_bar = [rows["bar", end]["beta_lower_real_rad_per_m"]*1e-6 for end in ("input", "output")]
        common_lower = np.allclose(lower, lower_bar, rtol=1e-10, atol=1e-12)
        ax.plot([0, length], lower, color=colors[2], lw=2.4, marker="o", ms=4,
                label="Lower guide - both states" if common_lower else "Lower guide - cross state")
        if not common_lower:
            ax.plot([0, length], lower_bar, "--", color=colors[2], label="Lower guide - bar state")
        beta_values = [r[f'beta_{guide}_real_rad_per_m']*1e-6 for r in rows.values() for guide in ('upper', 'lower')]
        label_separation = .13*(max(beta_values)-min(beta_values))
        for end, x, offset, align in (("input", 0, -.04*length, "right"),
                                       ("output", length, .04*length, "left")):
            label_y = {state: (rows[state, end]['beta_upper_real_rad_per_m']+rows[state, end]['beta_lower_real_rad_per_m'])*.5e-6
                       for state in ('cross', 'bar')}
            if abs(label_y['cross']-label_y['bar']) < label_separation:
                center = (label_y['cross']+label_y['bar'])/2
                label_y = {'cross': center+label_separation/2, 'bar': center-label_separation/2}
            for state, dx in (("cross", 0), ("bar", -.025*length if end == "input" else .025*length)):
                row = rows[state, end]
                upper_beta = row["beta_upper_real_rad_per_m"]*1e-6
                lower_beta = row["beta_lower_real_rad_per_m"]*1e-6
                ax.annotate("", (x+offset+dx, upper_beta), (x+offset+dx, lower_beta),
                            annotation_clip=False,
                            arrowprops={"arrowstyle": "<->", "color": "#263244", "lw": .9})
                delta = row["delta_beta_real_rad_per_m"]*1e-6
                ax.annotate(f"$\\Delta\\beta_{{\\mathrm{{{state},{'in' if end == 'input' else 'out'}}}}}$\n{delta:+.5f}",
                            (x+offset+dx, (upper_beta+lower_beta)/2),
                            xytext=(x+offset+dx+(-.02 if end == 'input' else .02)*length, label_y[state]),
                            annotation_clip=False,
                            ha=align, va="center", fontsize=9,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})
        ax.set_title(f"({'ab'[iw]})  {wavelength*1e9:.0f} nm", loc="left", fontsize=13, pad=12)
        ax.set(xlabel=r"Position $z$ ($\mu$m)", ylabel=r"Re($\beta$) (rad/$\mu$m)",
               xlim=(-.45*length, 1.05*length), xticks=np.linspace(0, length, 5))
        ax.margins(y=.17)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", color="#d4dce7", lw=.65)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Waveguide propagation constants", fontsize=20, fontweight="bold", y=.97)
    fig.text(.5, .90, "Linear interpolation between the two calculated endpoints", ha="center", color="#526278")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .86), ncol=len(labels), frameon=False)
    fig.text(.5, .13, f"Upper guide width: {endpoint.w1_input_m*1e6:.4f} to {endpoint.w1_output_m*1e6:.4f} um"
             f"     |     Assumed length L = {length:g} um", ha="center")
    fig.text(.5, .075, r"$\Delta\beta$ = Re($\beta_{upper}-\beta_{lower}$); detuning labels in rad/$\mu$m.", ha="center", fontsize=10)
    fig.text(.5, .025, "Dots: calculated endpoint values. Lines: linear interpolation; not a Block 2 profile simulation.", ha="center", fontsize=10)
    try:
        for suffix in ("png", "pdf"):
            fig.savefig(Path(output)/f"endpoint_betas_linear.{suffix}", dpi=180)
    finally:
        plt.close(fig)


def export_mode_profiles(output, profiles, endpoint):
    """Store complex Ex/Ey and SI coordinates losslessly, with a visual overview."""
    arrays, index, covered = {}, [], set()
    profiles = list(profiles)
    fig, axes = plt.subplots((len(profiles)+1)//2, 2, figsize=(11, 2.7*((len(profiles)+1)//2)), squeeze=False,
                             constrained_layout=True)
    try:
        for i, packet in enumerate(profiles):
            mode = packet["mode"]
            ex, ey = np.asarray(mode["ex"]), np.asarray(mode["ey"])
            x, y = np.asarray(packet["x_m"]).reshape(-1), np.asarray(packet["y_m"]).reshape(-1)
            if ex.shape != (len(x), len(y)) or ey.shape != ex.shape or not np.all(np.isfinite([ex, ey])):
                raise ValueError("Invalid Block 1 mode profile fields or coordinate grid")
            relevant = [r for r in endpoint.endpoint_records if (packet["end"] == "both" or r["end"] == packet["end"])
                        and (packet.get("shared_states", packet["guide"] == "lower" and packet["end"] == "both") or r["state"] == packet["state"])
                        and np.isclose(r["wavelength_m"], packet["wavelength_m"], rtol=0, atol=1e-15)]
            beta = 2*np.pi*complex(np.asarray(mode["neff"]).squeeze())/packet["wavelength_m"]
            for row in relevant:
                expected = complex(row[f"beta_{packet['guide']}_real_rad_per_m"], row[f"beta_{packet['guide']}_imag_rad_per_m"])
                if not np.isclose(beta, expected, rtol=1e-6, atol=1e-3):
                    raise ValueError("Saved mode profile disagrees with verified endpoint beta")
                covered.add((row['state'], row['end'], row['wavelength_m'], packet['guide']))
            if not relevant:
                raise ValueError("Saved mode profile has no matching endpoint record")
            prefix = f"mode_{i:02d}"
            arrays.update({f"{prefix}_ex": ex, f"{prefix}_ey": ey, f"{prefix}_x_m": x, f"{prefix}_y_m": y})
            index.append({k: v for k, v in packet.items() if k not in ("mode", "x_m", "y_m")} |
                         {"array_prefix": prefix, "neff": complex(np.asarray(mode["neff"]).squeeze()),
                          "identity": mode["identity"], "te_fraction": mode["te_fraction"]})
            intensity = np.abs(ex)**2+np.abs(ey)**2
            ax = axes.flat[i]
            img = ax.pcolormesh(x*1e6, y*1e6, (intensity/max(float(intensity.max()), 1e-300)).T,
                                shading="auto", cmap="inferno", vmin=0, vmax=1, rasterized=True)
            state_label = 'both states' if packet.get('shared_states', packet['guide'] == 'lower' and packet['end'] == 'both') else packet['state']
            end_label = 'both ends' if packet['end'] == 'both' else packet['end']+' end'
            ax.set(xlabel="x (um)", ylabel="y (um)", title=f"{packet['guide']} | {end_label} | {state_label} | {packet['wavelength_m']*1e9:.0f} nm")
            fig.colorbar(img, ax=ax, label="Relative transverse intensity")
        for ax in list(axes.flat)[len(profiles):]:
            ax.set_visible(False)
        expected = {(r['state'], r['end'], r['wavelength_m'], guide)
                    for r in endpoint.endpoint_records for guide in ('upper', 'lower')}
        if covered != expected:
            raise ValueError('Saved mode profiles do not cover every endpoint case')
        fig.suptitle("Block 1 selected isolated modes: |Ex|^2 + |Ey|^2 (each panel scaled to its peak)")
        np.savez_compressed(Path(output)/"endpoint_mode_profiles.npz", **arrays)
        fig.savefig(Path(output)/"endpoint_mode_profiles.png", dpi=150)
    finally:
        plt.close(fig)
    return index


def _collect_selected_profiles(backend, config, endpoint):
    """Original grid alternative: re-solve each saved guide/state/edge independently."""
    from .geometry import make_geometry
    from .materials import build_material_bank
    bank = build_material_bank(config)
    profiles, references = [], {}
    for row in sorted(endpoint.endpoint_records, key=lambda r: (r['end'], r['state'], r['wavelength_m'])):
        wavelength = row['wavelength_m']
        liquid = bank.liquid_cross if row['state'] == 'cross' else bank.liquid_bar
        geometry = make_geometry(config.geometry, w1_m=row['w1_m'], h_gap_m=endpoint.h_gap_m,
            h_rib_m=endpoint.h_rib_m, d_c2c_m=row['d_c2c_m'], state=row['state'],
            liquid_index=float(liquid.index(wavelength)))
        for guide in ('upper', 'lower'):
            key = (row['end'], row['state'], guide)
            mode, _ = backend.solve_isolated_mode(geometry, wavelength, guide=guide,
                reference=references.get(key), compact_x=config.solver.endpoint_compact_x, use_cache=False)
            references[key] = mode
            profiles.append(profile_packet(backend, mode, geometry, wavelength, guide, row['end']))
    return profiles


def export_endpoint_diagnostics(output, config, endpoint, *, backend=None, profiles=None, inspection_project=None):
    plot_endpoint_betas(output, config, endpoint)
    profiles = profiles if profiles is not None else getattr(backend, "_block1_mode_profiles", None)
    extra_profile_solves = 0
    if profiles is None and backend is not None:
        # The original grid search has no retained final verification fields.
        profiles = _collect_selected_profiles(backend, config, endpoint)
        extra_profile_solves = len(profiles)
    report = {"beta_plot": {"files": list(FILES[:2]), "length_m": config.device.length_m,
                            "scope": "real endpoint beta values; linear illustration only; no interior FDE samples"},
              "mode_profiles": {"status": "unavailable_from_saved_scalar_handoff"},
              "inspection_project": {"status": "unavailable_from_saved_scalar_handoff"},
              "validation_scope": "inspection exports; no additional convergence or physical-device validation"}
    if profiles:
        report["mode_profiles"] = {"status": "saved", "files": list(FILES[2:4]),
            "additional_isolated_solve_calls": extra_profile_solves,
            "representation": "complex normalized Ex/Ey on the solver's tracking grid; SI x/y coordinates; not raw vector fields",
            "modes": export_mode_profiles(output, profiles, endpoint)}
    if backend is not None and hasattr(backend, "save_endpoint_inspection"):
        report["inspection_project"] = backend.save_endpoint_inspection(Path(output)/FILES[4], endpoint)
    elif inspection_project is not None and Path(inspection_project).is_file():
        shutil.copy2(inspection_project, Path(output)/FILES[4])
        report["inspection_project"] = {"status": "saved", "file": FILES[4], "source": str(inspection_project),
                                         "end": "input", "state": "cross", "wavelength_m": float(endpoint.wavelengths_m[0])}
    write_json(Path(output)/FILES[5], report)
    return clean(report)
