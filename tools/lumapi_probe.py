"""Check the installed MODE API with a small native LSF waveguide sweep."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


API_DIRECTORY = Path(r"C:\Program Files\Lumerical\v221\api\python")
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "diagnostics" / "lumapi_probe"

SCRIPT = r'''
newproject;
addrect;
set("name","probe_cladding");
set("material","<Object defined dielectric>");
set("index",1.44);
set("x span",10e-6); set("y span",10e-6); set("z span",2e-6);
set("override mesh order from material database",1); set("mesh order",3);
addrect;
set("name","probe_core");
set("material","<Object defined dielectric>");
set("index",2.0);
set("x",0); set("y",0); set("z",0);
set("x span",0.8e-6); set("y span",0.35e-6); set("z span",2e-6);
addfde;
set("solver type",3); set("z",0);
set("x",0); set("x span",4e-6);
set("y",0); set("y span",3e-6);
set("mesh cells x",80); set("mesh cells y",60);
set("x min bc","PML"); set("x max bc","PML");
set("y min bc","PML"); set("y max bc","PML");
probe_widths=[0.8e-6;1.0e-6];
probe_counts=matrix(2,1);
probe_neff=matrix(2,1);
for (probe_i=1:2) {
    switchtolayout;
    setnamed("probe_core","x span",probe_widths(probe_i));
    setanalysis("wavelength",1.3e-6);
    setanalysis("number of trial modes",4);
    setanalysis("search","near n"); setanalysis("use max index",1);
    findmodes;
    probe_counts(probe_i)=nummodes;
    if (probe_counts(probe_i)>0) {
        probe_neff(probe_i)=getdata("mode1","neff");
    }
}
'''


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "waveguide_probe.lsf").write_text(SCRIPT, encoding="utf-8")
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "api_directory": str(API_DIRECTORY),
        "scope": "API connectivity and two coarse FDE solves; no device validation",
        "core_index": 2.0,
        "cladding_index": 1.44,
        "core_height_m": 0.35e-6,
        "wavelength_m": 1.3e-6,
        "mesh_cells": [80, 60],
        "steps": {},
    }
    started = time.perf_counter()
    session = None
    try:
        sys.path.insert(0, str(API_DIRECTORY))
        import lumapi

        report["lumapi_file"] = lumapi.__file__
        report["steps"]["import"] = "passed"
        print("Opening a new hidden MODE session", flush=True)
        session = lumapi.MODE(hide=True, serverArgs={"threads": "2"})
        report["steps"]["session"] = "passed"
        print("Testing variable transfer and native script execution", flush=True)
        session.putv("probe_input", np.array([1.0, 2.0, 3.0]))
        session.eval("probe_output=2*probe_input;")
        transferred = np.asarray(session.getv("probe_output")).reshape(-1)
        if not np.array_equal(transferred, [2.0, 4.0, 6.0]):
            raise RuntimeError("Unexpected API round-trip result")
        report["steps"]["round_trip"] = "passed"
        print("Running two FDE solves in one native LSF loop", flush=True)
        session.eval(SCRIPT)
        widths = np.asarray(session.getv("probe_widths")).reshape(-1)
        counts = np.asarray(session.getv("probe_counts")).reshape(-1)
        indices = np.asarray(session.getv("probe_neff")).reshape(-1)
        report["solves"] = [
            {"width_m": float(width), "mode_count": int(count),
             "mode1_neff_real": float(index.real), "mode1_neff_imag": float(index.imag)}
            for width, count, index in zip(widths, counts, indices)
        ]
        if np.any(counts < 1) or not np.all(np.isfinite(indices)):
            raise RuntimeError("One or more solves did not return a finite mode")
        report["steps"]["native_fde_loop"] = "passed"
        field = np.asarray(session.getdata("mode1", "Ex"))
        report["field_shape"] = list(field.shape)
        if not field.size or not np.all(np.isfinite(field)) or not np.any(np.abs(field) > 0):
            raise RuntimeError("MODE returned an empty or invalid field")
        report["steps"]["field_transfer"] = "passed"
        project = OUTPUT_DIRECTORY / "waveguide_probe.lms"
        session.save(project.as_posix())
        report["project"] = str(project)
        report["steps"]["save_project"] = "passed"
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        if session is not None:
            try:
                session.close()
                report["steps"]["close_session"] = "passed"
            except Exception as exc:
                report["steps"]["close_session"] = str(exc)
                report["status"] = "failed"
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        (OUTPUT_DIRECTORY / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
