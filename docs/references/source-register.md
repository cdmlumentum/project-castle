# Simulation source register

Record every reference used to make a workflow, model, data, or validation decision. Use a relative link for files in this project and a durable URL or access location for material stored elsewhere.

| ID | Source and owner | Version or date | Location | Relevance to workflow | Status |
| --- | --- | --- | --- | --- | --- |
| REF-001 | Fully Adiabatic Design of Waveguide Branches; bibliographic details pending | As supplied | [PDF](Fully_Adiabatic_Design_of_Waveguide_Branches.pdf) | Original conversation reference; technical claims await content review | pending review |
| REF-002 | Nonvolatile Liquid Controlled Adiabatic Silicon Photonics Switch; bibliographic details pending | As supplied | [PDF](Nonvolatile_Liquid_Controlled_Adiabatic_Silicon_Photonics_Switch.pdf) | Original conversation reference; technical claims await content review | pending review |
| REF-003 | Broadband and Temperature Tolerant Silicon Nitride Liquid Controlled Waveguide Coupler; bibliographic details pending | As supplied | [PDF](Broadband_and_Temperature_Tolerant_Silicon_Nitride_Liquid_Controlled_Waveguide_Coupler.pdf) | Original conversation reference; technical claims await content review | pending review |
| REF-004 | oe-26-16-19877; title and publisher pending | As supplied | [PDF](oe-26-16-19877.pdf) | Original conversation reference; technical claims await content review | pending review |
| REF-005 | Vertical liquid adiabatic coupler workflow; user supplied Python package | v23 / 0.23.0 | [Import manifest](../legacy/v23-import.json) | Geometry, numerical algorithms and saved block contract used as the implementation baseline | current |
| REF-006 | Endpoint propagation-constant plot; user supplied attachment | Received 2026-11-26 | `C:/Users/mos68521/Downloads/endpoint_betas_linear.pdf` | Block 1 two-panel beta-versus-position layout, signed detuning annotations and explicit linear interpolation between calculated endpoints; visual reference only, numeric values are not simulation inputs | current |
| REF-007 | Ansys Lumerical field retrieval and project persistence documentation | Accessed 2026-11-26 | [getdata command](https://optics.ansys.com/hc/en-us/articles/360034409834-getdata-Script-command), [save command](https://optics.ansys.com/hc/en-us/articles/360034410814-save-Script-command) | MODE modal field retrieval and solved project persistence; new archives retain the existing selected normalized transverse Ex/Ey representation | current |
| REF-008 | Ansys Optics: FDE solver simulation object | Accessed 2026-11-26 | [Official documentation](https://optics.ansys.com/hc/en-us/articles/360034916973-FDE-solver-Simulation-object) | Independent mesh/domain controls, PML and actual mesh considerations for Block 0 | current |
| REF-009 | Ansys Optics: Tunable CROW filter, FDE settings | Accessed 2026-11-26 | [Official example](https://optics.ansys.com/hc/en-us/articles/360051107554-Tunable-CROW-Filter) | Mesh and domain convergence guidance | current |
| REF-010 | Ansys Optics: addmesh | Accessed 2026-11-26 | [Official command](https://optics.ansys.com/hc/en-us/articles/360034924253-addmesh-Script-command) | Local override construction in the Python/native adapters | current |
| REF-011 | Ansys Optics: safe mode and MATLAB file persistence | Accessed 2026-11-26 | [Safe mode](https://optics.ansys.com/hc/en-us/articles/360044709054-Running-script-in-safe-mode), [matlabsave](https://optics.ansys.com/hc/en-us/articles/360034928113-matlabsave-Script-command), [matlabload](https://optics.ansys.com/hc/en-us/articles/360034408034-matlabload-Script-command) | Incomplete native handoffs without disabled rm/cp/mv commands; numeric MAT import | current |
| REF-012 | Ansys Optics: named object count and properties | Accessed 2026-11-26 | [getnamednumber](https://optics.ansys.com/hc/en-us/articles/360034408594-getnamednumber-Script-command), [getnamed](https://optics.ansys.com/hc/en-us/articles/360034408574-getnamed-Script-command) | Native persistent scene updates by object name | current |
| REF-013 | MathWorks MAT versions and h5py references | Accessed 2026-11-26 | [MAT versions](https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html), [HDF5 references](https://docs.h5py.org/en/stable/refs.html) | Offline v7.3 numeric struct/cell import, confirmed against a MAT file written by local MODE v221 | current |
| REF-014 | Ansys Optics: script error handling and property-name detection | Accessed 2026-11-26 | [try](https://optics.ansys.com/hc/en-us/articles/360034928513-try-Script-command), [findstring](https://optics.ansys.com/hc/en-us/articles/360034405954-findstring-Script-command) | Preserve native error details and detect the optional FDE autoscale property; installed v221 support was inspected directly, separately from current documentation | current |

Use a short, stable ID such as `REF-001`. In **Relevance to workflow**, name the requirement, model assumption, input, interface, or validation case supported by the source. In **Status**, use `current`, `superseded`, or `pending review`.

## Guarded tracking and monotone splines (2026-11-27)

- REF-009: [Ansys getdata](https://optics.ansys.com/hc/en-us/articles/360034409834-getdata-Script-command).
  Raw MODE field access; the installed v221 command is reused unchanged.
- REF-010: [Ansys overlap](https://optics.ansys.com/hc/en-us/articles/360034405254-overlap-Script-command).
  Documents mode alignment and the E/H power-overlap definition. The project's
  normalized transverse-E similarity is explicitly a different tracking metric;
  no post-v221 rectilinear-dataset overlap API is introduced.
- REF-011: [SciPy BSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html).
  Basis representation and derivative API. The ordered-coefficient guarantee
  follows from nonnegative lower-degree derivative basis functions; the new
  parameterization does not replace the cubic spline or its knots.
