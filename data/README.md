# Material data notes

green_si_2008_1260_1340.csv contains Green (2008) intrinsic-Si values at 300 K from Table 1, rows 1.26–1.34 µm, at the source's 10 nm wavelength spacing. The listed absorption coefficient is in cm^-1. The k column is calculated from k=alpha*lambda/(4*pi) after converting wavelength to cm and rounded for CSV readability. Dispersion code interpolates within this table and refuses extrapolation.

Source: M. A. Green, *Solar Energy Materials & Solar Cells* 92, 1305–1310 (2008), DOI [10.1016/j.solmat.2008.06.009](https://doi.org/10.1016/j.solmat.2008.06.009). Verify against the original table for publication-quality use.

In workflow v11 this table is retained for material-data validation only; its silicon material is not installed in MODE because the FDE cross section contains no silicon substrate.
