# openmc-xs-helpers

Utilities for quickly inspecting OpenMC **incident-neutron cross sections** from an installed `cross_sections.xml` library.

This package provides:

- **Reaction discovery** (available MTs across targets)
- **Tabular summaries** (peak cross section over an energy range, or cross section at a specific energy)
- **Interactive Plotly plots** with:
  - X-axis toggle: linear MeV ↔ log eV
  - Y-axis toggle: log ↔ linear
  - Mode toggle: microscopic ↔ microscopic×atom fraction ↔ macroscopic (weighted sum)
  - A configurable **log-floor** to avoid useless decades

The intended use-case is fast exploratory neutronics and materials comparisons (fusion-range defaults included).

---

## Features

- Reads native tabulated points from OpenMC HDF5 libraries (no resampling for plotted curves).
- Caches:
  - `cross_sections.xml` maps (nuclide → HDF5 file)
  - `IncidentNeutron` objects (HDF5 → parsed data)
- Works with:
  - `openmc.Material`
  - element symbols (e.g. `"W"`, `"Be"`) using isotopes present in your XS library
  - nuclides (e.g. `"W182"`, `"Li-6"`)
  - mixed lists, e.g. `[FirstWall_mat, "Be"]`

---

## Installation

### Local editable install (recommended during development)

From the repository root:

```bash
pip install -e .
