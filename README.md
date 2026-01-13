# openmc-xs-helpers

Notebook-friendly helpers for exploring **OpenMC incident-neutron cross sections** from a local
`cross_sections.xml` library.

This package is designed for fast “what reactions matter?” workflows:

- List **which reactions (MTs)** are available in your installed library
- Rank reactions by **peak cross section** over an energy range
- Rank reactions by **cross section at a specific energy**
- Plot cross sections with interactive Plotly toggles:
  - X axis: **linear (MeV)** / **log (eV)**
  - Y axis: **linear** / **log**
  - Mode (for materials/elements): **microscopic**, **microscopic × atom fraction**, **macroscopic**

## Scope and positioning

- **Complementary to OpenMC’s plotting** (e.g. `openmc.plot_xs`): this package focuses on
  Plotly figures, ranking tables, and **material/element-aware aggregation** (including macroscopic sums).
- **Different from XSPlot / dedicated UIs**: this is intended for **in-notebook analysis utilities**
  that fit directly into OpenMC workflows.

---

## Requirements

- A working OpenMC Python environment
- `openmc.config["cross_sections"]` points to a valid `cross_sections.xml`
- The referenced neutron HDF5 files exist and are readable (e.g. JEFF, ENDF/B, etc.)

> Note (Windows): `openmc` is typically not installable via PyPI on Windows.  
> Use this package **inside your OpenMC Docker/container environment** (recommended), or inside an
> environment where OpenMC is already installed.

---

## Installation

### Inside your OpenMC Docker environment (recommended)

From the repository root (where `pyproject.toml` lives):

```bash
python -m pip install -e .
