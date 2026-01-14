"""
openmc_xs_helpers.xs_helpers

Convenience helpers for inspecting and plotting OpenMC incident-neutron cross sections
directly from the OpenMC HDF5 libraries referenced by cross_sections.xml.

Public API:
    - AvailableReactions
    - available_library_reactions
    - available_reactions
    - peak_cross_section
    - cross_section_at_energy
    - peak_xs_table
    - find_xs_table
    - plot_xs
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Lazy install openmc and data
# -----------------------------

def _require_openmc():
    try:
        import openmc  # noqa: F401
    except Exception as e:
        raise ImportError(
            "openmc is required for this function. Install/run inside an OpenMC environment."
        ) from e
    import openmc
    return openmc


def _require_openmc_data():
    """
    Returns (openmc, IncidentNeutron, REACTION_NAME) with imports performed lazily.
    """
    openmc = _require_openmc()
    try:
        from openmc.data import IncidentNeutron
        from openmc.data.reaction import REACTION_NAME
    except Exception as e:
        raise ImportError(
            "openmc.data is required for this function. Use an OpenMC environment with nuclear data installed."
        ) from e
    return openmc, IncidentNeutron, REACTION_NAME

# Lazily populated OpenMC globals (so importing this module works without OpenMC)
openmc = None
IncidentNeutron = None
REACTION_NAME = None

def _ensure_openmc_loaded() -> None:
    """
    Populate module globals (openmc, IncidentNeutron, REACTION_NAME) once, on demand.
    """
    global openmc, IncidentNeutron, REACTION_NAME
    if openmc is not None and IncidentNeutron is not None and REACTION_NAME is not None:
        return
    openmc, IncidentNeutron, REACTION_NAME = _require_openmc_data()


# -----------------------------
# Defaults (fusion energy range)
# -----------------------------
FUSION_E_MIN_eV: float = 0.0253
FUSION_E_MAX_eV: float = 16.0e6
Y_LOG_FLOOR: float = 1e-15

__all__ = [
    "FUSION_E_MIN_eV",
    "FUSION_E_MAX_eV",
    "Y_LOG_FLOOR",
    "AvailableReactions",
    "available_library_reactions",
    "available_reactions",
    "peak_cross_section",
    "cross_section_at_energy",
    "peak_xs_table",
    "find_xs_table",
    "plot_xs",
]

# ==========================================================
# CACHES (XML map cache + IncidentNeutron cache)
# ==========================================================
# xs_xml_path (str) -> {"nuclide_to_h5": dict[str, Path], "element_to_nucs": dict[str, list[str]]}
_XSXML_MAP_CACHE: dict[str, dict[str, Any]] = {}

# h5_path (str) -> IncidentNeutron | None
_INCIDENT_NEUTRON_CACHE: dict[str, IncidentNeutron | None] = {}


# -----------------------------
# Small utility types
# -----------------------------
class AvailableReactions:
    """
    Iterable container of MT integers that also prints nicely.

    - Iterate -> MT integers
    - str(obj) -> "reaction (MT=..), reaction (MT=..), ..."
    """

    def __init__(self, mts: Iterable[int], label: str = ""):
        self.mts = sorted(set(int(m) for m in mts))
        self.label = label

    def __iter__(self):
        return iter(self.mts)

    def __len__(self):
        return len(self.mts)

    def __repr__(self):
        return f"AvailableReactions(n={len(self)}, label={self.label!r})"

    def __str__(self):
        if not self.mts:
            return "<no reactions>"
        rxmap = REACTION_NAME or {}
        return ", ".join(f"{rxmap.get(mt, f'MT{mt}')} (MT={mt})" for mt in self.mts)


# -----------------------------
# Generic parsing helpers
# -----------------------------
_NUMERIC_RE = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")

_ENERGY_RE = re.compile(
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]+)?"
)

_NUCLIDE_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x) if isinstance(x, (list, tuple, set)) else [x]


def _normalize_temperature(t: Any) -> str | None:
    """
    Accepts: "900K", "900", " 900 k ", 900 (int/float)
    Returns: "900K"
    """
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return f"{int(round(t))}K"
    s = str(t).strip().replace(" ", "")
    if s.lower().endswith("k"):
        s = s[:-1]
    if _NUMERIC_RE.fullmatch(s):
        return f"{int(round(float(s)))}K"
    if s.upper().endswith("K"):
        return s.upper()
    return s


def _parse_energy_to_eV(neutron_energy: Any) -> float | None:
    """
    Accepts:
      - float/int -> assumed eV
      - strings like "1 eV", "16 keV", "12 MeV", "1.25e6 eV", "1MeV"
    Returns float (eV) or None
    """
    if neutron_energy is None:
        return None
    if isinstance(neutron_energy, (int, float)):
        return float(neutron_energy)

    s = str(neutron_energy).strip().replace(" ", "")
    m = _ENERGY_RE.fullmatch(s)
    if not m:
        raise ValueError(
            f"Could not parse neutron_energy={neutron_energy!r}. "
            "Examples: '1 MeV', '16 keV', '1.25e6 eV', 1e6"
        )

    val = float(m.group(1))
    unit = (m.group(2) or "eV").lower()

    scale = {"ev": 1.0, "kev": 1e3, "mev": 1e6}.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported energy unit '{unit}'. Use eV, keV, or MeV.")
    return val * scale


def _normalize_nuclide_name(nuclide: Any) -> str:
    # NuclideTuple / tuple-like / OpenMC objects
    if not isinstance(nuclide, str):
        if hasattr(nuclide, "name"):
            nuclide = nuclide.name
        else:
            try:
                nuclide = nuclide[0]
            except Exception:
                nuclide = str(nuclide)

    s = str(nuclide).strip().replace(" ", "").replace("-", "")
    m = _NUCLIDE_RE.match(s)
    if not m:
        return str(nuclide).strip()
    el, A = m.group(1), m.group(2)
    el = el[0].upper() + el[1:].lower()
    return f"{el}{A}"


def _xs_xml_path(xs_xml_path: str | Path | None = None) -> Path:
    if xs_xml_path is not None:
        return Path(xs_xml_path)

    openmc = _require_openmc()  # <-- ensure openmc is loaded here

    try:
        p = openmc.config["cross_sections"]
    except Exception as e:
        raise KeyError(
            "openmc.config['cross_sections'] is not set. "
            "Set OPENMC_CROSS_SECTIONS or openmc.config['cross_sections']."
        ) from e

    return Path(p)



def _is_element_symbol(s: str) -> bool:
    s = s.strip()
    return bool(re.fullmatch(r"[A-Za-z]{1,2}", s)) and not bool(re.search(r"\d", s))


# ==========================================================
# cross_sections.xml parsing + IncidentNeutron caching
# ==========================================================
def _get_xsxml_maps(
    xs_xml_path: str | Path | None = None,
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """
    Parse cross_sections.xml once and cache:
      - nuclide_to_h5 (ONLY incident-neutron nuclide entries)
      - element_to_nucs (isotopes present in xs library for each element)

    Filters out (when present in the XML):
      - thermal scattering S(a,b)
      - photon libraries
      - any non-nuclide entries
    """
    xs_xml = _xs_xml_path(xs_xml_path).resolve()
    key = str(xs_xml)

    cached = _XSXML_MAP_CACHE.get(key)
    if cached is not None:
        return cached["nuclide_to_h5"], cached["element_to_nucs"]

    if not xs_xml.exists():
        raise FileNotFoundError(f"cross_sections.xml not found: {xs_xml}")

    tree = ET.parse(xs_xml)
    root = tree.getroot()

    nuclide_to_h5: dict[str, Path] = {}
    element_to_nucs: dict[str, list[str]] = {}

    for lib in root.findall("library"):
        name = (lib.get("materials") or lib.get("name") or "").strip()
        p = (lib.get("path") or "").strip()
        libtype = (lib.get("type") or "").strip().lower()

        if not name or not p:
            continue

        # If the XML provides a type, keep ONLY neutron/incident-neutron libraries
        if libtype and libtype not in ("neutron", "incident_neutron", "incident-neutron"):
            continue

        # Keep ONLY nuclide-like names (filters out S(a,b) entries like c_H_in_H2O)
        m = _NUCLIDE_RE.match(name.replace("-", "").replace(" ", ""))
        if not m:
            continue

        h5 = (xs_xml.parent / p).resolve()
        nuclide_to_h5[name] = h5

        el = m.group(1)
        el = el[0].upper() + el[1:].lower()
        element_to_nucs.setdefault(el, []).append(name)

    for el, lst in element_to_nucs.items():
        element_to_nucs[el] = sorted(set(lst))

    _XSXML_MAP_CACHE[key] = {
        "nuclide_to_h5": nuclide_to_h5,
        "element_to_nucs": element_to_nucs,
    }
    return nuclide_to_h5, element_to_nucs


def _find_h5_for_nuclide(nuclide: str, xs_xml_path: str | Path | None = None) -> Path | None:
    nuclide_to_h5, _ = _get_xsxml_maps(xs_xml_path=xs_xml_path)
    return nuclide_to_h5.get(nuclide, None)


def _get_incident_neutron(h5_path: Path):
    """
    Load an IncidentNeutron object from an OpenMC HDF5 file with caching.

    Returns:
        IncidentNeutron on success, or None if the file is not a valid
        incident-neutron nuclide HDF5 (e.g. S(a,b), photon, etc.).
    """
    _ensure_openmc_loaded()

    k = str(Path(h5_path).resolve())
    if k in _INCIDENT_NEUTRON_CACHE:
        return _INCIDENT_NEUTRON_CACHE[k]  # may be None

    try:
        obj = IncidentNeutron.from_hdf5(k)
    except (KeyError, OSError, ValueError):
        # KeyError commonly occurs when expected attrs like 'Z'/'A' are missing.
        obj = None

    _INCIDENT_NEUTRON_CACHE[k] = obj
    return obj



# -----------------------------
# Natural isotope weights (for elements)
# -----------------------------
def _natural_isotope_weights(element_symbol: str) -> dict[str, float] | None:
    """
    Returns dict like {"Gd152": 0.002, ...} for naturally occurring isotopes,
    normalized to sum to 1.

    Uses OpenMC's NATURAL_ABUNDANCE if available.

    Notes:
    - This function assumes `_ensure_openmc_loaded()` exists and sets the module-level
      `openmc` global (and that openmc.data is importable in the OpenMC environment).
    - If OpenMC isn't available (e.g., on Windows without OpenMC), it returns None.
    """
    try:
        _ensure_openmc_loaded()  # populates openmc / openmc.data availability
    except Exception:
        return None

    el = element_symbol[0].upper() + element_symbol[1:].lower()

    try:
        from openmc.data import NATURAL_ABUNDANCE  # type: ignore
    except Exception:
        return None

    # Keep only isotopes of this element with positive abundance
    out = {k: float(v) for k, v in NATURAL_ABUNDANCE.items() if k.startswith(el) and float(v) > 0.0}
    if not out:
        return None

    s = float(sum(out.values()))
    if s <= 0.0:
        return None

    return {k: v / s for k, v in out.items()}



def _element_to_nuclides(
    element_symbol: str,
    xs_xml_path: str | Path | None = None,
    natural_only: bool = True,
) -> list[str]:
    """
    Returns isotopes *present in cross_sections.xml* for an element.

    If natural_only=True, keeps only naturally occurring isotopes (OpenMC NATURAL_ABUNDANCE) when available.
    Otherwise falls back to all isotopes present in the xs library for that element.
    """
    _, element_to_nucs = _get_xsxml_maps(xs_xml_path=xs_xml_path)

    el = element_symbol[0].upper() + element_symbol[1:].lower()
    present = list(element_to_nucs.get(el, []))

    if not natural_only:
        return present

    w = _natural_isotope_weights(el)
    if not w:
        return present

    natural_present = [n for n in present if n in w]
    return natural_present if natural_present else present


def _element_isotope_rows(element_symbol: str, xs_xml_path=None, natural_only: bool = True):
    """
    Build nuclide_rows for an element target, including natural abundance
    where available, so tables can show Frac/Type.
    """
    el = element_symbol[0].upper() + element_symbol[1:].lower()
    nucs = _element_to_nuclides(el, xs_xml_path=xs_xml_path, natural_only=natural_only)

    w = _natural_isotope_weights(el)  # dict nuclide -> fraction (sums to 1), or None
    rows = []
    for n in nucs:
        if w and n in w:
            rows.append({"name": n, "percent": 100.0 * float(w[n]), "percent_type": "ao"})
        else:
            rows.append({"name": n, "percent": float("nan"), "percent_type": ""})
    return rows


# ==========================================================
# Target resolver
# ==========================================================
def _resolve_targets(targets: Any, xs_xml_path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Accepts:
      - openmc.Material
      - "Be" (element)   -> naturally occurring isotopes by default
      - "Li-6" (nuclide)
      - list mixing Materials and strings

    Returns list of groups (dicts).
    """
    openmc = _require_openmc()  # <-- critical: ensure openmc is loaded here

    items = _as_list(targets)
    if not items:
        raise ValueError("No targets provided.")

    has_material = any(isinstance(x, openmc.Material) for x in items)
    groups: list[dict[str, Any]] = []

    if has_material:
        for item in items:
            if isinstance(item, openmc.Material):
                mat = item
                title = mat.name
                default_t = f"{int(round(mat.temperature))}K" if mat.temperature is not None else "294K"
                rows = [{"name": n.name, "percent": n.percent, "percent_type": n.percent_type} for n in mat.nuclides]
                groups.append(
                    {
                        "title": title,
                        "nuclide_rows": rows,
                        "default_xs_temp": default_t,
                        "material": mat,
                        "target_kind": "material",
                        "element_symbol": None,
                    }
                )
            else:
                s = str(item).strip()
                if _is_element_symbol(s):
                    el = s[0].upper() + s[1:].lower()
                    rows = _element_isotope_rows(el, xs_xml_path=xs_xml_path, natural_only=True)
                    groups.append(
                        {
                            "title": f"{el}",
                            "nuclide_rows": rows,
                            "default_xs_temp": "294K",
                            "material": None,
                            "target_kind": "element",
                            "element_symbol": el,
                        }
                    )
                else:
                    nuc = _normalize_nuclide_name(s)
                    rows = [{"name": nuc, "percent": float("nan"), "percent_type": ""}]
                    groups.append(
                        {
                            "title": f"{nuc}",
                            "nuclide_rows": rows,
                            "default_xs_temp": "294K",
                            "material": None,
                            "target_kind": "nuclide",
                            "element_symbol": None,
                        }
                    )
        return groups

    # only strings -> one combined group
    all_nucs: list[str] = []
    label_parts: list[str] = []
    any_element = False

    for item in items:
        s = str(item).strip()
        label_parts.append(s)
        if _is_element_symbol(s):
            any_element = True
            el = s[0].upper() + s[1:].lower()
            all_nucs.extend(_element_to_nuclides(el, xs_xml_path=xs_xml_path, natural_only=True))
        else:
            all_nucs.append(_normalize_nuclide_name(s))

    all_nucs = sorted(set(all_nucs))
    rows = []
    for n in all_nucs:
        m = re.match(r"^([A-Za-z]{1,2})(\d+)$", n)
        if m:
            el = m.group(1)[0].upper() + m.group(1)[1:].lower()
            w = _natural_isotope_weights(el)
            if w and n in w:
                rows.append({"name": n, "percent": 100.0 * float(w[n]), "percent_type": "ao"})
                continue
        rows.append({"name": n, "percent": float("nan"), "percent_type": ""})

    return [
        {
            "title": ", ".join(label_parts),
            "nuclide_rows": rows,
            "default_xs_temp": "294K",
            "material": None,
            "target_kind": "mixed" if any_element else "nuclide",
            "element_symbol": None,
        }
    ]


# ==========================================================
# Reaction/MT parsing
# ==========================================================
def _reaction_string_to_mts(data: IncidentNeutron, reaction_str: str) -> tuple[list[int], list[int]]:
    key = reaction_str.strip().lower().replace(" ", "")
    key = key.replace("(n,alpha)", "(n,a)")
    mts_global = sorted(mt for mt, name in REACTION_NAME.items() if key in name.lower().replace(" ", ""))
    mts_present = [mt for mt in mts_global if mt in data.reactions]
    return mts_global, mts_present


def _parse_mt_or_reaction(data: IncidentNeutron, reaction_or_mt: Any, nuc: str) -> tuple[int | None, str | None]:
    if isinstance(reaction_or_mt, int):
        return int(reaction_or_mt), None

    s = str(reaction_or_mt).strip()
    m = re.match(r"^(?:mt\s*=?\s*)?(\d+)$", s.lower().replace(" ", ""))
    if m:
        return int(m.group(1)), None

    mts_global, mts_present = _reaction_string_to_mts(data, s)

    if len(mts_global) == 0:
        return None, f"Requested reaction '{reaction_or_mt}'. No MT name matches in REACTION_NAME."
    if len(mts_present) == 0:
        mt_list = ", ".join(f"{mt} ({REACTION_NAME.get(mt, f'MT{mt}')})" for mt in mts_global)
        return None, f"Requested '{reaction_or_mt}' -> MT candidates [{mt_list}] not present for {nuc}."
    if len(mts_present) > 1:
        return None, (
            f"Requested '{reaction_or_mt}' is ambiguous for {nuc}: MTs present = {mts_present}. "
            "Pass an explicit MT."
        )

    return int(mts_present[0]), None


# ==========================================================
# Public: available_reactions
# ==========================================================
def _excluded_default_mts(exclude_scattering: bool = True, exclude_derived: bool = True) -> set[int]:
    excluded: set[int] = set()
    if exclude_scattering:
        excluded |= set(range(53, 92)) # 
    if exclude_derived:
        excluded |= set(range(219, 999)) # 
    return excluded


def available_library_reactions(
    xs_xml_path=None,
    max_nuclides=None,
    exclude_scattering=True,
    exclude_derived=True,
    as_available_reactions=True,
    verbose=True,
):
    """
    Scan the entire cross_sections.xml neutron library and return the union of MTs
    present across all nuclides.

    This is a *global* library query (not target-specific).
    """
    

    nuclide_to_h5, _ = _get_xsxml_maps(xs_xml_path=xs_xml_path)

    items = list(nuclide_to_h5.items())
    if max_nuclides is not None:
        items = items[: int(max_nuclides)]

    excluded = _excluded_default_mts(
        exclude_scattering=exclude_scattering,
        exclude_derived=exclude_derived,
    )

    mts = set()
    missing = 0
    skipped = 0
    
    for nuc, h5 in items:
        h5 = Path(h5)
        if not h5.exists():
            missing += 1
            continue
        data = _get_incident_neutron(h5)
        if data is None:
            missing += 1
            skipped += 1
            continue
        
        for mt in map(int, data.reactions.keys()):
            if mt not in excluded:
                mts.add(mt)


    mts = sorted(mts)

    if verbose:
        xs_xml = _xs_xml_path(xs_xml_path)
        print(f"cross_sections.xml: {xs_xml}")
        print(f"nuclides scanned: {len(items)} (missing files: {missing}, skipped non-neutron: {skipped})")
        print(f"unique MTs found: {len(mts)}")

    return AvailableReactions(mts, label="library") if as_available_reactions else mts



def available_reactions(
    targets: Any,
    temperature: Any = None,  # accepted for call consistency; MT availability is temperature-independent
    xs_xml_path: str | Path | None = None,
    max_items: int | None = None,
    exclude_scattering: bool = True,
    exclude_derived: bool = True,
) -> AvailableReactions:
    groups = _resolve_targets(targets, xs_xml_path=xs_xml_path)
    excluded = _excluded_default_mts(exclude_scattering=exclude_scattering, exclude_derived=exclude_derived)

    all_mts: set[int] = set()
    for g in groups:
        for row in g["nuclide_rows"]:
            nuc = _normalize_nuclide_name(row["name"])
            h5 = _find_h5_for_nuclide(nuc, xs_xml_path=xs_xml_path)
            if h5 is None or not Path(h5).exists():
                continue
            data = _get_incident_neutron(h5)
            if data is None:
                continue
            for mt in map(int, data.reactions.keys()):
                if mt not in excluded:
                    all_mts.add(mt)


    mts = sorted(all_mts)
    if max_items is not None and len(mts) > max_items:
        mts = mts[:max_items]

    return AvailableReactions(mts, label=str(targets))


# ==========================================================
# Core: load native tabulated points for one nuclide + one MT
# ==========================================================
def _load_rx_xs_points(
    nuclide: Any,
    reaction_or_mt: Any,
    temperature: Any,
    energy_min_eV: float,
    energy_max_eV: float,
    xs_xml_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, int, str, str] | None:
    nuc = _normalize_nuclide_name(nuclide)

    h5 = _find_h5_for_nuclide(nuc, xs_xml_path=xs_xml_path)
    if h5 is None or not Path(h5).exists():
        return None

    data = _get_incident_neutron(h5)
    if data is None:
        return None
    
    mt, _ = _parse_mt_or_reaction(data, reaction_or_mt, nuc)
    if mt is None:
        return None

    rx = data.reactions.get(mt)
    if rx is None or rx.xs is None or len(rx.xs) == 0:
        return None

    temps = list(rx.xs.keys())
    T_req = _normalize_temperature(temperature) or ("294K" if "294K" in temps else temps[0])
    T_use = T_req if T_req in rx.xs else temps[0]

    xs_tab = rx.xs[T_use]
    E = np.asarray(xs_tab.x, float)
    y = np.asarray(xs_tab.y, float)

    mask = (E >= energy_min_eV) & (E <= energy_max_eV)
    if not np.any(mask):
        return None

    return E[mask], y[mask], int(mt), REACTION_NAME.get(int(mt), f"MT{int(mt)}"), T_use


def _interp_to(E_src_eV: np.ndarray, y_src: np.ndarray, E_dst_eV: np.ndarray) -> np.ndarray:
    E_src_eV = np.asarray(E_src_eV, float)
    y_src = np.asarray(y_src, float)
    E_dst_eV = np.asarray(E_dst_eV, float)
    return np.interp(E_dst_eV, E_src_eV, y_src, left=y_src[0], right=y_src[-1])


def _atom_fractions_for_group(group: dict[str, Any]) -> dict[str, float]:
    """
    Returns dict nuclide -> atom_fraction for scaling.

    - material: derived from atom densities
    - element:  natural abundance (if available)
    - else:     uniform
    """
    mat = group.get("material", None)
    nucs = [_normalize_nuclide_name(r["name"]) for r in group["nuclide_rows"]]

    if isinstance(mat, openmc.Material):
        nd = mat.get_nuclide_atom_densities()
        nd2 = {_normalize_nuclide_name(k): float(v) for k, v in nd.items()}
        vals = np.array([nd2.get(n, 0.0) for n in nucs], dtype=float)
        s = float(vals.sum())
        if s > 0:
            return {n: nd2.get(n, 0.0) / s for n in nucs}

    if group.get("target_kind") == "element" and group.get("element_symbol"):
        w = _natural_isotope_weights(group["element_symbol"])
        if w:
            ww = {n: float(w.get(n, 0.0)) for n in nucs}
            s = sum(ww.values())
            if s > 0:
                return {n: ww[n] / s for n in nucs}

    if len(nucs) == 0:
        return {}
    u = 1.0 / len(nucs)
    return {n: u for n in nucs}


# ==========================================================
# Peak XS and XS-at-energy for one nuclide + one MT
# ==========================================================
def peak_cross_section(
    nuclide: Any,
    reaction_or_mt: Any,
    temperature: str | None = None,
    energy_min_eV: float = FUSION_E_MIN_eV,
    energy_max_eV: float = FUSION_E_MAX_eV,
    xs_xml_path: str | Path | None = None,
) -> dict[str, Any] | None:
    temperature = _normalize_temperature("294K" if temperature is None else temperature)

    out = _load_rx_xs_points(
        nuclide, reaction_or_mt, temperature, energy_min_eV, energy_max_eV, xs_xml_path=xs_xml_path
    )
    if out is None:
        return None

    E_eV, xs_b, mt, rxname, T_use = out
    i_peak = int(np.asarray(xs_b).argmax())

    return {
        "nuclide": _normalize_nuclide_name(nuclide),
        "mt": int(mt),
        "reaction_name": rxname,
        "temperature": T_use,
        "E_eV_peak": float(E_eV[i_peak]),
        "E_MeV_peak": float(E_eV[i_peak]) / 1e6,
        "xs_b_peak": float(xs_b[i_peak]),
        "energy_min_eV": float(energy_min_eV),
        "energy_max_eV": float(energy_max_eV),
    }


def cross_section_at_energy(
    nuclide: Any,
    reaction_or_mt: Any,
    temperature: Any,
    neutron_energy: Any,
    xs_xml_path: str | Path | None = None,
) -> dict[str, Any] | None:
    nuc = _normalize_nuclide_name(nuclide)

    h5 = _find_h5_for_nuclide(nuc, xs_xml_path=xs_xml_path)
    if h5 is None or not Path(h5).exists():
        return None

    data = _get_incident_neutron(h5)
    if data is None:
        return None
    
    mt, _ = _parse_mt_or_reaction(data, reaction_or_mt, nuc)
    if mt is None:
        return None

    rx = data.reactions.get(mt)
    if rx is None or rx.xs is None or len(rx.xs) == 0:
        return None

    temperature = _normalize_temperature(temperature) or "294K"
    temps = list(rx.xs.keys())
    if temperature not in rx.xs:
        temperature = temps[0]

    E0 = _parse_energy_to_eV(neutron_energy)
    if E0 is None:
        return None

    xs_tab = rx.xs[temperature]

    try:
        xs_val = float(xs_tab(E0))
    except Exception:
        E = np.asarray(xs_tab.x, float)
        y = np.asarray(xs_tab.y, float)
        if E0 < E.min() or E0 > E.max():
            return None
        xs_val = float(np.interp(E0, E, y))

    return {
        "nuclide": nuc,
        "mt": int(mt),
        "reaction_name": REACTION_NAME.get(int(mt), f"MT{int(mt)}"),
        "temperature": temperature,
        "E_eV": float(E0),
        "E_MeV": float(E0) / 1e6,
        "xs_b": float(xs_val),
    }


# ==========================================================
# AUTO MAJOR-REACTION SELECTION + RANKING FOR TABLES
# ==========================================================
def _collect_available_mts_for_nuclide(
    nuc: str,
    excluded_mts: set[int],
    xs_xml_path: str | Path | None = None,
) -> list[int]:
    h5 = _find_h5_for_nuclide(nuc, xs_xml_path=xs_xml_path)
    if h5 is None or not Path(h5).exists():
        return []
    data = _get_incident_neutron(h5)
    if data is None:
        return []
    return [int(mt) for mt in map(int, data.reactions.keys()) if int(mt) not in excluded_mts]


def _auto_major_mts_for_group(group: dict[str, Any], xs_xml_path: str | Path | None = None) -> list[int]:
    excluded = _excluded_default_mts()
    mts: set[int] = set()
    for row in group["nuclide_rows"]:
        nuc = _normalize_nuclide_name(row["name"])
        for mt in _collect_available_mts_for_nuclide(nuc, excluded, xs_xml_path=xs_xml_path):
            mts.add(mt)
    return sorted(mts)


def _rank_rows_for_group(
    group: dict[str, Any],
    temperature: Any,
    reactions_or_mts: Any = None,  # None => auto major reactions
    neutron_energy: Any = None,  # None => rank by peak over range; else rank at point energy
    energy_min_eV: float = FUSION_E_MIN_eV,
    energy_max_eV: float = FUSION_E_MAX_eV,
    top_n: int | None = None,  # None => keep all
    xs_xml_path: str | Path | None = None,
    scale_by_atom_fraction: bool | None = None,  # None => defaults depend on target_kind
) -> tuple[list[dict[str, Any]], str, bool, bool]:
    """
    Returns:
      rows: list of dicts for printing
      requested_label: string for header
      scaled: bool (whether scaling was applied)
      is_point: bool (whether neutron_energy was provided)
    """
    T = _normalize_temperature("294K" if temperature is None else temperature)
    if T is None:
        T = "294K"

    kind = group.get("target_kind")
    if scale_by_atom_fraction is None:
        scale_by_atom_fraction = (kind == "material")

    # requested MT list
    if reactions_or_mts is None:
        req_mts = _auto_major_mts_for_group(group, xs_xml_path=xs_xml_path)
        requested_label = "major reactions"
    elif isinstance(reactions_or_mts, AvailableReactions):
        req_mts = list(reactions_or_mts)
        requested_label = str(reactions_or_mts)
    else:
        req_mts = _as_list(reactions_or_mts)
        requested_label = ", ".join(str(x) for x in req_mts)

    weights = _atom_fractions_for_group(group) if scale_by_atom_fraction else {}
    E0 = _parse_energy_to_eV(neutron_energy)

    rows: list[dict[str, Any]] = []

    for row in group["nuclide_rows"]:
        nuc = _normalize_nuclide_name(row["name"])
        frac = row["percent"]
        ptype = row["percent_type"]
        w = float(weights.get(nuc, 1.0)) if scale_by_atom_fraction else 1.0

        # Resolve req list per nuclide if reaction strings
        resolved: list[tuple[Any, int]] = []
        if req_mts and all(isinstance(x, int) for x in req_mts):
            resolved = [(int(mt), int(mt)) for mt in req_mts]  # (req, mt)
        else:
            h5 = _find_h5_for_nuclide(nuc, xs_xml_path=xs_xml_path)
            if h5 is None or not Path(h5).exists():
                continue
            data = _get_incident_neutron(h5)
            if data is None:
                continue
            
            for req in req_mts:
                mt, _ = _parse_mt_or_reaction(data, req, nuc)
                if mt is not None:
                    resolved.append((req, int(mt)))

        for req, mt in resolved:
            if E0 is None:
                pk = peak_cross_section(
                    nuc,
                    mt,
                    temperature=T,
                    energy_min_eV=energy_min_eV,
                    energy_max_eV=energy_max_eV,
                    xs_xml_path=xs_xml_path,
                )
                if pk is None:
                    continue
                xs_val = float(pk["xs_b_peak"])
                xs_scaled = (w * xs_val) if scale_by_atom_fraction else xs_val
                metric = xs_scaled
                
                rows.append(
                    {
                        "nuclide": nuc,
                        "frac": frac,
                        "ptype": ptype,
                        "req": req,
                        "mt": int(pk["mt"]),
                        "rxname": pk["reaction_name"],
                        "xs_b": xs_val,               # raw microscopic peak
                        "xs_b_scaled": xs_scaled,     # scaled (or raw if not scaling)
                        "E_MeV": float(pk["E_MeV_peak"]),
                        "T": pk["temperature"],
                        "metric": metric,
                        "mode": "peak",
                    }
                )
            else:
                cs = cross_section_at_energy(
                    nuc, mt,
                    temperature=T,
                    neutron_energy=E0,
                    xs_xml_path=xs_xml_path
                )
                if cs is None:
                    continue
            
                xs_val = float(cs["xs_b"])
                if (not np.isfinite(xs_val)) or (xs_val <= 0.0):
                    continue
                
                xs_scaled = (w * xs_val) if scale_by_atom_fraction else xs_val
                metric = xs_scaled  # rank by what you are actually showing
                
                rows.append({
                    "nuclide": nuc,
                    "frac": frac,
                    "ptype": ptype,
                    "req": req,
                    "mt": int(cs["mt"]),
                    "rxname": cs["reaction_name"],
                    "xs_b": xs_val,                # raw microscopic XS
                    "xs_b_scaled": xs_scaled,      # scaled (or same as raw if not scaling)
                    "E_MeV": float(cs["E_MeV"]),
                    "T": cs["temperature"],
                    "metric": metric,
                    "mode": "point",
})



    rows.sort(key=lambda r: r["metric"], reverse=True)
    if top_n is not None:
        rows = rows[: int(top_n)]
    return rows, requested_label, bool(scale_by_atom_fraction), (E0 is not None)


# ==========================================================
# Tables
# ==========================================================
def peak_xs_table(
    targets: Any,
    reactions_or_mts: Any = None,
    temperature: Any = None,
    energy_min_eV: float = FUSION_E_MIN_eV,
    energy_max_eV: float = FUSION_E_MAX_eV,
    xs_xml_path: str | Path | None = None,
    top_n: int | None = None,  # None => print all
    scale_by_atom_fraction: bool | None = None,
) -> None:
    groups = _resolve_targets(targets, xs_xml_path=xs_xml_path)
    temps_in = _as_list(temperature) if temperature is not None else None

    for g in groups:
        temp_list = (
            [_normalize_temperature(g["default_xs_temp"])]
            if temps_in is None
            else [_normalize_temperature(t) for t in temps_in]
        )
        temp_list = [t for t in temp_list if t is not None]

        for T in temp_list:
            rows, requested_label, scaled, _ = _rank_rows_for_group(
                g,
                temperature=T,
                reactions_or_mts=reactions_or_mts,
                neutron_energy=None,
                energy_min_eV=energy_min_eV,
                energy_max_eV=energy_max_eV,
                top_n=top_n,
                xs_xml_path=xs_xml_path,
                scale_by_atom_fraction=scale_by_atom_fraction,
            )

            scale_txt = " (scaled by atom fraction)" if scaled else ""

            xs_col = "XS_peak (b)×a" if scaled else "XS_peak (b)"
            header = (
                f"{'Nuclide':<8} "
                f"{'Frac':>10} "
                f"{'Type':>6} "
                f"{'Requested':<16} "
                f"{'MT':>4} "
                f"{'Reaction':<16} "
                f"{xs_col:>14} "
                f"{'E_peak (MeV)':>16} "
                f"{'XS_T':>8}"
            )

            print(f"\n{g['title']} | Requested: {requested_label}{scale_txt} | XS temperature: {T}")
            print(header)
            print("-" * len(header))

            for r in rows:
                frac = r["frac"]
                frac_str = f"{frac:10.4g}" if frac == frac else f"{'':>10}"
                xs_to_print = r.get("xs_b_scaled", r["xs_b"]) if scaled else r["xs_b"]

                print(
                    f"{r['nuclide']:<8} "
                    f"{frac_str} "
                    f"{r['ptype']:>6} "
                    f"{str(r['req']):<16} "
                    f"{r['mt']:4d} "
                    f"{r['rxname']:<16} "
                    f"{xs_to_print:14.4g} "
                    f"{r['E_MeV']:16.5g} "
                    f"{r['T']:>8}"
                )



def find_xs_table(
    targets: Any,
    reactions_or_mts: Any = None,
    temperature: Any = None,
    neutron_energy: Any = None,
    energy_min_eV: float = FUSION_E_MIN_eV,
    energy_max_eV: float = FUSION_E_MAX_eV,
    xs_xml_path: str | Path | None = None,
    top_n: int | None = None,  # None => print all
    scale_by_atom_fraction: bool | None = None,
) -> None:
    groups = _resolve_targets(targets, xs_xml_path=xs_xml_path)
    temps_in = _as_list(temperature) if temperature is not None else None
    E0 = _parse_energy_to_eV(neutron_energy)

    for g in groups:
        temp_list = (
            [_normalize_temperature(g["default_xs_temp"])]
            if temps_in is None
            else [_normalize_temperature(t) for t in temps_in]
        )
        temp_list = [t for t in temp_list if t is not None]

        for T in temp_list:
            rows, requested_label, scaled, _ = _rank_rows_for_group(
                g,
                temperature=T,
                reactions_or_mts=reactions_or_mts,
                neutron_energy=E0,
                energy_min_eV=energy_min_eV,
                energy_max_eV=energy_max_eV,
                top_n=top_n,
                xs_xml_path=xs_xml_path,
                scale_by_atom_fraction=scale_by_atom_fraction,
            )

            scale_txt = " (scaled by atom fraction)" if scaled else ""

            # Build header here so it can reflect whether scaling is active
            if E0 is None:
                xs_col = "XS_peak (b)×a" if scaled else "XS_peak (b)"
                header = (
                    f"{'Nuclide':<8} "
                    f"{'Frac':>10} "
                    f"{'Type':>6} "
                    f"{'Requested':<16} "
                    f"{'MT':>4} "
                    f"{'Reaction':<16} "
                    f"{xs_col:>14} "
                    f"{'E_peak (MeV)':>16} "
                    f"{'XS_T':>8}"
                )
            else:
                xs_col = "XS (b)×a" if scaled else "XS (b)"
                header = (
                    f"{'Nuclide':<8} "
                    f"{'Frac':>10} "
                    f"{'Type':>6} "
                    f"{'Requested':<16} "
                    f"{'MT':>4} "
                    f"{'Reaction':<16} "
                    f"{xs_col:>12} "
                    f"{'E (MeV)':>10} "
                    f"{'XS_T':>8}"
                )

            if E0 is None:
                print(f"\n{g['title']} | Requested: {requested_label}{scale_txt} | XS temperature: {T}")
            else:
                print(f"\n{g['title']} | Requested: {requested_label}{scale_txt} | XS temperature: {T} | E={E0/1e6:g} MeV")

            print(header)
            print("-" * len(header))

            for r in rows:
                frac = r["frac"]
                frac_str = f"{frac:10.4g}" if frac == frac else f"{'':>10}"
                xs_to_print = r.get("xs_b_scaled", r["xs_b"]) if scaled else r["xs_b"]

                if E0 is None:
                    print(
                        f"{r['nuclide']:<8} "
                        f"{frac_str} "
                        f"{r['ptype']:>6} "
                        f"{str(r['req']):<16} "
                        f"{r['mt']:4d} "
                        f"{r['rxname']:<16} "
                        f"{xs_to_print:14.4g} "
                        f"{r['E_MeV']:16.5g} "
                        f"{r['T']:>8}"
                    )
                else:
                    print(
                        f"{r['nuclide']:<8} "
                        f"{frac_str} "
                        f"{r['ptype']:>6} "
                        f"{str(r['req']):<16} "
                        f"{r['mt']:4d} "
                        f"{r['rxname']:<16} "
                        f"{xs_to_print:12.4g} "
                        f"{r['E_MeV']:10.5g} "
                        f"{r['T']:>8}"
                    )


# ==========================================================
# Plot helpers
# ==========================================================
def _decade_ticks_from_range(vmin: float, vmax: float) -> tuple[list[float] | None, list[str] | None]:
    vmin = float(vmin)
    vmax = float(vmax)
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin <= 0 or vmax <= 0:
        return None, None
    kmin = int(np.floor(np.log10(vmin)))
    kmax = int(np.ceil(np.log10(vmax)))
    tickvals = [10.0**k for k in range(kmin, kmax + 1)]
    ticktext = [f"10<sup>{k}</sup>" for k in range(kmin, kmax + 1)]
    return tickvals, ticktext


def _xaxis_linear_MeV() -> dict[str, Any]:
    return dict(
        title=dict(text="incident neutron energy [MeV]", standoff=30),
        type="linear",
        tickformat=".3e",
        exponentformat="e",
        showexponent="all",
        automargin=True,
    )


def _xaxis_log_eV(energy_min_eV: float, energy_max_eV: float) -> dict[str, Any]:
    tickvals, ticktext = _decade_ticks_from_range(energy_min_eV, energy_max_eV)
    return dict(
        title=dict(text="incident neutron energy [eV]", standoff=30),
        type="log",
        tickmode="array" if tickvals else "auto",
        tickvals=tickvals,
        ticktext=ticktext,
        tickformat=None,
        exponentformat="e",
        showexponent="all",
        automargin=True,
    )


def _yaxis_log_decades(y_arrays: list[np.ndarray], title_text: str, floor: float = Y_LOG_FLOOR) -> dict[str, Any]:
    ys: list[np.ndarray] = []
    for y in y_arrays:
        y = np.asarray(y, float)
        y = y[np.isfinite(y) & (y > 0)]
        if y.size:
            ys.append(y)

    if not ys:
        y_min = float(floor)
        y_max = y_min * 10.0
        tickvals, ticktext = _decade_ticks_from_range(y_min, y_max)
        return dict(
            type="log",
            title=dict(text=title_text),
            tickmode="array" if tickvals else "auto",
            tickvals=tickvals,
            ticktext=ticktext,
            tickformat=None,
            exponentformat="e",
            showexponent="all",
            range=[np.log10(y_min), np.log10(y_max)],
            autorange=False,
            automargin=True,
        )

    y_min_data = float(np.min(np.concatenate(ys)))
    y_max_data = float(np.max(np.concatenate(ys)))

    y_min = max(float(floor), y_min_data)
    y_max = y_max_data if y_max_data > y_min else (y_min * 10.0)

    tickvals, ticktext = _decade_ticks_from_range(y_min, y_max)

    return dict(
        type="log",
        title=dict(text=title_text),
        tickmode="array" if tickvals else "auto",
        tickvals=tickvals,
        ticktext=ticktext,
        tickformat=None,
        exponentformat="e",
        showexponent="all",
        range=[np.log10(y_min), np.log10(y_max)],
        autorange=False,
        automargin=True,
    )


def _join_english(items: Iterable[Any]) -> str:
    items = [str(x) for x in items if str(x).strip()]
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# ==========================================================
# Plot API
# ==========================================================
def plot_xs(
    targets: Any,
    reactions_or_mts: Any = None,  # None => auto top 10 major reactions
    temperature: Any = None,  # default 294K
    energy_min_eV: float = FUSION_E_MIN_eV,
    energy_max_eV: float = FUSION_E_MAX_eV,
    top_n: int = 10,
    xs_xml_path: str | Path | None = None,
    scale_by_atom_fraction: bool = True,
    verbose: bool = True,
) -> None:
    """
    Plots:
      - microscopic (nuclide) XS
      - microscopic × atom fraction
      - macroscopic (sum of weighted nuclide XS)

    Features:
      - Uses native tabulated points (no sampling on displayed curves)
      - X toggle: linear in MeV, log in eV (updates axis + trace x arrays)
      - Y toggle: log decades 10^k (with floor), or linear sci notation
      - Mode toggle: micro / micro×atom / macro
      - If reactions_or_mts is omitted, it plots the top N dominant reactions for each mode (N=top_n).

    """
    groups = _resolve_targets(targets, xs_xml_path=xs_xml_path)
    T = _normalize_temperature("294K" if temperature is None else temperature) or "294K"

    for group in groups:
        target_label = group["title"]
        nucs = [_normalize_nuclide_name(r["name"]) for r in group["nuclide_rows"]]

        # Single-nuclide target (e.g. "Gd157") => mode buttons are redundant
        single_nuclide = (group.get("target_kind") == "nuclide" and len(nucs) == 1)
        
        if verbose:
            print(f"[plot] Reading nuclide data for {target_label} at {T}...")

        weights = _atom_fractions_for_group(group) if scale_by_atom_fraction else {n: 1.0 for n in nucs}

        # requested MTs
        if reactions_or_mts is None:
            req_mts = _auto_major_mts_for_group(group, xs_xml_path=xs_xml_path)
            requested_is_auto = True
        elif isinstance(reactions_or_mts, AvailableReactions):
            req_mts = list(reactions_or_mts)
            requested_is_auto = False
        else:
            req_mts = _as_list(reactions_or_mts)
            requested_is_auto = False

        if not req_mts:
            print(f"{target_label}: no reactions/MTs to plot.")
            continue

        micro_curves: list[dict[str, Any]] = []
        scaled_curves: list[dict[str, Any]] | None = [] if not single_nuclide else None
        macro_components: dict[int, dict[str, Any]] | None = {} if not single_nuclide else None

        for nuc in nucs:
            # resolve req list per nuclide if reaction strings
            if req_mts and all(isinstance(x, int) for x in req_mts):
                resolved = [(int(mt), int(mt)) for mt in req_mts]
            else:
                h5 = _find_h5_for_nuclide(nuc, xs_xml_path=xs_xml_path)
                if h5 is None or not Path(h5).exists():
                    continue
                data = _get_incident_neutron(h5)
                if data is None:
                    continue
                resolved = []
                for req in req_mts:
                    mt, _ = _parse_mt_or_reaction(data, req, nuc)
                    if mt is not None:
                        resolved.append((req, int(mt)))

            w = float(weights.get(nuc, 1.0))

            for req, mt in resolved:
                out = _load_rx_xs_points(nuc, mt, T, energy_min_eV, energy_max_eV, xs_xml_path=xs_xml_path)
                if out is None:
                    continue
                E_eV, xs_b, mt_res, rxname, _T_use = out

                micro_curves.append({
                    "nuc": nuc, "req": req, "mt": mt_res, "rxname": rxname,
                    "E_eV": E_eV, "y": xs_b
                })
                
                if not single_nuclide:
                    scaled_curves.append({
                        "nuc": nuc, "req": req, "mt": mt_res, "rxname": rxname,
                        "E_eV": E_eV, "y": w * xs_b, "w": w
                    })
                
                    macro_components.setdefault(mt_res, {"rxname": rxname, "parts": []})
                    macro_components[mt_res]["parts"].append({"E": E_eV, "y": xs_b, "w": w})
                

        if not micro_curves:
            print(f"{target_label}: no curves found at {T}.")
            continue

        def _curve_metric(curve: dict[str, Any]) -> float:
            y = np.asarray(curve["y"], float)
            y = y[np.isfinite(y) & (y > 0)]
            return float(np.nanmax(y)) if y.size else 0.0

        if requested_is_auto:
            micro_sel = sorted(micro_curves, key=_curve_metric, reverse=True)[:top_n]
            if not single_nuclide:
                scaled_sel = sorted(scaled_curves, key=_curve_metric, reverse=True)[:top_n]
        else:
            micro_sel = sorted(micro_curves, key=_curve_metric, reverse=True)
            if not single_nuclide:
                scaled_sel = sorted(scaled_curves, key=_curve_metric, reverse=True)

        macro_sel: list[dict[str, Any]] = []
        if not single_nuclide:
            for mt, info in macro_components.items():
                parts = info["parts"]
                if not parts:
                    continue
                E_union = np.unique(np.concatenate([np.asarray(p["E"], float) for p in parts]))
                E_union.sort()
                y_sum = np.zeros_like(E_union, dtype=float)
                for p in parts:
                    y_sum += p["w"] * _interp_to(p["E"], p["y"], E_union)
                macro_sel.append({"mt": mt, "rxname": info["rxname"], "E_eV": E_union, "y": y_sum})
        if not single_nuclide:
            if requested_is_auto:
                macro_sel = sorted(macro_sel, key=_curve_metric, reverse=True)[:top_n]
            else:
                macro_sel = sorted(macro_sel, key=_curve_metric, reverse=True)

       # title (centered; no "Target:" prefix anywhere)
        if reactions_or_mts is None:
            n = int(top_n) if top_n is not None else 10
            title_text = f"Neutron cross sections for {target_label} at {T}."
            suffix = f" top {n}"
        else:
            mts = sorted(set([c["mt"] for c in micro_sel] + [c["mt"] for c in macro_sel]))
            rx_mt = [f"{REACTION_NAME.get(mt, f'MT{mt}')} (MT={mt})" for mt in mts]
            title_text = f"{_join_english(rx_mt)} neutron cross section for {target_label} at {T}."
            suffix = ""


        # y axis configurations (each includes floor)
        y_micro_log = _yaxis_log_decades([c["y"] for c in micro_sel], "cross section [b]")
        
        if not single_nuclide:
            y_scaled_log = _yaxis_log_decades([c["y"] for c in scaled_sel], "cross section [b]")
            y_macro_log = _yaxis_log_decades([c["y"] for c in macro_sel], "cross section [b]")

        fig = go.Figure()

        x_mev_all: list[np.ndarray] = []
        x_ev_all: list[np.ndarray] = []
        idx_micro: list[int] = []
        idx_scaled: list[int] = []
        idx_macro: list[int] = []

        def _add_trace(E_eV: np.ndarray, y: np.ndarray, name: str, visible: bool):
            E_eV = np.asarray(E_eV, float)
            x_ev = E_eV
            x_mev = E_eV / 1e6
            fig.add_trace(go.Scatter(x=x_mev, y=y, mode="lines", name=name, visible=visible))
            x_mev_all.append(x_mev)
            x_ev_all.append(x_ev)

        for c in micro_sel:
            _add_trace(c["E_eV"], c["y"], f"{c['nuc']} {c['rxname']} (MT={c['mt']})", True)
            idx_micro.append(len(fig.data) - 1)

        if not single_nuclide:
            for c in scaled_sel:
                _add_trace(
                    c["E_eV"],
                    c["y"],
                    f"{c['nuc']} a·{c['rxname']} (a={c.get('w',0):.3g}, MT={c['mt']})",
                    False,
                )
                idx_scaled.append(len(fig.data) - 1)
    
            for c in macro_sel:
                _add_trace(c["E_eV"], c["y"], f"{c['rxname']} (MT={c['mt']})", False)
                idx_macro.append(len(fig.data) - 1)

        n_tr = len(fig.data)
        vis_micro = [False] * n_tr
        vis_scaled = [False] * n_tr
        vis_macro = [False] * n_tr
        for i in idx_micro:
            vis_micro[i] = True
        for i in idx_scaled:
            vis_scaled[i] = True
        for i in idx_macro:
            vis_macro[i] = True

        if not single_nuclide:
            mode_buttons = [
                dict(
                    label=f"microscopic{suffix}",
                    method="update",
                    args=[{"visible": vis_micro}, {"yaxis": {**y_micro_log, "title": {"text": "cross section [b]"}}}],
                ),
                dict(
                    label=f"microscopic × atom fraction{suffix}",
                    method="update",
                    args=[{"visible": vis_scaled}, {"yaxis": {**y_scaled_log, "title": {"text": "cross section [b]"}}}],
                ),
                dict(
                    label=f"macroscopic{suffix}",
                    method="update",
                    args=[{"visible": vis_macro}, {"yaxis": {**y_macro_log, "title": {"text": "cross section [b]"}}}],
                ),
            ]

        x_buttons = [
            dict(label="X linear (MeV)", method="update", args=[{"x": x_mev_all}, {"xaxis": _xaxis_linear_MeV()}]),
            dict(
                label="X log (eV)",
                method="update",
                args=[{"x": x_ev_all}, {"xaxis": _xaxis_log_eV(energy_min_eV, energy_max_eV)}],
            ),
        ]

        # Y toggle based on all traces (still useful if user changed modes)
        all_y_pos: list[np.ndarray] = []
        for tr in fig.data:
            y = np.asarray(tr.y, float)
            y = y[np.isfinite(y) & (y > 0)]
            if y.size:
                all_y_pos.append(y)

        if all_y_pos:
            y_min_data = float(np.min(np.concatenate(all_y_pos)))
            y_max_data = float(np.max(np.concatenate(all_y_pos)))
            y_min = max(Y_LOG_FLOOR, y_min_data)
            y_max = y_max_data if y_max_data > y_min else y_min * 10.0
            y_tickvals, y_ticktext = _decade_ticks_from_range(y_min, y_max)
            y_log_range = [np.log10(y_min), np.log10(y_max)]
        else:
            y_tickvals, y_ticktext, y_log_range = None, None, None

        y_buttons = [
            dict(
                label="Y log",
                method="relayout",
                args=[
                    {
                        "yaxis.type": "log",
                        "yaxis.tickmode": "array" if y_tickvals else "auto",
                        "yaxis.tickvals": y_tickvals,
                        "yaxis.ticktext": y_ticktext,
                        "yaxis.tickformat": None,
                        "yaxis.exponentformat": "e",
                        "yaxis.showexponent": "all",
                        "yaxis.range": y_log_range,
                        "yaxis.autorange": False,
                    }
                ],
            ),
            dict(
                label="Y linear",
                method="relayout",
                args=[
                    {
                        "yaxis.type": "linear",
                        "yaxis.tickformat": ".2e",
                        "yaxis.exponentformat": "e",
                        "yaxis.showexponent": "all",
                        "yaxis.tickmode": "auto",
                        "yaxis.tickvals": None,
                        "yaxis.ticktext": None,
                    }
                ],
            ),
        ]

        show_mode_buttons = not single_nuclide  # <- replace single_nuclide with your actual flag if named differently
        
        # Build updatemenus dynamically
        menus = [
            # Y toggle: outside left of y-axis/title
            dict(
                type="buttons",
                direction="down",
                x=-0.25, xanchor="left",
                y=1.0,   yanchor="top",
                showactive=True,
                buttons=y_buttons,
                font=dict(size=11),
            ),
            # X toggle: below x-axis title/ticks
            dict(
                type="buttons",
                direction="right",
                x=0.50, xanchor="center",
                y=-0.18, yanchor="top",
                showactive=True,
                buttons=x_buttons,
                font=dict(size=11),
            ),
        ]
        
        # Only add the mode buttons row for material/element cases
        if show_mode_buttons:
            menus.append(
                dict(
                    type="buttons",
                    direction="right",
                    x=0.50, xanchor="center",
                    y=-0.30, yanchor="top",
                    showactive=True,
                    buttons=mode_buttons,
                    font=dict(size=11),
                )
            )
        
        # Adjust bottom margin depending on whether we have 2 rows of buttons or 1
        bottom_margin = 170 if not show_mode_buttons else 210
        
        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor="center"),
            height=650,
        
            # margins: reduce bottom space when mode buttons are removed
            margin=dict(t=90, b=bottom_margin, l=155, r=260),
        
            xaxis=_xaxis_linear_MeV(),
            yaxis=y_micro_log,
        
            legend=dict(
                orientation="v",
                x=1.02,
                xanchor="left",
                y=1.0,
                yanchor="top",
                font=dict(size=11),
                tracegroupgap=2,
                itemsizing="constant",
            ),
        
            updatemenus=menus,
        )

        fig.show()

