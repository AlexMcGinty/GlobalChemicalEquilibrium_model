"""Post-processing of a finished surface-temperature grid.

Everything here is driven by the network YAML, so element budgets and phase
masses follow automatically if species or elements are added to the network.

Typical use::

    from gce_tools.analysis import load_run
    run = load_run("run_toi561b.yaml",
                   planet={"surface_pressure_bar": 100},
                   grid={"run_dir": ".../100_bar"})
    run.T, run.x("gas", "SiO"), run.element_mass_kg("gas", "Si")
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .composition import state_from_row
from .grid import RunConfig
from .io import read_solver_output, check_columns

# Display names for the phases in plots. Phases not listed are title-cased.
PHASE_LABELS = {"gas": "Atmosphere", "silicate": "Silicate", "metal": "Core"}

# Elements that make a gas species count as "rock vapour" rather than a volatile.
ROCK_ELEMENTS = ("Si", "Mg", "Fe", "Na", "Al", "Ca", "Ti", "K")

_T_SENTINEL = "\x00T\x00"


# ── small helpers ───────────────────────────────────────────────────────────
def phase_label(phase: str) -> str:
    return PHASE_LABELS.get(phase, phase.title())


def tex_species(name: str) -> str:
    """'SiO2' -> 'SiO$_{2}$' for plot labels."""
    return re.sub(r"(\d+)", r"$_{\1}$", name)


def tex_sci(x: float, digits: int = 1) -> str:
    """7.05e-4 -> '$7 \\times 10^{-4}$' (digits = significant figures)."""
    mant, exp = f"{x:.{max(digits - 1, 0)}e}".split("e")
    return rf"${mant} \times 10^{{{int(exp)}}}$"


def rock_vapour_species(network, phase: str = "gas",
                        rock_elements=ROCK_ELEMENTS) -> list[str]:
    """Species in `phase` containing at least one rock-forming element."""
    return [sp for sp, st in network.phases[phase].species.items()
            if any(el in st for el in rock_elements)]


def select_row(df: pd.DataFrame, tol: float, chi_col: str = "chi^2"):
    """Converged row as in gce_tools.grid (last row with chi^2 <= tol).
    If nothing converged, fall back to the minimum-chi^2 row and flag it."""
    below = df[df[chi_col] <= tol]
    if not below.empty:
        return below.iloc[-1], True
    return df.loc[df[chi_col].idxmin()], False


def find_output_files(cfg: RunConfig) -> dict[float, Path]:
    """Map surface temperature -> output file for every file in run_dir
    matching grid.output_name. Falls back to the grid temperatures if the
    name template can't be turned into a glob."""
    try:
        name = cfg.grid.output_name.format(
            planet=cfg.planet.name, network=cfg.network.name,
            Ps=cfg.planet.surface_pressure_bar, T=_T_SENTINEL)
    except (ValueError, TypeError):
        files = {float(T): cfg.output_file(T) for T in cfg.grid.temperatures}
        return {T: f for T, f in files.items() if f.exists()}

    regex = re.compile(re.escape(name).replace(re.escape(_T_SENTINEL),
                                               r"(\d+(?:\.\d+)?)") + r"$")
    files = {}
    for f in cfg.run_dir.glob(name.replace(_T_SENTINEL, "*")):
        m = regex.match(f.name)
        if m:
            files[float(m.group(1))] = f
    return dict(sorted(files.items()))


def phase_element_moles(network, phase: str, row: pd.Series) -> dict[str, float]:
    """Moles of each element in one phase (solver units) from a solver row."""
    p = network.phases[phase]
    n_phase = float(row[p.moles_column])
    out = {el: 0.0 for el in network.elements}
    for sp, st in p.species.items():
        x = float(row[p.column(sp)])
        for el, k in st.items():
            out[el] += n_phase * x * k
    return out


# ── one model run ───────────────────────────────────────────────────────────
@dataclass
class RunResult:
    cfg: RunConfig
    df: pd.DataFrame          # one row per surface temperature (see load_run)
    tol: float
    atm_phase: str = "gas"

    @property
    def network(self):
        return self.cfg.network

    @property
    def T(self) -> np.ndarray:
        return self.df["T_surface"].to_numpy()

    @property
    def amf0(self) -> float:
        """Initial atmospheric mass fraction from the run YAML."""
        return self.cfg.planet.phase_mass_fractions()[self.atm_phase]

    @property
    def Ps0(self) -> float:
        return self.cfg.planet.surface_pressure_bar

    def label(self, digits: int = 1) -> str:
        return rf"AMF$_0$ = {tex_sci(self.amf0, digits)}"

    def col(self, name: str) -> np.ndarray:
        return self.df[name].to_numpy()

    def x(self, phase: str, species: str) -> np.ndarray:
        """Mole fraction of `species` in `phase`."""
        return self.col(self.network.phases[phase].column(species))

    def partial_pressure_bar(self, species: str) -> np.ndarray:
        return self.x(self.atm_phase, species) * self.col("P_surface_bar")

    def phase_mass_kg(self, phase: str) -> np.ndarray:
        return self.col(f"mass_{phase}_kg")

    def element_mass_kg(self, phase: str, element: str) -> np.ndarray:
        return self.col(f"m_{element}_{phase}_kg")

    def element_mass_change_kg(self, phase: str) -> np.ndarray:
        """Last-minus-first temperature change of each element's mass in `phase`,
        ordered as network.elements."""
        return np.array([self.element_mass_kg(phase, el)[-1] - self.element_mass_kg(phase, el)[0]
                         for el in self.network.elements])

    def zoom(self, T_min: float, T_max: float) -> np.ndarray:
        """Boolean mask for T_min <= T <= T_max."""
        return (self.T >= T_min) & (self.T <= T_max)


def load_run(config: str | Path, planet: dict | None = None, grid: dict | None = None,
             tol: float | None = None, atm_phase: str = "gas",
             mass_warn: float = 1e-3, verbose: bool = True) -> RunResult:
    """Load every output file of one temperature grid.

    config  : run YAML (network, planet, solver, grid).
    planet  : overrides for the `planet:` block, e.g. {"surface_pressure_bar": 100}.
    grid    : overrides for the `grid:` block, e.g. {"run_dir": ".../100_bar"} or
              {"output_name": "..."} for files written with a different name.
    tol     : chi^2 tolerance; defaults to grid.tol in the YAML.

    The returned DataFrame holds the selected solver row at each temperature plus
        T_surface, converged, mass_<phase>_kg, M_total_kg, M_total_over_M_planet,
        AMF (= M_atm / M_total), P_surface_bar, and m_<El>_<phase>_kg.
    """
    cfg = RunConfig.from_yaml(config)
    if planet:
        cfg.planet = replace(cfg.planet, **planet)
    if grid:
        grid = {k: (str(v) if k == "run_dir" else v) for k, v in grid.items()}
        cfg.grid = replace(cfg.grid, **grid)
    tol = cfg.grid.tol if tol is None else tol
    net = cfg.network

    files = find_output_files(cfg)
    if not files:
        raise FileNotFoundError(
            f"No files matching '{cfg.grid.output_name}' in {cfg.run_dir}. "
            "If the files were written with a different name, pass "
            "grid={'output_name': '...'}.")

    records = []
    for T, f in files.items():
        df = read_solver_output(f)
        check_columns(net, df)
        row, ok = select_row(df, tol)
        if not ok and verbose:
            print(f"WARNING: Ps0={cfg.planet.surface_pressure_bar:g} bar, T={T:.0f} K "
                  f"not converged, min chi^2 = {row['chi^2']:.3e} (tol {tol:.1e})")

        state = state_from_row(net, row)
        rec = row.to_dict()
        rec["T_surface"] = T
        rec["converged"] = ok

        masses_kg = {p: m / 1e3 for p, m in state.phase_masses_g().items()}
        for p, m in masses_kg.items():
            rec[f"mass_{p}_kg"] = m
        rec["M_total_kg"] = sum(masses_kg.values())
        rec["M_total_over_M_planet"] = rec["M_total_kg"] / cfg.planet.mass_kg
        rec["AMF"] = masses_kg[atm_phase] / rec["M_total_kg"]
        rec["P_surface_bar"] = state.surface_pressure_bar(cfg.planet, atm_phase)

        for p in net.phases:
            for el, n in phase_element_moles(net, p, row).items():
                rec[f"m_{el}_{p}_kg"] = n * net.mole_scale * net.atomic_masses[el] / 1e3
        records.append(rec)

    out = pd.DataFrame(records).sort_values("T_surface").reset_index(drop=True)
    res = RunResult(cfg, out, tol, atm_phase)

    if verbose:
        dev = np.abs(out["M_total_over_M_planet"] - 1).max()
        print(f"Loaded {len(out)} temperatures ({out['T_surface'].min():.0f}-"
              f"{out['T_surface'].max():.0f} K) from {cfg.run_dir}  "
              f"[{out['converged'].sum()}/{len(out)} converged, AMF0 = {res.amf0:.3e}]")
        if dev > mass_warn:
            warnings.warn(f"Ps0={cfg.planet.surface_pressure_bar:g} bar: converged planet mass "
                          f"differs from the YAML mass by up to {dev:.2e} (fractional). "
                          "Check that chem_input.dat matches the run YAML.")
    return res
