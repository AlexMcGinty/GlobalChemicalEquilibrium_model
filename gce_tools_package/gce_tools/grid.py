"""Run configuration and the surface-temperature grid driver."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .composition import Planet, initial_state, state_from_row, BulkState
from .io import (set_keys, write_element_totals, write_initial, read_solver_output,
                 check_columns, best_converged_row)
from .network import Network


@dataclass
class SolverConfig:
    base_dir: Path
    executable: str = "./solver"
    chem_input: str = "chem_input.dat"
    param: str = "param.dat"
    initial: str = "initial.dat"
    temperature_key: str = "T_AMOI"
    output_key: str = "Output file"
    eta_key: str = "eta"

    def path(self, name: str) -> Path:
        return self.base_dir / getattr(self, name)


@dataclass
class GridConfig:
    T_start: float
    T_end: float
    T_step: float
    run_dir: str
    tol: float = 1e-8
    output_name: str = "{planet}_{network}_Ps_{Ps:.0f}_Ts_{T}.dat"
    update_chem_input: bool = False     # carry element totals forward between T steps
    warm_start_initial: bool = False    # write last converged state to initial.dat
    stop_on_failure: bool = True
    # [[T_upper, eta], ...]; last entry may use null for "everything above"
    eta_schedule: list = field(default_factory=lambda: [[2000, 0.4], [2500, 0.2], [None, 0.1]])

    @property
    def temperatures(self) -> np.ndarray:
        return np.arange(self.T_start, self.T_end + self.T_step / 2, self.T_step)

    def eta(self, T: float) -> float:
        for upper, eta in self.eta_schedule:
            if upper is None or T < upper:
                return eta
        return self.eta_schedule[-1][1]


@dataclass
class RunConfig:
    network: Network
    planet: Planet
    guess: dict
    floor: float
    solver: SolverConfig
    grid: GridConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        cfg = yaml.safe_load(path.read_text())
        net_path = Path(cfg["network"])
        if not net_path.is_absolute():
            net_path = path.parent / net_path
        solver = dict(cfg["solver"])
        solver["base_dir"] = Path(solver["base_dir"]).expanduser()
        return cls(
            network=Network.from_yaml(net_path),
            planet=Planet(**cfg["planet"]),
            guess=cfg["initial_guess"],
            floor=float(cfg.get("floor", 1e-9)),
            solver=SolverConfig(**solver),
            grid=GridConfig(**cfg["grid"]),
        )

    def initial_state(self) -> BulkState:
        return initial_state(self.network, self.planet, self.guess, self.floor)

    @property
    def run_dir(self) -> Path:
        d = Path(self.grid.run_dir)
        return d if d.is_absolute() else self.solver.base_dir / d

    def output_file(self, T: float) -> Path:
        name = self.grid.output_name.format(
            planet=self.planet.name, network=self.network.name,
            Ps=self.planet.surface_pressure_bar, T=int(T))
        return self.run_dir / name


def _converged(cfg: RunConfig, out_file: Path) -> pd.Series | None:
    if not out_file.exists():
        return None
    df = read_solver_output(out_file)
    check_columns(cfg.network, df)
    return best_converged_row(df, cfg.grid.tol)


def run_solver_at(cfg: RunConfig, T: float) -> pd.Series | None:
    """Run the solver once at temperature T. Returns the converged row or None."""
    s = cfg.solver
    out_file = cfg.output_file(T)
    set_keys(s.path("param"), {s.eta_key: cfg.grid.eta(T)}, strict=False)
    set_keys(s.path("chem_input"), {s.temperature_key: float(T)})
    set_keys(s.path("param"), {s.output_key: out_file})
    out_file.unlink(missing_ok=True)

    subprocess.run([s.executable], cwd=s.base_dir, check=False)

    if not out_file.exists():
        raise RuntimeError(f"[T={int(T)}K] Solver produced no output file.")
    row = _converged(cfg, out_file)
    if row is None:
        best = read_solver_output(out_file)["chi^2"].min()
        print(f"  [T={int(T)}K] not converged: best chi^2 = {best:.3e} (tol {cfg.grid.tol:.1e})")
    else:
        print(f"  [T={int(T)}K] chi^2 = {row['chi^2']:.3e}  converged")
    return row


def run_grid(cfg: RunConfig) -> pd.DataFrame:
    """Sweep the temperature grid. Returns one row per converged temperature,
    with the solver output plus derived element totals."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for T in cfg.grid.temperatures:
        row = _converged(cfg, cfg.output_file(T))
        if row is not None:
            print(f"[T={int(T)}K] already converged, skipping")
        else:
            print(f"\n[T={int(T)}K] running solver...")
            row = run_solver_at(cfg, T)
            if row is None:
                if cfg.grid.stop_on_failure:
                    raise RuntimeError(f"Solver did not converge at T={int(T)}K")
                continue

        state = state_from_row(cfg.network, row)
        if cfg.grid.update_chem_input:
            write_element_totals(cfg.solver.path("chem_input"), state)
        if cfg.grid.warm_start_initial:
            write_initial(cfg.solver.path("initial"), state, update_only=True)

        rec = row.to_dict()
        rec["T_surface"] = float(T)
        rec.update({cfg.network.element_key.format(el=el): n
                    for el, n in state.element_moles.items()})
        # mass-conservation diagnostics
        rec["M_total_over_M_planet"] = state.total_mass_g() / cfg.planet.mass_g
        rec["P_surface_bar"] = state.surface_pressure_bar(cfg.planet)
        for p, m in state.phase_masses_g().items():
            rec[f"mass_frac_{p}"] = m / cfg.planet.mass_g
        results.append(rec)

    print("\nDone.")
    return pd.DataFrame(results)

