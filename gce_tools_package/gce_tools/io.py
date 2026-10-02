"""Reading and writing the solver's key = value files and its output tables."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from .composition import BulkState
from .network import Network


def set_keys(path: Path, values: dict[str, object], strict: bool = True) -> None:
    """Replace `key = ...` lines in a solver input file.

    With strict=True a key that is not found raises, so a network element that is
    missing from chem_input.dat can't be silently skipped.
    """
    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    found = set()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        for key, val in values.items():
            if re.match(rf"{re.escape(key)}\s*=", stripped):
                lines[i] = f"{key} = {val}\n"
                found.add(key)
                break
    missing = set(values) - found
    if missing:
        msg = f"Keys not found in {path.name}: {sorted(missing)}"
        if strict:
            raise KeyError(msg)
        warnings.warn(msg)
    path.write_text("".join(lines))


def write_element_totals(path: Path, state: BulkState, strict: bool = True) -> None:
    """Write nSi, nMg, ... (in solver units) into chem_input.dat."""
    net = state.network
    set_keys(path, {net.element_key.format(el=el): n for el, n in state.element_moles.items()},
             strict=strict)


def initial_values(state: BulkState) -> dict[str, float]:
    """Species fractions and phase moles keyed by solver column name."""
    vals = {}
    for pname, phase in state.network.phases.items():
        for sp in phase.species:
            vals[phase.column(sp)] = state.fractions[pname][sp]
    for pname, phase in state.network.phases.items():
        vals[phase.moles_column] = state.phase_moles[pname]
    return vals


def write_initial(path: Path, state: BulkState, update_only: bool = False) -> None:
    """Write the starting guess to initial.dat as `Column = value` lines.

    update_only=True edits matching lines of an existing file in place (keeps comments
    and any extra keys); otherwise the file is written from scratch.
    """
    vals = initial_values(state)
    if update_only:
        set_keys(path, vals, strict=True)
    else:
        Path(path).write_text("".join(f"{k} = {v}\n" for k, v in vals.items()))


def read_solver_output(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = df.columns.str.replace("#", "", regex=False).str.strip()
    return df


def check_columns(network: Network, df: pd.DataFrame) -> None:
    """Fail loudly if the solver output doesn't match the network definition."""
    missing = [c for c in network.solver_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Solver output is missing network columns: {missing}. "
                       "Is the compiled solver using a different network?")
    suffixes = tuple(f"_{p.column_suffix}" for p in network.phases.values())
    extra = [c for c in df.columns if c.endswith(suffixes) and c not in network.solver_columns]
    if extra:
        warnings.warn(f"Solver output has species not in the network (their mass is "
                      f"not counted): {extra}")


def best_converged_row(df: pd.DataFrame, tol: float, chi_col: str = "chi^2") -> pd.Series | None:
    below = df[df[chi_col] <= tol]
    return None if below.empty else below.iloc[-1]
