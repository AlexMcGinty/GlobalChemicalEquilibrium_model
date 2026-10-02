"""Planet bulk composition: initial guesses and element budgets."""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from .network import Network

G = 6.67430e-11          # m^3 kg^-1 s^-2
M_EARTH_KG = 5.97e24
R_EARTH_M = 6371e3


@dataclass
class Planet:
    name: str
    mass_earth: float
    radius_earth: float
    surface_pressure_bar: float
    metal_mass_fraction: float
    atm_mass_fraction: float | None = None   # None -> derived from surface pressure

    @property
    def mass_kg(self) -> float:
        return self.mass_earth * M_EARTH_KG

    @property
    def mass_g(self) -> float:
        return self.mass_kg * 1e3

    @property
    def gravity(self) -> float:
        return G * self.mass_kg / (self.radius_earth * R_EARTH_M) ** 2

    @property
    def area_m2(self) -> float:
        return 4 * math.pi * (self.radius_earth * R_EARTH_M) ** 2

    def phase_mass_fractions(self) -> dict[str, float]:
        """Mass fractions keyed by phase name (gas / silicate / metal)."""
        amf = self.atm_mass_fraction
        if amf is None:
            amf = self.surface_pressure_bar * 1e5 * self.area_m2 / (self.mass_kg * self.gravity)
        smf = 1.0 - amf - self.metal_mass_fraction
        if smf <= 0:
            raise ValueError(f"Silicate mass fraction is {smf:.3g}; check metal/atm fractions")
        return {"gas": amf, "silicate": smf, "metal": self.metal_mass_fraction}


@dataclass
class BulkState:
    """Phase moles + species mole fractions + derived element totals.

    All mole quantities are stored in solver units (i.e. divided by network.mole_scale).
    """
    network: Network
    phase_moles: dict[str, float]
    fractions: dict[str, dict[str, float]]

    @property
    def element_moles(self) -> dict[str, float]:
        return self.network.element_moles(self.phase_moles, self.fractions)

    def phase_masses_g(self) -> dict[str, float]:
        s = self.network.mole_scale
        return {p: self.phase_moles[p] * s * self.network.mean_molar_mass(p, self.fractions[p])
                for p in self.network.phases}

    def total_mass_g(self) -> float:
        """Planet mass implied by the element totals."""
        return self.network.element_mass(self.element_moles) * self.network.mole_scale

    def surface_pressure_bar(self, planet: Planet, atm_phase: str = "gas") -> float:
        """Surface pressure from the atmospheric mass, P = M_atm g / (4 pi R^2).
        Uses the nominal planet mass and radius for g."""
        m_atm_kg = self.phase_masses_g()[atm_phase] / 1e3
        return m_atm_kg * planet.gravity / planet.area_m2 / 1e5

    def ratio(self, num: str, den: str) -> float:
        el = self.element_moles
        return el[num] / el[den]

    def report(self, planet: Planet | None = None) -> str:
        net = self.network
        lines = [net.summary(), ""]
        for p, phase in net.phases.items():
            tot = sum(self.fractions[p].values())
            lines.append(f"{p}: {phase.moles_column} = {self.phase_moles[p]:.6e}  "
                         f"(sum x = {tot:.12f}, <M> = {net.mean_molar_mass(p, self.fractions[p]):.4f} g/mol)")
        lines.append("")
        lines.append("Element totals (solver units):")
        for el, n in self.element_moles.items():
            lines.append(f"  {net.element_key.format(el=el):<6} = {n:.10e}")
        if "C" in net.elements and "O" in net.elements:
            lines.append(f"C/O global = {self.ratio('C', 'O'):.4e}")
        if planet is not None:
            masses = self.phase_masses_g()
            total = net.element_mass(self.element_moles) * net.mole_scale
            lines.append("")
            for p, m in masses.items():
                lines.append(f"mass fraction {p:<9} = {m / planet.mass_g:.6e}")
            lines.append(f"sum of element masses / planet mass = {total / planet.mass_g:.10f}")
        return "\n".join(lines)


def normalise(x: dict[str, float]) -> dict[str, float]:
    tot = sum(x.values())
    if tot <= 0:
        raise ValueError("Mole fractions sum to zero")
    return {k: v / tot for k, v in x.items()}


def initial_state(network: Network, planet: Planet,
                  guess: dict[str, dict[str, float]], floor: float = 1e-9) -> BulkState:
    """Build a bulk state from a guessed composition.

    guess: {phase: {species: mole fraction}}. Species you leave out get `floor`;
    species you set explicitly to 0 stay 0. Each phase is renormalised to 1.
    """
    fractions = {}
    for pname, phase in network.phases.items():
        given = guess.get(pname, {})
        unknown = set(given) - set(phase.species)
        if unknown:
            raise KeyError(f"{sorted(unknown)} not in the '{pname}' phase of network '{network.name}'")
        fractions[pname] = normalise({s: float(given.get(s, floor)) for s in phase.species})

    mass_fracs = planet.phase_mass_fractions()
    missing = set(network.phases) - set(mass_fracs)
    if missing:
        raise KeyError(f"No mass fraction for phase(s) {sorted(missing)}")
    phase_moles = {
        p: mass_fracs[p] * planet.mass_g / network.mean_molar_mass(p, fractions[p]) / network.mole_scale
        for p in network.phases
    }
    return BulkState(network, phase_moles, fractions)


def state_from_row(network: Network, row: pd.Series) -> BulkState:
    """Rebuild a bulk state from one row of solver output."""
    phase_moles, fractions = {}, {}
    for pname, phase in network.phases.items():
        phase_moles[pname] = float(row[phase.moles_column])
        fractions[pname] = {s: float(row[phase.column(s)]) for s in phase.species}
    return BulkState(network, phase_moles, fractions)
