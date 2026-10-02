"""Chemical network definition.

A network is a set of phases (gas, silicate, metal, ...), each holding a list of
species. Stoichiometry and molar masses are derived from the chemical formulas,
so adding a species means adding one line to the network YAML file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Standard atomic weights (g/mol). The first seven match the values used in the
# original Comp_definer notebook. Override any of them in the network YAML.
ATOMIC_MASSES: dict[str, float] = {
    "H": 1.00784, "He": 4.002602, "C": 12.011, "N": 14.007, "O": 15.999,
    "Na": 22.98977, "Mg": 24.305, "Al": 26.98154, "Si": 28.0855,
    "P": 30.97376, "S": 32.06, "Cl": 35.45, "K": 39.0983, "Ca": 40.078,
    "Ti": 47.867, "Cr": 51.9961, "Mn": 54.93804, "Fe": 55.845, "Ni": 58.6934,
}

_TOKEN = re.compile(r"[A-Z][a-z]?|\(|\)|\d+")


def parse_formula(formula: str) -> dict[str, int]:
    """'Na2SiO3' -> {'Na': 2, 'Si': 1, 'O': 3}. Supports parentheses, e.g. 'Mg2(SiO4)'."""
    tokens = _TOKEN.findall(formula)
    if "".join(tokens) != formula:
        raise ValueError(f"Could not parse formula {formula!r}")
    stack: list[dict[str, int]] = [{}]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        mult = 1
        if tok == "(":
            stack.append({})
            continue
        if tok.isdigit():
            raise ValueError(f"Unexpected number in formula {formula!r}")
        if i < len(tokens) and tokens[i].isdigit():
            mult = int(tokens[i])
            i += 1
        if tok == ")":
            if len(stack) == 1:
                raise ValueError(f"Unbalanced ')' in {formula!r}")
            group = stack.pop()
        else:
            group = {tok: 1}
        for el, n in group.items():
            stack[-1][el] = stack[-1].get(el, 0) + n * mult
    if len(stack) != 1:
        raise ValueError(f"Unbalanced '(' in {formula!r}")
    return stack[0]


@dataclass
class Phase:
    name: str                     # e.g. "gas"
    column_suffix: str            # solver output column suffix, e.g. "gas" -> "H2_gas"
    moles_column: str             # solver output column of total phase moles, e.g. "Moles_atm"
    species: dict[str, dict[str, int]]   # species name -> stoichiometry

    def column(self, species: str) -> str:
        return f"{species}_{self.column_suffix}"

    @property
    def columns(self) -> list[str]:
        return [self.column(s) for s in self.species]


@dataclass
class Network:
    name: str
    phases: dict[str, Phase]
    elements: list[str]
    atomic_masses: dict[str, float] = field(default_factory=lambda: dict(ATOMIC_MASSES))
    mole_scale: float = 1e23              # solver works in units of mole_scale moles
    element_key: str = "n{el}"            # key format for element totals in chem_input.dat

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, cfg: dict) -> "Network":
        masses = dict(ATOMIC_MASSES)
        masses.update(cfg.get("atomic_masses", {}))
        phases = {}
        for pname, pcfg in cfg["phases"].items():
            species = {}
            for entry in pcfg["species"]:
                # entries are either "SiO2" or {"name": "X", "formula": "..."}
                if isinstance(entry, dict):
                    species[entry["name"]] = parse_formula(entry.get("formula", entry["name"]))
                else:
                    species[entry] = parse_formula(entry)
            phases[pname] = Phase(
                name=pname,
                column_suffix=pcfg.get("column_suffix", pname),
                moles_column=pcfg["moles_column"],
                species=species,
            )
        used = sorted({el for p in phases.values() for st in p.species.values() for el in st})
        elements = cfg.get("elements", used)
        net = cls(
            name=cfg.get("name", "unnamed"),
            phases=phases,
            elements=list(elements),
            atomic_masses=masses,
            mole_scale=float(cfg.get("mole_scale", 1e23)),
            element_key=cfg.get("element_key", "n{el}"),
        )
        net.validate()
        return net

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Network":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def validate(self) -> None:
        used = {el for p in self.phases.values() for st in p.species.values() for el in st}
        missing = used - set(self.elements)
        if missing:
            raise ValueError(f"Species contain elements not listed in 'elements': {sorted(missing)}")
        unknown = used - set(self.atomic_masses)
        if unknown:
            raise ValueError(f"No atomic mass for {sorted(unknown)}; add them under 'atomic_masses'")

    # ── chemistry ─────────────────────────────────────────────────────────
    def molar_mass(self, phase: str, species: str) -> float:
        st = self.phases[phase].species[species]
        return sum(n * self.atomic_masses[el] for el, n in st.items())

    def mean_molar_mass(self, phase: str, x: dict[str, float]) -> float:
        return sum(x[s] * self.molar_mass(phase, s) for s in self.phases[phase].species)

    def element_moles(self, phase_moles: dict[str, float],
                      fractions: dict[str, dict[str, float]]) -> dict[str, float]:
        """Total moles of each element given moles per phase and species mole fractions."""
        totals = {el: 0.0 for el in self.elements}
        for pname, phase in self.phases.items():
            n_phase = phase_moles[pname]
            for sp, st in phase.species.items():
                for el, k in st.items():
                    totals[el] += n_phase * fractions[pname][sp] * k
        return totals

    def element_mass(self, element_moles: dict[str, float]) -> float:
        return sum(n * self.atomic_masses[el] for el, n in element_moles.items())

    # ── solver I/O helpers ────────────────────────────────────────────────
    @property
    def solver_columns(self) -> list[str]:
        cols = []
        for p in self.phases.values():
            cols += [p.moles_column] + p.columns
        return cols

    def summary(self) -> str:
        lines = [f"Network '{self.name}': elements {', '.join(self.elements)}"]
        for p in self.phases.values():
            lines.append(f"  {p.name:<9} ({len(p.species):2d}): " + ", ".join(p.species))
        return "\n".join(lines)
