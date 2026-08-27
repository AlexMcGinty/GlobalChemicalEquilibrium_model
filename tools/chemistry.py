import numpy as np


def select_scaling_constants(planet_type: str = "Fe0675_MgSiO3_0325"):
    """
    Select scaling constants (m1, r1, k1, k2, k3) based on planet composition type.
    Data from Seager et al. (2007).
    """
    constants = {
        "Fe_gamma": {
            "m1": 4.34,
            "r1": 2.23,
            "k1": -0.20945,
            "k2": 0.0804,
            "k3": 0.394,
        },
        "MgSiO3": {
            "m1": 7.38,
            "r1": 3.58,
            "k1": -0.20945,
            "k2": 0.0804,
            "k3": 0.394,
        },
        "Fe0675_MgSiO3_0325": {
            "m1": 6.41,
            "r1": 3.19,
            "k1": -0.20945,
            "k2": 0.0804,
            "k3": 0.394,
        },
        "Fe03_MgSiO3_07": {
            "m1": 6.41,
            "r1": 2.84,
            "k1": -0.20945,
            "k2": 0.0804,
            "k3": 0.394,
        },
    }

    return constants[planet_type]


# Mixtures: [Fe, Mg, Si, O]
mixtures = {
    "Fe0675_MgSiO3_0325": np.array([0.675, 0.325, 0.325, 0.325 * 3]),
    "Fe03_MgSiO3_07": np.array([0.3, 0.7, 0.7, 0.7 * 3]),
    "Fe0225_MgSiO3_0525_H2O_025": np.array([0.225, 0.525, 0.525, 0.525 * 3 + 0.25]),
    "Fe0065_MgSiO3_0485_H2O_045": np.array([0.065, 0.485, 0.485, 0.485 * 3 + 0.45]),
    "Fe003_MgSiO3_022_H2O_075": np.array([0.03, 0.22, 0.22, 0.22 * 3 + 0.75]),
}
mixtures = {key: value / value.sum() for key, value in mixtures.items()}


def composition_from_chem_input(chem_input_path=None):
    """Read nFe, nMg, nSi, and nO from chem_input.dat and infer the mixture."""
    nFe = nMg = nSi = nO = None
    with open(chem_input_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("nFe") and "=" in line:
                nFe = float(line.split("=")[1].strip())
            elif line.startswith("nMg") and "=" in line:
                nMg = float(line.split("=")[1].strip())
            elif line.startswith("nSi") and "=" in line:
                nSi = float(line.split("=")[1].strip())
            elif line.startswith("nO") and "=" in line and not line.startswith("nO2") and not line.startswith("nO_"):
                nO = float(line.split("=")[1].strip())

    if nFe is None or nMg is None or nSi is None or nO is None:
        raise ValueError(f"Missing nFe/nMg/nSi/nO in chem_input file: {chem_input_path}")

    total = float(nFe + nMg + nSi + nO)
    composition = np.array([float(nFe) / total, float(nMg) / total, float(nSi) / total, float(nO) / total])
    distances = {name: np.linalg.norm(composition - mixture) for name, mixture in mixtures.items()}
    return min(distances, key=distances.get)