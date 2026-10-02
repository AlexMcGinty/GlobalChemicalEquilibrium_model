# gce_tools

Python front end for the Global Chemical Equilibrium solver.

## Layout
- `networks/*.yaml`: chemical networks (phases → species). One file per network.
- `run_*.yaml`: one model run (network, planet, initial guess, solver paths, T grid).
- `01_initial_composition.ipynb`: replaces Comp_definer.
- `02_temperature_grid.ipynb`: replaces Grid_claude.
- `gce_tools/`: the code (network, composition, io, grid).

## Changing the network
1. Copy `networks/mgsio3_h2.yaml` and add or remove species. Formulas are parsed
   automatically (`Na2SiO3`, `Mg2(SiO4)`, ...). Add new elements to `elements:`.
2. Point `network:` in your run YAML at the new file.
3. Make sure `chem_input.dat` has an `nX = ...` line for each new element.

The compiled solver must know the same species. If it doesn't, `check_columns`
stops with a list of the missing columns instead of silently dropping mass.

Copy this folder into the solver directory, or put it on your PYTHONPATH.
