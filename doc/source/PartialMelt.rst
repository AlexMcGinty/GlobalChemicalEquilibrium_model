.. _Partial_Melt_Version:

Partial Melt Version
====================

This partial-melt implementation is not described in any published papers.
``Partial_Melt_Version`` is a reduced solver for staged calculations after an initial full-melt equilibrium run.
It treats the core and crystallized silicate as inactive frozen reservoirs by progressively freezing silicate material stage by stage, removing the frozen fraction from the active equilibrium system, and solving equilibrium only for the remaining active silicate melt and gas atmosphere.
The version is derived from ``Sulfur_Nitrogen_Version`` but removes the active metal phase and adds bookkeeping variables such as ``M_frozen_core`` and ``M_frozen_solid``.
Partial-melt input files are generated through ``Example/run_partial_melt``, which converts a full-melt result into a sequence of lower melt-fraction equilibrium problems.
To run the workflow from the repository root, set the desired parameters in ``Example/run_partial_melt/run_partial_melt.py`` and execute ``PYTHONPATH=. python3 Example/run_partial_melt/run_partial_melt.py``.
Use this version through the partial-melt workflow rather than as a standalone solver setup.
