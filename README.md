# R2D Polycrystalline Electrochemical-Phase-Field-Mechanical Simulation

[English](README.md) | [简体中文](README.zh-CN.md)

This project uses FEniCSx/DOLFINx, PETSc, and MPI to model grain-boundary evolution and coupled electrochemical, phase-field, and mechanical behavior in a polycrystalline material.

The repository contains two independent workflows:

1. `generate_grain_boundaries.py` generates grain-boundary and eta fields from a Gmsh mesh.
2. `r2d.py` reads the mesh and generated grain-boundary fields and runs the coupled R2D simulation.

The two programs are intentionally separate and must be run separately.

## Workflow

```text
Gmsh mesh (.msh)
        |
        v
generate_grain_boundaries.py
        |
        +--> grain-boundary / eta fields (.npz)
                         |
                         v
              r2d.py --msh mesh.msh --gb grain_boundary.npz
                         |
                         +--> PNG / XDMF / NPZ / CSV
```

## Requirements

Run the project in a FEniCSx environment containing:

- Python 3.10 or newer
- FEniCSx / DOLFINx, UFL, and Basix
- PETSc and petsc4py
- MPI and mpi4py
- NumPy
- Matplotlib for visualization output

Run commands from the repository root.

## Generate Grain-Boundary Fields

```bash
mpirun -np 2 python generate_grain_boundaries.py
```

Equivalent package entry point:

```bash
mpirun -np 2 python -m grain_generation.main
```

The number of MPI processes can be changed according to available resources. This workflow reads the `.msh` file, creates eta grain fields, applies periodic constraints, solves the Allen-Cahn evolution, computes derived grain-boundary fields, and writes `.npz` and preview files.

## Run the R2D Simulation

```bash
mpirun -np 2 python r2d.py \
  --msh r2d_irregular_3_34.msh \
  --gb r2d_irregular_3_34.npz
```

Equivalent package entry point:

```bash
mpirun -np 2 python -m r2d_simulation.main \
  --msh r2d_irregular_3_34.msh \
  --gb r2d_irregular_3_34.npz
```

The R2D workflow creates the mixed finite-element state, initializes the physical fields, assembles the coupled equations, applies boundary and periodic constraints, advances charge/discharge phases with a Restricted Newton solver, and writes diagnostics and field data.

## Repository Layout

```text
.
├── generate_grain_boundaries.py    Grain-generation entry point
├── r2d.py                          R2D simulation entry point
├── grain_generation/               Grain-generation package
└── r2d_simulation/                 R2D simulation package
```

### `grain_generation/`

| Module | Responsibility |
|---|---|
| `main.py` | Command-line entry point |
| `workflow.py` | Grain-generation workflow orchestration |
| `config.py` | Parameters, region constants, and defaults |
| `mesh.py` | Mesh loading, tags, and geometry information |
| `initialization.py` | Grain and eta initial fields |
| `fields.py` | Finite-element and eta field operations |
| `constraints.py` | Periodic constraints |
| `equations.py` | Allen-Cahn weak form and Jacobian |
| `solver.py` | Newton/PETSc solver setup |
| `evolution.py` | Time stepping and convergence handling |
| `derived.py` | Grain-boundary indicators and derived fields |
| `output.py` | PNG, SVG, and NPZ output |

### `r2d_simulation/`

| Module | Responsibility |
|---|---|
| `main.py` | Command-line entry point |
| `workflow.py` | Top-level orchestration |
| `config.py` | `Params` and command-line parameter overrides |
| `state.py` | Shared `R2DContext` and state containers |
| `setup.py` | Mesh, scalar-space, and mixed-space construction |
| `simulation_setup.py` | Binding setup results to `R2DContext` |
| `state_builder.py` | Mixed-state, test-function, and trial-function views |
| `mesh.py` | Mesh and region processing |
| `regions.py` | Subspace and degree-of-freedom mappings |
| `fields.py` | Finite-element, grain-boundary, and output fields |
| `initialization.py` | Initial physical fields |
| `constitutive.py` | Material constitutive relations |
| `equations.py` | Auxiliary fields, residuals, and Jacobians |
| `constraints.py` | Boundary conditions, periodic constraints, and inactive DOFs |
| `solver.py` | Restricted Newton, PETSc, and profiling |
| `evolution.py` | Time-error estimation, adaptive time stepping, and retry |
| `evolution_loop.py` | Phase and accepted-step scheduling |
| `evolution_callbacks.py` | Single-step diagnostics, output, and cutoff handling |
| `cycling.py` | Charge/discharge phase construction |
| `diagnostics.py` | Voltage, SOC, concentration, and cutoff diagnostics |
| `output.py` | PNG, XDMF, NPZ, CSV, and phase output |
| `electrochemistry.py` | Electrochemical helper calculations |

## Outputs

Typical outputs include grain-boundary `.npz` fields, PNG/SVG previews, final R2D `.npz` state files, XDMF/HDF5 field files, diagnostic CSV files, delta fields, phase snapshots, and Newton/time-adaptation profiling records.

Output directories and filenames are controlled by the configuration parameters.

## Modular Architecture

```text
main.py
  -> workflow.run()
       -> config / setup
       -> fields / initialization
       -> equations / constraints
       -> solver
       -> evolution_loop
            -> evolution_callbacks
            -> diagnostics
            -> output
```

`workflow.py` connects modules and builds the runtime context. Numerical equations, material relations, constraints, solver behavior, time integration, diagnostics, and file output are implemented in their respective modules. Shared R2D objects are passed through `R2DContext` and `EvolutionRuntimeContext`.

## Development and Validation

```bash
python -m py_compile r2d.py r2d_simulation/*.py grain_generation/*.py
```

```bash
mpirun -np 2 python generate_grain_boundaries.py
mpirun -np 2 python r2d.py \
  --msh r2d_irregular_3_34.msh \
  --gb r2d_irregular_3_34.npz
```

For regression testing, compare initial potentials, periodic DOF counts, active and eliminated DOFs, Newton iterations, time-step sizes, residuals, runtimes, and generated output files.

## Notes

- `generate` and `r2d` are independent workflows.
- Physical region tags in the `.msh` file must match the configuration modules.
- `--msh` and `--gb` accept absolute paths or paths relative to the repository root.
- Choose the MPI process count according to mesh size and available memory.
- Default parameter values are defined in the relevant `config.py` files.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
