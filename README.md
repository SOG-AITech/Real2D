# Real 2D Solid-State Battery Model

[English](README.md) | [简体中文](README.zh-CN.md)

This repository contains a FEniCSx implementation of a Real 2D (R2D) full-cell model for solid-state batteries. The model resolves the coupled electrochemical, phase-field, and mechanical behavior of a lithium-metal anode, a polycrystalline solid electrolyte, a carbon/solid-electrolyte composite, and a multiparticle cathode under galvanostatic cycling.

The formulation is based primarily on:

- Z.-T. Sun *et al.*, "Real 2D Galvanostatic Model: Encoding Physicochemical Heterogeneity into a Full Battery," **Physical Review Letters** 135, 068001 (2025), DOI: [10.1103/4783-dkt8](https://doi.org/10.1103/4783-dkt8).
- The accompanying Supplemental Material (`SunXuZhouBo_PRLSI25.pdf`), which provides additional details on the geometry, material parameters, free-energy formulation, boundary conditions, Butler-Volmer kinetics, and numerical implementation.

The code implements the R2D modeling strategy described in these references for two-dimensional numerical simulations. Reproducing any particular published figure requires matching the mesh, parameters, solver configuration, and post-processing procedure used for that figure.

## Repository Contents

| File | Description |
| --- | --- |
| `r2d_irregular_3_34.msh` | Gmsh two-dimensional computational mesh. Physical tags identify the material regions: `Omega1` (lithium-metal/polycrystalline-solid-electrolyte phase-field region), `Omega2` (carbon/solid-electrolyte composite), and `Omega3` (cathode particles). Meshes containing tags `101` and `102` use them for the initialization-only lithium-cap and grain regions, respectively. |
| `generate_grain_boundaries.py` | Grain preprocessing and annealing program. It reads the `.msh` mesh, initializes multiple solid-electrolyte grain phase fields `eta_i`, performs Allen-Cahn relaxation, and computes the grain-boundary indicator `B` and auxiliary fields. The final fields are written to an `.npz` file together with PNG previews. |
| `r2d.py` | Production R2D solver. It reads the `.msh` mesh and grain `.npz` file, then solves for the lithium phase field `xi`, grain fields, lithium concentration, electric potentials, displacement/stress, and cathode-particle reactions during galvanostatic charge-discharge cycling. |

The grain `.npz` file is a preprocessing result, not a replacement mesh. The production solver reads it through `--gb` and uses the grain-boundary indicator in the effective conductivity, reaction, and mechanical coupling terms.

## Requirements

Use a Python environment compatible with the installed FEniCSx/DOLFINx version. The following packages are required:

- FEniCSx / DOLFINx
- PETSc, `petsc4py`, and `mpi4py` (MUMPS is recommended)
- UFL, Basix, and NumPy
- Matplotlib for PNG previews (optional for solver execution)
- Gmsh when creating or modifying the mesh

The workflows are MPI-based. Confirm that `python`, `mpirun`/`mpiexec`, and DOLFINx can be imported and launched in the target environment before starting a production run.

## Workflow

### 1. Generate the grain-boundary fields

Run the preprocessing script from the repository root:

```bash
mpirun -np 8 python generate_grain_boundaries.py
```

The current `__main__` entry point uses the default mesh and output names. It reads `r2d_irregular_3_34.msh` and produces:

- `r2d_120um_r3_comsol_layout1.npz`: grain phase fields and the grain-boundary indicator consumed by `r2d.py`;
- `grain_boundary_previews/`: grain labels, the `B` field, auxiliary fields, and other PNG diagnostics.

To use a different mesh or output filename, change the final call to `main(msh_file="your_mesh.msh", out_file="your_grains.npz")` or add command-line parsing to the preprocessing script.

The `GrainParams` dataclass controls the number of grains, random seed, initializer (`partition_random`, `comsol_random`, or `columnar`), and annealing parameters. Regenerate the `.npz` file after changing these settings.

### 2. Run the R2D simulation

Pass both the mesh and the generated grain file explicitly:

```bash
mpirun -np 16 python r2d.py \
  --msh r2d_irregular_3_34.msh \
  --gb r2d_120um_r3_comsol_layout1.npz \
  --cycles 7 \
  --current-density 10.0
```

Alternatively, rename or copy the generated file to `r2d_irregular_3_34.npz` and use the solver defaults. Production calculations can require substantial memory and runtime; choose the MPI process count according to the available memory and PETSc/MUMPS configuration.

Useful command-line examples are:

```bash
python r2d.py --help
python r2d.py --cycles 1 --t-end 100 --discharge-time 100 --progress-only
python r2d.py --linear-solver mumps --dt-max 5
```

The `--cycles` option sets the number of charge-discharge cycles. By default, the solver uses adaptive time stepping, a Newton/SNES nonlinear solve, post-processing clipping of grain fields, and a lifecycle stop criterion that terminates the run when the lithium interface reaches the prescribed cathode-side target.

## Output

By default, `r2d.py` writes results to `r2d_comsol_lr_periodic_tagged_side_120um_preview/`, including:

- `diagnostics.csv`: accepted-step diagnostics such as voltage, state of charge, residuals, time step, and interface metrics;
- `final_state.npz`: final solution and derived fields in NumPy format;
- PNG snapshots of the lithium phase field, state of charge, electric potentials, grain-boundary field, stress, and displacement;
- optional XDMF and sampled-field outputs, enabled with `--xdmf-interval` and `--field-output-interval`.

Output directories and output intervals can be changed in `Params` or overridden from the command line. PNG files are intended for rapid visual checks; quantitative post-processing should use `final_state.npz`, CSV, or XDMF data.

## Model Overview

The implementation follows the hybrid electrode-adaptive strategy of the reference work:

1. `Omega1` uses a multigrain phase-field formulation for lithium-metal morphology, the lithium/solid-electrolyte interface, and the solid-electrolyte grain structure.
2. `Omega2` uses a continuum electrochemical description for transport through the carbon/solid-electrolyte composite.
3. `Omega3` represents individual cathode particles with a sharp-interface/particle formulation for lithium intercalation, Butler-Volmer reaction kinetics, and chemical swelling.
4. Linear elasticity computes stress in the material regions, and hydrostatic stress contributes to the mechanical overpotential in the reaction kinetics.
5. A global galvanostatic constraint couples the anode and cathode currents during charge and discharge and determines the electric potentials consistently.

## Important Notes

- The physical tags in the `.msh` file must match the conventions used by the source code. When replacing the mesh, first inspect the cell/facet tag report printed by the preprocessing or solver script.
- The grain `.npz` file must be generated from a mesh with compatible topology and dimensions. Do not reuse a grain file generated from a different mesh.
- Production output can contain many PNG and diagnostic files. Consider adding runtime output directories to `.gitignore` before publishing the repository to GitHub.
- The PRL paper and its Supplemental Material define the model basis. The actual defaults used by this repository are those specified in `GrainParams`, `Params`, and the command-line help of the source files.

## Citation

If this code or a derivative is used in academic work, cite the PRL article and its Supplemental Material. Also report the mesh, parameter set, FEniCSx/PETSc versions, linear solver, and MPI configuration used for the simulation.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
