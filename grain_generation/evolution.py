from __future__ import annotations

from pathlib import Path
from .fields import clip_eta_components
from .output import save_final_outputs, save_initial_outputs, save_timed_previews


def run_time_loop(
    context, refresh_outputs, t=0.0,
):
    params = context.config
    mesh_state = context.mesh
    field_state = context.fields
    solver_state = context.solver
    problem = solver_state.problem
    eta = field_state.eta
    eta_n = field_state.eta_n
    periodic_constraints = solver_state.periodic_constraints
    msh = mesh_state.grain_mesh
    preview_dir = Path("grain_boundary_previews")
    step = 0
    next_preview_t = params.preview_interval
    while t < params.t_end - 1.0e-14:
        problem.solve()
        reason = int(problem.solver.reason)
        its = int(problem.solver.iterations)
        if reason <= 0:
            raise RuntimeError(
                f"Grain annealing SNES failed at t={t:g}, "
                f"iterations={its}, reason={reason}"
            )

        eta.x.scatter_forward()
        if params.clip_eta_after_solve:
            clip_eta_components(eta)
        for slave, master in periodic_constraints.items():
            eta.x.array[int(slave)] = eta.x.array[int(master)]
        eta.x.scatter_forward()
        eta_n.x.array[:] = eta.x.array

        t += params.dt
        step += 1
        if t >= next_preview_t - 0.5 * params.dt:
            refresh_outputs()
            save_timed_previews(
                field_state.V_scalar,
                field_state.eta_outputs,
                field_state.B,
                field_state.ce,
                params,
                t,
                output_dir=preview_dir,
                # Submesh contains only the grain region.
            )
            next_preview_t += params.preview_interval

        if msh.comm.rank == 0:
            print(
                f"step={step}, t={t:.4e} s, SNES iterations={its}, "
                f"reason={reason}"
            )

    return t, step


def run_context(context, output_file="r2d_generate_garin.npz"):
    """Run an already assembled modular grain-generation context."""
    from .derived import update_derived_fields
    from .fields import clamp_scalar_outputs, update_scalar_functions
    mesh_state = context.mesh
    field_state = context.fields
    params = context.config
    msh = mesh_state.grain_mesh
    preview_dir = Path("grain_boundary_previews")
    if msh.comm.rank == 0:
        preview_dir.mkdir(parents=True, exist_ok=True)
    msh.comm.barrier()

    def refresh_outputs():
        update_scalar_functions(field_state.eta, field_state.V_scalar, field_state.eta_outputs)
        if params.clip_eta_after_solve:
            clamp_scalar_outputs(field_state.eta_outputs)
        update_derived_fields(mesh_state, field_state, params)

    refresh_outputs()
    save_initial_outputs(
        field_state.V_scalar,
        field_state.eta_outputs,
        field_state.B,
        field_state.ce,
        params,
        output_dir=preview_dir,
    )

    time_value, _ = run_time_loop(context, refresh_outputs)
    refresh_outputs()
    save_final_outputs(
        output_file,
        field_state.V_scalar,
        field_state.eta_outputs,
        field_state.B,
        field_state.ce,
        params,
        time_value,
    )
    return context


