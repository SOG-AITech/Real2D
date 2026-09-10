"""Bind construction results into the shared :class:`R2DContext`."""


def attach_mesh_field_setup(context, setup):
    """Attach mesh, finite-element, and state objects to the context.

    The returned mapping is the compatibility view used while assembling UFL
    forms.  The authoritative objects remain available through ``context``.
    """
    context.runtime["simulation_setup"] = setup
    context.mesh.mesh = setup["msh"]
    context.mesh.cell_tags = setup["cell_tags"]
    context.mesh.facet_tags = setup["facet_tags"]
    context.mesh.dx = setup["dx"]
    context.mesh.ds = setup["ds"]
    context.mesh.dS = setup["dS"]
    context.mesh.cell_markers = setup["domain_markers"]
    context.fields.V_scalar = setup["V_scalar"]
    context.fields.ME = setup["ME"]
    context.fields.fields.update({
        "stats_dofs": setup["stats_dofs"],
        "boundary_probe_dofs": setup["boundary_probe_dofs"],
        "eta_regions": setup["eta_regions"],
        "n_grains": setup["n_grains"],
        "eta_initial_values": setup["eta_initial_values"],
        "state": setup["w"], "previous_state": setup["w_n"],
        "previous_previous_state": setup["w_nm1"], "older_state": setup["w_nm2"],
    })
    context.runtime["mesh_fields"] = {
        "mesh": setup["msh"], "V_scalar": setup["V_scalar"],
        "ME": setup["ME"], "state": setup["w"],
        "previous_state": setup["w_n"],
        "previous_previous_state": setup["w_nm1"], "older_state": setup["w_nm2"],
    }
    return setup


def context_view(context, name):
    """Return a named setup section from the context runtime registry."""
    return context.runtime.get(name, {})


__all__ = ["attach_mesh_field_setup", "context_view"]
