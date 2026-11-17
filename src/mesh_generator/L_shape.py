"""Functions to create meshes of the classical L-shaped domain."""

import gmsh

from ..mesh import GmshContextManager, GmshOptions, Mesh


def L_shape_refine(
    mesh_size: float, nb_refinement: int, gmsh_options: GmshOptions = GmshOptions()
) -> list[Mesh]:
    """Generate a 2D triangular mesh of the classical L-shaped domain with
    multiple refinement levels.

    The domain is the square ``[-1, 1] x [-1, 1]`` minus the square
    ``[0, 1] x [-1, 0]``. The re-entrant corner of interior angle ``3*pi/2``
    sits at the origin.

    The 2D domain is named ``"Omega"`` and its boundary is named ``"Gamma"``.

    Parameters
    ----------
    mesh_size : float
        Characteristic mesh size.
    nb_refinement : int
        The number of refinement steps to apply.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    Returns
    -------
    list[Mesh]
        List of meshes, one for each refinement level.
    """
    if mesh_size <= 0:
        raise ValueError(f"mesh_size={mesh_size} must be positive.")
    if nb_refinement < 0:
        raise ValueError(f"nb_refinement={nb_refinement} must be non-negative.")

    with GmshContextManager(gmsh_options):
        points: list[int] = [
            gmsh.model.geo.add_point(0, 0, 0, mesh_size),
            gmsh.model.geo.add_point(1, 0, 0, mesh_size),
            gmsh.model.geo.add_point(1, 1, 0, mesh_size),
            gmsh.model.geo.add_point(-1, 1, 0, mesh_size),
            gmsh.model.geo.add_point(-1, -1, 0, mesh_size),
            gmsh.model.geo.add_point(0, -1, 0, mesh_size),
        ]

        lines: list[int] = [
            gmsh.model.geo.add_line(points[k], points[(k + 1) % 6]) for k in range(6)
        ]

        loop: int = gmsh.model.geo.add_curve_loop(lines)
        surface: int = gmsh.model.geo.add_plane_surface([loop])

        gmsh.model.geo.synchronize()

        gmsh.model.add_physical_group(dim=1, tags=lines, name="Gamma")
        gmsh.model.add_physical_group(dim=2, tags=[surface], name="Omega")

        gmsh.model.mesh.generate(2)

        meshes: list[Mesh] = [Mesh.from_gmsh()]
        for _ in range(nb_refinement):
            gmsh.model.mesh.refine()
            meshes.append(Mesh.from_gmsh())

    return meshes


def L_shape(
    mesh_size: float,
    gmsh_options: GmshOptions = GmshOptions(),
) -> Mesh:
    """Generate a 2D triangular mesh of the classical L-shaped domain.

    The domain is the square ``[-1, 1] x [-1, 1]`` minus the square
    ``[0, 1] x [-1, 0]``. The re-entrant corner of interior angle ``3*pi/2``
    sits at the origin.

    The 2D domain is named ``"Omega"`` and its boundary is named ``"Gamma"``.

    Parameters
    ----------
    mesh_size : float
        Characteristic mesh size.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    Returns
    -------
    Mesh
        The generated mesh.
    """
    return L_shape_refine(mesh_size, 0, gmsh_options)[0]
