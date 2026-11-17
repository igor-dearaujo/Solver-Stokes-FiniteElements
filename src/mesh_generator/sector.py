"""Functions to create angular section (sector) shaped meshes."""

import gmsh
from numpy import cos, pi, sin

from ..mesh import GmshContextManager, GmshOptions, Mesh


def sector_refine(
    radius: float,
    angle: float,
    mesh_size: float,
    nb_refinement: int,
    gmsh_options: GmshOptions = GmshOptions(),
) -> list[Mesh]:
    """Generate an angular section (sector) shaped mesh with multiple refinement levels.

    Parameters
    ----------
    radius : float
        The radius of the sector.
    angle : float
        The opening angle of the sector in radians (must be in [0, 2π]).
    mesh_size : float
        The size of the mesh elements.
    nb_refinement : int
        The number of refinement steps to apply.
    gmsh_options : GmshOptions
        The GMSH options to use.

    Returns
    -------
    list[Mesh]
        List of meshes, one for each refinement level.
    """
    if radius <= 0:
        raise ValueError(f"radius={radius} must be positive.")
    if not (0 <= angle <= 2 * pi):
        raise ValueError(f"angle={angle} must be in [0, 2π].")
    if mesh_size <= 0:
        raise ValueError(f"mesh_size={mesh_size} must be positive.")
    if nb_refinement < 0:
        raise ValueError(f"nb_refinement={nb_refinement} must be non-negative.")

    with GmshContextManager(gmsh_options):
        # Angles of the arc endpoints (0 and angle) and of its midpoint (angle/2).
        pts: list[int] = [
            gmsh.model.geo.add_point(0, 0, 0, mesh_size),
            gmsh.model.geo.add_point(radius, 0, 0, mesh_size),
            gmsh.model.geo.add_point(
                radius * cos(angle / 2), radius * sin(angle / 2), 0, mesh_size
            ),
            gmsh.model.geo.add_point(
                radius * cos(angle), radius * sin(angle), 0, mesh_size
            ),
        ]

        # Each arc half sweeps angle/2 <= π, so gmsh traces the intended arc.
        arc_lines: list[int] = [
            gmsh.model.geo.add_circle_arc(pts[1], pts[0], pts[2]),
            gmsh.model.geo.add_circle_arc(pts[2], pts[0], pts[3]),
        ]

        lns: list[int] = [
            gmsh.model.geo.add_line(pts[0], pts[1]),
            *arc_lines,
            gmsh.model.geo.add_line(pts[3], pts[0]),
        ]

        wire_tag: int = gmsh.model.geo.add_curve_loop(lns)
        surfaces: list[int] = [gmsh.model.geo.add_plane_surface([wire_tag])]

        gmsh.model.geo.synchronize()

        gmsh.model.add_physical_group(dim=2, tags=surfaces, name="Omega")
        gmsh.model.add_physical_group(dim=1, tags=[lns[0], lns[-1]], name="Gamma")
        gmsh.model.add_physical_group(dim=1, tags=arc_lines, name="Sigma")

        gmsh.model.mesh.generate(2)

        meshes: list[Mesh] = [Mesh.from_gmsh()]
        for _ in range(nb_refinement):
            gmsh.model.mesh.refine()
            meshes.append(Mesh.from_gmsh())

    return meshes


def sector(
    radius: float,
    angle: float,
    mesh_size: float,
    gmsh_options: GmshOptions = GmshOptions(),
) -> Mesh:
    """Generate an angular section (sector) shaped mesh.

    Parameters
    ----------
    radius : float
        The radius of the sector.
    angle : float
        The opening angle of the sector in radians (must be in [0, 2π]).
    mesh_size : float
        The size of the mesh elements.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    Returns
    -------
    Mesh
        The generated mesh.
    """
    return sector_refine(radius, angle, mesh_size, 0, gmsh_options)[0]
