"""Functions to create Pac-Man shaped meshes."""

import gmsh
from numpy import cos, pi, sin, sqrt

from ..mesh import GmshContextManager, GmshOptions, Mesh


def pacman_refine(
    radius: float,
    theta: float,
    mesh_size: float,
    nb_refinement: int,
    gmsh_options: GmshOptions = GmshOptions(),
) -> list[Mesh]:
    """Generate a Pac-Man shaped mesh with multiple refinement levels.

    Parameters
    ----------
    radius : float
        The radius of the Pac-Man shape.
    theta : float
        The angle of the mouth opening in radians (must be in [0, 2π]).
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
    if not (0 <= theta <= 2 * pi):
        raise ValueError(f"theta={theta} must be in [0, 2π].")
    if mesh_size <= 0:
        raise ValueError(f"mesh_size={mesh_size} must be positive.")
    if nb_refinement < 0:
        raise ValueError(f"nb_refinement={nb_refinement} must be non-negative.")

    with GmshContextManager(gmsh_options):
        # Outer arc points
        xo, yo = radius * cos(theta / 2), radius * sin(theta / 2)

        pts_out: list[int] = [
            gmsh.model.geo.add_point(0, 0, 0, mesh_size),
            gmsh.model.geo.add_point(xo, yo, 0, mesh_size),
            gmsh.model.geo.add_point(-radius, 0, 0, mesh_size),
            gmsh.model.geo.add_point(xo, -yo, 0, mesh_size),
        ]

        lns_out: list[int] = [
            gmsh.model.geo.add_line(pts_out[0], pts_out[1]),
            gmsh.model.geo.add_circle_arc(pts_out[1], pts_out[0], pts_out[2]),
            gmsh.model.geo.add_circle_arc(pts_out[2], pts_out[0], pts_out[3]),
            gmsh.model.geo.add_line(pts_out[3], pts_out[0]),
        ]

        # Eye geometry
        xe, ye = xo / 2 - sqrt(3) * yo / 4, yo / 2 + sqrt(3) * xo / 4
        re = radius / 8

        pts_eye: list[int] = [
            gmsh.model.geo.add_point(xe, ye, 0, mesh_size),
            gmsh.model.geo.add_point(xe + re, ye, 0, mesh_size),
            gmsh.model.geo.add_point(xe, ye + re, 0, mesh_size),
            gmsh.model.geo.add_point(xe - re, ye, 0, mesh_size),
            gmsh.model.geo.add_point(xe, ye - re, 0, mesh_size),
        ]

        lns_eye: list[int] = [
            gmsh.model.geo.add_circle_arc(pts_eye[1], pts_eye[0], pts_eye[2]),
            gmsh.model.geo.add_circle_arc(pts_eye[2], pts_eye[0], pts_eye[3]),
            gmsh.model.geo.add_circle_arc(pts_eye[3], pts_eye[0], pts_eye[4]),
            gmsh.model.geo.add_circle_arc(pts_eye[4], pts_eye[0], pts_eye[1]),
        ]

        # Closed curves and associated surfaces
        wire_tags: list[int] = [
            gmsh.model.geo.add_curve_loop(lns_out),
            gmsh.model.geo.add_curve_loop(lns_eye),
        ]
        surfaces: list[int] = [gmsh.model.geo.add_plane_surface(wire_tags)]

        gmsh.model.geo.synchronize()

        # Define physical groups
        # Gamma-1: mouth edges and eye boundary
        gmsh.model.add_physical_group(
            dim=1, tags=[lns_out[0], lns_out[3], *lns_eye], name="Gamma-1"
        )
        # Gamma-2: outer arc edges
        gmsh.model.add_physical_group(
            dim=1, tags=[lns_out[1], lns_out[2]], name="Gamma-2"
        )
        gmsh.model.add_physical_group(dim=2, tags=surfaces, name="Omega")

        gmsh.model.mesh.generate(2)

        meshes: list[Mesh] = [Mesh.from_gmsh()]
        for _ in range(nb_refinement):
            gmsh.model.mesh.refine()
            meshes.append(Mesh.from_gmsh())

    return meshes


def pacman(
    radius: float,
    theta: float,
    mesh_size: float,
    gmsh_options: GmshOptions = GmshOptions(),
) -> Mesh:
    """Generate a Pac-Man shaped mesh.

    Parameters
    ----------
    radius : float
        The radius of the Pac-Man shape.
    theta : float
        The angle of the mouth opening in radians (must be in [0, 2π]).
    mesh_size : float
        The size of the mesh elements.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    Returns
    -------
    Mesh
        The generated mesh.
    """
    return pacman_refine(radius, theta, mesh_size, 0, gmsh_options)[0]
