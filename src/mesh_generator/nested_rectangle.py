"""Functions to create meshes with a rectangle nested inside an outer polygon."""

import gmsh

from ..mesh import GmshContextManager, GmshOptions, Mesh


def nested_rectangle_refine(
    outer_points: list[tuple[float, float]],
    inner_low: tuple[float, float],
    inner_high: tuple[float, float],
    mesh_size: float,
    nb_refinement: int,
    gmsh_options: GmshOptions = GmshOptions(),
) -> list[Mesh]:
    """Generate a mesh of an outer polygon Omega-1 containing a nested rectangle
    Omega-2, with multiple refinement levels.

    The outer polygon is defined by its vertices in counter-clockwise order.
    The rectangle Omega-2 must lie strictly inside the outer polygon and does not
    touch its boundary: the two domains share a conforming interface which is
    excluded from the "Gamma" physical group.

    Parameters
    ----------
    outer_points : list[tuple[float, float]]
        The (x, y) coordinates of the outer polygon Omega-1, counter-clockwise.
    inner_low : tuple[float, float]
        The (x, y) coordinates of the lower-left corner of Omega-2.
    inner_high : tuple[float, float]
        The (x, y) coordinates of the upper-right corner of Omega-2.
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
    n_points = len(outer_points)
    if n_points < 3:
        raise ValueError("outer_points must have at least three vertices.")

    ilx, ily = inner_low
    ihx, ihy = inner_high
    if ilx >= ihx or ily >= ihy:
        raise ValueError(
            f"inner_low={inner_low} must be strictly less than "
            f"inner_high={inner_high} in both dimensions."
        )
    if mesh_size <= 0:
        raise ValueError(f"mesh_size={mesh_size} must be positive.")
    if nb_refinement < 0:
        raise ValueError(f"nb_refinement={nb_refinement} must be non-negative.")

    with GmshContextManager(gmsh_options):
        pts_outer: list[int] = [
            gmsh.model.geo.add_point(px, py, 0, mesh_size) for px, py in outer_points
        ]
        lns_outer: list[int] = [
            gmsh.model.geo.add_line(pts_outer[k], pts_outer[(k + 1) % n_points])
            for k in range(n_points)
        ]

        pts_inner: list[int] = [
            gmsh.model.geo.add_point(ilx, ily, 0, mesh_size),
            gmsh.model.geo.add_point(ihx, ily, 0, mesh_size),
            gmsh.model.geo.add_point(ihx, ihy, 0, mesh_size),
            gmsh.model.geo.add_point(ilx, ihy, 0, mesh_size),
        ]
        lns_inner: list[int] = [
            gmsh.model.geo.add_line(pts_inner[k], pts_inner[(k + 1) % 4])
            for k in range(4)
        ]

        outer_loop = gmsh.model.geo.add_curve_loop(lns_outer)
        inner_loop = gmsh.model.geo.add_curve_loop(lns_inner)

        # Omega-1: outer polygon with the inner rectangle removed.
        # Omega-2: the inner rectangle, sharing the same curve loop so that the
        # two surfaces are meshed conformally across the interface.
        surfaces: list[int] = [
            gmsh.model.geo.add_plane_surface([outer_loop, inner_loop]),
            gmsh.model.geo.add_plane_surface([inner_loop]),
        ]

        gmsh.model.geo.synchronize()

        # Define physical groups
        # Gamma: outer boundary only (the Omega-1/Omega-2 interface is excluded).
        gmsh.model.add_physical_group(dim=1, tags=lns_outer, name="Gamma")
        gmsh.model.add_physical_group(dim=2, tags=[surfaces[0]], name="Omega-1")
        gmsh.model.add_physical_group(dim=2, tags=[surfaces[1]], name="Omega-2")

        gmsh.model.mesh.generate(2)

        meshes: list[Mesh] = [Mesh.from_gmsh()]
        for _ in range(nb_refinement):
            gmsh.model.mesh.refine()
            meshes.append(Mesh.from_gmsh())

    return meshes


def nested_rectangle(
    outer_points: list[tuple[float, float]],
    inner_low: tuple[float, float],
    inner_high: tuple[float, float],
    mesh_size: float,
    gmsh_options: GmshOptions = GmshOptions(),
) -> Mesh:
    """Generate a mesh of an outer polygon Omega-1 containing a nested rectangle
    Omega-2.

    See `nested_rectangle_refine` for details.

    Parameters
    ----------
    outer_points : list[tuple[float, float]]
        The (x, y) coordinates of the outer polygon Omega-1, counter-clockwise.
    inner_low : tuple[float, float]
        The (x, y) coordinates of the lower-left corner of Omega-2.
    inner_high : tuple[float, float]
        The (x, y) coordinates of the upper-right corner of Omega-2.
    mesh_size : float
        The size of the mesh elements.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    Returns
    -------
    Mesh
        The generated mesh.
    """
    return nested_rectangle_refine(
        outer_points, inner_low, inner_high, mesh_size, 0, gmsh_options
    )[0]
