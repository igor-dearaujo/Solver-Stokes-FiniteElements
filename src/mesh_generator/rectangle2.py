"""Functions to create rectangle meshes with multiple domains."""

import gmsh

from ..mesh import GmshContextManager, GmshOptions, Mesh


def rectangle2_refine(
    low_corner: tuple[float, float],
    high_corner: tuple[float, float],
    alpha: tuple[float, float],
    mesh_size: float,
    nb_refinement: int,
    gmsh_options: GmshOptions = GmshOptions(),
) -> list[Mesh]:
    """Generate a rectangle mesh with two domains and multiple refinement levels.

    The rectangle is split into two sub-domains using proportional parameters alpha.

    Parameters
    ----------
    low_corner : tuple[float, float]
        The (x, y) coordinates of the lower-left corner.
    high_corner : tuple[float, float]
        The (x, y) coordinates of the upper-right corner.
    alpha : tuple[float, float]
        Proportional split parameters for the left (alpha[0]) and right (alpha[1])
        sides of the rectangle. Must satisfy 0.01 < alpha[i] < 0.99.
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
    lx, ly = low_corner
    hx, hy = high_corner
    if lx >= hx or ly >= hy:
        raise ValueError(
            f"low_corner={low_corner} must be strictly less than "
            f"high_corner={high_corner} in both dimensions."
        )

    for i, a in enumerate(alpha):
        if not (0.01 < a < 0.99):
            raise ValueError(f"alpha[{i}]={a} must satisfy 0.01 < alpha[{i}] < 0.99.")

    if mesh_size <= 0:
        raise ValueError(f"mesh_size={mesh_size} must be positive.")
    if nb_refinement < 0:
        raise ValueError(f"nb_refinement={nb_refinement} must be non-negative.")

    with GmshContextManager(gmsh_options):
        lx, ly = low_corner
        hx, hy = high_corner
        a0, a1 = alpha

        points: list[int] = [
            gmsh.model.geo.add_point(lx, ly, 0, mesh_size),
            gmsh.model.geo.add_point(hx, ly, 0, mesh_size),
            gmsh.model.geo.add_point(hx, (1 - a1) * ly + a1 * hy, 0, mesh_size),
            gmsh.model.geo.add_point(hx, hy, 0, mesh_size),
            gmsh.model.geo.add_point(lx, hy, 0, mesh_size),
            gmsh.model.geo.add_point(lx, (1 - a0) * ly + a0 * hy, 0, mesh_size),
        ]

        lines: list[int] = [
            # horizontal lines
            gmsh.model.geo.add_line(points[0], points[1]),
            gmsh.model.geo.add_line(points[5], points[2]),
            gmsh.model.geo.add_line(points[4], points[3]),
            # vertical lines
            gmsh.model.geo.add_line(points[0], points[5]),
            gmsh.model.geo.add_line(points[5], points[4]),
            gmsh.model.geo.add_line(points[1], points[2]),
            gmsh.model.geo.add_line(points[2], points[3]),
        ]

        # Surfaces definitions
        surfaces: list[int] = [
            gmsh.model.geo.add_plane_surface([gmsh.model.geo.add_curve_loop(wire_tag)])
            for wire_tag in [
                [lines[0], lines[5], -lines[1], -lines[3]],
                [lines[1], lines[6], -lines[2], -lines[4]],
            ]
        ]

        gmsh.model.geo.synchronize()

        # Define physical groups
        # Gamma: all boundary edges
        gmsh.model.add_physical_group(
            dim=1, tags=[lines[i] for i in [0, 2, 3, 4, 5, 6]], name="Gamma"
        )
        gmsh.model.add_physical_group(dim=2, tags=[surfaces[0]], name="Omega-1")
        gmsh.model.add_physical_group(dim=2, tags=[surfaces[1]], name="Omega-2")

        gmsh.model.mesh.generate(2)

        meshes: list[Mesh] = [Mesh.from_gmsh()]
        for _ in range(nb_refinement):
            gmsh.model.mesh.refine()
            meshes.append(Mesh.from_gmsh())

    return meshes


def rectangle2(
    low_corner: tuple[float, float],
    high_corner: tuple[float, float],
    alpha: tuple[float, float],
    mesh_size: float,
    gmsh_options: GmshOptions = GmshOptions(),
) -> Mesh:
    """Generate a rectangle mesh with two different domains.

    The rectangle is split into two sub-domains using proportional parameters alpha.

    Parameters
    ----------
    low_corner : tuple[float, float]
        The (x, y) coordinates of the lower-left corner.
    high_corner : tuple[float, float]
        The (x, y) coordinates of the upper-right corner.
    alpha : tuple[float, float]
        Proportional split parameters for the left (alpha[0]) and right (alpha[1])
        sides of the rectangle. Must satisfy 0.01 < alpha[i] < 0.99.
    mesh_size : float
        The size of the mesh elements.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    Returns
    -------
    Mesh
        The generated mesh.
    """
    return rectangle2_refine(
        low_corner, high_corner, alpha, mesh_size, 0, gmsh_options
    )[0]
