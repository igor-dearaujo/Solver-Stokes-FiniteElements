"""Functions to create meshes of axis-aligned orthotopes.

An orthotope is a segment, rectangle, or cuboid.
"""

import gmsh

from ..mesh import GmshContextManager, GmshOptions, Mesh


def segment_refine(
    x_min: float,
    x_max: float,
    mesh_size: float,
    nb_refinement: int,
    gmsh_options: GmshOptions = GmshOptions(),
) -> list[Mesh]:
    """Generate a mesh of a segment with multiple refinement levels.

    Parameters
    ----------
    x_min : float
        The minimum x-coordinate of the segment.
    x_max : float
        The maximum x-coordinate of the segment.
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
    if x_min >= x_max:
        raise ValueError(f"x_min={x_min} must be less than x_max={x_max}.")
    if mesh_size <= 0:
        raise ValueError(f"mesh_size={mesh_size} must be positive.")
    if nb_refinement < 0:
        raise ValueError(f"nb_refinement={nb_refinement} must be non-negative.")

    with GmshContextManager(gmsh_options):
        # Points definition (in 3D, that's why there is a 0 in y and z positions)
        points: list[int] = [
            gmsh.model.geo.add_point(x_min, 0, 0, mesh_size),
            gmsh.model.geo.add_point(x_max, 0, 0, mesh_size),
        ]

        # Line definitions between points
        lines: list[int] = [gmsh.model.geo.add_line(points[0], points[1])]

        gmsh.model.geo.synchronize()

        # Define physical groups for boundary and domain
        gmsh.model.add_physical_group(dim=0, tags=points, name="Gamma")
        gmsh.model.add_physical_group(dim=1, tags=lines, name="Omega")

        gmsh.model.mesh.generate(1)

        meshes: list[Mesh] = [Mesh.from_gmsh()]
        for _ in range(nb_refinement):
            gmsh.model.mesh.refine()
            meshes.append(Mesh.from_gmsh())

    return meshes


def segment(
    x_min: float,
    x_max: float,
    mesh_size: float,
    gmsh_options: GmshOptions = GmshOptions(),
) -> Mesh:
    """Generate a mesh of a segment defined by its minimum and maximum x-coordinates.

    Parameters
    ----------
    x_min : float
        The minimum x-coordinate of the segment.
    x_max : float
        The maximum x-coordinate of the segment.
    mesh_size : float
        The size of the mesh elements.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    Returns
    -------
    Mesh
        The generated mesh.
    """
    return segment_refine(x_min, x_max, mesh_size, 0, gmsh_options)[0]


def rectangle_refine(
    low_corner: tuple[float, float],
    high_corner: tuple[float, float],
    mesh_size: float,
    nb_refinement: int,
    gmsh_options: GmshOptions = GmshOptions(),
) -> list[Mesh]:
    """Generate a mesh of an axis-aligned rectangle with multiple refinement levels.

    Parameters
    ----------
    low_corner : tuple[float, float]
        The (x, y) coordinates of the lower-left corner.
    high_corner : tuple[float, float]
        The (x, y) coordinates of the upper-right corner.
    mesh_size : float
        The size of the mesh elements.
    nb_refinement : int
        The number of refinement steps to apply.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

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
    if mesh_size <= 0:
        raise ValueError(f"mesh_size={mesh_size} must be positive.")
    if nb_refinement < 0:
        raise ValueError(f"nb_refinement={nb_refinement} must be non-negative.")

    with GmshContextManager(gmsh_options):
        # Points definition (in 3D, that's why there is a 0 in z position)
        points: list[int] = [
            gmsh.model.geo.add_point(lx, ly, 0, mesh_size),
            gmsh.model.geo.add_point(lx, hy, 0, mesh_size),
            gmsh.model.geo.add_point(hx, hy, 0, mesh_size),
            gmsh.model.geo.add_point(hx, ly, 0, mesh_size),
        ]

        # Line definitions between points
        lines: list[int] = [
            gmsh.model.geo.add_line(points[0], points[1]),
            gmsh.model.geo.add_line(points[1], points[2]),
            gmsh.model.geo.add_line(points[2], points[3]),
            gmsh.model.geo.add_line(points[3], points[0]),
        ]

        # Surface definitions
        wire_tags: list[int] = [gmsh.model.geo.add_curve_loop(lines)]
        surfaces: list[int] = [gmsh.model.geo.add_plane_surface(wire_tags)]

        gmsh.model.geo.synchronize()

        # Define physical groups for boundary and domain
        gmsh.model.add_physical_group(dim=1, tags=lines, name="Gamma")
        gmsh.model.add_physical_group(dim=2, tags=surfaces, name="Omega")

        gmsh.model.mesh.generate(2)

        meshes: list[Mesh] = [Mesh.from_gmsh()]
        for _ in range(nb_refinement):
            gmsh.model.mesh.refine()
            meshes.append(Mesh.from_gmsh())

    return meshes


def rectangle(
    low_corner: tuple[float, float],
    high_corner: tuple[float, float],
    mesh_size: float,
    gmsh_options: GmshOptions = GmshOptions(),
) -> Mesh:
    """Generate a mesh of an axis-aligned rectangle.

    Parameters
    ----------
    low_corner : tuple[float, float]
        The (x, y) coordinates of the lower-left corner.
    high_corner : tuple[float, float]
        The (x, y) coordinates of the upper-right corner.
    mesh_size : float
        The size of the mesh elements.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    Returns
    -------
    Mesh
        The generated mesh.
    """
    return rectangle_refine(low_corner, high_corner, mesh_size, 0, gmsh_options)[0]


def cuboid_refine(
    low_corner: tuple[float, float, float],
    high_corner: tuple[float, float, float],
    mesh_size: float,
    nb_refinement: int,
    gmsh_options: GmshOptions = GmshOptions(),
) -> list[Mesh]:
    """Generate a mesh of an axis-aligned cuboid with multiple refinement levels.

    Parameters
    ----------
    low_corner : tuple[float, float, float]
        The (x, y, z) coordinates of the lower-left-front corner.
    high_corner : tuple[float, float, float]
        The (x, y, z) coordinates of the upper-right-back corner.
    mesh_size : float
        The size of the mesh elements.
    nb_refinement : int
        The number of refinement steps to apply.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    Returns
    -------
    list[Mesh]
        List of meshes, one for each refinement level.
    """
    lx, ly, lz = low_corner
    hx, hy, hz = high_corner
    if lx >= hx or ly >= hy or lz >= hz:
        raise ValueError(
            f"low_corner={low_corner} must be strictly less than "
            f"high_corner={high_corner} in all dimensions."
        )
    if mesh_size <= 0:
        raise ValueError(f"mesh_size={mesh_size} must be positive.")
    if nb_refinement < 0:
        raise ValueError(f"nb_refinement={nb_refinement} must be non-negative.")

    with GmshContextManager(gmsh_options):
        # Points definition
        points: list[int] = [
            gmsh.model.geo.add_point(lx, ly, lz, mesh_size),  # 0
            gmsh.model.geo.add_point(lx, hy, lz, mesh_size),  # 1
            gmsh.model.geo.add_point(hx, hy, lz, mesh_size),  # 2
            gmsh.model.geo.add_point(hx, ly, lz, mesh_size),  # 3
            gmsh.model.geo.add_point(lx, ly, hz, mesh_size),  # 4
            gmsh.model.geo.add_point(lx, hy, hz, mesh_size),  # 5
            gmsh.model.geo.add_point(hx, hy, hz, mesh_size),  # 6
            gmsh.model.geo.add_point(hx, ly, hz, mesh_size),  # 7
        ]

        # Line definitions between points
        lines: list[int] = []
        for i, j in [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
        ]:
            lines.append(gmsh.model.geo.add_line(points[i], points[j]))

        # Surface definitions from wire tags
        surfaces: list[int] = []
        for tag_lines in [
            lines[0:4],  # Bottom face
            [lines[0], lines[5], -lines[8], -lines[4]],  # Left face
            [lines[1], lines[6], -lines[9], -lines[5]],  # Front face
            [lines[2], lines[7], -lines[10], -lines[6]],  # Right face
            [lines[3], lines[4], -lines[11], -lines[7]],  # Back face
            lines[8:12],  # Top face
        ]:
            wire_tags: list[int] = [gmsh.model.geo.add_curve_loop(tag_lines)]
            surfaces.append(gmsh.model.geo.add_plane_surface(wire_tags))

        # Volume definition from shell tags
        shell_tags = [gmsh.model.geo.add_surface_loop(surfaces)]
        volumes = [gmsh.model.geo.add_volume(shell_tags)]

        gmsh.model.geo.synchronize()

        # Define physical groups for boundary and domain
        gmsh.model.add_physical_group(dim=2, tags=surfaces, name="Gamma")
        gmsh.model.add_physical_group(dim=3, tags=volumes, name="Omega")

        gmsh.model.mesh.generate(3)

        meshes: list[Mesh] = [Mesh.from_gmsh()]
        for _ in range(nb_refinement):
            gmsh.model.mesh.refine()
            meshes.append(Mesh.from_gmsh())

    return meshes


def cuboid(
    low_corner: tuple[float, float, float],
    high_corner: tuple[float, float, float],
    mesh_size: float,
    gmsh_options: GmshOptions = GmshOptions(),
) -> Mesh:
    """Generate a mesh of an axis-aligned cuboid.

    Parameters
    ----------
    low_corner : tuple[float, float, float]
        The (x, y, z) coordinates of the lower-left-front corner.
    high_corner : tuple[float, float, float]
        The (x, y, z) coordinates of the upper-right-back corner.
    mesh_size : float
        The size of the mesh elements.
    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    Returns
    -------
    Mesh
        The generated mesh.
    """
    return cuboid_refine(low_corner, high_corner, mesh_size, 0, gmsh_options)[0]
