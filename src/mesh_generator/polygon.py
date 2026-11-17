"""Functions to create meshes of general polygons."""

from collections import defaultdict

import gmsh

from ..mesh import GmshContextManager, GmshOptions, Mesh


def _add_physical_groups(
    dimension: int,
    entity_tags: list[int],
    labels: dict[int, str] | None,
    default_name: str,
) -> None:
    """Create one Gmsh physical group for each distinct label.

    When ``labels`` is None, all entities are grouped under ``default_name``.

    Parameters
    ----------
    dimension:
        Dimension of the entities (0 for points, 1 for curves, 2 for surfaces).
    entity_tags:
        Gmsh tags of the entities.
    labels:
        Dictionary mapping entity indices to physical labels, or None.
    default_name:
        Name of the group when ``labels`` is None.
    """
    if labels is None:
        gmsh.model.add_physical_group(dimension, entity_tags, name=default_name)
        return

    groups: dict[str, list[int]] = defaultdict(list)

    for index, label in labels.items():
        groups[label].append(entity_tags[index])

    for label, tags in groups.items():
        gmsh.model.addPhysicalGroup(dimension, tags, name=label)


def _check_labels(
    n_points: int,
    point_labels: dict[int, str] | None,
    edge_labels: dict[int, str] | None,
) -> None:
    """Validate the point and edge label mappings.

    Parameters
    ----------
    n_points:
        Number of polygon vertices.
    point_labels:
        Mapping from vertex index to a physical label (or None).
    edge_labels:
        Mapping from edge index to a physical label (or None).
    """
    for name, labels in (
        ("point_labels", point_labels),
        ("edge_labels", edge_labels),
    ):
        if labels is None:
            continue

        invalid_indices = set(labels) - set(range(n_points))
        if invalid_indices:
            raise ValueError(f"{name} contains invalid indices: {invalid_indices}")

        for index, label in labels.items():
            if not isinstance(label, str) or not label:
                raise ValueError(f"{name}[{index}] must be a non-empty string.")


def polygon_refine(
    point_coords: list[tuple[float, float]],
    mesh_size: float,
    nb_refinement: int,
    *,
    gmsh_options: GmshOptions = GmshOptions(),
    point_labels: dict[int, str] | None = None,
    edge_labels: dict[int, str] | None = None,
) -> list[Mesh]:
    """Generate a 2D triangular mesh of a polygon with multiple refinement levels.

    The polygon is defined by its vertices in counter-clockwise order.
    Edge i connects points[i] to points[(i + 1) % n].

    Parameters
    ----------
    point_coords : list[tuple[float, float]]
        The list of the (x, y) coordinates of the polygon vertices.

    mesh_size : float
        Characteristic mesh size.

    nb_refinement : int
        The number of refinement steps to apply.

    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    point_labels: dict[int, str] | None, optional
        Mapping from vertex index to a physical label,
        e.g. {0: "Dirichlet", 1: "Neumann"},
        defaults to None.

    edge_labels: dict[int, str] | None, optional
        Mapping from edge index to a physical label,
        e.g. {0: "Dirichlet", 1: "Neumann"},
        defaults to None.

    Returns
    -------
    list[Mesh]
        List of meshes, one for each refinement level.
    """
    n_points = len(point_coords)  # number of points
    if n_points < 3:
        raise ValueError("A polygon must have at least three vertices.")
    if mesh_size <= 0:
        raise ValueError(f"mesh_size={mesh_size} must be positive.")
    if nb_refinement < 0:
        raise ValueError(f"nb_refinement={nb_refinement} must be non-negative.")

    _check_labels(n_points, point_labels, edge_labels)

    with GmshContextManager(gmsh_options):
        # Points definition (in 3D, that's why there is a 0 in z position)
        points: list[int] = [
            gmsh.model.geo.add_point(pk[0], pk[1], 0, mesh_size) for pk in point_coords
        ]

        # Line definitions between points
        lines: list[int] = [
            gmsh.model.geo.add_line(points[k], points[(k + 1) % n_points])
            for k in range(n_points)
        ]

        # Surface definitions
        wire_tags: list[int] = [gmsh.model.geo.add_curve_loop(lines)]
        surfaces: list[int] = [gmsh.model.geo.add_plane_surface(wire_tags)]

        gmsh.model.geo.synchronize()

        # Define physical groups for boundary and domain
        _add_physical_groups(0, points, point_labels, "Vertices")
        _add_physical_groups(1, lines, edge_labels, "Gamma")
        gmsh.model.add_physical_group(dim=2, tags=surfaces, name="Omega")

        gmsh.model.mesh.generate(2)

        meshes: list[Mesh] = [Mesh.from_gmsh()]
        for _ in range(nb_refinement):
            gmsh.model.mesh.refine()
            meshes.append(Mesh.from_gmsh())

    return meshes


def polygon(
    point_coords: list[tuple[float, float]],
    mesh_size: float,
    *,
    gmsh_options: GmshOptions = GmshOptions(),
    point_labels: dict[int, str] | None = None,
    edge_labels: dict[int, str] | None = None,
) -> Mesh:
    """Generate a 2D triangular mesh of a polygon.

    The polygon is defined by its vertices in counter-clockwise order.
    Edge i connects points[i] to points[(i + 1) % n].

    Parameters
    ----------
    point_coords : list[tuple[float, float]]
        The list of the (x, y) coordinates of the polygon vertices.

    mesh_size : float
        Characteristic mesh size.

    gmsh_options : GmshOptions, optional
        The GMSH options to use. Defaults to GmshOptions().

    point_labels: dict[int, str] | None, optional
        Mapping from vertex index to a physical label,
        e.g. {0: "Dirichlet", 1: "Neumann"},
        defaults to None.

    edge_labels: dict[int, str] | None, optional
        Mapping from edge index to a physical label,
        e.g. {0: "Dirichlet", 1: "Neumann"},
        defaults to None.

    Returns
    -------
    Mesh
        The generated mesh.
    """
    return polygon_refine(
        point_coords,
        mesh_size,
        0,
        gmsh_options=gmsh_options,
        point_labels=point_labels,
        edge_labels=edge_labels,
    )[0]
