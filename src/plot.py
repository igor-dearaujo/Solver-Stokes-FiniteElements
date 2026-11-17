"""Plotting utilities for meshes."""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.pyplot import colorbar
from mpl_toolkits.mplot3d import Axes3D
from numpy import float64
from numpy.typing import NDArray

from .mesh import Mesh


def plot2d_mesh(
    ax: Axes,
    mesh: Mesh,
    domains_kwargs: dict[str, Any] | None = None,
    *,
    show_legend: bool = True,
    show_nodes_indices: bool = False,
    show_nodes: bool = False,
) -> None:
    """Plot a 2D mesh. Higher order elements are drawn as P1 elements.

    Parameters
    ----------
    ax : Axes
        axes to plot on.
    mesh : Mesh
        mesh to plot.
    domains_kwargs : dict[str, Any] | None, optional, default None
        keyword arguments for each domain.
    show_legend : bool, optional, default True
        whether to show the legend.
    show_nodes_indices : bool, optional, default False
        whether to show the nodes indices.
    show_nodes: bool, optional, dafault False
        whether to show nodes (only if show_nodes_indices is False)
    """

    for i, (name, elements) in enumerate(mesh.domains.items()):
        base_kwargs: dict[str, Any] = {"color": f"C{i}"}
        if domains_kwargs is not None:
            base_kwargs.update(domains_kwargs.get(name, {}))

        for elements_type, data in elements.items():
            match elements_type:
                case "triangle":
                    kwargs = {
                        **base_kwargs,
                        "linewidth": base_kwargs.get("linewidth", 0.5),
                    }

                    ax.triplot(
                        mesh.nodes[:, 0],
                        mesh.nodes[:, 1],
                        data[:, :3],
                        label=f'"{name}"',
                        zorder=1,
                        **kwargs,
                    )

                case "line":
                    segments = [
                        (mesh.nodes[i, :], mesh.nodes[j, :]) for i, j in data[:, :2]
                    ]
                    ax.add_collection(
                        LineCollection(
                            segments,
                            label=f'"{name}"',
                            zorder=2,
                            **base_kwargs,
                        )
                    )

                case "point":
                    ax.scatter(
                        mesh.nodes[data[:, 0], 0],
                        mesh.nodes[data[:, 0], 1],
                        label=f'"{name}"',
                        zorder=3,
                        **base_kwargs,
                    )

                case _:
                    raise NotImplementedError(
                        f"Elements type {elements_type} is not supported."
                    )

    if show_nodes_indices:
        for i, node in enumerate(mesh.nodes):
            ax.text(
                node[0],
                node[1],
                str(i),
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"),
                zorder=3,
            )
    elif show_nodes:
        ax.scatter(mesh.nodes[:, 0], mesh.nodes[:, 1], s=5, color="black", zorder=4)

    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)


def plot2d_fct_trimesh(
    ax: Axes,
    values: NDArray[float64],
    mesh: Mesh,
    *,
    tripcolor_kwargs: dict[str, Any] | None = None,
    show_colorbar: bool = True,
) -> None:
    """Plot a function defined on a 2D mesh.

    Parameters
    ----------
    ax : Axes
        axes to plot on.
    values : NDArray
        values of the function.
    mesh : Mesh
        mesh on which the function is defined.
    tripcolor_kwargs : dict[str, Any] | None, optional
        Additional arguments for the tripcolor, by default None
    """

    kwargs = tripcolor_kwargs or {}

    im = None
    for elements in mesh.domains.values():
        for elements_type, data in elements.items():
            if elements_type != "triangle":
                continue

            im = ax.tripcolor(
                mesh.nodes[:, 0], mesh.nodes[:, 1], values, triangles=data, **kwargs
            )

    if im is not None and show_colorbar:
        colorbar(im, ax=ax)


def plot3d_fct_trimesh(
    ax: Axes3D,
    values: NDArray[float64],
    mesh: Mesh,
    *,
    trisurf_kwargs: dict[str, Any] | None = None,
    show_colorbar: bool = True,
) -> None:
    """Plot a function defined on a 2D mesh as a 3D surface.

    Parameters
    ----------
    ax : Axes3D
        axes to plot on.
    values : NDArray
        values of the function.
    mesh : Mesh
        mesh on which the function is defined.
    trisurf_kwargs : dict[str, Any] | None, optional
        Additional arguments for the trisurf, by default None
    """

    kwargs = trisurf_kwargs or {}

    im = None
    for elements in mesh.domains.values():
        for elements_type, data in elements.items():
            if elements_type != "triangle":
                continue

            im = ax.plot_trisurf(
                mesh.nodes[:, 0], mesh.nodes[:, 1], values, triangles=data, **kwargs
            )

    if im is not None and show_colorbar:
        colorbar(im, ax=ax)
