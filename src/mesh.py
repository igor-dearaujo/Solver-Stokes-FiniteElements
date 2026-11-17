"""Meshing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

import gmsh
from numpy import array, empty, float64, searchsorted, uint64, unique_values, vstack
from numpy.typing import NDArray
from scipy.spatial import KDTree


@dataclass(frozen=True, slots=True) #define automaticamente métodos especiais de POO com base nos atributos declarados
class Mesh:
    """Mesh class.

    Attributes
    ----------
    nodes : NDArray[float64]
        Node coordinates, array of shape (`nb_nodes`, `dim`).
    domains : dict[str, dict[str, NDArray[uint64]]]
        Dictionary containing physical names and their associated data.
        The keys are physical names, and the values are dictionaries with:
        - "element_type": NDArray[uint64], array of shape (`nb_elements`,
        `nb_node_per_element`). These inner arrays contain the indices of the nodes.
    """

    # of size (nb_nodes, dim)
    nodes: NDArray[float64] #type hint (__init__ is constructed by @dataclass)
    # { physical_name: { element_type: NDArray[uint64] } }
    domains: dict[str, dict[str, NDArray[uint64]]]

    @property
    def dimension(self: Mesh) -> int:  # -> int is a return type hint
        """Return the dimension of the mesh."""
        return self.nodes.shape[1]

    @property
    def nb_nodes(self: Mesh) -> int: #type hint self: Mesh
        """Return the number of nodes."""
        return self.nodes.shape[0]

    def nb_elements(self: Mesh, domain_name: str, element_type: str) -> int:
        """Return the total number of elements."""
        return self.domains[domain_name][element_type].shape[0]

    def domain_node_indices(
        self: Mesh, domain_name: str, element_type: str
    ) -> NDArray[uint64]:
        """Return the node indices for a given domain name and element type."""
        return unique_values(self.domains[domain_name][element_type])

    @staticmethod
    def from_gmsh() -> Mesh:
        """Create a Mesh instance from GMSH. This method must be used after defining
        the mesh and before `gmsh.finalize()`.

        Returns
        -------
        Mesh
        """

        domains_tmp: dict[str, dict[str, list[NDArray[uint64]]]] = {}
        tag_used: set[int] = set()

        # Collect all element node tags which are part of a physical group.
        for physical_dim, physical_tag in gmsh.model.get_physical_groups():
            physical_name = gmsh.model.get_physical_name(physical_dim, physical_tag)
            if physical_name == "":
                physical_name = str(physical_tag)

            if physical_name not in domains_tmp:
                domains_tmp[physical_name] = {}

            for entity_tag in gmsh.model.get_entities_for_physical_group(
                physical_dim, physical_tag
            ):
                for element_type, _, node_tags_flat in zip(
                    *gmsh.model.mesh.get_elements(physical_dim, entity_tag)
                ):
                    element_name, _, _, num_nodes, *_ = (
                        gmsh.model.mesh.get_element_properties(element_type)
                    )
                    element_name = element_name.split(" ")[0].lower()

                    # node_tags_flat is flat: [e1n1, ..., e1nN, e2n1, ...].
                    elements = array(node_tags_flat, dtype=uint64).reshape(
                        -1, num_nodes
                    )
                    tag_used.update(elements.ravel().tolist())

                    domains_tmp.setdefault(physical_name, {}).setdefault(
                        element_name, []
                    ).append(elements)

        # Stack the per-entity blocks once per (physical_name, element_name).
        domains: dict[str, dict[str, NDArray[uint64]]] = {
            name: {
                element_name: vstack(blocks)
                for element_name, blocks in element_types.items()
            }
            for name, element_types in domains_tmp.items()
        }

        # Sorted tags define the index order: index = position in sorted_tags.
        sorted_tags = array(sorted(tag_used), dtype=uint64)

        space_dim: int = gmsh.model.get_dimension()
        nb_nodes = sorted_tags.shape[0]

        # Get the node coordinates, keeping only the nodes used by the mesh.
        nodes = empty((nb_nodes, space_dim), dtype=float64)
        node_tags_all, node_coords_all, _ = gmsh.model.mesh.get_nodes()
        node_tags_arr = array(node_tags_all, dtype=uint64)
        node_coords_arr = array(node_coords_all, dtype=float64).reshape(-1, 3)[
            :, :space_dim
        ]

        pos = searchsorted(sorted_tags, node_tags_arr)
        in_used = pos < nb_nodes
        pos_clipped = pos.clip(max=nb_nodes - 1)
        mask = in_used & (sorted_tags[pos_clipped] == node_tags_arr)
        nodes[pos_clipped[mask]] = node_coords_arr[mask]

        # Change tag to index in domains.
        for name, element_types in domains.items():
            for element_name in element_types:
                idx = searchsorted(sorted_tags, element_types[element_name])
                domains[name][element_name] = idx.astype(uint64)

        return Mesh(nodes, domains)

    @staticmethod
    def from_file(
        filename: str, *, show_gui: bool = False, terminal_output: bool = False
    ) -> Mesh:
        """Create a Mesh instance from a file.

        Parameters
        ----------
        filename : str
            The name of the file to read.
        show_gui : bool, optional, default False
            Whether to show the GMSH GUI.
        terminal_output : bool, optional, default False
            Whether to enable terminal output.

        Returns
        -------
        Mesh
        """

        gmsh_options = GmshOptions(
            filename="",
            show_gui=show_gui,
            terminal_output=terminal_output,
        )

        with GmshContextManager(gmsh_options):
            gmsh.open(filename)
            mesh = Mesh.from_gmsh()

        return mesh

    def print_info(self: Mesh) -> None:
        print(f"nodes: {self.nb_nodes} nodes with {self.dimension} coordinates")
        print("domains:")
        for physical_name, element_types in self.domains.items():
            line = f'  "{physical_name}": '
            line += "; ".join(
                f"{element_type}, {elements.shape[0]} elements, "
                f"{elements.shape[1]} nodes per element"
                for element_type, elements in element_types.items()
            )
            print(line)


def project_fine_to_coarse(
    U_fine: NDArray[float64], mesh_fine: Mesh, mesh_coarse: Mesh
) -> NDArray[float64]:
    """Project a function from a fine mesh to a coarse mesh.

    Parameters
    ----------
    U_fine : NDArray[float64]
        The function values on the fine mesh.
    mesh_fine : Mesh
    mesh_coarse : Mesh

    Returns
    -------
    NDArray
        The projected function values on the coarse mesh.

    Raises
    ------
    ValueError
        If the coarse mesh nodes are not a subset of the fine mesh nodes.
    """

    tree = KDTree(mesh_fine.nodes)
    distances, indices = tree.query(mesh_coarse.nodes, eps=1e-8)

    if distances.max() >= 1e-8:  # type: ignore
        raise ValueError(
            "the coarse mesh nodes are not a subset of the fine mesh nodes"
        )

    return U_fine[indices]


@dataclass(frozen=True, kw_only=True, slots=True)
class GmshOptions:
    """GMSH options.

    Attributes
    ----------
    filename : str, optional, default ""
        If it is not an empty string, the name of the file to create.
    element_order : int, optional, default 1
        The order of the elements.
    show_gui : bool, optional, default False
        Whether to show the GMSH GUI.
    terminal_output : bool, optional, default False
        Whether to enable terminal output.
    optimize_method : str | None, optional, default None
        The optimization method to use. If None, no optimization is performed.
        See GMSH documentation for available methods.
    renumber_nodes : str | None, optional, default "RCMK"
        The renumbering method to use. If None, no renumbering is performed.
        See GMSH documentation for available methods.
    """

    filename: str = ""
    element_order: int = 1
    show_gui: bool = False
    terminal_output: bool = False
    optimize_method: str | None = None
    renumber_nodes: str | None = "RCMK"


@dataclass(frozen=True, slots=True)
class GmshContextManager:
    """Context manager for GMSH.

    Attributes
    ----------
    options : GmshOptions, optional, default GmshOptions()
    """

    options: GmshOptions = field(default_factory=GmshOptions)

    def __enter__(self: Self) -> Self:
        # Initialize the Gmsh API.
        gmsh.initialize()

        gmsh.option.set_number("General.Terminal", int(self.options.terminal_output))
        gmsh.option.set_number("Mesh.ElementOrder", self.options.element_order)

        # Add a new model and set it as the current model.
        if self.options.filename == "":
            gmsh.model.add("mesh")
        else:
            gmsh.model.add(self.options.filename)

        return self

    def __exit__(
        self: GmshContextManager,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None:
                if self.options.optimize_method is not None:
                    # optimize the mesh
                    gmsh.model.mesh.optimize(self.options.optimize_method)

                if self.options.renumber_nodes is not None:
                    # Renumber the nodes to improve the matrix bandwidth.
                    old, new = gmsh.model.mesh.compute_renumbering(
                        self.options.renumber_nodes
                    )
                    gmsh.model.mesh.renumber_nodes(old, new)

                if self.options.show_gui:
                    # Create and run the FLTK graphical user interface.
                    gmsh.fltk.run()

                if self.options.filename != "":
                    gmsh.write(f"{self.options.filename}.msh")
        finally:
            # Finalize the Gmsh API.
            gmsh.finalize()
