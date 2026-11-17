from dataclasses import dataclass

from numpy import array, float64
from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True, slots=True)
class Quadrature:
    order: int
    nodes: NDArray[float64]  # shape (nb nodes, dim)
    weight: NDArray[float64]  # shape (1,)

    def __str__(self) -> str:
        dim = self.nodes.shape[1]
        nodes_width = 8 * dim
        lines = [f"order: {self.order}", f"{'nodes':<{nodes_width}} | weight"]
        for node, w in zip(self.nodes, self.weight):
            coords = ", ".join(f"{c:.4f}" for c in node)
            lines.append(f"[{coords}] | {w:.4f}")
        return "\n".join(lines)


TRIANGLE_QUAD_P1 = Quadrature(
    order=1,
    nodes=array([[3.333333333333333e-01, 3.333333333333333e-01]], dtype=float64),
    weight=array([5.0e-01], dtype=float64),
)

TRIANGLE_QUAD_P2 = Quadrature(
    order=2,
    nodes=array(
        [
            [1.6666666666666666e-01, 1.6666666666666666e-01],
            [1.6666666666666666e-01, 6.666666666666666e-01],
            [6.666666666666666e-01, 1.6666666666666666e-01],
        ],
        dtype=float64,
    ),
    weight=array(
        [1.6666666666666666e-01, 1.6666666666666666e-01, 1.6666666666666666e-01],
        dtype=float64,
    ),
)

TRIANGLE_QUAD_P4 = Quadrature(
    order=4,
    nodes=array(
        [
            [1.0810301816807023e-01, 4.459484909159649e-01],
            [4.459484909159649e-01, 1.0810301816807023e-01],
            [4.459484909159649e-01, 4.459484909159649e-01],
            [9.157621350977074e-02, 9.157621350977074e-02],
            [9.157621350977074e-02, 8.168475729804585e-01],
            [8.168475729804585e-01, 9.157621350977074e-02],
        ],
        dtype=float64,
    ),
    weight=array(
        [
            1.1169079483900574e-01,
            1.1169079483900574e-01,
            1.1169079483900574e-01,
            5.4975871827660935e-02,
            5.4975871827660935e-02,
            5.4975871827660935e-02,
        ],
        dtype=float64,
    ),
)
