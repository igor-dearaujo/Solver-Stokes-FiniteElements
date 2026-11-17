from numpy.typing import NDArray
from scipy.sparse import sparray
from scipy.sparse.linalg import spsolve


def sparse_solve(A: sparray, b: NDArray) -> NDArray:
    """Solve the linear system Ax = b, where A is a sparse matrix.

    Parameters
    ----------
    A : sparse matrix (scipy.sparse)
        System matrix (any format supported by scipy.sparse).
    b : NDArray[float64 | complex128]
        Right-hand side.

    Returns
    -------
    x : NDArray[float64 | complex128]
        Solution of the system.
    """
    A = A.tocsc()
    return spsolve(A, b)
