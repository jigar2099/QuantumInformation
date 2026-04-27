from __future__ import annotations
from typing import Iterable, Sequence, TypeAlias
import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import (Statevector, DensityMatrix, operators,
                                 partial_trace, Pauli, state_fidelity)

from src.QuInfo.utils.DensityMatrix import (to_numpy_matrix, pretty_matrix, is_square_matrix, is_hermitian,
                                            eigenvalues_of_matrix,
                                            is_positive_semidefinite, trace_of_matrix)

np.set_printoptions(precision=4, suppress=True)

MatrixLike: TypeAlias = ArrayLike
VectorLike: TypeAlias = ArrayLike
ComplexMatrix: TypeAlias = NDArray[np.complex128]
Complexvector: TypeAlias = NDArray[np.complex128]
RealVector: TypeAlias = NDArray[np.float64]
POVM: TypeAlias = Sequence[MatrixLike]


def to_complex_vector(vector: VectorLike) -> Complexvector:
    """
    convert input vector to complexc 1s numpy vector
    :param vector:
    :return: complex numpy vector
    """
    vec = np.asanyarray(vector, dtype=np.complex128)
    if vec.ndim != 1:
        raise ValueError(
            f"Expected a 1D vector, got {vec.ndim}D vector with shape {vec.shape}"
        )
    return vec

def is_density_matrix(matrix: MatrixLike, tolerance: float = 1e-10, verbose: bool = False) -> bool:
    """
    apply is_square_matrix, is_hermitian, and is_positive_semidefinite to finalize
    the matrix is valid density matrix or not
    hermiticity: A = A-dagger(conjugate transpose of A)
    Trace: trace(A) = sum(eigenvalues) := 1
    Positive semidefinite: eigenvalues >= -tol
    :param matrix: input matrix
    :param tolerance: numerical constraint
    :param verbose: if true then print the report
    :return: bool (true/false)
    """
    try:
        rho = to_numpy_matrix(matrix)
        square = is_square_matrix(rho)
        hermitian = is_hermitian(rho, tolerance=tolerance)
        psd = is_positive_semidefinite(rho, tolerance=tolerance)
        trace_value = trace_of_matrix(rho)
        trace_one = np.isclose(trace_value, 1, atol=tolerance)
        valid = square and hermitian and psd and trace_one
    except ValueError:
        valid = False
        square = hermitian = psd = trace_one = False
        trace_value = np.nan

    if verbose:
        print("=" * 60)
        print("Density matrix validation")
        print("=" * 60)
        print("Square?       ", square)
        print("Hermitian?    ", hermitian)
        print("PSD?          ", psd)
        print("Trace:        ", trace_value)
        print("Trace = 1?    ", trace_one)
        print("Valid rho?    ", valid)
        print()
    return bool(valid)


def normalize_density_matrix(matrix: MatrixLike, tolerance: float = 1e-10) -> ComplexMatrix:
    """
    simply normalization of the density matrix
    :param matrix: input matrix
    :param tolerance: numerical constraint
    :return: Complex matrix

    NOTE: this function applies following calculation: Normalized_rho = rho/trace_of_rho
    the limitation of this functionis that the input matrix must be valid density matrix
    """
    rho = to_numpy_matrix(matrix)

    if not is_square_matrix(rho):
        raise ValueError("Input matrix must be a square matrix")
    tr = np.trace(rho)
    if abs(tr) < tolerance:
        raise ValueError(" Can't normalize a matrix with zero/near-zero trace")
    return rho / tr

def projector_from_state(state: VectorLike, normalize: bool=True)->ComplexMatrix:
    """
    construction of ranl-1 projector from state vector
    :param state: input statevector
    :param normalize: if true, then normalize the statevector before projector construction
    :return: complex matrix
    NOTE: projectors are the ideal projective measurement outcomes, |psi><psi|
    """
    vec = to_complex_vector(state)
    norm = np.linalg.norm(vec)

    if normalize:
        if norm < 1e-10:
            raise ValueError("Can't normalize a statevector with zero/near-zero norm")
        vec = vec/norm
    ket = vec.reshape(-1,1)
    bra = ket.conj().T

    return ket @ bra

def normalize_statevector(state:VectorLike, tolerance:float=1e-10)->Complexvector:
    """
    simply normalization of the statevector
    :param state: input statevector
    :param tolerance: numerical constraint
    :return: normalized statevector
    """
    vec= to_complex_vector(state)
    norm = np.linalg.norm(vec)

    if norm < tolerance:
        raise ValueError("can't normalize a statevector with zero/near-zero norm")
    return vec/norm











if __name__ == "__main__":
    print("*" * 25 + "Function-1" + "*" * 25)
    print(to_numpy_matrix(np.array([[1, 2], [3, 4]])))
    print("*" * 25 + "Function-2" + "*" * 25)
    pretty_matrix(np.array([[1, 2, 3], [3, 4, 5]]), "TestMatrix")
    print("*" * 25 + "Function-3" + "*" * 25)
    print(is_square_matrix(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])))
    print(is_square_matrix(np.array([[1, 2, 3], [3, 4, 5]])))
    print("*" * 25 + "Function-4" + "*" * 25)
    print(is_hermitian(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])))
    print(is_hermitian(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])))
    print("*" * 25 + "Function-5" + "*" * 25)
    #print(eigvals_hermitian(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])))
    #print(eigvals_hermitian(np.array([[1, 2, 3], [3, 4, 5]])))
    print("*" * 25 + "Function-6" + "*" * 25)
    #print(is_positive_semidefinite(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])))
    #print(is_positive_semidefinite(np.array([[1, 2, 3], [3, 4, 5]])))
    print("*" * 25 + "Function-7" + "*" * 25)
    print(is_density_matrix(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])))
    print(is_density_matrix(np.array([[1, 2, 3], [3, 4, 5]])))
    print("*" * 25 + "Function-8" + "*" * 25)
    print(normalize_density_matrix(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])))
    #print(normalize_density_matrix(np.array([[1, 2, 3], [3, 4, 5]])))
    print("*" * 25 + "Function-9" + "*" * 25)
    print(projector_from_state(np.array([1, 0, 0, 0])))
    print(projector_from_state(np.array([1, 0, 0, 0]), normalize=False))
    print("*" * 25 + "Function-10" + "*" * 25)
    print(normalize_statevector(np.array([1, 0, 0, 0])))
    print(normalize_statevector(np.array([1, 0, 0, 0]), tolerance=1e-10))