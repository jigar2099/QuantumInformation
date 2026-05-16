#from __future__ import annotations
from __future__ import annotations
import os
from platform import system

from scipy.linalg import eigvals

#os.chdir("..")

from typing import Sequence, TypeAlias
import numpy as np

from numpy.typing import NDArray
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from src.QuInfo.utils.DensityMatrix import (MatrixLike, to_numpy_matrix, is_square_matrix,
                                            is_valid_density_matrix, densityMatrix_from_label)
from src.QuInfo.utils.genMeasurements import (VectorLike, Complexvector, ComplexMatrix, to_complex_vector,
                                              normalize_statevector, projector_from_state, matrix_square_root_psd)

RealVector: TypeAlias = NDArray[np.float64]


def spectral_decomposition_density_matrix(matrix: MatrixLike, tolerance: float = 1e-12) -> tuple[RealVector, list[Complexvector]]:
    """
    Computes the spectral decomposition density matrix of a given matrix.

    :param matrix: input density matrix
    :param tolerance: tolerance parameter
    :return: tuple of non zero egvals and the corresponding egvects
    valueError: if the input is no t th valid density martix

    NOTE: every density matrix can be written as rho = sum_k p_k |u_k><u_k|
    here p_k is the probability and |u_k> are orthonormal egvecs.
    this decomposition represents a mixed quantum state as a classical mixture of pure states
    """
    rho = to_numpy_matrix(matrix)
    if not is_valid_density_matrix(rho, tolerance=1e-8):
        raise ValueError("input density matrix is not a valid density matrix")
    egvals, egvects = np.linalg.eig(rho)
    kept_values: list[float] = []
    kept_vectors: list[Complexvector] = []

    for idx, value in enumerate(egvals):
        if value > tolerance:
            kept_values.append(float(np.real_if_close(value)))
            kept_vectors.append(egvects[:, idx].astype(np.complex128))
    return np.asarray(kept_values, dtype=np.complex128), kept_vectors


def canonical_purification_from_spectral_decomposition(matrix: MatrixLike, tolerance: float = 1e-12) -> tuple[Complexvector, RealVector, list[Complexvector]]:
    """
    canonical purification of a denstiy matrix
    :param matrix: input density matrix
    :param tolerance: constraint, egval cutoff
    :return: tuple containing the purification statevector, the nonzero egvals, and related egvects
    ValueError: if the input is not a valid density matrix

    NOTE: if rho = sum_k p_k |u_k><u_k|, then one purification is
          |psi> = sum_k sqrt(p_k) |u_k> tensot |k>
          where |k> are orthonormal ancilla states. Tracing out the ancilla recovers the original
          density matrix rho
    """
    rho = to_numpy_matrix(matrix)
    egvals, egvects = spectral_decomposition_density_matrix(matrix, tolerance)
    system_dim = rho.shape[0]
    ancilla_dim = len(egvals)
    purification = np.zeros(system_dim * ancilla_dim, dtype=np.complex128)
    for idx, probability in enumerate(egvals):
        ancilla_basis = np.zeros(ancilla_dim, dtype=np.complex128)
        ancilla_basis[idx] = 1
        purification += np.sqrt(probability) * np.kron(egvects[idx], ancilla_basis)
    return normalize_statevector(purification), egvals, egvects

def canonical_purification_fixed_ancilla_dim(matrix:MatrixLike, ancilla_dim: int, tolerance: float = 1e-12) -> Complexvector:
    """
    Build canonical purificaiton using a chosen ancilla dimension
    :param matrix: input density matrix
    :param ancilla_dim: dimension of the ancilla Hilbert space
    :param tolerance:  nunerical constraint
    :return: complex vector; purification statevector in system tensor ancilla space
    ValueERror: if the input is not a valid density matrix
                if ancilla_dim is smaller than the rank of rho
    NOTE: A density matrix of rank r needs an ancilla space of dimension at least r for this canonical construction.
         chosing a large ancilla dimension is allowed and can be useful when comparing purificaitons in the same total Hilbert space.
    """
    rho = to_numpy_matrix(matrix)
    eigvals, egvects = spectral_decomposition_density_matrix(rho, tolerance=tolerance)
    rank = len(eigvals)
    system_dim = rho.shape[0]
    if ancilla_dim < rank:
        raise ValueError(f"Ancilla dimension {ancilla_dim} is smaller than the rank-{rank} of rho")
    purification = np.zeros(system_dim * ancilla_dim, dtype=np.complex128)
    for idx, probability in enumerate(eigvals):
        ancilla_basis = np.zeros(ancilla_dim, dtype=np.complex128)
        ancilla_basis[idx] = 1
        purification += np.sqrt(probability) * np.kron(egvects[idx], ancilla_basis)
    return normalize_statevector(purification)

def reduced_system_from_purification(state: VectorLike, system_dims: Sequence[int], trace_out_subsystems: Sequence[int])->ComplexMatrix:
    """
    ocmpute reduced density matrx from a purified state
    :param state: fulll putr state vector
    :param system_dims: dimensions of the subsystems
    :param trace_out_subsystems: indices of the subsystems to trace out
    :return: reduced density matrix after tracing out the selected subsystems
    ValueError: if the statevector length does not match the product of system_dims
                if the statevector cannot be normalized

    NOTE: if |psi> is a purification of rho, then rho is recovered by tracing out the extra ancilla/engironmant subsystem
        rho = Tr_ancilla(|psi><psi|)
    """
    psi = normalize_statevector(state)
    expected_dim = int(np.prod(system_dims))
    if psi.size != expected_dim:
        raise ValueError(f"psi size {psi.size} does not match expected dimension {expected_dim}")
    full_density_matrix = DensityMatrix(psi, dims=system_dims)
    reduced_density_matrix = partial_trace(full_density_matrix, list(trace_out_subsystems))
    return reduced_density_matrix.data.astype(np.complex128)

def schmidt_decomposition(state: VectorLike, dim_x:int, dim_y:int, tolerance: float = 1e-12 ) -> tuple[RealVector, list[Complexvector], list[Complexvector]]:
    """
    compute schmidt decomposition of a bipartite pure state
    :param state: bipartite pure statevector in X tensor Y
    :param dim_x: Dimension of subsystemm X
    :param dim_y: Dimension of subsystemm Y
    :param tolerance: numerical constraint
    :return: schmidt coefficients, left schmidt-vectors, and right schmidt-vectors
    ValueError: if the state vector length does not match the product of dim_x and dim_y
                if the statevecctor cannot be normalized

    NOTE: a bipartite pure state can be written as
        |psi> = sum_k s_k |x_k> tensor |y_k>
        where, s_k are nonnegative Schmidt coefficients. More than one nonzero
        Schmidt coefficient means the state is entangled.
    """
    psi = normalize_statevector(state)
    if psi.size != dim_x * dim_y:
        raise ValueError(f"psi size {psi.size} does not match expected dimension {dim_x * dim_y}")
    coefficient_matrix = psi.reshape(dim_x, dim_y)
    left_matrix, singular_values, right_matrix_dagger = np.linalg.svd(
        coefficient_matrix, full_matrices=False,
    )
    keep = singular_values > tolerance

    schmidt_coefficients = singular_values[keep].astype(np.float64)
    left_matrix = left_matrix[:, keep]
    right_matrix_dagger = right_matrix_dagger[keep, :]

    left_vectors = [
        left_matrix[:, idx].astype(np.complex128)
        for idx in range(len(schmidt_coefficients))]
    right_vectors = [
        right_matrix_dagger[idx, :].astype(np.complex128)
        for idx in range(len(schmidt_coefficients))]

    return schmidt_coefficients, left_vectors, right_vectors


def reconstruct_from_schmidt( schmidt_coefficients: Sequence[float], left_vectors: Sequence[VectorLike], right_vectors: Sequence[VectorLike]) -> Complexvector:
    """
    Reconstruct a bipartite statevector from schmidt decompositioin..
    :param schmidt_coefficients: schmidt coefficients, Left and right schmidt vectors
    :param left_vectors:
    :param right_vectors:
    :return: reconstructed normalized bipartite statevector
    ValuError: if the schmidt decomposition data is empty
               if the coefficient and vector lists do not have the same length

    NOTE: This reverses the Schmidt expansion
        |psi> = sum_k s_k |x_k> tensor |y_k>
        using the stored coefficients and subsystem vectors.
    """
    if len(schmidt_coefficients) == 0:
        raise ValueError(f"can not reconstruct from an empty schmidt decomposition")
    if not (len(schmidt_coefficients) == len(left_vectors) == len(right_vectors)):
        raise ValueError("Schmidt decomposition lengths do not match")

    dim_x = to_complex_vector(left_vectors[0]).size
    dim_y = to_complex_vector(right_vectors[0]).size
    psi = np.zeros(dim_x * dim_y, dtype=np.complex128)

    for coefficient, x_vector, y_vector in zip(schmidt_coefficients, left_vectors, right_vectors):
        psi += coefficient * np.kron(
            to_complex_vector(x_vector), to_complex_vector(y_vector)
        )
    return normalize_statevector(psi)

def random_unitary(dimension: int, rng:np.random.Generator | None = None) -> ComplexMatrix:
    """
    Generate random unitary matrix using QR decomposition
    :param dimension: dimension of the unitary matrix
    :param rng: optional Numpy random number generator
    :return: random unitary matrix of shape (dimension, dimension)
    ValueError: if dimension is not positive

    NOTE: A complex random matrix can be converted into a unitary matrix using QR decomposition, followed by a phase correction on the diagonal of R
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if rng is None:
        rng = np.random.default_rng(123)
    random_matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    q_matrix, r_matrix = np.linalg.qr(random_matrix)
    diagonal = np.diag(r_matrix)
    phases = diagonal / np.abs(diagonal)
    return (q_matrix @ np.diag(np.conj(phases))).astype(np.complex128)

def is_unitary(matrix:MatrixLike, tolerance: float = 1e-12) -> bool:
    """
    check if matrix is unitary
    :param matrix: matrix to be checked
    :param tolerance: numerical constraint
    :return: bool
    u_dagger * u = I
    """
    mat = to_numpy_matrix(matrix)
    if not is_square_matrix(mat):
        return False

    identity = np.eye(mat.shape[0], dtype=np.complex128)
    return bool(np.allclose(mat.conj().T @ mat, identity, atol=tolerance))

def apply_unitary_on_second_subsystem(state_xy: VectorLike, unitary_y:MatrixLike, dim_x:int, dim_y:int) -> Complexvector:
    """
    Apply unitary opertation on the second subsystem of a bipartite state
    :param state_xy: bipartite statevector in X and Y tensors
    :param unitary_y: unitary matrix acting on subsystem Y
    :param dim_x: dimension of subsystem X
    :param dim_y: dimension of subsystem Y
    :return: new statevector after applying I_X tensor U_Y
    ValueError: if the statevector length is not dim_x * dim_y
                if unitary_y does not have shape (dim_x, dim_y)
                if unitary_y is not unitary
    NOTE: Purifications of the same density matrix are related by a unitary acting only on the ancilla/environmens subsystem
    """
    psi = normalize_statevector(state_xy)
    unitary = to_numpy_matrix(unitary_y)
    if psi.size != dim_x * dim_y:
        raise ValueError("unitary_y does not have shape (dim_x, dim_y)")
    if unitary.shape != (dim_x, dim_y):
        raise ValueError(f" unitary must have shape {(dim_x, dim_y)}"
                         f" but got {unitary.shape}")
    if not is_unitary(unitary):
        raise ValueError("unitary is not unitary")

    full_unitary = np.kron(np.eye(dim_x, dtype=np.complex128), unitary)
    return (full_unitary @ psi).astype(np.complex128)

def overlap(state_1: VectorLike, state_2: VectorLike,squared: bool = False) -> float:
    """
    compute the absolute inner-product overlap between two pure states
    :param state_1: first statevector
    :param state_2: second statevector
    :param squared: if Truem return the squared overlap
    :return: absolute overlap or squared absolute overlap
    ValueError: if two statevectors have different dimensions
                if either statevector cannot be normalized
    NOTE: for pure quantum statesd, the squared overlap is |<phi|phi>|^2
          which is the same fidelity convention used by qiskit for pure state
    """
    psi = normalize_statevector(state_1)
    phi = normalize_statevector(state_2)
    if psi.size != phi.size:
        raise ValueError("psi and phi statevectors must have the same dimension")
    value  = np.abs(np.vdot(psi, phi))
    if squared:
        return float(value ** 2)
    return float(value)

def manual_fidelity(rho: MatrixLike, sigma: MatrixLike) -> float:
    """
    compute density-matrix fidelity manually
    :param rho: first density matrix
    :param sigma: second density matrix
    :return: fidelity value using the squared qiskit convention
    NOTE: this computes F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
          Fidelity measures how similar two quantum states are, a value of 1 means identical states, while 0 means perfectly distinguishable states
    """
    rho_matrix = to_numpy_matrix(rho)
    sigma_matrix = to_numpy_matrix(sigma)
    if rho_matrix.shape != sigma_matrix.shape:
        raise ValueError("rho and sigma must have the same dimension")
    if not is_valid_density_matrix(sigma_matrix, tolerance=1e-8):
        raise ValueError("sigma_matrix must have the same dimension")
    if not is_valid_density_matrix(rho_matrix, tolerance=1e-8):
        raise ValueError("rho_matrix must have the same dimension")
    sqrt_rho = matrix_square_root_psd(rho_matrix)
    middle = sqrt_rho @ sigma_matrix @ sqrt_rho
    sqrt_middle = matrix_square_root_psd(middle)
    root_value = np.real_if_close(np.trace(sqrt_middle))

    return float(root_value**2)

def root_fidelity(rho:MatrixLike, sigma:MatrixLike) -> float:
    """
    compute the root fidelity between two density matrices.

    :param rho: first density matrix
    :param sigma: second density matrix
    :return: root fidelity value
    ValueError: if rho and sigma do not have the same dimension, either input is not a valid density matrix
    NOTE: root fidelity is sqrt(F(rho, sigma))
          Uhlmann's theorem says that this equals the maximum possible ovelap between purifications of rho and sigma
    """
    return float(np.sqrt(manual_fidelity(rho, sigma)))




if __name__ == "__main__":
    rho_test = np.array([[0.5, 0.0 - 0.3j],
                         [0.0 + 0.3j, 0.5]])
    bell_state = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)

    print("*" * 25 + "Function-1" + "*" * 25)
    print(spectral_decomposition_density_matrix(rho_test))

    print("*" * 25 + "Function-2" + "*" * 25)
    purification, egvals, egvects = canonical_purification_from_spectral_decomposition(rho_test)
    print(purification, egvals, egvects)

    print("*" * 25 + "Function-3" + "*" * 25)
    fixed_purification = canonical_purification_fixed_ancilla_dim(
        rho_test,
        ancilla_dim=2,
        tolerance=1e-12,
    )
    print(fixed_purification)

    print("*" * 25 + "Function-4" + "*" * 25)
    print(reduced_system_from_purification(
        fixed_purification,
        system_dims=[2, 2],
        trace_out_subsystems=[1],
    ))

    print("*" * 25 + "Function-5" + "*" * 25)
    schmidt_coefficients, left_vectors, right_vectors = schmidt_decomposition(
        bell_state,
        dim_x=2,
        dim_y=2,
    )
    print(schmidt_coefficients, left_vectors, right_vectors)

    print("*" * 25 + "Function-6" + "*" * 25)
    print(reconstruct_from_schmidt(
        schmidt_coefficients,
        left_vectors,
        right_vectors,
    ))
    print("*" * 25 + "Function-7" + "*" * 25)
    print(random_unitary(
        2, rng=np.random.default_rng(123)
    ))
    print("*" * 25 + "Function-8" + "*" * 25)
    print(is_unitary(bell_state, tolerance=1e-12))
    print("*" * 25 + "Function-9" + "*" * 25)
    unitary_y = random_unitary(2, rng=np.random.default_rng(456))
    print(apply_unitary_on_second_subsystem(
        bell_state,
        unitary_y,
        dim_x=2,
        dim_y=2,
    ))

    print("*" * 25 + "Function-10" + "*" * 25)
    print(overlap(
        bell_state,
        reconstruct_from_schmidt(
            schmidt_coefficients,
            left_vectors,
            right_vectors,
        ),
        squared=True,
    ))

    print("*" * 25 + "Function-11" + "*" * 25)
    sigma_test = densityMatrix_from_label("+")
    print(manual_fidelity(
        rho_test,
        sigma_test,
    ))

    print("*" * 25 + "Function-12" + "*" * 25)
    print(root_fidelity(
        rho_test,
        sigma_test,
    ))