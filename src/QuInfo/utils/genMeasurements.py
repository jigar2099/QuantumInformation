from __future__ import annotations
from typing import Iterable, Sequence, TypeAlias
import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import (Statevector, DensityMatrix, Operator,
                                 partial_trace, Pauli, state_fidelity)

from src.QuInfo.utils.DensityMatrix import (to_numpy_matrix, pretty_matrix, is_square_matrix, is_hermitian,
                                            eigenvalues_of_matrix, densityMatrix_from_label, bloch_vector,
                                            is_positive_semidefinite, trace_of_matrix)

np.set_printoptions(precision=4, suppress=True)

MatrixLike: TypeAlias = ArrayLike
VectorLike: TypeAlias = ArrayLike
ComplexMatrix: TypeAlias = NDArray[np.complex128]
Complexvector: TypeAlias = NDArray[np.complex128]
RealVector: TypeAlias = NDArray[np.float64]
POVM: TypeAlias = Sequence[MatrixLike]


def matrix_dimension(matrix: MatrixLike)->int:
    """
    return hilbert-space dimention of the square matrix
    :param matrix: square matrix
    :return: matrix dimension
    """
    mat = to_numpy_matrix(matrix)
    if not is_square_matrix(mat):
        raise ValueError(
            f"Expected a square matrix, got {mat.ndim}D matrix with shape {mat.shape}"
        )
    return mat.shape[0]


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


def standard_basis_projectors() -> list[ComplexMatrix]:
    """
    return computational basis projectors for one qubit
    :return: list of |0><0| and |1><1|
    NOTE: these projector define standard Z-basis measurement
    """
    ket_0 = np.array([1, 0], dtype=np.complex128)
    ket_1 = np.array([0, 1], dtype=np.complex128)

    return [
        projector_from_state(ket_0),
        projector_from_state(ket_1)
    ]


def povm_sum(povm_elements: POVM) -> ComplexMatrix:
    """
    sum all povm elements
    :param povm_elements: seq of povem elements
    :return: matrix sum of all povm elements
    NOTE: valid povm must satisfy,
    p0+p1+p2+...+pn = 1
    """
    if len(povm_elements) == 0:
        raise ValueError("POVM list can not be empty")

    matrices = [to_numpy_matrix(P) for P in povm_elements]
    first_shape = matrices[0].shape

    for idx, P in enumerate(matrices):
        if P.shape != first_shape:
            raise ValueError(
                f"All POVM elements must have same shape"
                f" but got {P.shape} and {matrices[0].shape} at index {idx}"
            )
    total = np.zeros(first_shape, dtype=np.complex128)

    for P in matrices:
        total += P
    return total


def validate_povm(povm_elements: POVM, tolerance: float = 1e-10, name: str = "POVM", verbose: bool = True) -> bool:
    """
    Validate whether a collection of matrices forms a POVM.
    :param povm_elements: sequence of povmn elements
    :param tolerance: numerical constraint
    :param name: str, optional
    :param verbose: bool, optional
    :return: bool, if the input is a valid POVM
    NOTE: povm describe the most general quantum measurements, for a quantum state,
    rho, the probability of outcome 'a' is
    p(a) = Tr(P_a rho)
    besides povm should be hermitian, positive semidefinite, and sum of all elements equals the identity matrix
    """
    if len(povm_elements) == 0:
        raise ValueError("POVM list can not be empty")

    matrices = [to_numpy_matrix(P) for P in povm_elements]
    dim = matrix_dimension(matrices[0])
    identity = np.eye(dim, dtype=np.complex128)

    all_hermitian = True
    all_psd = True

    if verbose:
        print("=" * 60)
        print(f"{name} validation report")
        print("=" * 60)
    for idx, P in enumerate(matrices):
        if P.shape != (dim, dim):
            raise ValueError(
                f"All POVM elements must have shape ({dim}, {dim})"
                f" but got {P.shape} at index {idx}"
            )
        hermitian = is_hermitian(P, tolerance=tolerance)
        psd = is_positive_semidefinite(P, tolerance=tolerance)
        eigvals = eigenvalues_of_matrix(P)

        all_hermitian = all_hermitian and hermitian
        all_psd = all_psd and psd

    if verbose:
        print(f"POVM element {idx} validation report")
        print("=" * 60)
        print(f"Element P_{idx}")
        pretty_matrix(P, f"P_{idx}")
        print(f"Hermitian?    {hermitian}")
        print(f"PSD?          {psd}")
        print(f"Eigenvalues:  {eigvals}")
        print(f"Identity:     {np.allclose(P, identity)}")
        print()

    total = povm_sum(matrices)
    sums_to_identity = np.allclose(total, identity, atol=tolerance)

    if verbose:
        pretty_matrix(total, "POVM sum")
        print(f"POVM sum = 1?    {sums_to_identity}")
        print()

    return bool(all_hermitian and all_psd and sums_to_identity)


def measurement_probabilities(rho: MatrixLike, povm_elements: POVM, tolerance: float = 1e-10,
                              validate: bool = True) -> RealVector:
    """
    Calculate measurement outcome probs for a POVM
    for density matrix rho and povm elements "p_a", the probability
    of outcome "a" is
    p(a) = Tr(P_a rho)

    :param rho: density matrix (d,d)
    :param povm_elements:  sequence of povm elements each of shape (d,d)
    :param tolerance: numerical constraint
    :param validate: bool, optional
    :return: RealVector, 1d array of measurement probabilites
    ValueError: if dimensions are inconsistent, or povm is not valid

    NOTE: small img parts from floating point numbers are ignored using np.real_if_close
    """
    rho_mat = to_numpy_matrix(rho)
    dim = matrix_dimension(rho_mat)

    if validate:
        valid_povm = validate_povm(povm_elements, tolerance=tolerance, name="POVM", verbose=False)
        if not valid_povm:
            raise ValueError("POVM is not valid")
    probs: list[float] = []
    for idx, P in enumerate(povm_elements):
        P_mat = to_numpy_matrix(P)

        if P_mat.shape != (dim, dim):
            raise ValueError(
                f"POVM element {idx} must have shape ({dim}, {dim})"
                f" but got {P_mat.shape}"
            )
        probability = np.trace(P_mat @ rho_mat)
        probability = np.real_if_close(probability)
        probs.append(probability)
    probs_array = np.asarray(probs, dtype=np.float64)
    probs_array[np.abs(probs_array) < tolerance] = 0
    return probs_array


def measurement_channel_output(rho: MatrixLike, povm_elements: POVM, tolerance=1e-10) -> ComplexMatrix:
    """
    return classical output density matrix of a measurement channel
    rho -> sum_a Tr(P_a rho) |a><a|
    this produces a diagonal matrix, whose diagonal entries are the measurement probabilities
    :param rho: MatrixLike,
    :param povm_elements:POVM
    :param tolerance:numerical constraint
    :return: diagonal matrix coontaining the classical outcome probabilities

    NOTE: the output is diagonal because after measurement, we keep only classical information
    about which outcome occured
    """
    probabilities = measurement_probabilities(rho, povm_elements, tolerance=tolerance, validate=True)
    return np.diag(probabilities).astype(np.complex128)


# def bloch_coords_from_dm(rho:MatrixLike, tolerance:float=1e-10)->RealVector:
#     """
#     Compute the bloch vector of a one-qubit density matrix
#     any one-qubit density matrix rho can be represented in Bloch coordinates as
#     rho = 1/2 * (I + r_x X + r_y Y + r_z Z)
#     where
#         r_x = Tr(rho X)
#         r_y = Tr(rho Y)
#         r_z = Tr(rho Z)
#     :param rho: MatrixLike, one qubit matrix of shape (2,2)
#     :param tolerance:
#     :return: bloch vector of shape (3,)
#     ValueError: if rho is not a one-qubit density matrix (2,2)
#
#
#     """

def matrix_square_root_psd(matrix: MatrixLike, tolerance: float = 1e-10) -> ComplexMatrix:
    """
    compute square root of a psd hermitian matrix
    For a PSD matrix P with spectral decomposition
        P = V diag(lambda_i) V†
    the square root is
        sqrt(P) = V diag(sqrt(lambda_i)) V†
    :param matrix: psd hermitian matrix
    :param tolerance:
    :return: sqrt of input matrix
    ValueError: if the input is not hermitian or psd
    """
    mat = to_numpy_matrix(matrix)

    if not is_hermitian(mat, tol=tolerance):
        raise ValueError("Matrix square root requires a Hermitian matrix.")
    if not is_positive_semidefinite(mat, tol=tolerance):
        raise ValueError("Matrix square root requires a positive semidefinite matrix.")
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    # Clip tiny negative numerical values.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sqrt_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    return sqrt_matrix.astype(np.complex128)


def post_measurement_state_via_sqrtP(rho: MatrixLike, P: MatrixLike, tolerance: float = 1e-10) -> tuple[
    float, ComplexMatrix]:
    """
    compute post measurement state using the sqrt(P) update rule
    For POVM element P_a, one possible non-destructive update rule is
        rho_a = sqrt(P_a) rho sqrt(P_a) / Tr(P_a rho)
    :param rho: matrix
    :param P:matrix
    :param tolerance:
    :return:tuple (probability, post measurement state)
    ValueError: if outcome probability is approx 0

    NOTE: A povm alone gives probabilities.
    the post measurement state also depends on the measurement instrument
    the sqrt(P) rule is one natural choice, but not only possible choice
    """
    rho_mat = to_numpy_matrix(rho)
    P_mat = to_numpy_matrix(P)

    if rho_mat.shape != P_mat.shape:
        raise ValueError(
            f"POVM element P_a must have shape ({rho_mat.shape[0]}, {rho_mat.shape[1]})"
            f" but got {P_mat.shape}"
        )
    sqrt_P = matrix_square_root_psd(P_mat, tolerance=tolerance)
    probability = np.trace(P_mat @ rho_mat)
    probability = float(np.real_if_close(probability))
    if probability < tolerance:
        raise ValueError("Outcome probability is approx 0")
    post_state = sqrt_P @ rho_mat @ sqrt_P.conj().T
    post_state = post_state / probability
    return probability, post_state.astype(np.complex128)

def partial_measurement_probabilities(rho_XZ: MatrixLike, povm_X:POVM, subsystem_dim: int = 2, tolerance: float=1e-10)->RealVector:
    """
    Compute probabilities when measuring only the first subsystem.
    :param rho_XZ: matrix (full density matrix of the combined system of X and Z subsystems)
    :param povm_X: povm (povm of the X subsystem)
    :param subsystem_dim: int, optional (dimenstion of the second subsystem)
    :param tolerance: numerical constraint
    :return: realvector (1d array) of partial measurement probabilities (for the first subsystem)
    ValueError: if dimensions are inconsistent, or povm is not valid

    NOTE: what we want here is, we want to measure the probability of measuring the first subsystem
    for bipartite system of two qubits, the shape is (4,4), in this case the dimension of the second subsystem 2
    """
    rho_mat = to_numpy_matrix(rho_XZ)
    dim_total = matrix_dimension(rho_mat)

    I = np.eye(subsystem_dim, dtype=np.complex128)
    probabilities: list[float] = []
    for idx, P in enumerate(povm_X):
        P_mat = to_numpy_matrix(P)
        M = np.kron(P_mat, I)
        if M.shape != (dim_total, dim_total):
            raise ValueError(
                f"Measurement operator for POVM element {idx} must have shape ({M.shape})"
                f" but got {rho_mat.shape}"
            )
        probability = np.trace(M @ rho_mat)
        probability = float(np.real_if_close(probability))
        if abs(probability) < tolerance:
            probability = 0
        probabilities.append(probability)
    return np.asarray(probabilities, dtype=np.float64)


def conditional_state_of_second_qubit(rho_XZ: MatrixLike, P:MatrixLike, tolerance:float=1e-10)->[float, ComplexMatrix]:
    """
    compute the conditional state of the second qubit after measuring the first qubit
    if the first qubut is measured with povm element P_a, the conditional state of the second qubit can be given as,
    sigma_Z^(a) = Tr_X[(P_a ⊗ I) rho_XZ] / Tr[(P_a ⊗ I) rho_XZ]

    :param rho_XZ: matrix (full density matrix of the combined system of X and Z subsystems)
    :param P: one qubit povm element action on the first qubit
    :param tolerance:
    :return: tuple (probability, conditional state)
                probability: prob of the measurement outcome
                conditional_state: resulting state of the second qubit
    ValueError: if the probability of the outcome is approximately zero

    NOTE: for the qntangled states, measuring the first subsystem can change our conditional description of the second subsystem
    """
    rho_mat= to_numpy_matrix(rho_XZ)
    P_mat = to_numpy_matrix(P)
    if rho_mat.shape != (4,4):
        raise ValueError(f"rho_XZ must have shape (4,4) but got {rho_mat.shape}")
    if P_mat.shape != (2,2):
        raise ValueError(f"P must have shape (2,2) but got {P_mat.shape}")
    I = np.eye(2, dtype=np.complex128)
    M = np.kron(P_mat, I)

    unnormalized_joint = M @ rho_mat
    probability = np.trace(unnormalized_joint)
    probability = float(np.real_if_close(probability))

    if probability < tolerance:
        raise ValueError("outcome probability is approximately zero")
    reduced_state = partial_trace(DensityMatrix(unnormalized_joint), [0]).data
    conditional_state = reduced_state / probability

    return probability, conditional_state.astype(np.complex128)

def validate_probability_vector(probabilities: ArrayLike, tolerence:float=1e-10)->RealVector:
    """
    Validate and return a classical probability vector
    :param probabilities: ArrayLike, 1d array of probabilities
    :param tolerence:
    :return: validated probability vector
    VCalueError: if probabilities are negative or do not sum to 1
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"probabilities must be a 1d array, but got {probs.ndim}d array")
    if np.any(probs < -tolerence):
        raise ValueError(f"probabilities must be non-negative, but got {probs}")
    probs[np.abs(probs)<=tolerence] = 0
    total = np.sum(probs)
    if not np.isclose(total, 1, atol=tolerence):
        raise ValueError(f"probabilities must sum to 1, but got {total}")

    probs = probs / np.sum(probs)

    return probs.astype(np.float64)

def sample_measurement(probabilities: ArrayLike, shots: int =100, rng: np.random.Generator | None = None) ->NDArray[np.int64]:
    """
    sample classical measurement outcomes from a prabability distribution
    :param probabilities: array, probability vector
    :param shots: int, number of samples
    :param rng: numpy random number generator, if None, then default generator is used
    :return: array of sample outcomes
    ValueError: if probabilities are not valid
    NOTE: if the probability ie [.7, .3] and shots are 1000, then outcom 0 will appear 700 times, and outcome 1 300 times
    """
    if shots <= 0:
        raise ValueError(f"shots must be positive, but got {shots}")
    probs  = validate_probability_vector(probabilities)
    if rng is None:
        rng = np.random.default_rng(123)
    outcomes = np.arange(len(probs), dtype=np.int64)
    samples = rng.choice(outcomes, size=shots, p=probs)
    return samples.astype(np.int64)

def estimate_exception_from_pm1(samples:ArrayLike, plus_label:int=0, minus_label: int=1)->float:
    """
    estimate an expectation value from two outcome samples
    The labels are mapped as:
        plus_label  -> +1
        minus_label -> -1
    :param samples: array, sample measurement outcomes
    :param plus_label:
    :param minus_label:
    :return: estimated expectation value

    NOTE: this is to estimate Pauli expectation values such as <X>, <Y>, <Z> from measurement date
    """
    sample_array = np.asarray(samples, dtype=np.int64)
    allowed = {plus_label, minus_label}
    observed = set(sample_array.tolist())

    if not observed.issubset(allowed):
        raise ValueError(f"samples must contain only {plus_label} and {minus_label} but got {observed}")

    values = np.where(sample_array == plus_label, 1, -1)
    return float(np.mean(values))

def ancilla_measurement_distribution_via_cnot(system_state: Statevector)->tuple[ComplexMatrix, RealVector]:
    """
    Implement a simple ancilla-based measurement experiment using CNOT

    :param system_state: statevector,  one qubit qiskit statevector for the system qubit
    :return: tuple ( ancilla density matrix, ancilla probabilities)

    NOTE:
    consider the experiment:
    1. Prepare system qubit in the given state.
    2. Prepare ancilla qubit in |0>.
    3. Apply CNOT with system as control and ancilla as target.
    4. Trace out the system and inspect the ancilla probabilities.

    something like Naimark-style,
    system + ancilla + unitary + standard measurement
    It is not the full generic POVM implementation, but it is a good pattern
    """

    if system_state.dim != 2:
        raise ValueError(f"system_state must be a one-qubit statevector, but got {system_state.dim} qubits")
    ancilla_zero = Statevector.from_label("0")

    joint_state = system_state.tensor(ancilla_zero)

    cnot = QuantumCircuit(2)
    cnot.cx(0,1)

    U = Operator(cnot)
    final_state = joint_state.evolve(U)
    final_density_matrix = DensityMatrix(final_state)

    ancilla_dm = partial_trace(final_density_matrix, [0]).data
    probabilities = np.real_if_close(np.diag(ancilla_dm)).astype(np.float)

    return ancilla_dm.astype(np.complex128), probabilities.astype(np.float64)










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
    # print(eigvals_hermitian(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])))
    # print(eigvals_hermitian(np.array([[1, 2, 3], [3, 4, 5]])))
    print("*" * 25 + "Function-6" + "*" * 25)
    # print(is_positive_semidefinite(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])))
    # print(is_positive_semidefinite(np.array([[1, 2, 3], [3, 4, 5]])))
    print("*" * 25 + "Function-7" + "*" * 25)
    print(is_density_matrix(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])))
    print(is_density_matrix(np.array([[1, 2, 3], [3, 4, 5]])))
    print("*" * 25 + "Function-8" + "*" * 25)
    print(normalize_density_matrix(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])))
    # print(normalize_density_matrix(np.array([[1, 2, 3], [3, 4, 5]])))
    print("*" * 25 + "Function-9" + "*" * 25)
    print(projector_from_state(np.array([1, 0, 0, 0])))
    print(projector_from_state(np.array([1, 0, 0, 0]), normalize=False))
    print("*" * 25 + "Function-10" + "*" * 25)
    print(normalize_statevector(np.array([1, 0, 0, 0])))
    print(normalize_statevector(np.array([1, 0, 0, 0]), tolerance=1e-10))
    print("*" * 25 + "Function-11" + "*" * 25)
    print(standard_basis_projectors())
    print("*" * 25 + "Function-12" + "*" * 25)
    print(povm_sum([np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]))
    print("*" * 25 + "Function-13" + "*" * 25)
    print(validate_povm([np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]))
    print("*" * 25 + "Function-14" + "*" * 25)
    print(validate_povm([np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])], tolerance=1e-10))
    print("*" * 25 + "Function-15" + "*" * 25)
    print(
        measurement_probabilities(np.array([[1, 0], [0, 0]]), [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]))
    print("*" * 25 + "Function-16" + "*" * 25)
    print(measurement_channel_output(np.array([[1, 1], [1, 1]]),
                                     [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]))
    print("*" * 25 + "Function-17" + "*" * 25)
    print(
        measurement_probabilities(np.array([[1, 0], [0, 0]]), [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])],
                                  tolerance=1e-10))
    print("*" * 25 + "Function-18" + "*" * 25)
    print(measurement_channel_output(np.array([[1, 1], [1, 1]]),
                                     [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])], tolerance=1e-10))
    print("*" * 25 + "Function-19" + "*" * 25)
    print(partial_measurement_probabilities(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                                            [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]))
    print("*" * 25 + "Function-20" + "*" * 25)
    print(partial_measurement_probabilities(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                                            [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])], tolerance=1e-10))
    print("*" * 25 + "Function-21" + "*" * 25)
    print(conditional_state_of_second_qubit(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                                            np.array([[1, 0], [0, 0]])))
    print("*" * 25 + "Function-22" + "*" * 25)
