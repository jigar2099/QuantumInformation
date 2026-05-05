from __future__ import annotations
#import os
#os.chdir("..")

from typing import Iterable, Sequence, TypeAlias, Callable, Mapping
import numpy as np

from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import (Statevector, DensityMatrix, Operator,
                                 partial_trace, Pauli, state_fidelity)
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix
from qiskit.quantum_info import partial_trace as qiskit_partial_trace

from src.QuInfo.utils.DensityMatrix import *
from src.QuInfo.utils import *


MatrixLike: TypeAlias = NDArray[np.complexfloating] | Sequence[Sequence[complex]]
VectroLike: TypeAlias = NDArray[np.complexfloating] | Sequence[complex]
KrausList: Sequence[MatrixLike]
ChannelFunction: TypeAlias = Callable[[MatrixLike], NDArray[np.complexfloating]]


def apply_unitary_channel(matrix: MatrixLike, unitary: MatrixLike) -> NDArray[np.complex128]:
    """
    apply unitary quantum channel to a density matrix
    Phi(rho) = U rho U^dagger
    :param martix:
    :param unitary:
    :return:
    ValueError: if the matrix dimensionas are incompatible
    NOTE: unitary channels are described by close-system quantum evolution, and preserve purity and reversibility
    """
    rho = to_numpy_matrix(matrix)
    U = to_numpy_matrix(unitary)

    if rho.shape[0] != rho.shape[1]:
        raise ValueError("Input matrix must be a square matrix")
    if U.shape != rho.shape:
        raise ValueError(f"Unitarty must have shape {rho.shape} but got {U.shape}")
    return U @ rho @ U.conj().T


def convex_combine_channels(channel_outputs: Sequence[MatrixLike], probabilities: Sequence[float],
                            tolerance: float = 1e-10) -> NDArray[np.complex128]:
    """
    Combine several channel outputs using classical probabilities

    Given outputs: phi_1(rho), phi_2(rho), ..., phi_n(rho)
    and probabilities: p_1, p_2, ..., p_n
    this function computes the convex combination of the outputs as,
    sum_k p_k phi_k(rho)
    where p_k is the probability of choosing the kth output

    :param channel_outputs: list of output density matrices from a different channels
    :param probabilities: classical probabilities associate with each channel output
    :param tolerance: numerical constraint
    :return:
    ValueError:
            if the number of outputs and probabilities do not match,
            if probabilities are negative
            if probabilities do not sum to 1
            if output martices do not have the same shape
    NOTE: a aoncvex combination of channels models a situation where one of several physical processes happens randomly
    For ExAMPLE,
            with prob p do nothing,
            with prob 1-p do a quantum channel,
            and with prob q do a classical channel
    """
    if len(channel_outputs) == 0:
        raise ValueError("Channel outputs list can not be empty")
    if len(channel_outputs) != len(probabilities):
        raise ValueError("Number of channel outputs and probabilities must match")
    probs = np.asarray(probabilities, dtype=np.float64)

    if np.any(probs < -tolerance):
        raise ValueError("Probabilities must be non-negative")
    if not np.isclose(np.sum(probs), 1, atol=tolerance):
        raise ValueError(f"Probabilities must sum to 1, but got {np.sum(probs)}")
    outputs = [to_numpy_matrix(output) for output in channel_outputs]
    first_shape = outputs[0].shape

    for output in outputs:
        if output.shape != first_shape:
            raise ValueError("All channel outputs must have same shape")

    total = np.zeros(first_shape, dtype=np.complex128)

    for probability, output in zip(probs, outputs):
        total += probability * output

    return total


def apply_kraus_channel(matrix: MatrixLike, kraus_ops: KrausList) -> NDArray[np.complex128]:
    """
    Apply quantum channel using kraus representation
    phi(rho) = sum_k A_k rho A_k^dagger
    :param matrix: input density matrix
    :param kraus_ops: sequence of kraus operators A_k
    :return: output density matrix after applying kraus channel
    ValueError:
        if no kraus operators are provided,
        if dimensions are incompatible
    NOTE: kraus operators describe general open system quantum evolution, including noise, measurement-like effects, and interaction with an environment
    """
    rho = to_numpy_matrix(matrix)
    if len(kraus_ops) == 0:
        raise ValueError("Kraus operators list can not be empty")
    output_dim = to_numpy_matrix(kraus_ops[0]).shape[0]
    out = np.zeros((output_dim, output_dim), dtype=np.complex128)

    for A_like in kraus_ops:
        A = to_numpy_matrix(A_like)
        if A.shape[1] != rho.shape[0]:
            raise ValueError(f"Kraus operator must have shape ({rho.shape[0]}, {rho.shape[0]}) but got {A.shape}")
        out += A @ rho @ A.conj().T
    return out


def kraus_completeness_check(kraus_ops: KrausList) -> NDArray[np.complex128]:
    """
    compute the kraus completeness operator
    for a trace-preserving quantum channel, the kraus operators satisfy:
    sum_k A_k^dagger A_k = I
    :param kraus_ops: seq. of kraus operators
    :return: matrix sum_k A_k^dagger A_k
    ValueError: if no kraus operators are provided
    NOTE: This condition guarantees that the output density matrix has trace 1 whenever the input density matrix has trace 1
    """
    if len(kraus_ops) == 0:
        raise ValueError("Kraus operators list can not be empty")
    first = to_numpy_matrix(kraus_ops[0])
    input_dim = first.shape[1]
    total = np.zeros((input_dim, input_dim), dtype=np.complex128)

    for A_like in kraus_ops:
        A = to_numpy_matrix(A_like)
        if A.shape[0] != input_dim:
            raise ValueError(f"Kraus operator must have shape ({input_dim}, {input_dim}) but got {A.shape}")
        total += A.conj().T @ A
    return total


def reset_channel_qubit(matrix: MatrixLike) -> NDArray[np.complex128]:
    """
    Apply the one-qubit reset channel
    lambda(rho) = tr(rho) |0><0|
    for valid density matrix, Tr(rho) = 1, therefore: lambda(rho) = |0><0|
    :param matrix: one-qubit input matrix with shape (2,2)
    :return: output state |0><0|
    ValueError: if the input is not a 2X2 matrix
    NOTE: the reset channel erases the input state and prepares the qubit in |0> state. this is non-unitary and irreversible
    """
    rho = to_numpy_matrix(matrix)
    if rho.shape != (2, 2):
        raise ValueError("Input matrix must be a 2x2 square matrix")
    ket0 = np.array([[1], [0]], dtype=np.complex128)
    proj0 = ket0 @ ket0.conj().T
    return np.trace(rho) * proj0


def dephasing_channel_qubit(matrix: MatrixLike) -> NDArray[np.complex128]:
    """
    apply the complete one-qubit dephasing channel
    the complete dephasing channel removes off-diagonal entries of the density matrix
    :param matrix:
    :return: dephased output matrix
    ValueError: if the input is not a 2X2 matrix
    NOTE: dephasing destroys coherence in the computational basis while preserves the classical probabilities on the diagonal elements
    """
    rho = to_numpy_matrix(matrix)
    if rho.shape != (2, 2):
        raise ValueError(f"Dephasing channel expects a 2x2 square matrix, got {rho.shape}")
    out = np.array(rho, copy=True)
    out[0, 1] = 0
    out[1, 0] = 0
    return out


def depolarizing_channel_qubit(matrix: MatrixLike) -> NDArray[np.complex128]:
    """
    apply the complete one-qubit depolarizing channel
    Omega(rho) = Tr(rho) I/2
    For a valid density matrix: Omega(who) = I/2
    :param matrix: one qubit input matrix with shape (2,2)
    :return: maximally mixed output matrix
    ValueError: if the input is not a 2X2 matrix

    NOTE: depolarization removes all the information about the input state, and sends every valid qubit state to the center of the bloch sphere
    """
    rho = to_numpy_matrix(matrix)
    if rho.shape != (2, 2):
        raise ValueError(f"Depolarizing channel expects a 2x2 square matrix, got {rho.shape}")
    return np.trace(rho) * np.eye(2, dtype=np.complex128) / 2


def noisy_dephasing_channel_qubit(matrix: MatrixLike, epsilon: float) -> NDArray[np.complex128]:
    """
    apply noisy dephasing channel, it is just a convex combination
    Delta_epsilon(rho) = (1-epsilon) rho + epsilon Delta(rho)
    Delta is complete dephasing channel

    :param matrix: one qubit input matrix with shape (2,2)
    :param epsilon: noise strength
    :return: matrix after noisy dephasing
    ValueError: if epsilon is outside [0,1]

    NOTE: epsilon controls how strongly the channel removes coherence, 0 epsilon means no noise, 1 epsilon means complete dephasing
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    rho = to_numpy_matrix(matrix)
    return (1 - epsilon) * rho + epsilon * dephasing_channel_qubit(rho)


def noisy_depolarizing_channel_qunit(matrix: MatrixLike, epsilon: float) -> NDArray[np.complex128]:
    """
    apply noisy depolarizing channel, which is just a convex combination
    Omega_epsilon(rho) = (1-epsilon) rho + epsilon Omega(rho)
    Omega is complete depolarizing channel
    :param matrix: one qubit input matrix with shape (2,2)
    :param epsilon: noise strength
    :return: density matrix after noisy depolarizing
    ValueError: if epsilon is outside [0,1]
    NOTE: same as above "noisy_dephasing_channel_qubit"
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    rho = to_numpy_matrix(matrix)
    return (1 - epsilon) * rho + epsilon * depolarizing_channel_qubit(rho)


def apply_channel_to_first_qubit(matrix: MatrixLike, channel_func: ChannelFunction) -> NDArray[np.complex128]:
    """
    apply a one qubit channel to first qubit of a two-qubit channel
    The input state rho_AB is decomposed into blocks: rho_AB = sum_{a,b} |a><b| tensor rho_ab
    Then the channel acts only on the first subsystem: Phi_A tensor I_B
    giving: sum_{a,b} Phi(|a><b|) tensor rho_ab
    :param matrix: two qubit input defnsity matrix with shape (4,4)
    :param channel_func: channelFunction, that maps 2X2 matrix to another 2X2 matrix
    :return: OUtput two-qubit density matrix applying the channel to the first qubit
    ValueError: If the input is not a 4X4 matrix
    NOTE: THis demonstrrates local noise: one part of an entangled system interacts with an environment, while the other one is untouched
    """
    rho_AB = to_numpy_matrix(matrix)
    if rho_AB.shape != (4, 4):
        raise ValueError(
            f"This funciton expectsa two-qubit density matrix with shape (4,4), got {rho_AB.shape}"
        )
    out = np.zeros_like(rho_AB, dtype=np.complex128)

    # block decomposition w.r.t first qubit
    rho00 = rho_AB[0:2, 0:2]
    rho01 = rho_AB[0:2, 2:4]
    rho10 = rho_AB[2:4, 0:2]
    rho11 = rho_AB[2:4, 2:4]

    E00 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    E01 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    E10 = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    E11 = np.array([[0, 0], [0, 1]], dtype=np.complex128)

    blocks = [
        (channel_func(E00), rho00),
        (channel_func(E01), rho01),
        (channel_func(E10), rho10),
        (channel_func(E11), rho11),
    ]

    for left, right in blocks:
        out += np.kron(left, right)
    return out


def choi_from_definition(channel_func: ChannelFunction, dim: int = 2) -> NDArray[np.complex128]:
    """
    construct the choi matrix of a qunatum channel from its definition
    The choi matrix is: J(phi) = sum{a,b} |a><b| tensor Phi(|a><b|)
    :param channel_func: function representing the channel action on a dim X dim matrix
    :param dim: input dimension of the quantum system, for one qubit, dim = 2
    :return: Choi matrix of shape (dim * dim, dim * dim)
    ValueError: if dim is not positive
    NOTE: the choi matrix converts a qunatum channel into a matrrx representation.
    A map is completely positive if and only if its choi matrix is PSD
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    J = np.zeros((dim * dim, dim * dim), dtype=np.complex128)
    for a in range(dim):
        for b in range(dim):
            Eab = np.zeros((dim, dim), dtype=np.complex128)
            Eab[a, b] = 1

            left = np.zeros((dim, dim), dtype=np.complex128)
            left[a, b] = 1
            right = channel_func(Eab)
            J += np.kron(left, right)
    return J


def partial_trace_output_system_of_choi(choi_matrix: MatrixLike, input_dim: int = 2, output_dim: int = 2) -> NDArray[
    np.complex128]:
    """
    Compute the partial trace over the output system of a Choi matrx
    For a trace-preserving channel: Tr_output(J(Phi)) = I_input
    :param choi_matrix: choi matrix with shape (input_dim*output_dim, input_dim*output_dim)
    :param input_dim: input system dimension
    :param output_dim: output system dimension
    :return: partial trace over the output system
    VlaueError: if the choi matrix has incompatible shape

    NOTE: This chechk tells us whether a completely positive map presertves trace
    If it does, then valid density matrix remains normalized
    """
    J = to_numpy_matrix(choi_matrix)

    expected_shape = (input_dim * output_dim, input_dim * output_dim)

    if J.shape != expected_shape:
        raise ValueError(f"Expected Choi shape {expected_shape}, got {J.shape}.")

    J_reshaped = J.reshape(input_dim, output_dim, input_dim, output_dim)

    traced = np.zeros((input_dim, input_dim), dtype=np.complex128)

    for output_index in range(output_dim):
        traced += J_reshaped[:, output_index, :, output_index]

    return traced


def apply_channel_via_choi(matrix: MatrixLike, choi_matrix: MatrixLike) -> NDArray[np.complex128]:
    """
    Apply a quantum channel to a state using its Choi matrix.
    For a qubit-to-qubit channel:Phi(rho) = Tr_input[(rho^T tensor I) J(Phi)]
    ValueError: If rho does not have shape (2, 2), or if the Choi matrix does not have shape (4, 4).
    NOTE:
    This shows that the Choi matrix contains complete information about the
    quantum channel.
    """
    rho = to_numpy_matrix(matrix)
    J = to_numpy_matrix(choi_matrix)
    from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix
    from qiskit.quantum_info import partial_trace as qiskit_partial_trace

    if rho.shape != (2, 2):
        raise ValueError(f"Input rho must have shape (2, 2), got {rho.shape}.")

    if J.shape != (4, 4):
        raise ValueError(f"Choi matrix must have shape (4, 4), got {J.shape}.")

    temp = np.kron(rho.T, np.eye(2, dtype=np.complex128)) @ J

    # Trace out the input system.
    output = qiskit_partial_trace(QiskitDensityMatrix(temp), [0]).data

    return np.asarray(output, dtype=np.complex128)


if __name__ == "__main__":
    print("*" * 25 + "Function-1" + "*" * 25)
    print(apply_unitary_channel(np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 1]])))
    print("*" * 25 + "Function-2" + "*" * 25)
    print(convex_combine_channels([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])], [0.5, 0.5]))
    print("*" * 25 + "Function-3" + "*" * 25)
    print(apply_kraus_channel(np.array([[1, 2], [3, 4]]), [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]))
    print("*" * 25 + "Function-4" + "*" * 25)
    print(kraus_completeness_check([np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]))
    print("*" * 25 + "Function-5" + "*" * 25)
    print(reset_channel_qubit(np.array([[1, 2], [3, 4]])))
    print("*" * 25 + "Function-6" + "*" * 25)
    print(dephasing_channel_qubit(np.array([[1, 2], [3, 4]])))
    print("*" * 25 + "Function-7" + "*" * 25)
    print(depolarizing_channel_qubit(np.array([[1, 2], [3, 4]])))
    print("*" * 25 + "Function-8" + "*" * 25)
    print(noisy_dephasing_channel_qubit(np.array([[1, 2], [3, 4]]), 0.5))
    print("*" * 25 + "Function-9" + "*" * 25)
    print(noisy_depolarizing_channel_qunit(np.array([[1, 2], [3, 4]]), 0.5))
    print("*" * 25 + "Function-10" + "*" * 25)
    print(apply_channel_to_first_qubit(np.eye(4, dtype=np.complex128) / 4, dephasing_channel_qubit))
    print("*" * 25 + "Function-11" + "*" * 25)
    print(choi_from_definition(dephasing_channel_qubit))
    print("*" * 25 + "Function-12" + "*" * 25)
    print(partial_trace_output_system_of_choi(choi_from_definition(dephasing_channel_qubit)))
    print("*" * 25 + "Function-13" + "*" * 25)
    print(apply_channel_via_choi(np.array([[1, 2], [3, 4]]), choi_from_definition(dephasing_channel_qubit)))
