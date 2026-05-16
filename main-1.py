"""
Showcase script for the QuantumInformation utility modules.

Run from the QuantumInformation project root:
    python main-1.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import partial_trace as qiskit_partial_trace
from typing import Callable, Sequence, TypeAlias

import src.QuInfo.utils.genMeasurements as gen_measurements
from src.QuInfo.utils.DensityMatrix import (
    bloch_vector,
    density_from_label_one_qubit,
    densityMatrix_from_label,
    eigenvalues_of_matrix,
    is_hermitian,
    is_power_of_two,
    is_positive_semidefinite,
    is_square_matrix,
    is_valid_density_matrix,
    measurement_probabilities_standard_basis,
    measurement_probabilities_standard_basis_1qubit,
    plot_bloch_vectors,
    pretty_matrix,
    purity,
    report_density_matrix,
    to_numpy_matrix,
    trace_of_matrix,
)
from src.QuInfo.utils.genMeasurements import (
    ancilla_measurement_distribution_via_cnot,
    conditional_state_of_second_qubit,
    estimate_exception_from_pm1,
    is_density_matrix,
    matrix_dimension,
    matrix_square_root_psd,
    measurement_channel_output,
    measurement_probabilities,
    normalize_density_matrix,
    normalize_statevector,
    partial_measurement_probabilities,
    post_measurement_state_via_sqrtP,
    povm_sum,
    projector_from_state,
    sample_measurement,
    standard_basis_projectors,
    to_complex_vector,
    validate_povm,
    validate_probability_vector,
)
from src.QuInfo.utils.PurificationFidelity import (
    apply_unitary_on_second_subsystem,
    canonical_purification_fixed_ancilla_dim,
    canonical_purification_from_spectral_decomposition,
    is_unitary,
    manual_fidelity,
    overlap,
    random_unitary,
    reconstruct_from_schmidt,
    reduced_system_from_purification,
    root_fidelity,
    schmidt_decomposition,
    spectral_decomposition_density_matrix,
)

np.set_printoptions(precision=4, suppress=True)

MatrixLike: TypeAlias = NDArray[np.complexfloating] | Sequence[Sequence[complex]]
KrausList: TypeAlias = Sequence[MatrixLike]
ChannelFunction: TypeAlias = Callable[[MatrixLike], NDArray[np.complexfloating]]


def install_local_compatibility_shims() -> None:
    """
    Keep this showcase runnable with the current utility code and NumPy.

    These shims do not edit the source modules. They only patch this process so
    methods with small keyword/deprecation issues can still be demonstrated.
    """
    if not hasattr(np, "float"):
        np.float = float

    def is_hermitian_compatible(matrix, tolerance: float = 1e-10, tol: float | None = None):
        return is_hermitian(matrix, tolerance=tolerance if tol is None else tol)

    def is_positive_semidefinite_compatible(matrix, tolerance: float = 1e-10, tol: float | None = None):
        return is_positive_semidefinite(matrix, tolerance=tolerance if tol is None else tol)

    gen_measurements.is_hermitian = is_hermitian_compatible
    gen_measurements.is_positive_semidefinite = is_positive_semidefinite_compatible
    plt.show = lambda *args, **kwargs: None


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def subsection(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def show_result(label: str, value) -> None:
    print(f"\n{label}:")
    print(value)


def run_safely(label: str, func) -> None:
    """Run a showcase call without stopping the rest of the demo."""
    print(f"\n{label}:")
    try:
        result = func()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
    else:
        if result is not None:
            print(result)


def apply_unitary_channel(matrix: MatrixLike, unitary: MatrixLike) -> NDArray[np.complex128]:
    rho = to_numpy_matrix(matrix)
    unitary_matrix = to_numpy_matrix(unitary)

    if rho.shape[0] != rho.shape[1]:
        raise ValueError("Input matrix must be square.")
    if unitary_matrix.shape != rho.shape:
        raise ValueError(f"Unitary must have shape {rho.shape}, got {unitary_matrix.shape}.")

    return unitary_matrix @ rho @ unitary_matrix.conj().T


def convex_combine_channels(
    channel_outputs: Sequence[MatrixLike],
    probabilities: Sequence[float],
    tolerance: float = 1e-10,
) -> NDArray[np.complex128]:
    if len(channel_outputs) == 0:
        raise ValueError("Channel outputs list cannot be empty.")
    if len(channel_outputs) != len(probabilities):
        raise ValueError("Number of channel outputs and probabilities must match.")

    probs = np.asarray(probabilities, dtype=np.float64)
    if np.any(probs < -tolerance):
        raise ValueError("Probabilities must be non-negative.")
    if not np.isclose(np.sum(probs), 1, atol=tolerance):
        raise ValueError(f"Probabilities must sum to 1, got {np.sum(probs)}.")

    outputs = [to_numpy_matrix(output) for output in channel_outputs]
    first_shape = outputs[0].shape
    if any(output.shape != first_shape for output in outputs):
        raise ValueError("All channel outputs must have the same shape.")

    total = np.zeros(first_shape, dtype=np.complex128)
    for probability, output in zip(probs, outputs):
        total += probability * output
    return total


def apply_kraus_channel(matrix: MatrixLike, kraus_ops: KrausList) -> NDArray[np.complex128]:
    rho = to_numpy_matrix(matrix)
    if len(kraus_ops) == 0:
        raise ValueError("Kraus operators list cannot be empty.")

    output_dim = to_numpy_matrix(kraus_ops[0]).shape[0]
    out = np.zeros((output_dim, output_dim), dtype=np.complex128)

    for kraus_like in kraus_ops:
        kraus = to_numpy_matrix(kraus_like)
        if kraus.shape[1] != rho.shape[0]:
            raise ValueError(f"Kraus operator has incompatible shape {kraus.shape}.")
        out += kraus @ rho @ kraus.conj().T

    return out


def kraus_completeness_check(kraus_ops: KrausList) -> NDArray[np.complex128]:
    if len(kraus_ops) == 0:
        raise ValueError("Kraus operators list cannot be empty.")

    first = to_numpy_matrix(kraus_ops[0])
    input_dim = first.shape[1]
    total = np.zeros((input_dim, input_dim), dtype=np.complex128)

    for kraus_like in kraus_ops:
        kraus = to_numpy_matrix(kraus_like)
        if kraus.shape[1] != input_dim:
            raise ValueError(f"Kraus operator has incompatible shape {kraus.shape}.")
        total += kraus.conj().T @ kraus

    return total


def reset_channel_qubit(matrix: MatrixLike) -> NDArray[np.complex128]:
    rho = to_numpy_matrix(matrix)
    if rho.shape != (2, 2):
        raise ValueError(f"Reset channel expects a 2x2 matrix, got {rho.shape}.")

    ket0 = np.array([[1], [0]], dtype=np.complex128)
    return np.trace(rho) * (ket0 @ ket0.conj().T)


def dephasing_channel_qubit(matrix: MatrixLike) -> NDArray[np.complex128]:
    rho = to_numpy_matrix(matrix)
    if rho.shape != (2, 2):
        raise ValueError(f"Dephasing channel expects a 2x2 matrix, got {rho.shape}.")

    out = np.array(rho, copy=True, dtype=np.complex128)
    out[0, 1] = 0
    out[1, 0] = 0
    return out


def depolarizing_channel_qubit(matrix: MatrixLike) -> NDArray[np.complex128]:
    rho = to_numpy_matrix(matrix)
    if rho.shape != (2, 2):
        raise ValueError(f"Depolarizing channel expects a 2x2 matrix, got {rho.shape}.")

    return np.trace(rho) * np.eye(2, dtype=np.complex128) / 2


def noisy_dephasing_channel_qubit(matrix: MatrixLike, epsilon: float) -> NDArray[np.complex128]:
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1].")

    rho = to_numpy_matrix(matrix)
    return (1 - epsilon) * rho + epsilon * dephasing_channel_qubit(rho)


def noisy_depolarizing_channel_qubit(matrix: MatrixLike, epsilon: float) -> NDArray[np.complex128]:
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1].")

    rho = to_numpy_matrix(matrix)
    return (1 - epsilon) * rho + epsilon * depolarizing_channel_qubit(rho)


def apply_channel_to_first_qubit(
    matrix: MatrixLike,
    channel_func: ChannelFunction,
) -> NDArray[np.complex128]:
    rho_ab = to_numpy_matrix(matrix)
    if rho_ab.shape != (4, 4):
        raise ValueError(f"Expected a two-qubit density matrix with shape (4, 4), got {rho_ab.shape}.")

    blocks = [
        (np.array([[1, 0], [0, 0]], dtype=np.complex128), rho_ab[0:2, 0:2]),
        (np.array([[0, 1], [0, 0]], dtype=np.complex128), rho_ab[0:2, 2:4]),
        (np.array([[0, 0], [1, 0]], dtype=np.complex128), rho_ab[2:4, 0:2]),
        (np.array([[0, 0], [0, 1]], dtype=np.complex128), rho_ab[2:4, 2:4]),
    ]

    out = np.zeros_like(rho_ab, dtype=np.complex128)
    for left, right in blocks:
        out += np.kron(channel_func(left), right)
    return out


def choi_from_definition(channel_func: ChannelFunction, dim: int = 2) -> NDArray[np.complex128]:
    if dim <= 0:
        raise ValueError("dim must be positive.")

    choi = np.zeros((dim * dim, dim * dim), dtype=np.complex128)
    for a in range(dim):
        for b in range(dim):
            basis = np.zeros((dim, dim), dtype=np.complex128)
            basis[a, b] = 1
            choi += np.kron(basis, channel_func(basis))

    return choi


def partial_trace_output_system_of_choi(
    choi_matrix: MatrixLike,
    input_dim: int = 2,
    output_dim: int = 2,
) -> NDArray[np.complex128]:
    choi = to_numpy_matrix(choi_matrix)
    expected_shape = (input_dim * output_dim, input_dim * output_dim)
    if choi.shape != expected_shape:
        raise ValueError(f"Expected Choi shape {expected_shape}, got {choi.shape}.")

    reshaped = choi.reshape(input_dim, output_dim, input_dim, output_dim)
    traced = np.zeros((input_dim, input_dim), dtype=np.complex128)
    for output_index in range(output_dim):
        traced += reshaped[:, output_index, :, output_index]

    return traced


def apply_channel_via_choi(matrix: MatrixLike, choi_matrix: MatrixLike) -> NDArray[np.complex128]:
    rho = to_numpy_matrix(matrix)
    choi = to_numpy_matrix(choi_matrix)

    if rho.shape != (2, 2):
        raise ValueError(f"Input rho must have shape (2, 2), got {rho.shape}.")
    if choi.shape != (4, 4):
        raise ValueError(f"Choi matrix must have shape (4, 4), got {choi.shape}.")

    temp = np.kron(rho.T, np.eye(2, dtype=np.complex128)) @ choi
    output = qiskit_partial_trace(QiskitDensityMatrix(temp), [0]).data
    return np.asarray(output, dtype=np.complex128)


def showcase_density_matrix_helpers() -> None:
    section("DensityMatrix.py helpers")

    subsection("Create example density matrices")

    rho_mixed = np.array(
        [
            [0.5, -0.3j],
            [0.3j, 0.5],
        ],
        dtype=np.complex128,
    )
    rho_zero = density_from_label_one_qubit("0")
    rho_plus = densityMatrix_from_label("+")
    rho_three_qubit = densityMatrix_from_label("r+1")

    pretty_matrix(to_numpy_matrix(rho_zero), "|0><0|")
    pretty_matrix(to_numpy_matrix(rho_plus), "|+><+|")
    pretty_matrix(rho_mixed, "rho_mixed")

    subsection("Full density-matrix diagnostic report")
    report_density_matrix(rho_mixed, name="rho_mixed")

    subsection("Individual validation and matrix-property helpers")
    show_result("is_square_matrix(rho_mixed)", is_square_matrix(rho_mixed))
    show_result("is_hermitian(rho_mixed)", is_hermitian(rho_mixed))
    show_result("trace_of_matrix(rho_mixed)", trace_of_matrix(rho_mixed))
    show_result("eigenvalues_of_matrix(rho_mixed)", eigenvalues_of_matrix(rho_mixed))
    show_result("is_positive_semidefinite(rho_mixed)", is_positive_semidefinite(rho_mixed))
    show_result("is_valid_density_matrix(rho_mixed)", is_valid_density_matrix(rho_mixed))
    show_result("purity(rho_mixed)", purity(rho_mixed))

    subsection("Measurement probabilities and Bloch-vector helpers")
    show_result(
        "measurement_probabilities_standard_basis_1qubit(rho_mixed)",
        measurement_probabilities_standard_basis_1qubit(rho_mixed),
    )
    show_result(
        "measurement_probabilities_standard_basis(densityMatrix_from_label('r+1'))",
        measurement_probabilities_standard_basis(rho_three_qubit),
    )
    show_result("is_power_of_two(8)", is_power_of_two(8))
    show_result("bloch_vector(rho_mixed)", bloch_vector(rho_mixed))

    plot_bloch_vectors(
        {
            "|0>": bloch_vector(density_from_label_one_qubit("0")),
            "|1>": bloch_vector(density_from_label_one_qubit("1")),
            "|+>": bloch_vector(density_from_label_one_qubit("+")),
            "rho_mixed": bloch_vector(rho_mixed),
        },
        "Example Bloch vectors",
    )
    plt.savefig("plots/main-1-bloch-showcase.png", bbox_inches="tight")
    plt.close()
    print("\nSaved Bloch sphere plot to plots/main-1-bloch-showcase.png")


def showcase_general_measurements() -> None:
    section("genMeasurements.py helpers")

    subsection("State, vector, and density-matrix preparation")

    rho_plus = to_numpy_matrix(densityMatrix_from_label("+"))
    rho_zero = to_numpy_matrix(densityMatrix_from_label("0"))
    z_projectors = standard_basis_projectors()
    x_projectors = [
        projector_from_state(np.array([1, 1])),
        projector_from_state(np.array([1, -1])),
    ]

    pretty_matrix(rho_plus, "rho_plus")
    show_result("matrix_dimension(rho_plus)", matrix_dimension(rho_plus))
    show_result("to_complex_vector([1, -1j])", to_complex_vector([1, -1j]))
    show_result("is_density_matrix(rho_plus)", is_density_matrix(rho_plus, verbose=True))
    show_result("normalize_density_matrix(2 * rho_plus)", normalize_density_matrix(2 * rho_plus))
    show_result("normalize_statevector([1, 1])", normalize_statevector([1, 1]))

    subsection("Projectors, POVM validation, and measurement probabilities")
    pretty_matrix(projector_from_state([1, 1]), "Projector |+><+|")
    show_result("standard_basis_projectors()", z_projectors)
    show_result("povm_sum(standard_basis_projectors())", povm_sum(z_projectors))
    show_result("validate_povm(standard_basis_projectors())", validate_povm(z_projectors, name="Z basis POVM"))

    show_result(
        "measurement_probabilities(rho_plus, Z projectors)",
        measurement_probabilities(rho_plus, z_projectors),
    )
    show_result(
        "measurement_probabilities(rho_plus, X projectors)",
        measurement_probabilities(rho_plus, x_projectors),
    )
    show_result(
        "measurement_channel_output(rho_plus, Z projectors)",
        measurement_channel_output(rho_plus, z_projectors),
    )

    subsection("Partial measurements on a two-qubit Bell state")
    bell = np.zeros((4, 4), dtype=np.complex128)
    bell[0, 0] = bell[0, 3] = bell[3, 0] = bell[3, 3] = 0.5
    pretty_matrix(bell, "Bell state density matrix |Phi+><Phi+|")
    show_result(
        "partial_measurement_probabilities(bell, Z projectors)",
        partial_measurement_probabilities(bell, z_projectors),
    )
    prob, conditional = conditional_state_of_second_qubit(bell, z_projectors[0])
    show_result("conditional_state_of_second_qubit probability for outcome 0", prob)
    pretty_matrix(conditional, "Conditional second-qubit state")

    subsection("Classical probability validation, sampling, and expectation estimate")
    probabilities = validate_probability_vector([0.5, 0.5])
    samples = sample_measurement(probabilities, shots=20, rng=np.random.default_rng(7))
    show_result("validate_probability_vector([0.5, 0.5])", probabilities)
    show_result("sample_measurement([0.5, 0.5], shots=20)", samples)
    show_result("estimate_exception_from_pm1(samples)", estimate_exception_from_pm1(samples))

    subsection("Post-measurement state and ancilla measurement demo")
    show_result("matrix_square_root_psd(z_projectors[0])", matrix_square_root_psd(z_projectors[0]))
    prob_post, post_state = post_measurement_state_via_sqrtP(rho_plus, z_projectors[0])
    show_result("post_measurement_state_via_sqrtP probability for outcome 0", prob_post)
    pretty_matrix(post_state, "Post-measurement state")

    ancilla_dm, ancilla_probs = ancilla_measurement_distribution_via_cnot(Statevector.from_label("+"))
    pretty_matrix(ancilla_dm, "Ancilla density matrix after CNOT")
    show_result("Ancilla standard-basis probabilities", ancilla_probs)


def showcase_purification_fidelity() -> None:
    section("PurificationFidelity.py helpers")

    rho_test = np.array(
        [
            [0.5, -0.3j],
            [0.3j, 0.5],
        ],
        dtype=np.complex128,
    )
    sigma_test = densityMatrix_from_label("+")
    bell_state = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)

    subsection("Spectral decomposition and canonical purification")
    eigvals, eigvects = spectral_decomposition_density_matrix(rho_test)
    purification, purification_eigvals, purification_eigvects = (
        canonical_purification_from_spectral_decomposition(rho_test)
    )
    fixed_purification = canonical_purification_fixed_ancilla_dim(
        rho_test,
        ancilla_dim=2,
        tolerance=1e-12,
    )

    pretty_matrix(rho_test, "rho_test")
    show_result("spectral_decomposition_density_matrix(rho_test)", (eigvals, eigvects))
    show_result(
        "canonical_purification_from_spectral_decomposition(rho_test)",
        (purification, purification_eigvals, purification_eigvects),
    )
    show_result(
        "canonical_purification_fixed_ancilla_dim(rho_test, ancilla_dim=2)",
        fixed_purification,
    )
    pretty_matrix(
        reduced_system_from_purification(
            fixed_purification,
            system_dims=[2, 2],
            trace_out_subsystems=[1],
        ),
        "Reduced system from fixed purification",
    )

    subsection("Schmidt decomposition and reconstruction")
    schmidt_coefficients, left_vectors, right_vectors = schmidt_decomposition(
        bell_state,
        dim_x=2,
        dim_y=2,
    )
    reconstructed_bell = reconstruct_from_schmidt(
        schmidt_coefficients,
        left_vectors,
        right_vectors,
    )
    show_result("schmidt_decomposition(bell_state, dim_x=2, dim_y=2)", (
        schmidt_coefficients,
        left_vectors,
        right_vectors,
    ))
    show_result("reconstruct_from_schmidt(...)", reconstructed_bell)
    show_result("overlap(bell_state, reconstructed_bell, squared=True)", overlap(
        bell_state,
        reconstructed_bell,
        squared=True,
    ))

    subsection("Ancilla unitary action and fidelity")
    unitary_y = random_unitary(2, rng=np.random.default_rng(456))
    pretty_matrix(unitary_y, "random_unitary(2)")
    show_result("is_unitary(random_unitary(2))", is_unitary(unitary_y, tolerance=1e-12))
    show_result(
        "apply_unitary_on_second_subsystem(bell_state, unitary_y, dim_x=2, dim_y=2)",
        apply_unitary_on_second_subsystem(
            bell_state,
            unitary_y,
            dim_x=2,
            dim_y=2,
        ),
    )
    show_result("manual_fidelity(rho_test, densityMatrix_from_label('+'))", manual_fidelity(
        rho_test,
        sigma_test,
    ))
    show_result("root_fidelity(rho_test, densityMatrix_from_label('+'))", root_fidelity(
        rho_test,
        sigma_test,
    ))


def showcase_quantum_channels() -> None:
    section("Quantum channel helpers")

    rho = np.array([[1, 2], [3, 4]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    bit_flip = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    dephasing_choi = choi_from_definition(dephasing_channel_qubit)

    subsection("Unitary, convex, and Kraus-channel examples")
    show_result("apply_unitary_channel(rho, I)", apply_unitary_channel(rho, identity))
    show_result(
        "convex_combine_channels([rho, 2*rho], [0.5, 0.5])",
        convex_combine_channels([rho, 2 * rho], [0.5, 0.5]),
    )
    show_result("apply_kraus_channel(rho, [I, X])", apply_kraus_channel(rho, [identity, bit_flip]))
    show_result("kraus_completeness_check([I, X])", kraus_completeness_check([identity, bit_flip]))

    subsection("Standard one-qubit channels")
    show_result("reset_channel_qubit(rho)", reset_channel_qubit(rho))
    show_result("dephasing_channel_qubit(rho)", dephasing_channel_qubit(rho))
    show_result("depolarizing_channel_qubit(rho)", depolarizing_channel_qubit(rho))
    show_result("noisy_dephasing_channel_qubit(rho, 0.5)", noisy_dephasing_channel_qubit(rho, 0.5))
    show_result("noisy_depolarizing_channel_qubit(rho, 0.5)", noisy_depolarizing_channel_qubit(rho, 0.5))

    subsection("Local channels and Choi representation")
    show_result(
        "apply_channel_to_first_qubit(I_4/4, dephasing_channel_qubit)",
        apply_channel_to_first_qubit(np.eye(4, dtype=np.complex128) / 4, dephasing_channel_qubit),
    )
    show_result("choi_from_definition(dephasing_channel_qubit)", dephasing_choi)
    show_result(
        "partial_trace_output_system_of_choi(choi)",
        partial_trace_output_system_of_choi(dephasing_choi),
    )
    show_result("apply_channel_via_choi(rho, choi)", apply_channel_via_choi(rho, dephasing_choi))


def main() -> None:
    install_local_compatibility_shims()
    showcase_density_matrix_helpers()
    showcase_general_measurements()
    showcase_purification_fidelity()
    showcase_quantum_channels()


if __name__ == "__main__":
    main()
