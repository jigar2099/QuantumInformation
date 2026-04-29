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
from qiskit.quantum_info import Statevector

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

np.set_printoptions(precision=4, suppress=True)


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


def main() -> None:
    install_local_compatibility_shims()
    showcase_density_matrix_helpers()
    showcase_general_measurements()


if __name__ == "__main__":
    main()
