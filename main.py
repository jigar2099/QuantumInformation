import numpy as np
from src.QuInfo.utils.DensityMatrix import (report_density_matrix, densityMatrix_from_label,
                                            measurement_probabilities_standard_basis,
                                            plot_bloch_vectors)
from src.QuInfo.utils.genMeasurements import (is_density_matrix,projector_from_state)

print("*"*130)
print("="*50+"Density Matrix"+"="*50)
print("*"*130)

report_density_matrix(
    np.array([[0.5, 0.0 - 0.3j], [0.0 + 0.3j, 0.5]]),
    name="TestMatrix",
    tolerance=1e-10
)

sampleDensityMatrix = np.array(
    densityMatrix_from_label("r+1"),
    dtype=float)
print(sampleDensityMatrix)
print(measurement_probabilities_standard_basis(
                                            sampleDensityMatrix)
                                            )
plot_bloch_vectors({"0-vec": [1, 0, 1],
                                "1-vec": [0.1, -0.5, -1],
                                "2-vec": [-0.5, -1, 0.5],
                                "3-vec": [-1, 0.5, 0.5]},
                               "Bloch Sphere")

print("*"*130)
print("="*50+"General Measurements"+"="*50)
print("*"*130)
print(projector_from_state(np.array([1, 0, 0, 0]), normalize=False))
is_density_matrix(np.array([[1, 2, 3],
                            [3, 4, 5],
                            [5, 6, 7]]),
                  tolerance=1e-10,
                  verbose=True)

