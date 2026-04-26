import numpy as np
from src.QuInfo.utils.DensityMatrix import (report_density_matrix, densityMatrix_from_label,
                                            measurement_probabilities_standard_basis,
                                            plot_bloch_vectors)

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

