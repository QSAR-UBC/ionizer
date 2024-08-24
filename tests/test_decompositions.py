# Some of the code in this file is based on code written for PennyLane.
# The appropriate copyright notice is included below.

# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test the decompositions of standard operations in to the {GPI, GPI2, MS} gate set.
"""

import pytest

import pennylane as qml
from pennylane import math
import numpy as np

from ionizer.decompositions import decomp_map, gpi_single_qubit_unitary


non_parametrized_ops = [
    (qml.PauliX, decomp_map["PauliX"]),
    (qml.PauliY, decomp_map["PauliY"]),
    (qml.PauliZ, decomp_map["PauliZ"]),
    (qml.Hadamard, decomp_map["Hadamard"]),
    (qml.SX, decomp_map["SX"]),
    (qml.CNOT, decomp_map["CNOT"]),
]

parametrized_ops = [
    (qml.RX, decomp_map["RX"]),
    (qml.RY, decomp_map["RY"]),
    (qml.RZ, decomp_map["RZ"]),
]

single_qubit_unitaries = [
    (np.eye(2), []),
    (
        np.array([[0.0, 0.54030231 - 0.84147098j], [0.54030231 + 0.84147098j, 0.0]]),
        ["GPI"],
    ),
    (qml.RZ.compute_matrix(0.2), ["GPI", "GPI"]),
    (
        np.array(
            [
                [0.70710678, -0.59500984 - 0.38205142j],
                [0.59500984 - 0.38205142j, 0.70710678],
            ]
        ),
        ["GPI2"],
    ),
    (qml.RY.compute_matrix(1.0), ["GPI2", "GPI", "GPI2"]),
    (
        np.array(
            [
                [-0.07696599 - 0.27774849j, 0.13976596 - 0.94731066j],
                [-0.47145233 + 0.83346549j, 0.2313124 - 0.17193766j],
            ]
        ),
        ["GPI2", "GPI", "GPI2"],
    ),
    (
        np.array(
            [
                [0.10705033 - 0.21821853j, -0.68049246 - 0.69126761j],
                [-0.01322673 + 0.96992059j, -0.22825748 - 0.08353216j],
            ]
        ),
        ["GPI2", "GPI", "GPI2"],
    ),
]


class TestDecompositions:
    """Tests that decompositions of non-parametric operations are correct
    up to phases."""

    @pytest.mark.parametrize("gate,decomp_function", non_parametrized_ops)
    def test_non_parametric_decompositions(self, gate, decomp_function):
        """Test decompositions of non-parametric operations."""
        expected_mat = gate.compute_matrix()
        obtained_mat = qml.matrix(decomp_function, wire_order=range(gate.num_wires))(
            wires=range(gate.num_wires)
        )
        mat_product = qml.math.dot(expected_mat, qml.math.conj(qml.math.T(obtained_mat)))
        mat_product = mat_product / mat_product[0, 0]
        assert qml.math.allclose(mat_product, qml.math.eye(mat_product.shape[0]))

    @pytest.mark.parametrize(
        "angle", [-np.pi, -np.pi / 2, -0.3, 0.0, 0.2, np.pi / 3, np.pi / 2, np.pi]
    )
    @pytest.mark.parametrize("gate,decomp_function", parametrized_ops)
    def test_parametric_decompositions(self, gate, decomp_function, angle):
        """Test decompositions of parametric operations."""
        expected_mat = gate.compute_matrix(angle)
        obtained_mat = qml.matrix(decomp_function, wire_order=range(gate.num_wires))(
            angle, wires=range(gate.num_wires)
        )
        mat_product = qml.math.dot(expected_mat, qml.math.conj(qml.math.T(obtained_mat)))
        mat_product = mat_product / mat_product[0, 0]
        assert qml.math.allclose(mat_product, qml.math.eye(mat_product.shape[0]))

    @pytest.mark.parametrize("unitary, decomp_list", single_qubit_unitaries)
    def test_single_qubit_unitary_decomposition(self, unitary, decomp_list):
        """Test decompositions of single-qubit unitary matrices."""
        obtained_decomp_list = gpi_single_qubit_unitary(unitary, [0])

        assert all(
            op.name == expected_name for op, expected_name in zip(obtained_decomp_list, decomp_list)
        )

        if len(obtained_decomp_list) > 0:
            with qml.tape.QuantumTape() as tape:
                for op in obtained_decomp_list:
                    qml.apply(op)

            obtained_matrix = qml.matrix(tape, wire_order=tape.wires)
            mat_product = math.dot(obtained_matrix, math.conj(math.T(unitary)))
            mat_product = mat_product / mat_product[0, 0]
            assert math.allclose(mat_product, math.eye(2))
