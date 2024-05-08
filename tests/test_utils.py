"""
Test utility functions.
"""

import pytest

import numpy as np
import pennylane as qml
from pennylane import math

from ionizer.ops import GPI, GPI2, MS
from ionizer.utils import (
    are_mats_equivalent,
    rescale_angles,
    extract_gpi2_gpi_gpi2_angles,
    tape_to_json,
)

from test_decompositions import single_qubit_unitaries


class TestMatrixAngleUtilities:
    """Test utility functions for matrix and angle manipulations."""

    @pytest.mark.parametrize(
        "mat1, mat2",
        [
            (np.eye(2), 1j * np.eye(2)),
            (qml.PauliX.compute_matrix(), np.exp(1j * np.pi / 2) * qml.PauliX.compute_matrix()),
            (qml.SX.compute_matrix(), np.exp(-1j * np.pi / 2) * qml.SX.compute_matrix()),
            (qml.Hadamard.compute_matrix(), -qml.Hadamard.compute_matrix()),
        ],
    )
    def test_equivalent_matrices(self, mat1, mat2):
        """Test that we correcty identify matrices that are equivalent
        up to a global phase."""
        assert are_mats_equivalent(mat1, mat2)

    @pytest.mark.parametrize(
        "mat1, mat2",
        [
            (np.eye(2), qml.PauliX.compute_matrix()),
            (qml.RZ.compute_matrix(0.3), GPI.compute_matrix(0.3)),
            (qml.Hadamard.compute_matrix(), GPI2.compute_matrix(0.2)),
        ],
    )
    def test_inequivalent_matrices(self, mat1, mat2):
        """Test that we correcty identify matrices that are equivalent
        up to a global phase."""
        assert not are_mats_equivalent(mat1, mat2)

    @pytest.mark.parametrize(
        "angles,rescaled_angles",
        [
            (0.0, 0.0),
            (np.pi, np.pi),
            (-np.pi / 2, -np.pi / 2),
            (3 * np.pi / 2, -np.pi / 2),
            (np.array([np.pi / 2, 5 * np.pi / 4]), np.array([np.pi / 2, -3 * np.pi / 4])),
        ],
    )
    def test_rescale_angles(self, angles, rescaled_angles):
        """Test angles are correctly rescaled between -np.pi and np.pi."""
        obtained_angles = rescale_angles(angles)
        assert math.allclose(obtained_angles, rescaled_angles)

    @pytest.mark.parametrize(
        "angles,rescaled_angles",
        [
            (0.0, 0.0),
            (np.pi, 0.5),
            (-np.pi / 2, -0.25),
            (3 * np.pi / 2, -0.25),
            (np.array([np.pi / 3, 5 * np.pi / 4]), np.array([1.0 / 6, -0.375])),
        ],
    )
    def test_rescale_angles_hardware(self, angles, rescaled_angles):
        """Test angles are correctly rescaled and renormalized into the range
        accepted by hardware (-1 to 1, corresponding to -2*np.pi to 2*np.pi)."""
        obtained_angles = rescale_angles(angles, renormalize_for_json=True)
        print(obtained_angles)
        assert math.allclose(obtained_angles, rescaled_angles)

    @pytest.mark.parametrize("unitary", [test_case[0] for test_case in single_qubit_unitaries])
    def test_extract_gpi_gpi2_angles(self, unitary):
        """Test that extracting GPI/GPI2 angles yields the correct operation."""
        gamma, beta, alpha = extract_gpi2_gpi_gpi2_angles(unitary)

        with qml.tape.QuantumTape() as tape:
            GPI2(gamma, wires=0)
            GPI(beta, wires=0)
            GPI2(alpha, wires=0)

        assert are_mats_equivalent(qml.matrix(tape), unitary)


class TestConvertToJSON:
    """Test that quantum tapes are correctly converted to JSON objects."""

    def test_single_gate_circuit(self):
        """Test a circuit with a single gate."""
        with qml.tape.QuantumTape() as tape:
            GPI(0.3, wires=0)

        circuit_json = tape_to_json(tape, shots=1000, name="1qc")

        assert circuit_json["name"] == "1qc"
        assert circuit_json["shots"] == 1000
        assert circuit_json["target"] == "simulator"

        assert circuit_json["body"]["gateset"] == "native"
        assert circuit_json["body"]["qubits"] == 1

        circuit_contents = circuit_json["body"]["circuit"]
        assert len(circuit_contents) == 1
        assert circuit_contents[0]["phase"] == 0.3 / (2 * np.pi)
        assert circuit_contents[0]["target"] == 0

    def test_multi_gate_circuit(self):
        """Test a circuit with multiple gates on the same wire."""
        with qml.tape.QuantumTape() as tape:
            GPI(0.3, wires=1)
            GPI2(-0.4, wires=1)
            GPI2(0.5, wires=1)

        circuit_json = tape_to_json(tape, name="3gates", target="qpu")

        assert circuit_json["name"] == "3gates"
        assert circuit_json["shots"] == 100
        assert circuit_json["target"] == "qpu"

        assert circuit_json["body"]["qubits"] == 1

        circuit_contents = circuit_json["body"]["circuit"]
        assert len(circuit_contents) == 3

        assert circuit_contents[0]["phase"] == 0.3 / (2 * np.pi)
        assert circuit_contents[0]["target"] == 1
        assert circuit_contents[1]["phase"] == -0.4 / (2 * np.pi)
        assert circuit_contents[1]["target"] == 1
        assert circuit_contents[2]["phase"] == 0.5 / (2 * np.pi)
        assert circuit_contents[2]["target"] == 1

    def test_multi_gate_multi_wire_circuit(self):
        """Test a circuit with multiple gates on different wires."""
        with qml.tape.QuantumTape() as tape:
            GPI(0.3, wires=0)
            GPI2(-0.4, wires=2)
            GPI2(0.5, wires=1)

        circuit_json = tape_to_json(tape, name="3wires", target="qpu")

        assert circuit_json["name"] == "3wires"
        assert circuit_json["shots"] == 100
        assert circuit_json["target"] == "qpu"

        assert circuit_json["body"]["qubits"] == 3

        circuit_contents = circuit_json["body"]["circuit"]
        assert len(circuit_contents) == 3

        assert circuit_contents[0]["phase"] == 0.3 / (2 * np.pi)
        assert circuit_contents[0]["target"] == 0
        assert circuit_contents[1]["phase"] == -0.4 / (2 * np.pi)
        assert circuit_contents[1]["target"] == 2
        assert circuit_contents[2]["phase"] == 0.5 / (2 * np.pi)
        assert circuit_contents[2]["target"] == 1

    def test_multi_wire_with_ms_gates(self):
        """Test a circuit with multiple MS gates on different wires in different orders."""
        with qml.tape.QuantumTape() as tape:
            GPI(0.3, wires=0)
            MS(wires=[0, 1])
            GPI2(0.5, wires=1)
            MS(wires=[1, 0])

        circuit_json = tape_to_json(tape, name="ms_gates", target="qpu")

        assert circuit_json["name"] == "ms_gates"
        assert circuit_json["shots"] == 100
        assert circuit_json["target"] == "qpu"

        assert circuit_json["body"]["qubits"] == 2

        circuit_contents = circuit_json["body"]["circuit"]
        assert len(circuit_contents) == 4

        assert circuit_contents[0]["phase"] == 0.3 / (2 * np.pi)
        assert circuit_contents[0]["target"] == 0
        assert circuit_contents[1]["phases"] == [0, 0]
        assert circuit_contents[1]["targets"] == [0, 1]
        assert circuit_contents[2]["phase"] == 0.5 / (2 * np.pi)
        assert circuit_contents[2]["target"] == 1
        assert circuit_contents[3]["phases"] == [0, 0]
        assert circuit_contents[3]["targets"] == [1, 0]
