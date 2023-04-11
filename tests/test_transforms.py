"""
Test the suite of transpilation transforms. 
"""
import pytest

import pennylane as qml
from pennylane import math
import numpy as np

from ionizer.utils import are_mats_equivalent
from ionizer.transforms import (
    commute_through_ms_gates,
    virtualize_rz_gates,
    single_qubit_fusion_gpi,
    convert_to_gpi,
    ionize,
)
from ionizer.ops import GPI, GPI2, MS


def _compare_tape_contents(tape1, tape2):
    """Test if two tapes are equal."""
    assert len(tape1.operations) == len(tape2.operations)

    for op1, op2 in zip(tape1.operations, tape2.operations):
        assert op1.name == op2.name
        assert op1.num_params == op2.num_params
        if op1.num_params > 0:
            assert math.allclose(op1.data, op2.data)

    return True


class TestCommuteThroughMSGates:
    """Tests that we correctly commute GPI/GPI2 gates through MS gates."""

    def test_no_commutation_either_qubit(self):
        """Test that when no gates on either qubits commute, nothing happens."""
        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI(0.3, wires=0)
            GPI2(0.2, wires=1)
            MS(wires=[0, 1])

        transformed_tape = commute_through_ms_gates.tape_fn(tape)
        assert _compare_tape_contents(tape, transformed_tape)

        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI2(0.3, wires=0)
            GPI(0.1, wires=0)
            MS(wires=[0, 1])

        transformed_tape = commute_through_ms_gates.tape_fn(tape)
        assert _compare_tape_contents(tape, transformed_tape)

    def test_commutation_both_qubits(self):
        """Test that when no gates on either qubits commute, nothing happens."""
        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI(0, wires=0)
            GPI2(-np.pi, wires=1)
            MS(wires=[0, 1])

        with qml.tape.QuantumTape() as expected_tape:
            MS(wires=[0, 1])
            MS(wires=[0, 1])
            GPI2(-np.pi, wires=1)  # Most recent gate always pushed ahead
            GPI(0, wires=0)

        transformed_tape = commute_through_ms_gates.tape_fn(tape)
        assert _compare_tape_contents(transformed_tape, expected_tape)

    def test_commutation_one_qubit(self):
        """Test case where gates on one of the qubits commutes."""
        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI(0, wires=0)
            GPI2(0.3, wires=1)
            MS(wires=[0, 1])

        with qml.tape.QuantumTape() as expected_tape:
            MS(wires=[0, 1])
            GPI2(0.3, wires=1)
            MS(wires=[0, 1])
            GPI(0, wires=0)

        transformed_tape = commute_through_ms_gates.tape_fn(tape)
        assert _compare_tape_contents(transformed_tape, expected_tape)

        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI2(0.3, wires=0)
            GPI(np.pi, wires=1)
            MS(wires=[0, 1])

        with qml.tape.QuantumTape() as expected_tape:
            MS(wires=[0, 1])
            GPI2(0.3, wires=0)
            MS(wires=[0, 1])
            GPI(np.pi, wires=1)

        transformed_tape = commute_through_ms_gates.tape_fn(tape)
        assert _compare_tape_contents(transformed_tape, expected_tape)

    def test_commutation_one_qubit_multiple_ms(self):
        """Test case where gates on one of the qubits commutes through multiple MS gates."""
        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI(0, wires=0)
            MS(wires=[0, 1])
            MS(wires=[0, 2])

        with qml.tape.QuantumTape() as expected_tape:
            MS(wires=[0, 1])
            MS(wires=[0, 1])
            MS(wires=[0, 2])
            GPI(0, wires=0)

        transformed_tape = commute_through_ms_gates.tape_fn(tape)
        assert _compare_tape_contents(transformed_tape, expected_tape)

    def test_commutation_two_qubits_multiple_ms(self):
        """Test case where gates on one of the qubits commutes."""
        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI(0, wires=0)
            GPI2(np.pi, wires=2)
            MS(wires=[0, 1])
            MS(wires=[0, 2])

        with qml.tape.QuantumTape() as expected_tape:
            MS(wires=[0, 1])
            MS(wires=[0, 1])
            MS(wires=[0, 2])
            GPI(0, wires=0)
            GPI2(np.pi, wires=2)

        transformed_tape = commute_through_ms_gates.tape_fn(tape)
        assert _compare_tape_contents(transformed_tape, expected_tape)


class TestVirtualizeRZGates:
    """Tests that virtual RZ gates are implemented correctly."""

    def test_rz_gpi(self):
        """Test that RZ is correctly pushed through a GPI gate."""
        with qml.tape.QuantumTape() as tape:
            qml.RZ(0.3, wires=0)
            GPI(0.6, wires=0)

        transformed_tape = virtualize_rz_gates.tape_fn(tape)
        print(transformed_tape.operations)

        assert len(transformed_tape.operations) == 1
        assert are_mats_equivalent(qml.matrix(tape), qml.matrix(transformed_tape))

    def test_rz_gpi2(self):
        """Test that RZ is correctly pushed through a GPI2 gate."""
        with qml.tape.QuantumTape() as tape:
            qml.RZ(0.3, wires=0)
            GPI2(0.6, wires=0)

        transformed_tape = virtualize_rz_gates.tape_fn(tape)

        assert len(transformed_tape.operations) == 3
        assert all(
            op.name == name for op, name in zip(transformed_tape.operations, ["GPI2", "GPI", "GPI"])
        )
        assert are_mats_equivalent(qml.matrix(tape), qml.matrix(transformed_tape))

    def test_rz_gpi_ms(self):
        """Test that non-GPI gates stop RZ from going through."""
        with qml.tape.QuantumTape() as tape:
            GPI2(0.5, wires=0)
            qml.RZ(0.3, wires=0)
            MS(wires=[0, 1])

        transformed_tape = virtualize_rz_gates.tape_fn(tape)

        assert len(transformed_tape.operations) == 4
        assert all(
            op.name == name
            for op, name in zip(transformed_tape.operations, ["GPI2", "GPI", "GPI", "MS"])
        )
        assert are_mats_equivalent(qml.matrix(tape), qml.matrix(transformed_tape))

    def test_rz_gpi_multiqubit(self):
        """Test that RZ gates are virtualized on multiple qubits."""
        with qml.tape.QuantumTape() as tape:
            qml.RZ(0.3, wires=0)
            GPI2(0.5, wires=0)
            qml.RZ(0.2, wires=1)
            GPI(0.5, wires=1)

        transformed_tape = virtualize_rz_gates.tape_fn(tape)

        assert len(transformed_tape.operations) == 4
        assert all(
            op.name == name
            for op, name in zip(transformed_tape.operations, ["GPI2", "GPI", "GPI", "GPI"])
        )
        assert are_mats_equivalent(qml.matrix(tape), qml.matrix(transformed_tape))

    def test_rz_gpi_multiqubit_multims(self):
        """Test that RZ gates are virtualized on multiple qubits with gates in between."""
        with qml.tape.QuantumTape() as tape:
            qml.RZ(0.3, wires=0)
            GPI2(0.5, wires=0)
            MS(wires=[1, 0])
            qml.RZ(0.2, wires=1)
            GPI(0.5, wires=1)
            MS(wires=[1, 0])

        transformed_tape = virtualize_rz_gates.tape_fn(tape)

        assert len(transformed_tape.operations) == 6
        assert all(
            op.name == name
            for op, name in zip(
                transformed_tape.operations, ["GPI2", "GPI", "GPI", "MS", "GPI", "MS"]
            )
        )
        assert are_mats_equivalent(qml.matrix(tape), qml.matrix(transformed_tape))


class TestSingleQubitGPIFusion:
    """Tests that gate fusion and identity application happens correctly."""

    def test_no_fusion(self):
        """Test that if there are no gates to fuse that nothing happens."""
        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI(0.3, wires=0)
            MS(wires=[0, 1])
            GPI(0.3, wires=1)
            GPI(0.2, wires=0)
            GPI2(0, wires=2)
            MS(wires=[1, 2])

        transformed_tape = single_qubit_fusion_gpi.tape_fn(tape)
        assert _compare_tape_contents(tape, transformed_tape)
        assert are_mats_equivalent(qml.matrix(tape), qml.matrix(transformed_tape))

    def test_no_fusion_multiple_gates(self):
        """Test that if there are no gates to fuse that nothing happens, even
        when there is more than one gate on a qubit."""
        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI(0.3, wires=0)
            GPI2(0.3, wires=0)
            MS(wires=[0, 1])
            GPI(0.3, wires=1)
            GPI2(0.2, wires=0)
            GPI2(0, wires=2)
            MS(wires=[1, 2])

        transformed_tape = single_qubit_fusion_gpi.tape_fn(tape)
        assert _compare_tape_contents(tape, transformed_tape)
        assert are_mats_equivalent(qml.matrix(tape), qml.matrix(transformed_tape))

    def test_fusion_three_gates(self):
        """Test that if a GPI2/GPI/GPI2 sequence already exists that we don't fuse."""
        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI2(0.3, wires=0)
            GPI(0.2, wires=0)
            GPI2(0.3, wires=0)
            MS(wires=[0, 1])
            GPI(0.3, wires=1)
            GPI2(0.2, wires=0)
            GPI2(0, wires=2)
            MS(wires=[1, 2])

        transformed_tape = single_qubit_fusion_gpi.tape_fn(tape)
        assert _compare_tape_contents(tape, transformed_tape)
        assert are_mats_equivalent(qml.matrix(tape), qml.matrix(transformed_tape))

    def test_fusion_four_gates(self):
        """Test that more than three gates get properly fused."""
        with qml.tape.QuantumTape() as tape:
            MS(wires=[0, 1])
            GPI2(0.3, wires=0)
            GPI(0.2, wires=0)
            GPI2(np.pi / 2, wires=0)
            GPI2(0.3, wires=0)
            GPI2(0.4, wires=0)  # Fuse with one less gate
            MS(wires=[0, 1])
            GPI(0.3, wires=1)
            GPI2(0.2, wires=0)
            GPI2(0, wires=2)
            GPI2(1.0, wires=2)
            GPI2(2.0, wires=2)  # Fuse but same number of gates
            MS(wires=[1, 2])

        transformed_tape = single_qubit_fusion_gpi.tape_fn(tape)
        assert len(transformed_tape.operations) == len(tape.operations) - 2
        assert are_mats_equivalent(qml.matrix(tape), qml.matrix(transformed_tape))


class TestConvertToGPI:
    """Tests that operations on a tape are correctly converted to GPI gates."""

    def test_convert_tape_to_gpi_known_gates(self):
        """Test that known gates are correctly converted to GPI gates."""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[2, 0])
            qml.Hadamard(wires=2)

        transformed_tape = convert_to_gpi.tape_fn(tape)

        assert all(op.name in ["GPI", "GPI2", "MS"] for op in transformed_tape.operations)
        assert are_mats_equivalent(qml.matrix(tape), qml.matrix(transformed_tape))

    def test_convert_tape_to_gpi_known_gates_exclusion(self):
        """Test that known gates are correctly converted to GPI gates and
        excluded gates are kept as-is."""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[2, 0])
            qml.Hadamard(wires=2)

        transformed_tape = convert_to_gpi.tape_fn(tape, exclude_list=["RY"])

        assert all(op.name in ["GPI", "GPI2", "MS", "RY"] for op in transformed_tape.operations)
        assert transformed_tape.operations[3].name == "RY"
        assert are_mats_equivalent(
            qml.matrix(tape, wire_order=range(3)), qml.matrix(transformed_tape, wire_order=range(3))
        )


class TestIonize:
    """Integration test for full ionize transform."""

    def test_ionize_tape(self):
        """Test ionize transform on a single tape."""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[2, 0])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[0, 1])
            qml.SX(wires=3)
            qml.RZ(-np.pi / 2, wires=2)
            qml.CNOT(wires=[1, 3])

        transformed_tape = ionize.tape_fn(tape)

        assert all(op.name in ["GPI", "GPI2", "MS"] for op in transformed_tape.operations)
        assert are_mats_equivalent(
            qml.matrix(tape, wire_order=range(4)), qml.matrix(transformed_tape, wire_order=range(4))
        )

    def test_ionize_qnode(self):
        """Test ionize transform on a QNode."""
        dev = qml.device("default.qubit", wires=5)

        def quantum_function():
            qml.RX(0.3, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[2, 0])
            qml.Hadamard(wires=2)
            qml.SX(wires=3)
            qml.CNOT(wires=[3, 0])

        @qml.qnode(dev)
        def normal_qnode():
            quantum_function()
            return qml.expval(qml.PauliX(0) @ qml.PauliY(wires=1))

        @qml.qnode(dev)
        @ionize
        def ionized_qnode():
            quantum_function()
            return qml.expval(qml.PauliX(0) @ qml.PauliY(wires=1))

        assert math.allclose(normal_qnode(), ionized_qnode())

        assert all(op.name in ["GPI", "GPI2", "MS"] for op in ionized_qnode.qtape.operations)
        assert are_mats_equivalent(
            qml.matrix(normal_qnode, wire_order=range(4))(),
            qml.matrix(ionized_qnode, wire_order=range(4))(),
        )

    @pytest.mark.parametrize(
        "params",
        [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.0, 0.0, 0.0]),
            np.array([-1.0, -2.0, -0.5]),
            np.array([-0.54, 0.68, 0.11]),
        ],
    )
    def test_ionize_parametrized_qnode(self, params):
        """Test ionize transform on a QNode."""
        dev = qml.device("default.qubit", wires=5)

        def quantum_function(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[2, 0])
            qml.Hadamard(wires=2)
            qml.RZ(params[2], wires=3)
            qml.CNOT(wires=[3, 0])
            qml.CRZ(0.1, wires=[0, 2])

        @qml.qnode(dev)
        def normal_qnode(params):
            quantum_function(params)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(wires=2))

        @qml.qnode(dev)
        @ionize
        def ionized_qnode(params):
            quantum_function(params)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(wires=2))

        assert math.allclose(normal_qnode(params), ionized_qnode(params))

        assert all(op.name in ["GPI", "GPI2", "MS"] for op in ionized_qnode.qtape.operations)
        assert are_mats_equivalent(
            qml.matrix(normal_qnode, wire_order=range(4))(params),
            qml.matrix(ionized_qnode, wire_order=range(4))(params),
        )
