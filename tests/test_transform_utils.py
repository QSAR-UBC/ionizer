"""
Test the utility functions for the transpilation transforms. 
"""

import pytest
from functools import partial

import pennylane as qml
from pennylane import math
import numpy as np

from ionizer.utils import are_mats_equivalent
from ionizer.transform_utils import (
    search_and_apply_two_gate_identities,
    search_and_apply_three_gate_identities,
)

from ionizer.ops import GPI, GPI2, MS


def _compare_op_lists(ops1, ops2):
    """Test if two tapes are equal."""
    assert len(ops1) == len(ops2)

    for op1, op2 in zip(ops1, ops2):
        assert op1.name == op2.name
        assert op1.num_params == op2.num_params
        if op1.num_params > 0:
            assert math.allclose(op1.data, op2.data)

    return True


class TestSearchAndApplyTwoGateIdentities:
    """Tests that two-gate identities are correctly applied."""

    def test_invalid_length(self):
        """Test that sequences with incorrect length throw errors."""
        gates_to_apply = [GPI(0.3, wires=0)]

        with pytest.raises(ValueError, match="Only sets of 2 gates"):
            _ = search_and_apply_two_gate_identities(gates_to_apply)

        with pytest.raises(ValueError, match="Only sets of 2 gates"):
            _ = search_and_apply_two_gate_identities(gates_to_apply * 3)

    def test_invalid_gates(self):
        """Test that when invalid gates are passed, an error is thrown."""
        expected_gates_to_apply = [qml.Hadamard(wires=0), qml.RX(0.2, wires=0)]

        with pytest.raises(
            ValueError, match="only 2- and 3-gate circuit identities on GPI/GPI2 gates"
        ):
            _ = search_and_apply_two_gate_identities(expected_gates_to_apply)

    def test_double_gpi2_identity_valid_angles(self):
        """Test that the identity GPI2(x) GPI2(x) = GPI(x) is found."""
        gates_to_apply = [GPI2(0.3, wires=0), GPI2(0.3, wires=0)]

        obtained_gates_to_apply = search_and_apply_two_gate_identities(gates_to_apply)

        assert len(obtained_gates_to_apply) == 1
        assert obtained_gates_to_apply[0].name == "GPI"
        assert math.isclose(obtained_gates_to_apply[0].data[0], 0.3)

        initial_tape = qml.tape.QuantumTape(gates_to_apply, [])
        obtained_tape = qml.tape.QuantumTape(obtained_gates_to_apply, [])
        assert are_mats_equivalent(qml.matrix(initial_tape), qml.matrix(obtained_tape))

    def test_double_gpi2_identity_invalid_angles(self):
        """Test that if angles are different, GPI2(x) GPI2(x) = GPI(x) is not used."""
        gates_to_apply = [GPI2(0.3, wires=0), GPI2(0.4, wires=0)]

        obtained_gates_to_apply = search_and_apply_two_gate_identities(gates_to_apply)

        assert _compare_op_lists(gates_to_apply, obtained_gates_to_apply)

    def test_double_gpi2_identity_valid_angles_different_wires(self):
        """Test that GPI2(x) GPI2(x) = GPI(x) is not used if wires are different."""
        gates_to_apply = [GPI2(0.3, wires=0), GPI2(0.4, wires=1)]

        with pytest.raises(ValueError, match="Gates must share wires to find identities."):
            _ = search_and_apply_two_gate_identities(gates_to_apply)

    @pytest.mark.parametrize(
        "input_ops,expected_ops",
        [
            ([GPI2(0, wires=0), GPI(0, wires=0)], [GPI2(-np.pi, wires=0)]),
            ([GPI2(0, wires=2), GPI2(np.pi, wires=2)], []),
            ([GPI2(np.pi, wires=0), GPI2(0, wires=0)], []),
            ([GPI2(np.pi, wires=1), GPI(0, wires=1)], [GPI2(0, wires=1)]),
        ],
    )
    def test_valid_identities(self, input_ops, expected_ops):
        """Test a set of known identities."""
        obtained_ops = search_and_apply_two_gate_identities(input_ops)
        assert _compare_op_lists(obtained_ops, expected_ops)

        # Add identity op to account for cases where the simplification returned is identity
        initial_tape = qml.tape.QuantumTape(input_ops, [])
        obtained_tape = qml.tape.QuantumTape(
            [qml.Identity(wires=input_ops[0].wires)] + obtained_ops, []
        )
        assert are_mats_equivalent(qml.matrix(initial_tape), qml.matrix(obtained_tape))


class TestSearchAndApplyThreeGateIdentities:
    """Tests that three-gate identities are correctly applied."""

    def test_invalid_length(self):
        """Test that sequences with incorrect length throw errors."""
        gates_to_apply = [GPI(0.3, wires=0)]

        with pytest.raises(ValueError, match="Only sets of 3 gates"):
            _ = search_and_apply_three_gate_identities(gates_to_apply)

        with pytest.raises(ValueError, match="Only sets of 3 gates"):
            _ = search_and_apply_three_gate_identities(gates_to_apply * 4)

    def test_invalid_gates(self):
        """Test that when invalid gates are passed, an error is thrown."""
        expected_gates_to_apply = [
            qml.Hadamard(wires=0),
            qml.Hadamard(wires=0),
            qml.RX(0.2, wires=0),
        ]

        with pytest.raises(
            ValueError, match="only 2- and 3-gate circuit identities on GPI/GPI2 gates"
        ):
            _ = search_and_apply_three_gate_identities(expected_gates_to_apply)

    def test_invalid_identity(self):
        """Test that if angles are different, no identity gets applied"""
        gates_to_apply = [GPI2(0.3, wires=0), GPI2(0.4, wires=0), GPI2(0.5, wires=0)]

        obtained_gates_to_apply = search_and_apply_three_gate_identities(gates_to_apply)

        assert _compare_op_lists(gates_to_apply, obtained_gates_to_apply)

    def test_valid_identity_different_wires(self):
        """Test that for a valid identity on inconsistent wires, no identity gets applied"""
        gates_to_apply = [GPI2(0, wires=0), GPI(np.pi / 4, wires=1), GPI2(0, wires=0)]

        with pytest.raises(ValueError, match="Gates must share wires to find identities."):
            _ = search_and_apply_three_gate_identities(gates_to_apply)

    @pytest.mark.parametrize(
        "input_ops,expected_ops",
        [
            ([GPI2(np.pi, wires=0), GPI(0, wires=0), GPI2(-np.pi, wires=0)], []),  # Simplifies to I
            (
                [GPI2(0, wires=0), GPI(np.pi / 4, wires=0), GPI2(0, wires=0)],
                [GPI2(-np.pi / 2, wires=0)],
            ),  # 3-gate identity
            (
                [GPI2(0, wires=0), GPI2(np.pi, wires=0), GPI2(0.3, wires=0)],
                [GPI2(0.3, wires=0)],
            ),  # First two gates only
            (
                [GPI2(0.3, wires=0), GPI2(0, wires=0), GPI2(np.pi, wires=0)],
                [GPI2(0.3, wires=0)],
            ),  # Second two gates only
            (
                [GPI2(-np.pi / 2, wires=0), GPI(-np.pi / 4, wires=0), GPI2(-np.pi / 2, wires=0)],
                [GPI2(-np.pi, wires=0)],
            ),  # 3-gate identity
        ],
    )
    def test_valid_identities(self, input_ops, expected_ops):
        """Test a set of known identities."""
        obtained_ops = search_and_apply_three_gate_identities(input_ops)
        assert _compare_op_lists(obtained_ops, expected_ops)

        # Add identity op to account for cases where the simplification returned is identity
        initial_tape = qml.tape.QuantumTape(input_ops, [])
        obtained_tape = qml.tape.QuantumTape(
            [qml.Identity(wires=input_ops[0].wires)] + obtained_ops, []
        )
        assert are_mats_equivalent(qml.matrix(initial_tape), qml.matrix(obtained_tape))
