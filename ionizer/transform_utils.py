"""
Utility functions for transpiling normal gates into trapped-ion gates, such
as circuit identities.
"""

import pennylane as qml

from .ops import GPI
from .identity_hunter import lookup_gate_identity


def search_and_apply_two_gate_identities(gates_to_apply):
    """Try to simplify a sequence of two gates.

    Sequences that are found are queued to the current context; if no identity
    is found, we simply queue the provided sequence of gates.

    Args:
        gates_to_apply (List[pennylane.Operation]): A sequence of two gates
            we would like to simplify.

    Returns:
        List[pennylane.Operation]: The simplified or alternate gate sequence that
        will be applied within the transform.
    """
    if len(gates_to_apply) != 2:
        raise ValueError(
            "Only sets of 2 gates can be passed to search_and_apply_two_gate_identities"
        )

    # Make sure the gates share wires
    if len(qml.wires.Wires.shared_wires([gates_to_apply[0].wires, gates_to_apply[1].wires])) != 1:
        raise ValueError("Gates must share wires to find identities.")

    # Special case with no fixed angles: GPI2(x) GPI2(x) = GPI(x)
    if gates_to_apply[0].name == "GPI2" and gates_to_apply[1].name == "GPI2":
        if qml.math.isclose(gates_to_apply[0].data[0], gates_to_apply[1].data[0]):
            return [GPI(gates_to_apply[0].data[0], wires=gates_to_apply[0].wires)]

    # Search for non-special identity
    with qml.QueuingManager.stop_recording():
        identity_to_apply = lookup_gate_identity(gates_to_apply)

    # If there is an identity, return it; otherwise just return the gates
    return identity_to_apply if identity_to_apply is not None else gates_to_apply


def search_and_apply_three_gate_identities(gates_to_apply):
    """Try to simplify a sequence of three gates.

    Sequences that are found are queued to the current context; if no identity
    is found, we simply queue the provided sequence of gates.

    Args:
        gates_to_apply (List[pennylane.Operation]): A sequence of three gates
             we would like to simplify.

    Returns:
        List[pennylane.Operation]: The simplified or alternate gate sequence that
        will be applied within the transform.
    """
    if len(gates_to_apply) != 3:
        raise ValueError(
            "Only sets of 3 gates can be passed to search_and_apply_three_gate_identities"
        )

    # Make sure the gates share wires
    if len(qml.wires.Wires.shared_wires([gate.wires for gate in gates_to_apply])) != 1:
        raise ValueError("Gates must share wires to find identities.")

    # First, check if we can apply an identity to all three gates
    with qml.QueuingManager.stop_recording():
        three_gate_identity_to_apply = lookup_gate_identity(gates_to_apply)

    if three_gate_identity_to_apply is not None:
        return three_gate_identity_to_apply

    # If we can't apply a 3-gate identity, see if there is a 2-gate one on the
    # first two gates.
    with qml.QueuingManager.stop_recording():
        identity_to_apply = search_and_apply_two_gate_identities(gates_to_apply[:2])

    if identity_to_apply is not None:
        if len(identity_to_apply) < 2:
            return identity_to_apply + [gates_to_apply[2]]

    # If not, apply the first gate, then check if there is anything to be
    # done between the second and third.
    identity_to_apply = search_and_apply_two_gate_identities(gates_to_apply[1:])
    if identity_to_apply is not None:
        if len(identity_to_apply) < 2:
            return [gates_to_apply[0]] + identity_to_apply

    return gates_to_apply
