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
"""Transforms for transpiling textbook gates into native trapped-ion gates.

The main transform in this module, :func:`ionizer.transforms.ionize`,
performs end-to-end transpilation and optimization of circuits. It calls a
number of helper transforms which can also be used individually.

All transforms contain a mechanism for under-the-hood equivalence checking (up
to a global phase) through the ``verify_equivalence`` flag. When set, an error
will be raised if the transpiled circuit is not equivalent to the original. For
details and example usage see :ref:`basic_usage-equivalence_validation` and
:func:`ionizer.utils.flag_non_equivalence`.

"""

from typing import Sequence, Callable
from functools import partial

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization.optimization_utils import find_next_gate

from .utils import flag_non_equivalence, rescale_angles, extract_gpi2_gpi_gpi2_angles
from .decompositions import decomp_map
from .ops import GPI, GPI2
from .identity_hunter import (
    search_and_apply_two_gate_identities,
    search_and_apply_three_gate_identities,
)


@qml.transform
def commute_through_ms_gates(
    tape: QuantumTape, direction="right", verify_equivalence=False
) -> (Sequence[QuantumTape], Callable):
    r"""Commute :math:`GPI` and :math:`GPI2` gates with special angle values
    through :class:`~ionizer.ops.MS` gates.

    The following gates commute with :math:`MS` gates when applied to either
    qubit in the :math:`MS` gate: :math:`GPI2(0)`, :math:`GPI2(\pm \pi)`,
    :math:`GPI(0)`, :math:`GPI(\pm \pi)`.

    When there are multiple adjacent :math:`MS` gates, commuting :math:`GPI` and
    :math:`GPI2` gates are pushed as far as possible in the specified direction
    (see example).

    This function is based on PennyLane's `commute_controlled  <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.commute_controlled.html>`_
    transform.

    Args:
        tape (pennylane.QuantumTape): A quantum tape to transform.
        direction (str): Either ``"right"`` (default) or ``"left"`` to indicate
            the direction gates should move (from a circuit diagram perspective).
        verify_equivalence (bool): Whether to perform background equivalence
            checking (up to global phase) of the circuit before and after the
            transform.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape],
        function]: The transformed circuit as described in :func:`qml.transform
        <pennylane.transform>`.

    **Example**

    .. code::

        import pennylane as qml
        from pennylane import numpy as np
        from functools import partial

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        @partial(commute_through_ms_gates, direction="left")
        def circuit():
            MS(wires=[0, 1])
            MS(wires=[1, 2])
            GPI2(np.pi, wires=0)
            GPI(0, wires=1)
            GPI(0.3, wires=2)
            return qml.probs()

    .. code::

        >>> qml.draw(circuit)()
        0: ──GPI2(3.14)─╭MS────────────────┤  Probs
        1: ──GPI(0.00)──╰MS─╭MS────────────┤  Probs
        2: ─────────────────╰MS──GPI(0.30)─┤  Probs

    """
    if direction not in ["left", "right"]:
        raise ValueError(
            f"Direction for commute_through_ms_gates must be 'left' or 'right'. Got {direction}."
        )

    list_copy = tape.operations.copy()
    new_operations = []

    # If we want to push left, easier to go right but in reverse.
    if direction == "left":
        list_copy = list_copy[::-1]

    with qml.QueuingManager.stop_recording():
        with qml.tape.QuantumTape() as _:
            while len(list_copy) > 0:
                current_gate = list_copy[0]
                list_copy.pop(0)

                # Apply MS as we find them
                if len(current_gate.wires) == 2:
                    new_operations.append(current_gate)
                    continue

                # Find the next gate that acts on the same wires
                next_gate_idx = find_next_gate(current_gate.wires, list_copy)

                # If there is no next gate, just apply this one and move on
                if next_gate_idx is None:
                    new_operations.append(current_gate)
                    continue

                next_gate = list_copy[next_gate_idx]

                # To limit code duplication, decide whether to apply next gate or not.
                apply_current_gate = True

                # Check if next gate is MS and see if we can commute through it.
                if next_gate.name == "MS":
                    # The following commute through MS gates on either qubit:
                    # GPI2(0), GPI2(π), GPI2(-π), GPI(0), GPI(π), GPI(-π)
                    if current_gate.name in ["GPI", "GPI2"]:
                        angle = math.abs(current_gate.data[0])
                        if math.isclose(angle, 0.0) or math.isclose(angle, np.pi):
                            list_copy.insert(next_gate_idx + 1, current_gate)
                            apply_current_gate = False

                # If we didn't commute this gate through, apply it.
                if apply_current_gate:
                    new_operations.append(current_gate)

    if direction == "left":
        new_operations = new_operations[::-1]

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    if verify_equivalence:
        flag_non_equivalence(tape, new_tape)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


@qml.transform
def virtualize_rz_gates(tape: QuantumTape, verify_equivalence=False) -> (Sequence[QuantumTape], Callable):
    r"""Apply :math:`RZ` gates virtually by adjusting the phase of adjacent
    :math:`GPI` and :math:`GPI2` gates.

    This transform reads a circuit from left to right, and applies the
    following circuit identities (expressed in matrix order):

     - :math:`GPI(x) RZ(z) = GPI(x - z/2)`
     - :math:`GPI2(x) RZ(z) = RZ(z) GPI2(x - z)`

    :math:`RZ` are pushed as far right as possible, until :math:`MS` gates are
    encountered or the end of the circuit is reached.  Any :math:`RZ(\phi)` that
    are not absorbed into native gates are then implemented as

    .. math:: RZ(\phi) = GPI(0) GPI(-\phi/2).

    Args:
        tape (pennylane.QuantumTape): A quantum tape to transform
        verify_equivalence (bool): Whether to perform background equivalence
            checking (up to global phase) of the circuit before and after the
            transform.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape],
        function]: The transformed circuit as described in :func:`qml.transform
        <pennylane.transform>`.

    **Example**

    .. code::

        import pennylane as qml
        from pennylane import numpy as np

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        @virtualize_rz_gates
        def circuit():
            qml.RZ(0.3, wires=0)
            GPI(0.5, wires=0)

            GPI2(0.2, wires=1)
            qml.RZ(np.pi/2, wires=1)
            GPI2(0.4, wires=1)
            GPI(-np.pi/2, wires=1)

            MS(wires=[0, 1])

            qml.RZ(0.8, wires=1)
            return qml.probs()

    .. code::

        >>> qml.draw(circuit)()
        0: ──GPI(0.35)───────────────────────────╭MS────────────────────────┤  Probs
        1: ──GPI2(0.20)──GPI2(-1.17)──GPI(-2.36)─╰MS──GPI(-0.40)──GPI(0.00)─┤  Probs

    """
    list_copy = tape.operations.copy()
    new_operations = []

    with qml.QueuingManager.stop_recording():
        while len(list_copy) > 0:
            current_gate = list_copy[0]
            list_copy.pop(0)

            if current_gate.name == "RZ":
                next_gate_idx = find_next_gate(current_gate.wires, list_copy)

                # No gate afterwards; just apply this one but use GPI gates
                if next_gate_idx is None:
                    new_operations.append(GPI(-current_gate.data[0] / 2, wires=current_gate.wires))
                    new_operations.append(GPI(0.0, wires=current_gate.wires))
                    continue

                # As long as there are more single-qubit gates afterwards, push the
                # RZ through and apply the phase-adjusted gates
                accumulated_rz_phase = current_gate.data[0]
                apply_accumulated_phase_gate = True

                while next_gate_idx is not None:
                    next_gate = list_copy[next_gate_idx]

                    # If the next gate is an RZ, accumulate its phase into this gate,
                    # then remove it from the queue and don't apply anything yet.
                    if next_gate.name == "RZ":
                        accumulated_rz_phase += next_gate.data[0]
                        list_copy.pop(next_gate_idx)
                    # Apply the identity GPI(θ) RZ(ϕ) = GPI(θ - ϕ/2); then there are no more
                    # RZs to process so we leave the loop.
                    elif next_gate.name == "GPI":
                        new_operations.append(
                            GPI(
                                rescale_angles(next_gate.data[0] - accumulated_rz_phase / 2),
                                wires=current_gate.wires,
                            )
                        )
                        apply_accumulated_phase_gate = False
                        list_copy.pop(next_gate_idx)
                        break
                    # Apply the identity GPI2(θ) RZ(ϕ) = RZ(ϕ) GPI2(θ - ϕ); apply the GPI2 gate
                    # with adjusted phase.
                    elif next_gate.name == "GPI2":
                        new_operations.append(
                            GPI2(
                                rescale_angles(next_gate.data[0] - accumulated_rz_phase),
                                wires=current_gate.wires,
                            )
                        )
                        list_copy.pop(next_gate_idx)
                    # If it's anything else, we want to just apply it normally
                    else:
                        break

                    next_gate_idx = find_next_gate(current_gate.wires, list_copy)

                # Once we pass through all the gates, if there is any remaining
                # accumulated phase, apply the RZ gate as two GPI gates. Apply the GPI(0)
                # last, because this gate is likely to be right before an MS gate, and
                # this way we can commute it through.
                if apply_accumulated_phase_gate:
                    new_operations.append(
                        GPI(
                            -rescale_angles(accumulated_rz_phase) / 2,
                            wires=current_gate.wires,
                        )
                    )
                    new_operations.append(GPI(0.0, wires=current_gate.wires))

            else:
                new_operations.append(current_gate)

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    if verify_equivalence:
        flag_non_equivalence(tape, new_tape)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


@qml.transform
def single_qubit_fusion_gpi(tape: QuantumTape, verify_equivalence=False) -> (Sequence[QuantumTape], Callable):
    r"""Simplify sequences of :math:`GPI` and :math:`GPI2` gates using gate
    fusion and circuit identities.

    Any sequence of more than 3 gates will be fused and re-implemented up to
    global phase using :math:`GPI` and :math:`GPI2` (see
    :func:`ionizer.utils.extract_gpi2_gpi_gpi2_angles`).

    Sequences of two or three gates (including those obtained through gate
    fusion) are then checked against the database of known circuit identities
    for simplifications (see :func:`ionizer.identity_hunter.lookup_gate_identity`).

    This transform is based on PennyLane's `single_qubit_fusion
    <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.single_qubit_fusion.html>`_
    transform.

    Args:
        tape (pennylane.QuantumTape): A quantum tape to transform.
        verify_equivalence (bool): Whether to perform background equivalence
            checking (up to global phase) of the circuit before and after the
            transform.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape],
        function]: The transformed circuit as described in :func:`qml.transform
        <pennylane.transform>`.

    **Example**

    .. code::

        import pennylane as qml
        from pennylane import numpy as np

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        @single_qubit_fusion_gpi
        def circuit():
            # Known circuit identity
            GPI(np.pi/4, wires=0)
            GPI2(-3 * np.pi/4, wires=0)

            # Already three gates, but no known identity
            GPI2(0.2, wires=1)
            GPI2(0.4, wires=1)
            GPI(-np.pi/2, wires=1)

            # Squished down to three gates
            GPI2(0.1, wires=2)
            GPI2(0.2, wires=2)
            GPI(0.3, wires=2)
            GPI(0.4, wires=2)
            GPI2(0.5, wires=2)

            return qml.probs()

    .. code::

        >>> qml.draw(circuit)()
        0: ──GPI2(0.79)───────────────────────────┤  Probs
        1: ──GPI2(0.20)───GPI2(0.40)──GPI(-1.57)──┤  Probs
        2: ──GPI2(-1.57)──GPI(2.65)───GPI2(-0.97)─┤  Probs

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()
    new_operations = []

    with qml.QueuingManager.stop_recording():
        while len(list_copy) > 0:
            current_gate = list_copy[0]
            list_copy.pop(0)

            # Ignore 2-qubit gates
            if len(current_gate.wires) > 1:
                new_operations.append(current_gate)
                continue

            # Find the next gate that acts on the same wires
            next_gate_idx = find_next_gate(current_gate.wires, list_copy)

            # If there is no next gate, just apply this one
            if next_gate_idx is None:
                new_operations.append(current_gate)
                continue

            gates_to_apply = [current_gate]

            # Loop as long as a valid next gate exists
            while next_gate_idx is not None:
                next_gate = list_copy[next_gate_idx]

                if len(next_gate.wires) > 1:
                    break

                gates_to_apply.append(next_gate)
                list_copy.pop(next_gate_idx)

                next_gate_idx = find_next_gate(current_gate.wires, list_copy)

            # We should only actually do fusion if we find more than 3 gates
            # Otherwise, we just apply them normally
            if len(gates_to_apply) == 1:
                new_operations.append(gates_to_apply[0])
            # Try applying identities to sequences of two-qubit gates.
            elif len(gates_to_apply) == 2:
                new_operations.extend(search_and_apply_two_gate_identities(gates_to_apply))
            # If we have exactly three gates, try applying identities to the sequence
            elif len(gates_to_apply) == 3:
                new_operations.extend(search_and_apply_three_gate_identities(gates_to_apply))
            # If we have more than three gates, we need to fuse.
            else:
                running_matrix = qml.matrix(gates_to_apply[0])

                for gate in gates_to_apply[1:]:
                    running_matrix = np.dot(qml.matrix(gate), running_matrix)

                gamma, beta, alpha = extract_gpi2_gpi_gpi2_angles(running_matrix)

                # If all three angles are the same, GPI2(θ) GPI(θ) GPI2(θ) = I
                if all(math.isclose([gamma, beta, alpha], [gamma])):
                    continue

                # Construct the three new operations to apply
                gates_to_apply = [
                    GPI2(gamma, wires=current_gate.wires),
                    GPI(beta, wires=current_gate.wires),
                    GPI2(alpha, wires=current_gate.wires),
                ]
                new_operations.extend(search_and_apply_three_gate_identities(gates_to_apply))

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    if verify_equivalence:
        flag_non_equivalence(tape, new_tape)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


@qml.transform
def convert_to_gpi(tape: QuantumTape, exclude_list=None, verify_equivalence=False) -> (Sequence[QuantumTape], Callable):
    r"""Transpile desired gates in a circuit to trapped-ion gates.

    Args:
        tape (pennylane.QuantumTape): A quantum tape to transform.
        exclude_list (list[str]): A list of names of gates to exclude from
            conversion (see the ionize transform for an example).
        verify_equivalence (bool): Whether to perform background equivalence
            checking (up to global phase) of the circuit before and after the
            transform.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape],
        function]: The transformed circuit as described in :func:`qml.transform
        <pennylane.transform>`.

    **Example**

    .. code::

        import pennylane as qml
        from pennylane import numpy as np

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        @partial(convert_to_gpi, exclude_list=["IsingYY"])
        def circuit():
            # Gates with known decompositions
            qml.Hadamard(wires=0)
            qml.PauliX(wires=1)
            qml.RX(0.2, wires=2)

            # Should not be decomposed
            qml.IsingYY(0.1, wires=[0, 1])
            qml.IsingYY(0.2, wires=[0, 2])

            # Gets expanded into two RY and two CNOT gates, which are then decomposed
            qml.CRY(0.3, wires=[0, 1])

            return qml.probs()

    .. code::

        >>> qml.draw(circuit)()
        0: ──GPI(0.00)───GPI2(-1.57)─╭IsingYY(0.10)─╭IsingYY(0.20)──GPI2(1.57)────────────────────────╭MS───GPI2(3.14)──GPI2(-1.57)──GPI2(1.57)─────────────╭MS──GPI2(3.14)──GPI2(-1.57)─┤  Probs
        1: ──GPI(0.00)───────────────╰IsingYY(0.10)─│───────────────GPI2(3.14)──GPI(0.07)──GPI2(3.14)─╰MS───GPI2(3.14)──GPI2(3.14)───GPI(-0.07)──GPI2(3.14)─╰MS──GPI2(3.14)──────────────┤  Probs
        2: ──GPI2(1.57)──GPI(-1.47)───GPI2(1.57)────╰IsingYY(0.20)───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤  Probs

    """
    if exclude_list is None:
        exclude_list = []

    new_operations = []

    # We define a custom expansion function here to convert everything except
    # the gates identified in exclude_list to gates with known decompositions.
    def stop_at(op):
        return op.name in decomp_map or op.name in exclude_list

    custom_expand_fn = qml.transforms.create_expand_fn(depth=9, stop_at=stop_at)

    with qml.QueuingManager.stop_recording():
        expanded_tape = custom_expand_fn(tape)

        for op in expanded_tape.operations:
            if op.name not in exclude_list and op.name in decomp_map:
                if op.num_params > 0:
                    new_operations.extend(decomp_map[op.name](*op.data, op.wires))
                else:
                    new_operations.extend(decomp_map[op.name](op.wires))
            else:
                new_operations.append(op)

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    if verify_equivalence:
        flag_non_equivalence(tape, new_tape)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


@qml.transform
def ionize(tape: QuantumTape, verify_equivalence=False) -> (Sequence[QuantumTape], Callable):
    r"""Apply a sequence of passes to transpile and optimize a circuit
    over the trapped-ion gate set :math:`GPI`, :math:`GPI2`, and :math:`MS`.

    The following sequence of passes is performed:

        - Decompose all operations into Paulis/Pauli rotations, Hadamard, and :math:`CNOT`
        - Cancel inverses and merge single-qubit rotations
        - Convert everything except :math:`RZ` to :math:`GPI`, :math:`GPI2`, and :math:`MS` gates
        - Virtually apply :math:`RZ` gates
        - Repeatedly apply single-qubit gate fusion and commutation through
          :math:`MS` gates, and perform simplification based on a database of
          circuit identities.

    .. note::

        When ``verify_equivalence`` is set to ``True``, equivalence checking up
        to a global phase is performed with respect to the initial and final
        circuit matrices only. It is not checked for intermediate transforms.
    
    Args:
        tape (pennylane.QuantumTape): A quantum tape to transform.
        verify_equivalence (bool): Whether to perform background equivalence
            checking (up to global phase) of the circuit before and after the
            transform.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape],
        function]: The transformed circuit as described in :func:`qml.transform
        <pennylane.transform>`.

    **Example**

    .. code::

        import pennylane as qml
        from ionizer.transforms import ionize

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        @ionize
        def circuit():
            qml.Hadamard(wires=0)
            qml.PauliX(wires=1)
            qml.RX(0.2, wires=2)
            qml.CRY(0.3, wires=[0, 1])
            return qml.probs()

    .. code::

        >>> qml.draw(circuit)()
        0: ─────────────────────────────────────╭MS──────────────────────────────────────╭MS──GPI2(-1.57)─┤  Probs
        1: ──GPI2(1.57)──GPI(-0.86)──GPI2(1.42)─╰MS──GPI2(-1.57)──GPI(2.43)──GPI2(-1.42)─╰MS──────────────┤  Probs
        2: ──GPI2(1.57)──GPI(-1.47)──GPI2(1.57)───────────────────────────────────────────────────────────┤  Probs

    """

    def stop_at(op):
        return op.name in decomp_map

    custom_expand_fn = qml.transforms.create_expand_fn(depth=9, stop_at=stop_at)

    with qml.QueuingManager.stop_recording():
        # Initial set of passes to decompose and translate the tape and virtualize RZ
        optimized_tape = custom_expand_fn(tape)
        optimized_tape, _ = qml.transforms.cancel_inverses(optimized_tape)
        optimized_tape, _ = qml.transforms.merge_rotations(optimized_tape)
        optimized_tape, _ = partial(convert_to_gpi, exclude_list=["RZ"])(optimized_tape[0])
        optimized_tape, _ = virtualize_rz_gates(optimized_tape[0])

        # Actual optimization passes
        # TODO: how many iterations do we actually have to do?
        for _ in range(5):
            optimized_tape, _ = partial(commute_through_ms_gates, direction="left")(
                optimized_tape[0]
            )
            optimized_tape, _ = single_qubit_fusion_gpi(optimized_tape[0])

    new_tape = type(tape)(optimized_tape[0].operations, tape.measurements, shots=tape.shots)

    if verify_equivalence:
        flag_non_equivalence(tape, new_tape)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
