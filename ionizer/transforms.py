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
Transforms for transpiling normal gates into trapped-ion gates.

The main transform, @ionizer.transforms.ionize, will perform a full sequence of
expansions and simplifications of the tape. The transforms it uses during this
process can also be called individually.
"""
from typing import Sequence, Callable
from functools import partial

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization.optimization_utils import find_next_gate

from .utils import rescale_angles, extract_gpi2_gpi_gpi2_angles
from .decompositions import decomp_map
from .ops import GPI, GPI2
from .transform_utils import (
    search_and_apply_two_gate_identities,
    search_and_apply_three_gate_identities,
)


@qml.transform
def commute_through_ms_gates(
    tape: QuantumTape, direction="right"
) -> (Sequence[QuantumTape], Callable):
    """Apply a transform that passes through a tape and pushes GPI/GPI2
    gates with appropriate (commuting) angles through MS gates.

    More specifically, the following commute through MS gates on either qubit:
        GPI2(0), GPI2(π), GPI2(-π), GPI(0), GPI(π), GPI(-π)

    This function is modelled off PennyLane's commute_controlled transform
    https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.commute_controlled.html

    Args:
        tape (pennylane.QuantumTape): A quantum tape to transform.
        direction (str): Which direction to push the commuting gates in.
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
        with qml.tape.QuantumTape() as commuted_tape:
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

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


@qml.transform
def virtualize_rz_gates(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    """When dealing with GPI/GPI2/MS gates, RZ gates can be implemented virtually
    by pushing them through such gates and simply adjusting the phases of the
    gates we pushed them through:
        - GPI(x) RZ(z) = GPI(x - z/2)
        - RZ(z) GPI(x) = GPI(x + z/2)
        - GPI2(x) RZ(z) = RZ(z) GPI2(x - z)
        - RZ(z) GPI2(x) = GPI2(x + z) RZ(z)

    This transform rolls through a tape, and adjusts the circuits so that
    all the RZs get implemented virtually.

    Args:
        tape (pennylane.QuantumTape): A quantum tape to transform.
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
                        GPI(-rescale_angles(accumulated_rz_phase) / 2, wires=current_gate.wires)
                    )
                    new_operations.append(GPI(0.0, wires=current_gate.wires))

            else:
                new_operations.append(current_gate)

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


@qml.transform
def single_qubit_fusion_gpi(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    """Perform single-qubit fusion of all sequences of single-qubit gates into
    no more than 3 GPI/GPI2 gates.

    This transform is based on PennyLane's single_qubit_fusion transform.
    https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.single_qubit_fusion.html

    Args:
        tape (pennylane.QuantumTape): A quantum tape to transform.
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
                first_gate = GPI2(gamma, wires=current_gate.wires)
                second_gate = GPI(beta, wires=current_gate.wires)
                third_gate = GPI2(alpha, wires=current_gate.wires)

                gates_to_apply = [first_gate, second_gate, third_gate]
                new_operations.extend(search_and_apply_three_gate_identities(gates_to_apply))

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


@qml.transform
def convert_to_gpi(tape: QuantumTape, exclude_list=[]) -> (Sequence[QuantumTape], Callable):
    """Transpile a tape directly to native trapped ion gates.

    Any operation without a decomposition in decompositions.py will remain
    as-is.

    Args:
        tape (pennylane.QuantumTape): A quantum tape to transform.
        exclude_list (list[str]): A list of names of gates to exclude from
            conversion (see the ionize transform for an example).
    """
    new_operations = []

    with qml.QueuingManager.stop_recording():
        for op in tape.operations:
            if op.name not in exclude_list and op.name in decomp_map.keys():
                if op.num_params > 0:
                    new_operations.extend(decomp_map[op.name](*op.data, op.wires))
                else:
                    new_operations.extend(decomp_map[op.name](op.wires))
            else:
                new_operations.append(op)

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


@qml.transform
def ionize(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    """A full set of transpilation passes to apply to convert the circuit
    into native gates and optimize it.

    It performs the following sequence of steps:
        - Decomposes all operations into Paulis/Pauli rotations, Hadamard, and CNOT
        - Merges all single-qubit rotations
        - Converts everything except RZ to GPI/GPI2/MS gates
        - Virtually applies all RZ gates
        - Repeatedly applies gate fusion and commutation through MS gate
          which performs simplification based on some circuit identities.

    Args:
        tape (pennylane.QuantumTape): A quantum tape to transform.
    """

    # The tape will first be expanded into known operations
    def stop_at(op):
        return op.name in list(decomp_map.keys())

    custom_expand_fn = qml.transforms.create_expand_fn(depth=9, stop_at=stop_at)

    with qml.QueuingManager.stop_recording():
        # Initial set of passes to decompose and translate the tape and virtualize RZ
        optimized_tape = custom_expand_fn(tape)
        optimized_tape, _ = qml.transforms.merge_rotations(optimized_tape)
        optimized_tape, _ = partial(convert_to_gpi, exclude_list=["RZ"])(optimized_tape[0])
        optimized_tape, _ = virtualize_rz_gates(optimized_tape[0])

        # TODO: how many iterations do we actually have to do?
        for _ in range(5):
            optimized_tape, _ = partial(commute_through_ms_gates, direction="left")(
                optimized_tape[0]
            )
            optimized_tape, _ = single_qubit_fusion_gpi(optimized_tape[0])

    new_tape = type(tape)(optimized_tape[0].operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
