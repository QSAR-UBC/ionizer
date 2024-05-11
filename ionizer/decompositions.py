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
Custom decompositions of operations into the {GPI, GPI2, MS} native gate set.
"""
from pennylane import math
import numpy as np

from .ops import GPI, GPI2, MS
from .utils import extract_gpi2_gpi_gpi2_angles


# Non-parametrized operations (up to phases)
def gpi_pauli_x(wires):
    """PauliX decomposition as a GPI gate.

    Args:
        wires (Sequence[int] or pennylane.Wires): The wires this gate is acting on.

    Returns:
        List[Operation]: The sequence of GPI rotations that implements the gate.
    """
    return [GPI(0.0, wires=wires)]


def gpi_pauli_y(wires):
    """PauliY decomposition as a GPI gate.

    Args:
        wires (Sequence[int] or pennylane.Wires): The wires this gate is acting on.

    Returns:
        List[Operation]: The sequence of GPI rotations that implements the gate.
    """
    return [
        GPI(np.pi / 2, wires=wires)]


def gpi_pauli_z(wires):
    """PauliZ decomposition into GPI gates.

    Args:
        wires (Sequence[int] or pennylane.Wires): The wires this gate is acting on.

    Returns:
        List[Operation]: The sequence of GPI rotations that implements
        the gate up to a global phase.
    """
    return [GPI(0.0, wires=wires), GPI(-np.pi / 2, wires=wires)]


def gpi_hadamard(wires):
    """Hadamard decomposition into GPI/GPI2 gates.

    Args:
        wires (Sequence[int] or pennylane.Wires): The wires this gate is acting on.

    Returns:
        List[Operation]: The sequence of GPI/GPI2 rotations that implements
        the gate up to a global phase.
    """
    return [GPI(0.0, wires=wires), GPI2(-np.pi / 2, wires=wires)]


def gpi_sx(wires):
    """Square-root of PauliX as a GPI2 gate.

    Args:
        wires (Sequence[int] or pennylane.Wires): The wires this gate is acting on.

    Returns:
        List[Operation]: The GPI2 rotation that implements the gate.
    """
    return [GPI2(0.0, wires=wires)]


def gpi_cnot(wires):
    """CNOT decomposition into GPI2 and MS gate.

    Args:
        wires (Sequence[int] or pennylane.Wires): The wires this gate is acting on.

    Returns:
        List[Operation]: The sequence of GPI2 rotations that implements
        the gate up to a global phase.
    """
    return [
        GPI2(np.pi / 2, wires=wires[0]),
        MS(wires=wires),
        GPI2(np.pi, wires=wires[0]),
        GPI2(np.pi, wires=wires[1]),
        GPI2(-np.pi / 2, wires=wires[0]),
    ]


# Parametrized operations (up to phases)
def gpi_rx(phi, wires):
    """RX decomposition into GPI/GPI2 gates.

    Args:
        phi (tensor): Rotation angle
        wires (Sequence[int] or pennylane.Wires): The wires this gate is acting on.

    Returns:
        List[Operation]: The sequence of GPI/GPI2 rotations that implements
        the gate up to a global phase.
    """
    return [
        GPI2(np.pi / 2, wires=wires),
        GPI(phi / 2 - np.pi / 2, wires=wires),
        GPI2(np.pi / 2, wires=wires),
    ]


def gpi_ry(phi, wires):
    """RY decomposition into GPI/GPI2 gates.

    Args:
        phi (tensor): Rotation angle
        wires (Sequence[int] or pennylane.Wires): The wires this gate is acting on.

    Returns:
        List[Operation]: The sequence of GPI/GPI2 rotations that implements
        the gate up to a global phase.
    """
    return [GPI2(np.pi, wires=wires), GPI(phi / 2, wires=wires), GPI2(np.pi, wires=wires)]


def gpi_rz(phi, wires):
    """RZ decomposition into GPI/GPI2 gates.

    Args:
        phi (tensor): Rotation angle
        wires (Sequence[int] or pennylane.Wires): The wires this gate is acting on.

    Returns:
        List[Operation]: The sequence of GPI/GPI2 rotations that implements
        the gate up to a global phase.
    """
    return [GPI(-phi / 2, wires=wires), GPI(0.0, wires=wires)]


def gpi_single_qubit_unitary(unitary, wires):
    """Single-qubit unitary matrix decomposition into GPI/GPI2 gates.

    This function is modeled off of PennyLane's unitary_to_rot transform:
    https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.unitary_to_rot.html

    Args:
        unitary (tensor): A unitary matrix.
        wires (Sequence[int] or pennylane.Wires): The wires this gate is acting on.

    Returns:
        List[Operation]: The sequence of GPI/GPI2 rotations that implements
        the desired unitary up to a global phase.
    """
    # Check in case we have the identity
    if math.allclose(unitary, math.eye(2)):
        return []

    # Special case: if we have off-diagonal elements this is a single GPI
    if math.isclose(unitary[0, 0], 0.0):
        angle = math.angle(unitary[1, 0])
        return [GPI(angle, wires=wires)]

    # Special case: if we have off-diagonal 0s but it is not the identity,
    # this is an RZ which is a sequence of two GPIs.
    if math.allclose([unitary[0, 1], unitary[1, 0]], [0.0, 0.0]):
        return gpi_rz(2 * math.angle(unitary[1, 1]), wires)

    # Special case: if both diagonal elements are 1/sqrt(2), this is a GPI2
    if math.allclose([unitary[0, 0], unitary[1, 1]], [1 / np.sqrt(2), 1 / np.sqrt(2)]):
        angle = math.angle(unitary[1, 0]) + np.pi / 2
        return [GPI2(angle, wires=wires)]

    # In the general case we must compute and return all three angles.
    gamma, beta, alpha = extract_gpi2_gpi_gpi2_angles(unitary)

    return [GPI2(gamma, wires=wires), GPI(beta, wires=wires), GPI2(alpha, wires=wires)]


decomp_map = {
    "PauliX": gpi_pauli_x,
    "PauliY": gpi_pauli_y,
    "PauliZ": gpi_pauli_z,
    "Hadamard": gpi_hadamard,
    "SX": gpi_sx,
    "CNOT": gpi_cnot,
    "RX": gpi_rx,
    "RY": gpi_ry,
    "RZ": gpi_rz,
}
