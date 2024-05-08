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
Utility functions.
"""
import numpy as np
from pennylane import math


def are_mats_equivalent(unitary1, unitary2):
    """Checks the equivalence of two unitary matrices.

     Args:
        unitary1 (tensor): First unitary matrix.
        unitary2 (tensor): Second unitary matrix.

    Returns:
        bool: True if the two matrices are equivalent up to a global phase,
        False otherwise.
    """
    mat_product = math.dot(unitary1, math.conj(math.T(unitary2)))

    # If the top-left entry is not 0, divide everything by it and test against identity
    if not math.isclose(mat_product[0, 0], 0.0):
        mat_product = mat_product / mat_product[0, 0]

        if math.allclose(mat_product, math.eye(mat_product.shape[0])):
            return True

    return False


def rescale_angles(angles, renormalize_for_json=False):
    r"""Rescale gate rotation angles into a fixed range between -np.pi and np.pi.

    Args:
        angles (tensor): The angles to rescale.
        renormalize_for_json (bool): By default, we rescale into the range -np.pi to
            np.pi. If this is set to True, rescale instead into the range -1 to
            1 (-2\pi to 2\pi) as this the range of angles accepted by IonQ's
            native gate input specs.

    Return:
        (tensor): The rescaled angles.
    """
    rescaled_angles = math.arctan2(math.sin(angles), math.cos(angles))

    if renormalize_for_json:
        return rescaled_angles / (2 * np.pi)

    return rescaled_angles


def extract_gpi2_gpi_gpi2_angles(unitary):
    r"""Given a matrix U, recovers a set of three angles alpha, beta, and
    gamma such that
        U = GPI2(alpha) GPI(beta) GPI2(gamma)
    up to a global phase.

    This function is loosely based on the zyz_decomposition function implemented
    in the PennyLane decomposition transform, adjusted for a different gate set.
    https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.zyz_decomposition.html

    Args:
        unitary (tensor): A unitary matrix.

    Returns:
        tensor: Rotation angles for the GPI/GPI2 gates. The order of the
        returned angles corresponds to the order in which they would be
        implemented in the circuit.
    """
    det = math.angle(math.linalg.det(unitary))
    su2_mat = math.exp(-1j * det / 2) * unitary

    phase_00 = math.angle(su2_mat[0, 0])
    phase_10 = math.angle(su2_mat[1, 0])

    # Extract the angles; note that we clip the input to the arccos because due
    # to finite precision, the number may be slightly greater than 1 when their
    # input matrix element is 0, which would throw an error
    alpha = phase_10 - phase_00 + np.pi
    beta = math.arccos(math.clip(math.abs(su2_mat[0, 0]), 0, 1)) + phase_10 + np.pi
    gamma = phase_10 + phase_00 + np.pi

    return rescale_angles([gamma, beta, alpha])


def tape_to_json(tape, name, shots=100, target="simulator"):
    """Convert a quantum tape expressed in terms of GPI/GPI2/MS operations
    into a JSON object suitable for job submission to hardware.

    Args:
        tape (QuantumTape): The quantum tape of the circuit to send.
        name (str): Desired name of the job.
        shots (int): Number of shots to be executed by the job.
        target (str): Where the job will be executed, e.g., "simulator" or a
            particular hardware device.

    Returns:
        Dict: JSON formatted for submission to hardware.
    """

    circuit_json = {}
    circuit_json["lang"] = "json"
    circuit_json["shots"] = shots
    circuit_json["target"] = target
    circuit_json["name"] = name

    circuit_json["body"] = {}
    circuit_json["body"]["gateset"] = "native"
    circuit_json["body"]["qubits"] = len(tape.wires)

    circuit_op_list = []

    for op in tape.operations:
        gate_dict = {
            "gate": str.lower(op.name),
        }

        if op.name == "MS":
            gate_dict["phases"] = [0, 0]
            gate_dict["targets"] = [int(x) for x in op.wires]
        else:
            gate_dict["phase"] = rescale_angles(float(op.data[0]), renormalize_for_json=True)
            gate_dict["target"] = int(op.wires[0])

        circuit_op_list.append(gate_dict)

    circuit_json["body"]["circuit"] = circuit_op_list

    return circuit_json
