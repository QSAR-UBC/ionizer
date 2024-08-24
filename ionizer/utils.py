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

import pennylane as qml
import numpy as np
from pennylane import math


def are_mats_equivalent(unitary1, unitary2):
    r"""Checks the equivalence of two unitary matrices up to a global phase.

    Args:
        unitary1 (tensor): First unitary matrix.
        unitary2 (tensor): Second unitary matrix.

    Returns:
        bool: True if the two matrices are equivalent up to a global phase,
        False otherwise.

    **Example**

    .. code::

        >>> matrix_T = np.diag([1, 1j])
        >>> matrix_RZ = np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])
        >>> are_mats_equivalent(matrix_T, matrix_RZ)
        True
    """
    mat_product = math.dot(unitary1, math.conj(math.T(unitary2)))

    # If the top-left entry is not 0, divide everything by it and test against identity
    if not math.isclose(mat_product[0, 0], 0.0):
        mat_product = mat_product / mat_product[0, 0]

        if math.allclose(mat_product, math.eye(mat_product.shape[0])):
            return True

    return False


def flag_non_equivalence(tape1, tape2):
    """Check equivalence of two circuits up to a global phase.

    Args:
        tape1 (pennylane.QuantumTape): a quantum tape
        tape2 (pennylane.QuantumTape): quantum tape to compare with ``tape1``

    Raises:
        ValueError if the two circuits are not equivalent.

    **Example**

    .. code::

        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.T(wires=1)

        with qml.tape.QuantumTape() as tape2_equiv:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(np.pi/4, wires=1)

        with qml.tape.QuantumTape() as tape3_nonequiv:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(np.pi/2, wires=1)

    .. code::

        >>> flag_non_equivalence(tape1, tape2_equiv)
        >>> flag_non_equivalence(tape1, tape3_nonequiv)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "ionizer/utils.py", line 98, in flag_non_equivalence
            raise ValueError("Quantum circuits are not equivalent after transform.")
        ValueError: Quantum circuits are not equivalent after transform.

    """
    # Compute matrix representation using a consistent wire order
    joint_wires = qml.wires.Wires.all_wires([tape1.wires, tape2.wires])
    matrix_1 = qml.matrix(tape1, wire_order=joint_wires)
    matrix_2 = qml.matrix(tape2, wire_order=joint_wires)

    if not are_mats_equivalent(matrix_1, matrix_2):
        raise ValueError("Quantum circuits are not equivalent after transform.")



def rescale_angles(angles, renormalize_for_json=False):
    r"""Rescale gate rotation angles into a fixed range.

    By default, rescales between :math:`-\pi` and :math:`\pi`. However, IonQ's
    native gate parameters are defined in terms of "turns", rather than radians,
    where 1 turn is equivalent to :math:`2\pi`. Setting ``renormalize_for_json``
    to ``True`` converts radians to turns.  See the `IonQ
    documentation <https://docs.ionq.com/api-reference/v0.3/native-gates-api>`_
    for more information.

    Args:
        angles (tensor): The angles to rescale.
        renormalize_for_json (bool): If ``True``, rescale angles into the range
            :math:`-1` to :math:`1` (:math:`-2\pi` to :math:`2\pi`). Otherwise,
            rescale to within :math:`-\pi` to :math:`\pi`.

    Returns:
        tensor: The rescaled angles.

    **Example**

    .. code::

        >>> angles = np.array([-4, -2, 0, -2, 4])
        >>> rescale_angles(angles)
        [ 2.28318531 -2.          0.         -2.         -2.28318531]
        >>> rescale_angles(angles, renormalize_for_json=True)
        [ 0.36338023 -0.31830989  0.         -0.31830989 -0.36338023]

    """
    rescaled_angles = math.arctan2(math.sin(angles), math.cos(angles))

    if renormalize_for_json:
        return rescaled_angles / (2 * np.pi)

    return rescaled_angles


def extract_gpi2_gpi_gpi2_angles(unitary):
    r"""Given unitary matrix, recovers a set of three angles :math:`\alpha`,
    :math:`\beta`, and :math:`\gamma` such that

    .. math::

        U = GPI2(\alpha) GPI(\beta) GPI2(\gamma)

    up to a global phase.

    This function is loosely based on PennyLane's
    `zyz_decomposition <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.zyz_decomposition.html>`_.

    Args:
        unitary (tensor): A unitary matrix.

    Returns:
        tensor: Rotation angles for the :math:`GPI` and :math:`GPI2`
        gates. The order of the returned angles corresponds to the order in
        which they would appear in a circuit, i.e., ``[gamma, beta, alpha]``.

    **Example**

    .. code::

        >>> matrix_RZ = np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])
        >>> angles = extract_gpi2_gpi_gpi2_angles(matrix_RZ)
        >>> angles
        [ 2.35619449  3.14159265 -2.35619449]
        >>> recovered_matrix = np.linalg.multi_dot(
        ...     [op.compute_matrix(angle) for op, angle in zip([GPI2, GPI, GPI2], angles[::-1])]
        ... )
        >>> are_mats_equivalent(matrix_RZ, recovered_matrix)
        True

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
    """Convert a quantum tape consisting of :math:`GPI`, :math:`GPI2` and
    :math:`MS` operations into a JSON object suitable for job submission to hardware.

    Please see the `IonQ webpage
    <https://docs.ionq.com/api-reference/v0.3/native-gates-api>`_ for full documentation of the
    job submission API.

    Note that this function is not tested against the API in an automated, or
    manual way. If this function needs to be updated to work with API changes,
    please `open an issue <https://github.com/QSAR-UBC/ionizer/issues>`_.

    Args:
        tape (QuantumTape): The quantum tape of the circuit to send.
        name (str): Desired name of the job.
        shots (int): Number of shots to be executed by the job.
        target (str): Where the job will be executed, e.g., "simulator" or a
            particular hardware device.

    Returns:
        Dict: JSON formatted for submission to IonQ hardware.

    **Example**

    .. code::

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        @ionize
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs()

        # Note that the circuit must be executed once to generate the tape.
        circuit()

        json_object = tape_to_json(circuit.qtape, "job_name", 1024, "simulator")

    .. code::

        >>> pprint(json_object)
        {'input': {'circuit': [{'gate': 'gpi2', 'phase': 0.0, 'target': 0},
                              {'gate': 'gpi2', 'phase': 0.5, 'target': 1},
                              {'gate': 'ms', 'phases': [0, 0], 'targets': [0, 1]},
                              {'gate': 'gpi2', 'phase': -0.25, 'target': 0}],
                  'gateset': 'native',
                  'qubits': 2},
         'lang': 'json',
         'name': 'job_name',
         'shots': 1024,
         'target': 'simulator'}

    """

    circuit_json = {}
    circuit_json["lang"] = "json"
    circuit_json["shots"] = shots
    circuit_json["target"] = target
    circuit_json["name"] = name

    circuit_json["input"] = {}
    circuit_json["input"]["gateset"] = "native"
    circuit_json["input"]["qubits"] = len(tape.wires)

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

    circuit_json["input"]["circuit"] = circuit_op_list

    return circuit_json
