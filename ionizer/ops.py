# Some of the code in this file is repurposed from PennyLane.
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
Native gates for IonQ hardware as PennyLane operations.
"""

import numpy as np

import pennylane as qml
from pennylane.operation import Operation


class GPI(Operation):
    r"""The single-qubit :math:`GPI` rotation

    .. math:: GPI(\phi) = \begin{bmatrix}
        0 & e^{-i\phi} \\ e^{i\phi} & 0
        \end{bmatrix}.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    """

    num_wires = 1
    num_params = 1
    ndim_params = (0,)

    # Note: disable pylint complaint about redefined built-in, since the id
    # value itself is coming from the class definition of Operators in PennyLane proper.
    def __init__(self, phi, wires, id=None):  # pylint: disable=redefined-builtin
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Canonical matrix representation in computational basis.

        Args:
            phi (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> GPI.compute_matrix(0.3)
        array([[0.        +0.j        , 0.95533649-0.29552021j],
               [0.95533649+0.29552021j, 0.        +0.j        ]])

        """
        return qml.math.stack([[0, qml.math.exp(-1j * phi)], [qml.math.exp(1j * phi), 0]])

    def adjoint(self):
        # The GPI gate is its own adjoint.
        return GPI(self.data[0], self.wires)


class GPI2(Operation):
    r"""The single-qubit :math:`GPI2` rotation

    .. math:: GPI2(\phi) = \frac{1}{\sqrt{2}} \begin{bmatrix}
        1 & -ie^{-i\phi} \\ -ie^{i\phi} & 1
        \end{bmatrix}.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    """

    num_wires = 1
    num_params = 1
    ndim_params = (0,)

    def __init__(self, phi, wires, id=None):  # pylint: disable=redefined-builtin
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Canonical matrix representation in computational basis.

        Args:
            phi (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> GPI2.compute_matrix(0.3)
        array([[ 0.70710678+0.j        , -0.33900505-0.62054458j],
               [ 0.33900505-0.62054458j,  0.70710678+0.j        ]])

        """
        exponent = -1j * phi
        return qml.math.stack(
            [
                [1, -1j * qml.math.exp(exponent)],
                [-1j * qml.math.exp(qml.math.conj(exponent)), 1],
            ]
        ) / np.sqrt(2)

    def adjoint(self):
        return GPI2(self.data[0] + np.pi, self.wires)


class MS(Operation):
    r"""The two-qubit Mølmer-Sørensen (MS) operation.

    In general this is a parametrized operation, but the IonQ hardware permits
    only this version, where both parameters are 0.

    .. math:: MS = \frac{1}{\sqrt{2}} \begin{bmatrix}
                1 & 0 & 0 & -i \\
                0 & 1 & -i & 0 \\
                0 & -i & 1 & 0 \\
                -i & 0 & 0 & 1 \\
              \end{bmatrix}.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    """

    num_wires = 2
    num_params = 0

    def __init__(self, wires, id=None):  # pylint: disable=redefined-builtin
        super().__init__(wires=wires, id=id)

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Canonical matrix representation in computational basis.

        Args:
            phi (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> MS.compute_matrix()
        array([[ 0.70710678+0.j        ,  0.        +0.j        ,
                 0.        +0.j        , -0.        -0.70710678j],
               [ 0.        +0.j        ,  0.70710678+0.j        ,
                -0.        -0.70710678j,  0.        +0.j        ],
               [ 0.        +0.j        , -0.        -0.70710678j,
                 0.70710678+0.j        ,  0.        +0.j        ],
               [-0.        -0.70710678j,  0.        +0.j        ,
                 0.        +0.j        ,  0.70710678+0.j        ]])

        """
        return qml.math.stack(
            [[1, 0, 0, -1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [-1j, 0, 0, 1]]
        ) / np.sqrt(2)
