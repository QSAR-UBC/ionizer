r"""Submodule to generate and store a database of circuit identities involving
up to three successive :math:`GPI` and :math:`GPI2` gates.

This module is primarily for internal use. It exposes only functions to query the
database for an identity involving a specified gate sequence. The database is
included as a set of pickle files with the package.

"""

from importlib.resources import files

from itertools import product
import pickle
import numpy as np

import pennylane as qml
from pennylane import math

from .ops import GPI, GPI2
from .utils import are_mats_equivalent

DOUBLE_IDENTITY_FILE = files("ionizer.resources").joinpath("double_gate_identities.pkl")
TRIPLE_IDENTITY_FILE = files("ionizer.resources").joinpath("triple_gate_identities.pkl")


def _test_inclusion_in_identity_db(db_subset, single_gates, candidate_angles, candidate_matrix):
    """Helper function to test if a candidate gate identity was already found.

    Args:
        db_subset (Dict[str, Tuple(Tuple(float), str, float)]): Subset of database we
            wish to search for presence of identity.
        single_gates (Dict[str, List[Tuple(float, tensor)]]): Dictionary containing
            which gates to generate identities from, along with list of special
            cases of angles/matrices to use in identity generation.
        candidate_angles (List[float]): List of angles used in identity generation.
        candidate_matrix (tensor): Unitary matrix for this particular set of angles.

    Returns:
        Tuple(str or None, float or None): If a valid identity is found, returns the name
        of the gate and its parameter. Otherwise, returns None, None.
    """
    # Loop over GPI/GPI2 for all the special angles
    for id_gate, gate_angle_list in single_gates.items():
        # Get their explicit reference angles and matrix representations
        angles, matrices = (
            [x[0] for x in gate_angle_list],
            [x[1] for x in gate_angle_list],
        )

        # Test each reference against the candidate to see if any are equivalent
        for ref_angle, ref_matrix in zip(angles, matrices):
            if are_mats_equivalent(candidate_matrix, ref_matrix):
                # If they are equivalent, return the identity if not already in database
                if not any(
                    np.allclose(candidate_angles, database_angles)
                    for database_angles in [identity[0] for identity in db_subset]
                ):
                    return id_gate, ref_angle

    return None, None


def _generate_gate_identities(single_gates, id_angles, identity_length):
    """Generates all identities involving specified angles for a sequence
    of :math:`GPI` and :math:`GPI2` gates.

    Args:
        single_gates (Dict[str, List[Tuple(float, tensor)]]): Dictionary containing
            which gates to generate identities from, along with list of special
            cases of angles/matrices to use in identity generation.
        id_angles (List[float]): Special values of angles used in identity generation.
        identity_length (int): How long a gate sequence to test. Must be 2 or 3.

    Returns:
        Dict[str, Tuple(Tuple(float), str, float)]: Dictionary of identities
        where the key is a concatenated string of gates, and the value contains
        the angles involved in the identity, the resultant gate, and its argument.

    Example:
        If ``identity_length=2``, an example return dictionary with one entry is

            ``{'GPIGPI2': ((0.7853981633974483, -2.356194490192345), 'GPI2', 0.7853981633974483)}``

        This represents the equality :math:`GPI(0.785398) GPI2(-2.356194) = GPI2(0.78539)`.
    """

    gate_identities = {}

    # Generate combinations of gates to test if they are equivalent to a single one
    for gate_list in product([GPI, GPI2], repeat=identity_length):
        combo_name = "".join([gate.__name__ for gate in gate_list])

        if combo_name not in gate_identities:
            gate_identities[combo_name] = []

        for angle_list in product(id_angles, repeat=identity_length):
            matrix = math.linalg.multi_dot(
                [gate.compute_matrix(angle) for gate, angle in zip(gate_list, angle_list)]
            )

            # Test in case we produced the identity
            if are_mats_equivalent(matrix, np.eye(2)):
                gate_identities[combo_name].append((angle_list, "Identity", 0.0))
                continue

            # Check if we produced something else instead; if so, add to database
            equivalent_gate, equivalent_angle = _test_inclusion_in_identity_db(
                gate_identities[combo_name], single_gates, angle_list, matrix
            )

            if equivalent_gate is not None:
                gate_identities[combo_name].append((angle_list, equivalent_gate, equivalent_angle))

    return gate_identities


def _generate_gate_identity_database():
    r"""Generates all 2- and 3-gate identities involving :math:`GPI`
    and :math:`GPI2` and special angles.

    Special angles include: :math:`0, \pm \pi/4, \pm \pi/2, \pm 3\pi/4, \pm \pi`.

    Results are stored in ``.pkl`` files which can be used later on.
    """

    id_angles = [
        -np.pi,
        -3 * np.pi / 4,
        -np.pi / 2,
        -np.pi / 4,
        0.0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
        np.pi,
    ]

    single_gates = {
        "GPI": [([angle], GPI.compute_matrix(angle)) for angle in id_angles],
        "GPI2": [([angle], GPI2.compute_matrix(angle)) for angle in id_angles],
    }

    double_gate_identities = _generate_gate_identities(single_gates, id_angles, 2)
    triple_gate_identities = _generate_gate_identities(single_gates, id_angles, 3)

    with DOUBLE_IDENTITY_FILE.open("wb") as outfile:
        pickle.dump(double_gate_identities, outfile)

    with TRIPLE_IDENTITY_FILE.open("wb") as outfile:
        pickle.dump(triple_gate_identities, outfile)


def lookup_gate_identity(gates):
    """Given a sequence of two or three single-qubit gates, query a database of
    known circuit identities for a shorter implementation.

    Args:
        gates (List[Operation]): A list of two or three ``GPI`` and/or ``GPI2`` operations.
            These should be ordered as they appear in the circuit diagram.

    Returns:
        List[Operation]: If an equivalent but shorter sequence of ``GPI`` and ``GPI2`` gates is
        found in the identity database, this will be returned. If no equivalent sequence
        is found, the empty list is returned.

    **Example**

    .. code::

        >>> gate_list = [GPI(np.pi / 4, wires=0), GPI2(-3 * np.pi / 4)]
        >>> lookup_gate_identity(gate_list)
        [GPI2(0.7853981633974483, wires=[0])]

    """

    if len(gates) not in [2, 3]:
        raise ValueError("Currently only 2- and 3-gate circuit identities are supported.")

    if any(op.name not in ["GPI", "GPI2"] for op in gates):
        raise ValueError(
            "Currently only 2- and 3-gate circuit identities on GPI/GPI2 gates are supported."
        )

    gate_identities = {}

    if len(gates) == 2:
        try:
            with DOUBLE_IDENTITY_FILE.open("rb") as infile:
                gate_identities = pickle.load(infile)
        except FileNotFoundError:
            # Generate the file first and then load it
            _generate_gate_identity_database()
            with DOUBLE_IDENTITY_FILE.open("rb") as infile:
                gate_identities = pickle.load(infile)
    elif len(gates) == 3:
        try:
            with TRIPLE_IDENTITY_FILE.open("rb") as infile:
                gate_identities = pickle.load(infile)
        except FileNotFoundError:
            # Generate the file first and then load it
            _generate_gate_identity_database()
            with TRIPLE_IDENTITY_FILE.open("rb") as infile:
                gate_identities = pickle.load(infile)

    # Get the information about this particular combination of gates. Note that
    # the database is constructed using matrix multiplication so we will need to
    # exchange the order of the gates.
    combo_name = "".join([gate.name for gate in gates[::-1]])
    combo_angles = [float(gate.data[0]) for gate in gates[::-1]]

    combo_db = gate_identities[combo_name]
    all_angle_combos = [combo[0] for combo in combo_db]
    angle_check = [np.allclose(combo_angles, test_angles) for test_angles in all_angle_combos]

    if any(angle_check):
        idx = np.where(angle_check)[0][0]
        new_gate_name = combo_db[idx][1]
        new_gate_angle = combo_db[idx][2]

        if new_gate_name == "GPI":
            return [GPI(*new_gate_angle, wires=gates[0].wires)]
        if new_gate_name == "GPI2":
            return [GPI2(*new_gate_angle, wires=gates[0].wires)]
        if new_gate_name == "Identity":
            return []

    return None


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
