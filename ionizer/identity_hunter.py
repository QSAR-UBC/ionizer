"""
Submodule to generate and store a database of circuit identities involving
up to 3 GPI/GPI2 gates.
"""
from importlib.resources import files

from itertools import product
import pickle
import numpy as np

from pennylane import math

from .ops import GPI, GPI2
from .utils import are_mats_equivalent

DOUBLE_IDENTITY_FILE = files("resources").joinpath("double_gate_identities.pkl")
TRIPLE_IDENTITY_FILE = files("resources").joinpath("triple_gate_identities.pkl")


def generate_gate_identities():
    """Generates all 2- and 3-gate identities involving GPI/GPI2 and special angles.

    Results are stored in pkl files which can be used later on.
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

    double_gate_identities = {}

    # Check which combinations of 2 gates reduces to a single one
    for gate_1, gate_2 in product([GPI, GPI2], repeat=2):
        combo_name = gate_1.__name__ + gate_2.__name__

        for angle_1, angle_2 in product(id_angles, repeat=2):
            matrix = math.dot(gate_1.compute_matrix(angle_1), gate_2.compute_matrix(angle_2))

            # Test in case we produced the identity;
            if not math.isclose(matrix[0, 0], 0.0):
                if math.allclose(matrix / matrix[0, 0], math.eye(2)):
                    if combo_name not in list(double_gate_identities.keys()):
                        double_gate_identities[combo_name] = []
                    double_gate_identities[combo_name].append(([angle_1, angle_2], "Identity", 0.0))
                    continue

            for id_gate in list(single_gates.keys()):
                angles, matrices = [x[0] for x in single_gates[id_gate]], [
                    x[1] for x in single_gates[id_gate]
                ]

                for ref_angle, ref_matrix in zip(angles, matrices):
                    if are_mats_equivalent(matrix, ref_matrix):
                        if combo_name not in list(double_gate_identities.keys()):
                            double_gate_identities[combo_name] = []

                        if not any(
                            np.allclose([angle_1, angle_2], database_angles)
                            for database_angles in [
                                identity[0] for identity in double_gate_identities[combo_name]
                            ]
                        ):
                            double_gate_identities[combo_name].append(
                                ([angle_1, angle_2], id_gate, ref_angle)
                            )

    with DOUBLE_IDENTITY_FILE.open("wb") as outfile:
        pickle.dump(double_gate_identities, outfile)

    triple_gate_identities = {}

    # Check which combinations of 2 gates reduces to a single one
    for gate_1, gate_2, gate_3 in product([GPI, GPI2], repeat=3):
        combo_name = gate_1.__name__ + gate_2.__name__ + gate_3.__name__

        for angle_1, angle_2, angle_3 in product(id_angles, repeat=3):
            matrix = math.linalg.multi_dot(
                [
                    gate_1.compute_matrix(angle_1),
                    gate_2.compute_matrix(angle_2),
                    gate_3.compute_matrix(angle_3),
                ]
            )

            # Test in case we produced the identity;
            if not math.isclose(matrix[0, 0], 0.0):
                if math.allclose(matrix / matrix[0, 0], math.eye(2)):
                    if combo_name not in list(triple_gate_identities.keys()):
                        triple_gate_identities[combo_name] = []
                    triple_gate_identities[combo_name].append(
                        ([angle_1, angle_2, angle_3], "Identity", 0.0)
                    )
                    continue

            for id_gate, _ in single_gates.items():
                angles, matrices = [x[0] for x in single_gates[id_gate]], [
                    x[1] for x in single_gates[id_gate]
                ]

                for ref_angle, ref_matrix in zip(angles, matrices):
                    if are_mats_equivalent(matrix, ref_matrix):
                        if combo_name not in list(triple_gate_identities.keys()):
                            triple_gate_identities[combo_name] = []

                        if not any(
                            np.allclose([angle_1, angle_2, angle_3], database_angles)
                            for database_angles in [
                                identity[0] for identity in triple_gate_identities[combo_name]
                            ]
                        ):
                            triple_gate_identities[combo_name].append(
                                ([angle_1, angle_2, angle_3], id_gate, ref_angle)
                            )

    with TRIPLE_IDENTITY_FILE.open("wb") as outfile:
        pickle.dump(triple_gate_identities, outfile)


def lookup_gate_identity(gates):
    """Given a pair of input gates in the order they come in the circuit,
    look up if there is a circuit identity in our database. Note that the
    database is constructed using matrix multiplication so we will need to
    exchange the order of the gates."""

    if len(gates) not in [2, 3]:
        raise ValueError("Currently only 2- and 3-gate circuit identities are supported.")

    if any(op.name not in ["GPI", "GPI2"] for op in gates):
        raise ValueError(
            "Currently only 2- and 3-gate circuit identities on GPI/GPI2 gates are supported."
        )

    if len(gates) == 2:
        try:
            with DOUBLE_IDENTITY_FILE.open("rb") as infile:
                gate_identities = pickle.load(infile)
        except FileNotFoundError:
            # Generate the file first and then load it
            generate_gate_identities()
            with DOUBLE_IDENTITY_FILE.open("rb") as infile:
                gate_identities = pickle.load(infile)
    elif len(gates) == 3:
        try:
            with TRIPLE_IDENTITY_FILE.open("rb") as infile:
                gate_identities = pickle.load(infile)
        except FileNotFoundError:
            # Generate the file first and then load it
            generate_gate_identities()
            with TRIPLE_IDENTITY_FILE.open("rb") as infile:
                gate_identities = pickle.load(infile)

    # Get the information about this particular combination of gates
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
