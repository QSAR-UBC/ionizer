# The Ionizer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10761367.svg)](https://doi.org/10.5281/zenodo.10761367)

Transpile and optimize your [PennyLane](https://github.com/pennylaneai/pennylane) circuits into
IonQ's native trapped-ion gate set (GPI, GPI2, MS) with just a single extra line
of code!

```python
from ionizer.transforms import ionize


@qml.qnode(dev)
@ionize
def circuit(x):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(x, wires=1)
    return qml.expval(qml.PauliZ(0))
```

```pycon
>>> qml.draw(circuit)(0.3)
0: ──GPI2(0.00)─╭MS──GPI2(-1.57)─────────────────────────┤  <Z>
1: ──GPI2(3.14)─╰MS──GPI2(1.57)───GPI(-1.42)──GPI2(1.57)─┤
```

## Installation

Requirements:

- PennyLane >= 0.33

The Ionizer is not currently available via a package manager. To install, clone the repository and run

```
python -m pip install .
```

or

```
python setup.py install
```

If you need to run Ionizer with a version of PennyLane between 0.29 and 0.32,
please use version 0.1.2 of the package.

## Examples

The Ionizer is implemented using [quantum function
transforms](https://arxiv.org/abs/2202.13414), similar to PennyLane's [existing
compilation
tools](https://docs.pennylane.ai/en/stable/introduction/compiling_circuits.html). To
compile and execute the circuit using trapped ion gates, the
`@ionize` decorator performs the following steps:

- Decomposes all operations into Paulis/Pauli rotations, Hadamard, and CNOT
- Merges all single-qubit rotations
- Converts everything except RZ to GPI/GPI2/MS gates (`@ionizer.transforms.convert_to_gpi`)
- Virtually applies all RZ gates (`@ionizer.transforms.virtualize_rz_gates`)
- Repeatedly applies gate fusion and commutation through MS gates which performs simplification based on some circuit identities (`@ionizer.transforms.single_qubit_fusion_gpi` and `@ionizer.transforms.commute_through_ms_gates`)

```python
from ionizer.transforms import ionize


@qml.qnode(dev)
@ionize
def circuit_ionized(params):
    for idx in range(5):
        qml.Hadamard(wires=idx)

    for idx in range(4):
        qml.RY(params[idx], wires=idx)
        qml.CNOT(wires=[idx + 1, idx])

    for wire in dev.wires:
        qml.PauliX(wires=wire)

    return qml.expval(qml.PauliX(0))
```

```pycon
>>> circuit_ionized(params)
tensor(0.99500417, requires_grad=True)
>>> qml.draw(circuit_ionized)(params)
0: ──GPI2(-1.57)──GPI(-1.52)──GPI2(-3.04)─╭MS───────────────────────────────────────────────────
1: ──GPI2(-1.92)──GPI(3.14)───GPI2(-1.22)─╰MS──GPI2(2.36)──GPI(1.67)──GPI2(0.99)─╭MS────────────
2: ──GPI2(-1.92)──GPI(3.14)───GPI2(-1.22)────────────────────────────────────────╰MS──GPI2(2.36)
3: ──GPI2(-1.92)──GPI(3.14)───GPI2(-1.22)───────────────────────────────────────────────────────
4: ──GPI2(-1.92)──GPI(3.14)───GPI2(-1.22)───────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────────────────┤  <X>
────────────────────────────────────────────────────────────────────────────────────────────┤
───GPI(1.72)──GPI2(1.09)─╭MS────────────────────────────────────────────────────────────────┤
─────────────────────────╰MS──GPI2(2.36)──GPI(1.77)──GPI2(1.19)─╭MS─────────────────────────┤
────────────────────────────────────────────────────────────────╰MS──GPI2(0.00)──GPI2(1.57)─┤

```

Note that while this comes packaged together as the `@ionize` transform, the
individual transforms can also be accessed and used independently.

There is currently not direct support for other frameworks. However, if you would like to do this with a Qiskit circuit, it can be accomplished as follows through the [`pennylane-qiskit`](https://github.com/PennyLaneAI/pennylane-qiskit) package.

```python
qiskit_circuit = QuantumCircuit(...)

# Turns a Qiskit circuit into a PennyLane quantum function
qfunc = qml.from_qiskit(qiskit_circuit)


@qml.qnode(dev)
@ionize
def pennylane_circuit():
    qfunc()
    return qml.expval(qml.PauliX(0))
```

## Notes

This package is a work in progress. While it has been verified to work on some
fairly large circuits, we still need to work on:

- finding circuit identities involving the 2-qubit gate
- improving the documentation and usage instructions
- ensuring differentiability of variational parameters
- writing more tests (compile at your own risk!)

## Resources

- [IonQ documentation](https://ionq.com/docs/getting-started-with-native-gates)
- [Basic circuit compilation techniques for an ion-trap quantum machine](https://arxiv.org/abs/1603.07678)

## Citing

If you use the Ionizer as part of your workflow, we would appreciate if you cite it using the BibTeX below.

```
@software{di_matteo_2024_10761367,
  author       = {Di Matteo, Olivia},
  title        = {The Ionizer},
  month        = mar,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.2},
  doi          = {10.5281/zenodo.10761367},
  url          = {https://doi.org/10.5281/zenodo.10761367}
}
```
