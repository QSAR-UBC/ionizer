# The Ionizer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13370396.svg)](https://doi.org/10.5281/zenodo.13370396)
<a href="https://ionizer.readthedocs.io/en/stable/" target="_blank"><img src="https://readthedocs.org/projects/ionizer/badge/?version=stable"></a>

Transpile and optimize your PennyLane circuits into
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


The Ionizer is available via PyPI:

```
pip install ionizer
```

The core requirement is [PennyLane](https://pennylane.ai/) 0.37.

Python versions 3.10-3.12 are supported and tested against.

To install from source, clone this repository and use
[Poetry](https://python-poetry.org/) to install the dependencies listed in the
`pyproject.toml` file.

## Examples

For more detailed explanations and usage examples, please check the full
[online documentation](https://ionizer.readthedocs.io/en/stable/).

The Ionizer is implemented using [quantum function
transforms](https://arxiv.org/abs/2202.13414), similar to PennyLane's [existing
compilation
tools](https://docs.pennylane.ai/en/stable/introduction/compiling_circuits.html). To
compile and execute the circuit using trapped ion gates, the
`@ionize` transform will

 - Decompose all operations into Paulis/Pauli rotations, Hadamard, and CNOT
 - Cancel inverses and merge single-qubit rotations
 - Convert everything except RZ to GPI, GPI2, and MS gates
 - Virtually apply RZ gates
 - Repeatedly apply single-qubit gate fusion and commutation through MS gates,
   and perform simplification based on a database of circuit identities.

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

The consistuent transforms can also be accessed and used independently.

There is currently not direct support for other frameworks. However, if you
would like to apply the transforms to Qiskit circuits, this can be accomplished
using the
[`pennylane-qiskit`](https://github.com/PennyLaneAI/pennylane-qiskit) package as
shown belown.

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
- ensuring differentiability of variational parameters
- writing more tests (compile at your own risk!)

## Resources

- [IonQ documentation](https://ionq.com/docs/getting-started-with-native-gates)
- [Basic circuit compilation techniques for an ion-trap quantum machine](https://arxiv.org/abs/1603.07678)


## Contributing

The Ionizer is available open source under the MIT License.  Contributions are
welcome. Please open an issue if you are interested in contributing, or if you
encounter a bug.


## Reference

If you use the Ionizer as part of your workflow, we would appreciate if you cite it using the BibTeX below.

```
@software{di_matteo_2024_13370396,
  author       = {Di Matteo, Olivia},
  title        = {The Ionizer},
  month        = aug,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.3},
  doi          = {10.5281/zenodo.13370396},
  url          = {https://doi.org/10.5281/zenodo.13370396}
}
```
