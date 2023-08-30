import pennylane as qml
from pennylane import numpy as np
import pennylane.math as math
from ionizer.ops import GPI


def circuit(phi):
    qml.RX(0.2, wires=0)
    qml.RY(1.1, wires=0)
    qml.RX(0.3, wires=0)
    GPI(phi, wires=0)
    return qml.expval(qml.PauliY(wires=0))

phi = np.random.rand()
dev = qml.device("default.qubit", wires=1)
qnode_GPI = qml.QNode(circuit, dev)

grad_GPI = qml.grad(qnode_GPI, argnum=0)(phi)
GPI_val_p = qnode_GPI(phi+np.pi/2)
GPI_val_m = qnode_GPI(phi+np.pi/2)

print(grad_GPI)
print(GPI_val_p)
print(grad_GPI/GPI_val_p)
print(GPI_val_m)
print(grad_GPI/GPI_val_m)