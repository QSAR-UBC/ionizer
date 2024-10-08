.. _basic_usage:

Basic usage
===========

The goal of the package is to provide a single, easy-to-use transform that will
both transpile and optimize a circuit expressed using arbitrary gates into the
native gate set of IonQ's trapped-ion devices, :class:`ionizer.ops.GPI`,
:class:`ionizer.ops.GPI2`, and :class:`ionizer.ops.MS`.

For more information about these gates, see the `PennyLane blog post
<https://pennylane.ai/blog/2023/06/the-ionizer-building-a-hardware-specific-transpiler-using-pennylane/>`_
and `IonQ documentation <https://ionq.com/docs/getting-started-with-native-gates>`_.

Circuits expressed using any regular PennyLane gates can be decorated with the
:func:`ionizer.transforms.ionize` transform, and executed as normal.

.. code::

    import pennylane as qml
    from ionizer.transforms import ionize

    dev = qml.device("default.qubit", wires=2)


    @qml.qnode(dev)
    @ionize
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RX(x, wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

.. code::

   >>> circuit(0.3)
   0.9553364891256048
   >>> qml.draw(circuit)(0.3)
   0: ──GPI2(0.00)─╭MS──GPI2(-1.57)─────────────────────────┤ ╭<Z@Z>
   1: ──GPI2(3.14)─╰MS──GPI2(1.57)───GPI(-1.42)──GPI2(1.57)─┤ ╰<Z@Z>

.. warning::

   Compiled circuits are not guaranteed to be optimal. However, they should be
   significantly smaller than what one would obtain by performing a naive 1-1
   mapping to native gates.


Transforms can also be applied to QNodes directly, even after they are
constructed:

.. code::

    @qml.qnode(dev)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RX(x, wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    ionized_qnode = ionize(circuit)

.. code::

    >>> ionized_qnode(0.3)
    0.9553364891256048

.. _basic_usage-equivalence_validation:

Equivalence validation
----------------------

While all implemented transforms should preserve the behaviour of circuits, they
nevertheless contain a mechanism for under-the-hood equivalence checking (up to
a global phase). If the ``verify_equivalence`` flag is ``True``, an error will
be raised if the transpiled circuit is not equivalent to the original. This flag
is ``False`` by default because equivalence is checked at the unitary matrix
level.

.. code::

    from functools import partial
    import pennylane as qml
    from ionizer.transforms import ionize

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    @partial(ionize, verify_equivalence=True)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RX(x, wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

For more details, see :func:`ionizer.utils.flag_non_equivalence`.

Ionizer and gradients
---------------------

.. warning::

   Full automatic differentiation is not currently supported by the Ionizer.

To execute circuits that evaluate gradients, is recommended to first construct
the required circuits with the desired parameters, then transpile those circuit
for execution. This can be done by applying the transform to one or more quantum
tapes directly.

For example, the following code transpiles both quantum circuits required to the
parameter shift gradient of the circuit defined at the top of this page.

.. code::

   from pennylane import numpy as np

   # Execute the circuit once to construct the tape
   x = np.array(0.5, requires_grad=True)
   circuit(x)

   # Compute tapes required for gradient, and processing function that
   # evaluates the gradients based on results
   gradient_tapes, gradient_fn = qml.gradients.param_shift(circuit.qtape)

   for tape in gradient_tapes:
       print(tape.draw(decimals=2))
       print()

   results = dev.execute(gradient_tapes)
   print(f"Gradient from original tape execution = {gradient_fn(results)}", end="\n\n")

   # Transpile each of the gradient tapes. The same processing function
   # can be applied to the results of the transpiled tapes.
   transpiled_gradient_tapes, _ = ionize(gradient_tapes)

   for tape in transpiled_gradient_tapes:
       print(tape.draw(decimals=2))
       print()

   transpiled_results = dev.execute(transpiled_gradient_tapes)
   print(f"Gradient from transpiled tape execution = {gradient_fn(transpiled_results)}")

The following output, showing both original and transpiled versions of the
gradient tape, is

.. code::

   0: ──H─╭●───────────┤ ╭<Z@Z>
   1: ────╰X──RX(2.07)─┤ ╰<Z@Z>

   0: ──H─╭●────────────┤ ╭<Z@Z>
   1: ────╰X──RX(-1.07)─┤ ╰<Z@Z>

   Gradient from original tape execution = -0.479425538604203

   0: ──GPI2(0.00)─╭MS──GPI2(-1.57)─────────────────────────┤ ╭<Z@Z>
   1: ──GPI2(3.14)─╰MS──GPI2(1.57)───GPI(-0.54)──GPI2(1.57)─┤ ╰<Z@Z>

   0: ──GPI2(0.00)─╭MS──GPI2(-1.57)─────────────────────────┤ ╭<Z@Z>
   1: ──GPI2(3.14)─╰MS──GPI2(1.57)───GPI(-2.11)──GPI2(1.57)─┤ ╰<Z@Z>

   Gradient from transpiled tape execution = -0.479425538604203
