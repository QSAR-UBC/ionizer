The Ionizer
===========

The Ionizer is a set of tools for transpiling and optimizing PennyLane circuits
into native trapped-ion gates.

These documentation pages focus on usage of the transpilation tools. Technical
details and intuition about trapped ion gates can be found in this `PennyLane
blog post
<https://pennylane.ai/blog/2023/06/the-ionizer-building-a-hardware-specific-transpiler-using-pennylane/>`_
about the Ionizer, and IonQ's documentation page, `Getting started with Native
Gates <https://ionq.com/docs/getting-started-with-native-gates>`_.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   notes/installation
   notes/basic_usage

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   api/ops
   api/transforms
   api/decompositions
   api/identity_hunter
   api/utils

Contributing
------------

The Ionizer is available open source on `GitHub
<https://github.com/QSAR-UBC/ionizer>`_ under the MIT License.  Contributions
are welcome. Please open an issue if you are interested in contributing, or if
you encounter a bug.

Reference
---------

The Ionizer is developed and maintained by the `Quantum Software and Algorithms
Research Lab <https://glassnotes.github.io/qsar.html>` at UBC.

If you use the Ionizer as part of your project, we would appreciate if you cite
it using the BibTeX below.

.. code::

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
