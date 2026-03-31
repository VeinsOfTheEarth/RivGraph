.. _quickstart:

==========
Quickstart
==========

The fastest way to get oriented is:

1. install RivGraph using :doc:`../install/index`
2. open one of the canonical notebooks in ``examples/``
3. run it from the repository root so the example-data paths resolve cleanly

Canonical notebooks
-------------------

- ``examples/delta_example.ipynb``
- ``examples/braided_river_example.ipynb``
- ``examples/mouse_brain_example.ipynb``

Minimal usage patterns
----------------------

These snippets show the first few steps only. For full workflows, use the notebooks.

Delta
~~~~~

.. code-block:: python

   from rivgraph.classes import delta

   d = delta(
       name="Colville",
       path_to_mask="examples/data/Colville_Delta/Colville_mask.tif",
       results_folder="examples/data/Colville_Delta/Results",
       verbose=True,
   )
   d.skeletonize()
   d.compute_network()

River
~~~~~

.. code-block:: python

   from rivgraph.classes import river

   r = river(
       name="Brahma",
       path_to_mask="examples/data/Brahmaputra_Braided_River/Brahmaputra_mask.tif",
       results_folder="examples/data/Brahmaputra_Braided_River/Results",
       exit_sides="NS",
       verbose=True,
   )
   r.skeletonize()
   r.compute_network()

What happens next
-----------------

- Delta workflows then need shoreline and inlet-node vectors before pruning and direction setting.
- River workflows typically proceed to pruning, centerline and mesh generation, and direction setting.
- Geovector export defaults to GeoPackage.

Next places to go:

- :doc:`../examples/index`
- :doc:`../shoreline/index`
- :doc:`../maskmaking/index`
- :doc:`../apiref/index`
