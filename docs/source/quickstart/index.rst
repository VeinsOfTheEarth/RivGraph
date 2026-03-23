.. _quickstart:

==========
Quickstart
==========

1. Install RivGraph using the steps in :doc:`../install/index`.
2. Open one of the canonical example notebooks in ``examples/``:

   - ``examples/delta_example.ipynb``
   - ``examples/braided_river_example.ipynb``
   - ``examples/mouse_brain.ipynb``

3. Run the notebook from the repository root so that the relative example data paths resolve cleanly.

Minimal usage patterns
----------------------

Delta
~~~~~

::

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

::

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

For a full workflow, use the notebooks rather than this minimal snippet. They show the required pruning, direction-setting, plotting, and export steps.
