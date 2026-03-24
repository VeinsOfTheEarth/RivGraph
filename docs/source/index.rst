Welcome to RivGraph's documentation!
====================================

RivGraph provides two primary classes for working with binary channel masks: ``delta`` and ``river``.
These classes expose the main workflows for extracting channel networks, computing metrics, assigning
flow directions, and exporting georeferenced outputs.

If you are new to RivGraph, start with :doc:`installation instructions <install/index>`, then move to the
canonical notebooks in ``examples/``. Delta workflows additionally require shoreline and inlet definitions,
while river workflows require image exit sides.

This site is organized as lightweight documentation around the source code and example notebooks:

- notebooks in ``examples/`` are the canonical runnable examples
- narrative pages explain inputs, assumptions, and common workflow steps
- API pages are generated from in-code docstrings

If you cannot find what you need here, please open a request in the `issue tracker <https://github.com/VeinsOfTheEarth/RivGraph/issues>`_.

Documentation
-------------

.. toctree::
   :maxdepth: 1

   quickstart/index
   background/index
   install/index
   examples/index
   maskmaking/index
   shoreline/index
   linksnodes/index
   issues/index
   featuredevelopment/index
   contributing/index
   gallery/index
   apiref/index
