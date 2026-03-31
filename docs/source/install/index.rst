.. _install:

============
Installation
============

RivGraph can be installed either as a ready-to-use package from conda-forge or from source for development.
For source work, ``environment.yml`` is the canonical environment.

Install from conda-forge
------------------------

If you want to use RivGraph without modifying the code, install the conda-forge package into a fresh environment:

::

   conda create -n rivgraph_env rivgraph -c conda-forge
   conda activate rivgraph_env

Install from source
-------------------

For development, testing, or documentation work, start from the repository root and create the source environment:

::

   conda env create -f environment.yml
   conda activate rivgraph

Then install RivGraph in editable mode:

::

   pip install -e . --no-deps

Using ``--no-deps`` is intentional here so that pip does not try to re-resolve compiled geospatial dependencies that are already managed by conda.

Verify a source install
-----------------------

From the repository root, run the test suite:

::

   pytest -ra

If you want a faster smoke test focused on geospatial IO and regression coverage, run:

::

   pytest -ra tests/test_geospatial_roundtrip.py tests/regression

Build the documentation locally
-------------------------------

After activating the source environment, install the docs extras:

::

   pip install -e ".[docs]"

Then build the HTML docs from the repository root with a cross-platform command:

::

   python -m sphinx -b html docs/source docs/build/html

Open ``docs/build/html/index.html`` in your browser to review the rendered site.

Convenience commands
~~~~~~~~~~~~~~~~~~~~

You can also build from the ``docs`` directory:

**macOS / Linux**
::

   cd docs
   make html

**Windows**
::

   cd docs
   make.bat html

Use ``linkcheck`` separately when you want to validate external links:

::

   python -m sphinx -b linkcheck docs/source docs/build/linkcheck
