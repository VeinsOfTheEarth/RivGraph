.. _install:

=========================
Installation Instructions
=========================

.. note::
   *RivGraph* requires the installation of common geospatial Python packages such as `GDAL <https://gdal.org/>`_.
   These packages can be difficult to install properly and often create dependency errors.
   Because of this, we recommend using `Anaconda <https://www.anaconda.com/products/individual>`_ to create a virtual environment for *RivGraph*, and to manage the installation of Python libraries as it will handle package versions and dependencies for you.

Installation via *conda*
--------------------------

The latest 'stable' version of *RivGraph* can be installed via `conda`.
We recommend installing *RivGraph* into a fresh conda environment to minimize the risk of dependency clashes.
The easiest way to do this is by first downloading the `environment.yml <https://github.com/jonschwenk/RivGraph/blob/master/environment.yml>`_, (right-click, "Save As...") opening Terminal (Mac/Unix) or Anaconda Prompt (Windows) and typing:
::

   $ conda env create --file /path/to/environment.yml

.. note::
   The default environment name is 'rivgraph' (per the `environment.yml` file), but you can change the environment file to name it anything you like.

If you would rather install *RivGraph* into a pre-existing environment "myenv", you can use the following commands:
::

   conda activate myenv
   conda install rivgraph -c jschwenk

.. note::

 *RivGraph* dependencies may be pinned to specific versions of packages (e.g. geopandas 0.7) that may not mesh with your existing environment.

Installation from source
------------------------

If you would prefer to install the *RivGraph* package from source, then follow these steps:

1. Clone the repository
::

   $ git clone https://github.com/jonschwenk/RivGraph.git

2. From the cloned folder, run the following in the command line:
::

   $ python setup.py install

to install the RivGraph package.

.. note::
  If you encounter an error during step 2, check that you've installed the required dependencies.
  A list of the *RivGraph* dependencies can be found in the `environment.yml <https://github.com/jonschwenk/RivGraph/blob/master/environment.yml>`_ file.

3. To test your installation, from the cloned folder you can run the unit tests with the following command:
::

   $ pytest
