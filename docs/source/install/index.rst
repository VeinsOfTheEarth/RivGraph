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
The easiest way to do this is by first downloading the `environment.yml <https://github.com/VeinsOfTheEarth/RivGraph/blob/master/environment.yml>`_ (go to link, click "Raw", then copy the contents into a text editor and save as 'environment.yml'), opening Terminal (Mac/Unix) or Anaconda Prompt (Windows) and typing:
::

   $ conda env create --file /path/to/environment.yml

.. tip::
   The default environment name is 'rivgraph' (per the `environment.yml` file), but you can change the environment file to name it anything you like.

If you would rather install *RivGraph* into a pre-existing environment "myenv", you can use the following commands:
::

   conda activate myenv
   conda install rivgraph -c jschwenk

.. warning::

 *RivGraph* dependencies may be pinned to specific versions of packages (e.g. geopandas 0.7) that may not mesh with your existing environment.
 Check the `environment file <https://github.com/VeinsOfTheEarth/RivGraph/blob/master/environment.yml>`_ for these cases.

Installation from source
------------------------

If you would prefer to install the *RivGraph* package from source, then follow these steps:

.. warning::

   *Rivgraph* uses many geospatial dependencies (e.g. GDAL) that can be
   difficult to install. Note that so long as the continuous integration
   workflows on GitHub are working (denoted by a green check next to the latest
   commit message on the source repository home page), the latest version of
   the source code is stable and installation from source will work if the
   dependencies are correctly installed.

1. Clone the repository
::

   $ git clone https://github.com/jonschwenk/RivGraph.git

2. Install dependencies; note these can be installed via conda from the
`environment.yml <https://github.com/VeinsOfTheEarth/RivGraph/blob/master/environment.yml>`_ file, however a list is also
provided below with links to the homepage for each dependency.

**RivGraph Dependencies:**
   - `python <https://www.python.org/>`_ (tested on v3.6)
   - `GDAL <https://gdal.org/>`_
   - `NumPy <https://numpy.org/>`_
   - `GeoPandas <https://geopandas.org/>`_ (v0.7.0)
   - `scikit-image <https://scikit-image.org/>`_
   - `OpenCV <https://github.com/skvark/opencv-python>`_
   - `NetworkX <https://networkx.org/>`_
   - `Matplotlib <https://matplotlib.org/>`_
   - `pyproj <https://pyproj4.github.io/pyproj/stable/>`_
   - `Shapely <https://shapely.readthedocs.io/en/latest/>`_
   - `Fiona <https://fiona.readthedocs.io/en/latest/>`_
   - `FastDTW <https://github.com/slaypni/fastdtw>`_

3. From the cloned folder, run the following in the command line:
::

   $ python setup.py install

to install the *RivGraph* package.

.. note::
   If you run into issues installing *RivGraph* at this stage, please check
   to see whether you've installed all of the required dependencies.

4. To test your installation, you need to install the `pytest <https://docs.pytest.org/en/stable/index.html>`_ package.
Then from the cloned folder you can run the unit tests with the following command:
::

   $ pytest
