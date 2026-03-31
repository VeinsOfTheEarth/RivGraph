[![DOI](https://joss.theoj.org/papers/10.21105/joss.02952/status.svg)](https://doi.org/10.21105/joss.02952)

[![RivGraph logo](docs/logos/rg_logo_full.png)](https://VeinsOfTheEarth.github.io/RivGraph/ "Go to documentation.")

# RivGraph

[![tests](https://github.com/VeinsOfTheEarth/RivGraph/actions/workflows/tests.yml/badge.svg?branch=version_1_dev)](https://github.com/VeinsOfTheEarth/RivGraph/actions/workflows/tests.yml)
[![docs](https://github.com/VeinsOfTheEarth/RivGraph/actions/workflows/docs.yml/badge.svg?branch=version_1_dev)](https://github.com/VeinsOfTheEarth/RivGraph/actions/workflows/docs.yml)

RivGraph is a Python package for converting a binary mask of a channel network into a directed, weighted graph of connected links and nodes. It is designed for river and delta channel networks derived from remote sensing imagery and geospatial masks.

![Core functionality of RivGraph](examples/images/rivgraph_overview_white.PNG)

## Core capabilities

- morphologic metrics such as widths, lengths, and branching characteristics
- graph-based and algebraic representations of channel networks
- topologic and dynamic metrics such as alternative paths, flux sharing, and entropy-based measures
- tools for cleaning and preparing binary channel masks
- island detection, metrics, and filtering
- mesh generation for along-river analysis
- geospatial export of links, nodes, rasters, and derived products

RivGraph preserves georeferencing information throughout the workflow. If you start with a georeferenced mask, exported rasters and vectors remain aligned with the source data and can be used directly in GIS workflows.

The flow-directionality logic and validation are described in our [ESurf Dynamics paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html). General package usage is described in our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.02952).

## Canonical examples

The main runnable notebooks live in `examples/`:

- `examples/delta_example.ipynb`
- `examples/braided_river_example.ipynb`
- `examples/mouse_brain_example.ipynb`

## Installation

RivGraph v1 targets Python 3.12. For a source install, create the conda environment and then install RivGraph in editable mode:

```bash
conda env create -f environment.yml
conda activate rivgraph
pip install -e . --no-deps
```

Using `--no-deps` is intentional here: the geospatial stack is managed by the conda environment file, which avoids pip trying to re-resolve compiled dependencies that are already pinned in conda.

To verify the install, run:

```bash
python -m pytest -ra
```

## Building the docs

Install the documentation extras into the same activated environment:

```bash
pip install ".[docs]"
```

Then build the HTML docs:

```bash
make -C docs html
```

On Windows, you can instead run `make.bat html` from the `docs/` directory.

The built site will be written to `docs/build/html`.

## Getting started

Start with the [documentation](https://VeinsOfTheEarth.github.io/RivGraph/) and the notebooks in `examples/`.

RivGraph contains two primary classes, `delta` and `river`, that organize the main processing workflows. The notebooks show the end-to-end usage patterns, while the API docs are generated from the source docstrings.

## Contributing

Bug reports, documentation improvements, and pull requests are welcome. The easiest way to start is to open an issue in the [tracker](https://github.com/VeinsOfTheEarth/RivGraph/issues).

## Citing RivGraph

If you use RivGraph in your research, please consider citing it.

If you use RivGraph's flow-directionality algorithms, please cite our [ESurf Dynamics paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html). If you publish work that uses RivGraph more generally, please also cite our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.02952).

## License

RivGraph is distributed under the BSD 3-clause license. A copy is provided in [LICENSE.txt](LICENSE.txt).
