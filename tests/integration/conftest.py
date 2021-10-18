"""pytest fixture to initialize some rivgraph classes for the tests."""
import pytest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

# from inspect import getsourcefile
# basepath = os.path.dirname(
#     os.path.dirname(
#         os.path.abspath(getsourcefile(lambda: 0))))
# sys.path.insert(0, basepath)

# print(basepath)
# basepath = os.path.join(os.path.sep, *(basepath.split(os.path.sep)[:-1]))
# print(basepath)
# basepath = r'C:\Users\Jon\Documents\GitHub\RivGraph'
# print(os.path.realpath(os.path.dirname(__file__)+".."))
# sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
# sys.path.insert(0, os.path.join(basepath))

from rivgraph.classes import delta
from rivgraph.classes import river

# pytest fixture functionality to create rivgraph.delta classes that can be
# used by other test functions but save us from re-defining the example class
# for each test fct. https://docs.pytest.org/en/latest/fixture.html#fixtures


@pytest.fixture(scope="module")
def test_net(tmp_path_factory):
    """Define the test network."""
    np.random.seed(1)
    return delta(
        "colville",
        os.path.normpath(
            "tests/integration/data/Colville/Colville_islands_filled.tif"),
        os.path.join(
            tmp_path_factory.mktemp('colville')))


@pytest.fixture(scope="module")
def known_net(tmp_path_factory):
    """Define the known network to test against."""
    np.random.seed(1)
    known_net = delta(
        'known',
        os.path.normpath(
            'tests/integration/data/Colville/Colville_islands_filled.tif'),
        os.path.join(tmp_path_factory.mktemp('known')))
    known_net.load_network(
        path=os.path.normpath(
            'tests/integration/data/Colville/Colville_network.pkl'))
    return known_net


@pytest.fixture(scope="module")
def test_river(tmp_path_factory):
    """Define the test river network."""
    np.random.seed(1)
    return river(
        'Brahmclip',
        os.path.normpath('tests/integration/data/Brahma/brahma_mask_clip.tif'),
        os.path.join(tmp_path_factory.mktemp('brahma')),
        exit_sides='ns')


@pytest.fixture(scope="module")
def known_river(tmp_path_factory):
    """Define the known river network."""
    np.random.seed(1)
    known_river = river(
        'Brahmclip',
        os.path.normpath(
            'tests/integration/data/Brahma/brahma_mask_clip.tif'),
        os.path.join(tmp_path_factory.mktemp('brahma')), exit_sides='ns')
    known_river.load_network(
        path=os.path.normpath(
            'tests/integration/data/Brahma/Brahmclip_network.pkl'))
    return known_river


@pytest.fixture(scope="module")
def synthetic_cycles():
    """Creation of synthetic skeleton."""
    np.random.seed(1)
    # create synthetic binary skeleton
    synthetic = np.zeros((15, 10))
    synthetic[0, 7] = 1
    synthetic[1, 6] = 1
    synthetic[2, 5] = 1
    synthetic[2, 3] = 1
    synthetic[3, 3:6] = 1
    synthetic[4, 2:4] = 1
    synthetic[4, 5:7] = 1
    synthetic[5, 1:3] = 1
    synthetic[5, 6:8] = 1
    synthetic[6, 1:3] = 1
    synthetic[6, 6:8] = 1
    synthetic[7, 2:4] = 1
    synthetic[7, 5:7] = 1
    synthetic[8, 3:6] = 1
    synthetic[9:, 4] = 1

    # visualize synthetic case as a png to look at and a tif to use
    # plt.imshow(synthetic)
    # plt.savefig(os.path.join(basepath, 'tests/data/SyntheticCycle/skeleton.png'))
    # plt.close()
    # skimage.io.imsave(os.path.join(basepath, 'tests/data/SyntheticCycle/skeleton.tif'), synthetic)

    # create and return rivgraph.delta object
    return delta(
        'synthetic_cycles',
        os.path.normpath('tests/integration/data/SyntheticCycle/skeleton.tif'),
        os.path.normpath('tests/integration/results/synthetic_cycles/'))
