import pytest
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph.classes import delta

# pytest fixture functionality to create rivgraph.delta classes that can be
# used by other test functions but save us from re-defining the example class
# for each test fct. https://docs.pytest.org/en/latest/fixture.html#fixtures

@pytest.fixture(scope="module")
def test_net():
    return delta('colville','tests/data/Colville/Colville_islands_filled.tif','tests/results')

@pytest.fixture(scope="module")
def known_net():
    known_net = delta('known','tests/data/Colville/Colville_islands_filled.tif','tests/results')
    known_net.load_network(path='tests/data/Colville/Colville_network.pkl')
    return known_net

@pytest.fixture(scope="module")
def synthetic_cycles():
    # create synthetic binary skeleton
    synthetic = np.zeros((10,10))
    synthetic[0,7]=1; synthetic[1,6]=1; synthetic[2,5]=1; synthetic[2,2]=1
    synthetic[2,3]=1; synthetic[3,1]=1; synthetic[3,4]=1; synthetic[4,2]=1
    synthetic[4,3]=1; synthetic[4,5]=1; synthetic[5,5]=1
    synthetic[6,5]=1; synthetic[7,4]=1; synthetic[8,4]=1; synthetic[9,4]=1

    # visualize synthetic case as a png to look at and a tif to use
    plt.imshow(synthetic)
    plt.savefig('tests/data/SyntheticCycle/skeleton.png')
    plt.close()
    skimage.io.imsave('tests/data/SyntheticCycle/skeleton.tif',synthetic)

    # create and return rivgraph.delta object
    return delta('synthetic_cycles','tests/data/SyntheticCycle/skeleton.tif','tests/results')
