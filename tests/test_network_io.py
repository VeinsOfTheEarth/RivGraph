import pytest
import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph.classes import delta

def test_save(known_net):
    # test saving functionality
    known_net.save_network()
    # assert that the file now exists
    assert os.path.isfile('tests/results/known/known_network.pkl') == True

def test_load(known_net):
    # test loading functionality
    known_net.load_network()
    # assert that path used is correct
    assert known_net.paths['network_pickle'] == 'tests/results/known/known_network.pkl'

def test_outvec_json(known_net):
    # test default functionality should write network to json
    known_net.to_geovectors()
    # check that files exist
    assert os.path.isfile(known_net.paths['links']) == True
    assert os.path.isfile(known_net.paths['nodes']) == True

def test_outvec_shp(known_net):
    # test default functionality should write network to shp
    known_net.to_geovectors(export='network',ftype='shp')
    # check that files exist
    assert os.path.isfile(known_net.paths['links']) == True
    assert os.path.isfile(known_net.paths['nodes']) == True

def test_to_geotiff(known_net):
    # have to re-create skeleton
    known_net.skeletonize()
    # have to generate distances
    known_net.compute_distance_transform()
    # test writing of geotiff to disk
    known_net.to_geotiff('directions')
    known_net.to_geotiff('distance')
    known_net.to_geotiff('skeleton')
    # check that expected files exist
    assert os.path.isfile(known_net.paths['linkdirs']) == True
    assert os.path.isfile(known_net.paths['Idist']) == True
    assert os.path.isfile(known_net.paths['Iskel']) == True

def test_plotnetwork(known_net):
    # make plots with various kwargs specified
    # default
    f1 = known_net.plot()
    plt.savefig('tests/results/known/testall.png')
    plt.close()
    # network
    f2 = known_net.plot('network')
    plt.savefig('tests/results/known/testnetwork.png')
    plt.close()
    # directions
    f3 = known_net.plot('directions')
    plt.savefig('tests/results/known/testdirections.png')
    plt.close()
    # assert that figures were made
    assert os.path.isfile('tests/results/known/testall.png') == True
    assert os.path.isfile('tests/results/known/testnetwork.png') == True
    assert os.path.isfile('tests/results/known/testdirections.png') == True

def test_delete_files():
    # delete created files at the end
    for i in os.listdir('tests/results/known/'):
        os.remove('tests/results/known/'+i)
    # check directory is empty
    assert os.listdir('tests/results/known/') == []
