import pytest
import sys, os
import io
import numpy as np

sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph.classes import delta

def test_plot_failure(test_net):
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # enable verbosity
    test_net.verbose = True
    # try plot
    test_net.plot()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == 'Network has not been computed yet; cannot plot.'

def test_save_networkcatch(test_net):
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # enable verbosity
    test_net.verbose = True
    # try save
    test_net.save_network()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == 'Network has not been computed yet. Use the compute_network() method first.'

def test_compute_verbosity(test_net):
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # enable verbosity
    test_net.verbose = True
    # run compute network
    test_net.compute_network()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[0:26] == 'Skeletonizing mask...done.'
    assert capturedOutput.getvalue()[27:-1] == 'Resolving links and nodes...done.'

def test_distance_verbosity(test_net):
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # network part
    test_net.verbose = True
    test_net.compute_distance_transform()
    # capture output
    sys.stdout = sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == 'Computing distance transform...done.'

def test_plot_dirs(test_net):
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # try plot
    test_net.verbose = True
    test_net.plot('directions')
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == 'Must assign link directions before plotting link directions.'

def test_compute_link_verbosity(test_net):
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # network part
    test_net.verbose = True
    test_net.compute_link_width_and_length()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == 'Computing link widths and lengths...done.'

def test_junction_vebosity(test_net):
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # network part
    test_net.verbose = True
    test_net.compute_junction_angles()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == 'Junction angles cannot be computed before link directions are set.'
