"""Tests associated with verbosity outputs."""
import pytest
import sys
import os
import io
from rivgraph.classes import delta
from rivgraph.classes import river


def test_to_geovectors(test_net):
    """Test not being able to write geovectors."""
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # enable verbosity
    test_net.verbose = True
    # try geovector
    test_net.to_geovectors(export='all')
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == 'Links have not been computed and thus cannot be exported.\nNodes have not been computed and thus cannot be exported.\nMesh has not been computed and thus cannot be exported.\nCenterlines has not been computed and thus cannot be exported.\nSmoothed centerline has not been computed and thus cannot be exported.'


def test_to_geotiff(test_net):
    """Test invalid export value."""
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # enable verbosity
    test_net.verbose = True
    # try geovector
    test_net.to_geotiff('invalid')
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == "Cannot write invalid. Choose from ['directions', 'distance', 'skeleton']."


def test_plot_failure(test_net):
    """Test network not computed output."""
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
    """Test network not computed output."""
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
    """Test skeletonization / resolution of links and nodes."""
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
    """Test distance transform output."""
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
    """Link direction warning."""
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
    """Computing links/widths/lengths output."""
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
    """Testing junction verbosity."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # network part
    test_net.verbose = True
    test_net.compute_junction_angles()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == 'Junction angles cannot be computed before link directions are set.'


def test_skeletonize(test_river):
    """Test river skeletonization."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # do skeletonization
    test_river.verbose = True
    test_river.skeletonize()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == 'Skeletonizing mask...done.'


def test_compute_centerline(test_river):
    """Test compute_centerline function."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # do skeletonization
    test_river.verbose = True
    test_river.compute_centerline()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == 'Computing centerline...done.'


def test_compute_mesh(test_river):
    """Test compute_mesh function."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # do skeletonization
    test_river.verbose = True
    test_river.compute_mesh()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == 'Resolving links and nodes...done.\nComputing distance transform...done.\nComputing link widths and lengths...done.\nGenerating mesh...done.'
