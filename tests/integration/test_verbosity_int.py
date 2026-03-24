"""Tests associated with verbosity outputs."""
import sys
import io


def test_to_geovectors(test_net):
    """Test not being able to write geovectors."""
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # enable verbosity
    test_net.verbose = True
    test_net.init_logger()
    # try geovector
    test_net.to_geovectors(export='all')
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == '---------- New Run ----------\nLinks have not been computed and thus cannot be exported.\nNodes have not been computed and thus cannot be exported.\nMesh has not been computed and thus cannot be exported.\nCenterlines has not been computed and thus cannot be exported.\nSmoothed centerline has not been computed and thus cannot be exported.'


def test_to_geotiff(test_net):
    """Test invalid export value."""
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # enable verbosity
    test_net.verbose = True
    test_net.init_logger()
    # try geovector
    test_net.to_geotiff('invalid')
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == "---------- New Run ----------\nCannot write invalid. Choose from ['directions', 'distance', 'skeleton']."


def test_plot_failure(test_net):
    """Test network not computed output."""
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # enable verbosity
    test_net.verbose = True
    test_net.init_logger()
    # try plot
    test_net.plot()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == '---------- New Run ----------\nNo path is available to load the network.'


def test_save_networkcatch(test_net):
    """Test network not computed output."""
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # enable verbosity
    test_net.verbose = True
    test_net.init_logger()
    # try save
    test_net.save_network()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == '---------- New Run ----------\nNetwork has not been computed yet. Use the compute_network() method first.'

def test_compute_verbosity(test_net):
    """Test skeletonization / resolution of links and nodes."""
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # enable verbosity
    test_net.verbose = True
    test_net.init_logger()
    # run compute network
    test_net.compute_network()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == '---------- New Run ----------\nSkeletonizing mask...\ndone skeletonization.\nResolving links and nodes...\nlinks and nodes have been resolved.'


def test_distance_verbosity(test_net):
    """Test distance transform output."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # network part
    test_net.verbose = True
    test_net.init_logger()
    test_net.compute_distance_transform()
    # capture output
    sys.stdout = sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == '---------- New Run ----------\nComputing distance transform...\ndistance transform done.'


def test_plot_dirs(test_net):
    """Link direction warning."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # try plot
    test_net.verbose = True
    test_net.init_logger()
    test_net.plot('directions')
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == '---------- New Run ----------\nMust assign link directions before plotting link directions.'


def test_compute_link_verbosity(test_net):
    """Computing links/widths/lengths output."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # network part
    test_net.verbose = True
    test_net.init_logger()
    test_net.compute_link_width_and_length()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:5] == '-----'
    assert capturedOutput.getvalue()[-5:-1] == 'ted.'


def test_junction_verbosity(known_net):
    """Testing junction verbosity."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # network part
    known_net.verbose = True
    known_net.init_logger()
    known_net.compute_junction_angles()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:5] == '-----'


def test_skeletonize(test_river):
    """Test river skeletonization."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # do skeletonization
    test_river.verbose = True
    test_river.init_logger()
    test_river.skeletonize()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == '---------- New Run ----------\nSkeletonizing mask...\nskeletonization is done.'


def test_compute_centerline(test_river):
    """Test compute_centerline function."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # do skeletonization
    test_river.verbose = True
    test_river.init_logger()
    test_river.compute_centerline()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:-1] == '---------- New Run ----------\nComputing centerline...\ncenterline computation is done.'


def test_compute_mesh(test_river):
    """Test compute_mesh function."""
    # capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # do skeletonization
    test_river.verbose = True
    test_river.init_logger()
    test_river.compute_mesh()
    sys.stdout == sys.__stdout__
    # make assertion
    assert capturedOutput.getvalue()[:5] == '-----'
    assert capturedOutput.getvalue()[-5:-1] == 'one.'
