"""Tests for `rivgraph.river.river_utils` functions."""
import pytest
import sys
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import shapely

from inspect import getsourcefile
basepath = os.path.dirname(os.path.dirname(os.path.abspath(getsourcefile(lambda:0))))
sys.path.insert(0, basepath)
from rivgraph.rivers import river_utils as ru
from rivgraph.rivers import centerline_utils as cu
from rivgraph.classes import centerline


class TestFindInletsOutlets:
    """Testing the find_inlet_outlet_nodes() function."""

    def test_find_inletoutlet_ne(self, known_river):
        """Test using north and east exit sides."""
        nodes = known_river.nodes
        links = known_river.links
        known_river.skeletonize()
        Iskel = known_river.Iskel
        exit_sides = ['n', 'e']
        nodes = ru.find_inlet_outlet_nodes(links, nodes, exit_sides, Iskel)
        # assert that correct nodes were assigned as inlets/outlets
        assert nodes['inlets'] == [247]
        assert nodes['outlets'] == [872]

    def test_find_inletoutlet_ns(self, known_river):
        """Test using north and south exit sides."""
        nodes = known_river.nodes
        links = known_river.links
        known_river.skeletonize()
        Iskel = known_river.Iskel
        exit_sides = ['n', 's']
        nodes = ru.find_inlet_outlet_nodes(links, nodes, exit_sides, Iskel)
        # assert that correct nodes were assigned as inlets/outlets
        assert nodes['inlets'] == [247]
        assert nodes['outlets'] == [1919]

    def test_find_inletoutlet_nw(self, known_river):
        """Test using north and west exit sides."""
        nodes = known_river.nodes
        links = known_river.links
        known_river.skeletonize()
        Iskel = known_river.Iskel
        exit_sides = ['n', 'w']
        nodes = ru.find_inlet_outlet_nodes(links, nodes, exit_sides, Iskel)
        # assert that correct nodes were assigned as inlets/outlets
        assert nodes['inlets'] == [247]
        assert nodes['outlets'] == [150]

    def test_find_inletoutlet_es(self, known_river):
        """Test using east and south exit sides."""
        nodes = known_river.nodes
        links = known_river.links
        known_river.skeletonize()
        Iskel = known_river.Iskel
        exit_sides = ['e', 's']
        nodes = ru.find_inlet_outlet_nodes(links, nodes, exit_sides, Iskel)
        # assert that correct nodes were assigned as inlets/outlets
        assert nodes['inlets'] == [872]
        assert nodes['outlets'] == [1919]

    def test_find_inletoutlet_ew(self, known_river):
        """Test using east and west exit sides."""
        nodes = known_river.nodes
        links = known_river.links
        known_river.skeletonize()
        Iskel = known_river.Iskel
        exit_sides = ['e', 'w']
        nodes = ru.find_inlet_outlet_nodes(links, nodes, exit_sides, Iskel)
        # assert that correct nodes were assigned as inlets/outlets
        assert nodes['inlets'] == [872]
        assert nodes['outlets'] == [150]

    def test_find_inletoutlet_sw(self, known_river):
        """Test using south and west exit sides."""
        nodes = known_river.nodes
        links = known_river.links
        known_river.skeletonize()
        Iskel = known_river.Iskel
        exit_sides = ['s', 'w']
        nodes = ru.find_inlet_outlet_nodes(links, nodes, exit_sides, Iskel)
        # assert that correct nodes were assigned as inlets/outlets
        assert nodes['inlets'] == [1919]
        assert nodes['outlets'] == [150]


def test_offset_linestring():
    """
    Test MultiLineString catch.

    Example of this condition
    taken from https://github.com/Toblerity/Shapely/issues/656
    """
    line = shapely.wkt.loads("LINESTRING (0.615 -0.7176374373233079, 0.44875 -0.6083988835322591, 0.30875 -0.491886287971077, 0.19625 -0.3892309735881384, 0.11 -0.276535372660264, 0.04875 -0.1824690907834349, 0.0125 -0.04856212366768966, 0 0)")
    offset = 0.91
    offset_line = line.parallel_offset(offset, side='right')
    OLS = cu.offset_linestring(line, offset, 'right')

    # make assertions
    assert type(offset_line) == shapely.geometry.multilinestring.MultiLineString
    assert type(OLS) == shapely.geometry.linestring.LineString


def test_curvars():
    """Test unwrap==False."""
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ys = [1, 4, 9, 16, 25, 16, 9, 4, 1, 0]
    C, Areturn, s = cu.curvars(xs, ys, unwrap=False)

    # make assertions
    assert np.shape(C) == (9,)
    assert np.shape(Areturn) == (9,)
    assert np.shape(s) == (9,)
    assert C[0] == pytest.approx(-0.03131767154315412)
    assert Areturn[4] == pytest.approx(-1.460139105621001)
    assert s[7] == pytest.approx(48.775500247528115)


def test_inflection_pts():
    """Test inflection_points()."""
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ys = [1, 4, 9, 16, 25, 16, 9, 4, 1, 0]
    C, _, _ = cu.curvars(xs, ys, unwrap=False)
    # run function
    infs = cu.inflection_points(C)

    # make assertion
    assert np.all(infs == np.array([3, 5]))


def test_eBI_avg():
    """Test compute_eBI() with method='avg'."""
    path_meshlines = os.path.join(basepath, os.path.normpath('tests/data/Brahma/Brahmclip_meshlines.json'))
    path_links = os.path.join(basepath, os.path.normpath('tests/data/Brahma/Brahmclip_links.json'))
    eBI, BI = ru.compute_eBI(path_meshlines, path_links, method='avg')

    # make assertions
    assert np.shape(eBI) == (169,)
    assert np.shape(BI) == (169,)
    assert eBI[5] == pytest.approx(2.4833617464718523)
    assert eBI[25] == pytest.approx(5.062946753237394)
    assert eBI[100] == pytest.approx(6.389362214783173)
    assert eBI[120] == pytest.approx(8.543247059514744)
    assert eBI[140] == pytest.approx(4.683156935890404)
    assert np.all(BI == np.array([ 0,  2,  4,  4,  3,  3,  5,  5,  4,  6,  4,  3,  4,  3,  3,  3,  5,
        4,  6,  7,  7,  7,  7,  7,  7,  6,  7,  5,  7,  7,  8,  7,  9,  7,
        7,  8,  7, 11, 10, 10, 11,  8,  8,  7, 10, 11, 12, 10, 10,  8,  8,
        9, 11, 13,  8,  8,  8, 11, 13,  9,  9,  7,  6,  6,  6, 10,  9, 10,
        9,  8,  8,  9,  5,  5,  5,  3,  2,  3,  3,  5,  6,  5,  6,  8,  8,
        5,  4,  4,  3,  2,  3,  3,  3,  3,  5,  6,  7,  4,  4,  7,  9,  8,
        7,  6,  8,  6,  7,  7,  8, 10,  8,  9,  9,  7,  8,  6, 10,  9,  6,
       11,  9, 14, 12, 12, 13, 12, 10, 10, 10,  8,  7,  9,  6,  4,  4,  6,
        6,  4,  3,  5,  6,  8,  7, 10,  9,  8,  8,  7,  7,  7,  5,  5,  5,
        4,  3,  3,  3,  4,  6,  4,  5,  6,  4,  5,  5,  2,  2,  2,  2]))


class TestCenterline:
    """Testing the river_utils.centerline() class."""

    def test_init(self):
        """Init the class."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        CL = centerline(x, y)
        # make assertions
        assert np.all(CL.xo == x)
        assert np.all(CL.yo == y)

    def test_init_const_attr(self):
        """Init with constant attribute."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        attribs = {}
        attribs['width'] = 10.
        CL = centerline(x, y, attribs)
        # make assertions
        assert np.all(CL.xo == x)
        assert np.all(CL.yo == y)
        assert CL.width == 10.0

    def test_init_attr(self):
        """Init with attribute list."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        attribs = {}
        attribs['width'] = np.ones((11,))*10
        CL = centerline(x, y, attribs)
        # make assertions
        assert np.all(CL.xo == x)
        assert np.all(CL.yo == y)
        assert np.all(CL.width == np.ones((11,))*10)

    def test_get_xy(self):
        """Test get xy function."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        CL = centerline(x, y)
        x, y, vers = CL._centerline__get_x_and_y()
        # make assertions
        assert np.all(x == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert np.all(y == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1])
        assert vers == 'original'

    def test_get_xy_rs(self):
        """Test get xy function with rs."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        xrs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        yrs = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11,  11]
        attribs = {}
        attribs['xrs'] = xrs
        attribs['yrs'] = yrs
        CL = centerline(x, y, attribs=attribs)
        x, y, vers = CL._centerline__get_x_and_y()
        # make assertions
        assert np.all(x == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        assert np.all(y == [11, 11, 11, 11, 11, 11, 11, 11, 11, 11,  11])
        assert vers == 'resampled'

    def test_get_xy_sm(self):
        """Test get xy function with xs."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        xs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        ys = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11,  11]
        attribs = {}
        attribs['xs'] = xs
        attribs['ys'] = ys
        CL = centerline(x, y, attribs=attribs)
        x, y, vers = CL._centerline__get_x_and_y()
        # make assertions
        assert np.all(x == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        assert np.all(y == [11, 11, 11, 11, 11, 11, 11, 11, 11, 11,  11])
        assert vers == 'smooth'

    def test_s(self):
        """Test the s() function."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        CL = centerline(x, y)
        sss = CL.s()
        # make assertions
        assert np.all(sss == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_ds(self):
        """Test the ds() function."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        CL = centerline(x, y)
        dss = CL.ds()
        # make assertions
        assert np.all(dss == [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    def test_C(self):
        """Test the C() function."""
        x = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        CL = centerline(x, y)
        Cs = CL.C()
        # make assertions
        assert pytest.approx(Cs == np.array([-0., 0., 0., 0., 0.,
                                             1.57079633, 1.57079633,
                                             0., 0., 0., 0.]))


def test_sine_curvature():
    """Use a sine wave to compute curvature."""
    xs = np.linspace(0, 100, 101)
    ys = np.sin(xs) + 5
    C, Areturn, sdist = cu.curvars(xs, ys)
    # make some simple assertions about shape of outputs
    assert C.shape == (100,)
    assert Areturn.shape == (100,)
    assert sdist.shape == (100,)
    # now define this as a centerline
    CL = centerline(xs, ys)
    # smooth the centerline
    CL.window_cl = 10
    CL.smooth(n=2)
    # make some assertions about the smoothing
    assert CL.xs.shape == (101,)
    assert CL.ys.shape == (101,)
    assert np.sum(CL.xs != xs) > 0
    assert np.sum(CL.ys != ys) > 0
    # resample the centerline to 50 points
    CL.resample(50)
    # assert resampled dimensions are as expected
    assert CL.xrs.shape == (50,)
    assert CL.yrs.shape == (50,)


def test_smooth_nowindow():
    """smooth() without providing a window."""
    xs = np.linspace(0, 100, 101)
    ys = np.sin(xs) + 5
    CL = centerline(xs, ys)
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # smooth the centerline
    CL.smooth()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == 'Must provide a smoothing window.'


def test_sine_csmooth():
    """Use a sine wave to compute curvature."""
    xs = np.linspace(0, 100, 101)
    ys = np.sin(xs) + 5
    CL = centerline(xs, ys)
    CL.window_C = 11
    # smooth the centerline
    Cs = CL.Csmooth()
    # check that the smoothed shape is what we expect
    assert Cs.shape == (101,)


def test_csmooth_nowindow():
    """csmooth() without providing a window."""
    xs = np.linspace(0, 100, 101)
    ys = np.sin(xs) + 5
    CL = centerline(xs, ys)
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # smooth the centerline
    Cs = CL.Csmooth()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == 'Must provide a smoothing window.'


def test_sine_plot():
    """Use sine wave to test plotting of CenterLine."""
    xs = np.linspace(0, 100, 101)
    ys = np.sin(xs) + 5
    CL = centerline(xs, ys)
    CL.plot()
    plt.savefig(os.path.join(basepath, os.path.normpath('tests/results/synthetic_cycles/sinewave.png')))
    plt.close()
    # assert file exists now
    assert os.path.isfile(os.path.join(basepath, os.path.normpath('tests/results/synthetic_cycles/sinewave.png'))) == True


def test_long_sine():
    """Stretched sine wave."""
    xs = np.linspace(0, 10000, 10001)
    ys = 20*(np.sin(xs/1000) + 50)
    idx, smoothline = cu.inflection_pts_oversmooth(xs, ys, 40)
    # make assertions
    assert idx[-1] == 10000
    assert np.shape(smoothline) == (2, 10001)


def test_long_sine_CLinfs():
    """Long sine wave into CL and using infs function."""
    xs = np.linspace(0, 10000, 10001)
    ys = 20*(np.sin(xs/1000) + 50)
    CL = centerline(xs, ys)
    CL.infs(40)
    # make assertion
    assert CL.infs_os[-1] == 10000


def test_plot_withattrs():
    """Testing CenterLine plotting with various attributes."""
    xs = np.linspace(0, 100, 101)
    ys = np.sin(xs) + 5
    CL = centerline(xs, ys)
    CL.infs_os = [0]
    CL.ints_all = [1]
    CL.ints = [2]
    CL.plot()
    plt.savefig(os.path.join(basepath, os.path.normpath('tests/results/synthetic_cycles/sinewaveattrs.png')))
    plt.close()
    # assert file exists now
    assert os.path.isfile(os.path.join(basepath, os.path.normpath('tests/results/synthetic_cycles/sinewaveattrs.png'))) == True


def test_zs_noinflection():
    """zs_plot without inflection points."""
    xs = np.linspace(0, 100, 101)
    ys = np.sin(xs) + 5
    CL = centerline(xs, ys)
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # call plot
    CL.zs_plot()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == 'Must compute inflection points first.'


def test_zs_noints():
    """zs_plot without intersection points."""
    xs = np.linspace(0, 100, 101)
    ys = np.sin(xs) + 5
    CL = centerline(xs, ys)
    CL.infs_os = [1]
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # call plot
    CL.zs_plot()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == 'Must compute intersections first.'


def test_zs_nomigrates():
    """zs_plot without migration rates."""
    xs = np.linspace(0, 100, 101)
    ys = np.sin(xs) + 5
    CL = centerline(xs, ys)
    CL.infs_os = [1]
    CL.ints = [2]
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # call plot
    CL.zs_plot()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == 'Must compute migration rates first.'


# Delete data created by tests in this file ...

def test_delete_files():
    """Delete created files at the end."""
    for i in os.listdir(os.path.join(basepath, os.path.normpath('tests/results/synthetic_cycles/'))):
        os.remove(os.path.join(basepath, os.path.normpath('tests/results/synthetic_cycles/'+i)))
    # check directory is empty
    assert os.listdir(os.path.join(basepath, os.path.normpath('tests/results/synthetic_cycles/'))) == []
