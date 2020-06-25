"""Tests for `rivgraph.river.river_utils` functions."""
import pytest
import sys
import os
import numpy as np
import shapely
from shapely.geometry import MultiLineString
from shapely.geometry import LineString
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph.rivers import river_utils as ru


def test_offset_linestring():
    """
    Test MultiLineString catch.

    Example of this condition
    taken from https://github.com/Toblerity/Shapely/issues/656
    """
    line = shapely.wkt.loads("LINESTRING (0.615 -0.7176374373233079, 0.44875 -0.6083988835322591, 0.30875 -0.491886287971077, 0.19625 -0.3892309735881384, 0.11 -0.276535372660264, 0.04875 -0.1824690907834349, 0.0125 -0.04856212366768966, 0 0)")
    offset = 0.91
    offset_line = line.parallel_offset(offset, side='right')
    OLS = ru.offset_linestring(line, offset, 'right')

    # make assertions
    assert type(offset_line) == shapely.geometry.multilinestring.MultiLineString
    assert type(OLS) == shapely.geometry.linestring.LineString


def test_curvars():
    """Test unwrap==False."""
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ys = [1, 4, 9, 16, 25, 16, 9, 4, 1, 0]
    C, Areturn, s = ru.curvars(xs, ys, unwrap=False)

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
    C, _, _ = ru.curvars(xs, ys, unwrap=False)
    # run function
    infs = ru.inflection_points(C)

    # make assertion
    assert np.all(infs == np.array([3, 5]))


def test_eBI_avg():
    """Test compute_eBI() with method='avg'."""
    path_meshlines = 'tests/data/Brahma/Brahmclip_meshlines.json'
    path_links = 'tests/data/Brahma/Brahmclip_links.json'
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
        CL = ru.centerline(x, y)
        # make assertions
        assert np.all(CL.xo == x)
        assert np.all(CL.yo == y)

    def test_init_const_attr(self):
        """Init with constant attribute."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        attribs = {}
        attribs['width'] = 10.
        CL = ru.centerline(x, y, attribs)
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
        CL = ru.centerline(x, y, attribs)
        # make assertions
        assert np.all(CL.xo == x)
        assert np.all(CL.yo == y)
        assert np.all(CL.width == np.ones((11,))*10)

    def test_get_xy(self):
        """Test get xy function."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        CL = ru.centerline(x, y)
        x, y, vers = CL._centerline__get_x_and_y()
        # make assertions
        assert np.all(x == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert np.all(y == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1])
        assert vers == 'original'

    def test_s(self):
        """Test the s() function."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        CL = ru.centerline(x, y)
        sss = CL.s()
        # make assertions
        assert np.all(sss == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_ds(self):
        """Test the ds() function."""
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]
        CL = ru.centerline(x, y)
        dss = CL.ds()
        # make assertions
        assert np.all(dss == [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
