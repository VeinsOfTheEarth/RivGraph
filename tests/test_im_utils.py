# -*- coding: utf-8 -*-
"""Unit tests for im_utils.py."""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from rivgraph import im_utils


def test_neighbor_idcs():
    """Test of neighbors_idcs() function."""
    x, y = im_utils.neighbor_idcs(2, 6)
    assert x == [1, 2, 3, 1, 3, 1, 2, 3]
    assert y == [5, 5, 5, 6, 6, 7, 7, 7]


def test_neighbor_xy():
    """Test of neighbor_xy() function."""
    x, y = im_utils.neighbor_xy(2, 6, 0)
    assert x == 1
    assert y == 5

    x1, y1 = im_utils.neighbor_xy(2, 6, 7)
    assert x1 == 3
    assert y1 == 7


def test_getarray():
    """Test get_array() function."""
    idx = [2, 2]
    I = np.zeros((5, 5))
    I[2, 2] = 1
    I[1, 1] = 0.5
    I[3, 3] = 1.5
    size = (3, 3)
    array, row, col = im_utils.get_array(idx, I, size)
    # make assertions
    assert array[1, 1] == 1.
    assert array[0, 0] == 0.5
    assert array[2, 2] == 1.5
    assert row == 1
    assert col == 1


def test_edge_coords():
    """Test edge_coords() function."""
    I = np.zeros((3, 3))
    sizeI = np.shape(I)
    edgepts = im_utils.edge_coords(sizeI, dtype='xy')
    # make assertions to known edge coordinates in x and y
    assert np.all(edgepts[0] == [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 1, 0])
    assert np.all(edgepts[1] == [0, 1, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0])


def test_imshowpair():
    """Test imshowpair() function."""
    I1 = np.array([[0, 1, 0], [0, 1, 0]], dtype=int)
    I2 = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
    im_utils.imshowpair(I1, I2)
    plt.draw()
    plt.close()
    # if plot can be drawn and closed then pass the test
    assert True


class TestNeighbors_Flat:
    """Tests associated with the im_utils.neighbors_flat() function."""

    def test_neighbors_flat(self):
        """Test 1."""
        I = np.zeros((5, 5))
        I[1, 1] = 1.0
        I[2, 2] = 3.0
        Iflat = np.ravel(I)
        idx = 11
        ncols = 5
        filt = 'none'
        idcs, vals = im_utils.neighbors_flat(idx, Iflat, ncols, filt=filt)
        # make assertion
        assert np.all(idcs == [5,  6,  7, 10, 12, 15, 16, 17])
        assert np.all(vals == [0., 1., 0., 0., 3., 0., 0., 0.])

    def test_neighbors_flatsmall(self):
        """Test 2."""
        I = np.zeros((5, 5))
        I[1, 2] = 1.0
        Iflat = np.ravel(I)
        idx = 3
        ncols = 5
        filt = 'none'
        idcs, vals = im_utils.neighbors_flat(idx, Iflat, ncols, filt=filt)
        # make assertion
        assert np.all(idcs == [2, 4, 7, 8, 9])
        assert np.all(vals == [0., 0., 1., 0., 0.])

    def test_neighbors_flatmid(self):
        """Test 3."""
        I = np.zeros((5, 5))
        I[1, 1] = 1.0
        Iflat = np.ravel(I)
        idx = 10
        ncols = 5
        filt = 'none'
        idcs, vals = im_utils.neighbors_flat(idx, Iflat, ncols, filt=filt)
        # make assertion
        assert np.all(idcs == [5,  6, 11, 15, 16])
        assert np.all(vals == [0., 1., 0., 0., 0.])

    def test_neighbors_midcol(self):
        """Test 4."""
        I = np.zeros((5, 5))
        I[4, 3] = 1.0
        Iflat = np.ravel(I)
        idx = 19
        ncols = 5
        filt = 'none'
        idcs, vals = im_utils.neighbors_flat(idx, Iflat, ncols, filt=filt)
        # make assertion
        assert np.all(idcs == [13, 14, 18, 23, 24])
        assert np.all(vals == [0., 0., 0., 1., 0.])

    def test_neighbors_flatfirst(self):
        """Test 5."""
        I = np.zeros((5, 5))
        I[1, 1] = 1.0
        I[2, 2] = 3.0
        Iflat = np.ravel(I)
        idx = 0
        ncols = 5
        filt = 'none'
        idcs, vals = im_utils.neighbors_flat(idx, Iflat, ncols, filt=filt)
        # make assertion
        assert np.all(idcs == [1, 5, 6])
        assert np.all(vals == [0., 0., 1.])

    def test_neighbors_flatlast(self):
        """Test 6."""
        I = np.zeros((5, 5))
        I[4, 3] = 3.0
        Iflat = np.ravel(I)
        idx = 22
        ncols = 5
        filt = 'none'
        idcs, vals = im_utils.neighbors_flat(idx, Iflat, ncols, filt=filt)
        # make assertion
        assert np.all(idcs == [16, 17, 18, 21, 23])
        assert np.all(vals == [0., 0., 0., 0., 3.])

    def test_neighbors_flattop(self):
        """Test 7."""
        I = np.zeros((5, 5))
        I[0, 3] = 1.0
        I[1, 4] = 3.0
        Iflat = np.ravel(I)
        idx = 4
        ncols = 5
        filt = 'none'
        idcs, vals = im_utils.neighbors_flat(idx, Iflat, ncols, filt=filt)
        # make assertion
        assert np.all(idcs == [3, 8, 9])
        assert np.all(vals == [1., 0., 3.])

    def test_neighbors_flatbottom(self):
        """Test 8."""
        I = np.zeros((5, 5))
        I[3, 4] = 1.0
        Iflat = np.ravel(I)
        idx = 24
        ncols = 5
        filt = 'none'
        idcs, vals = im_utils.neighbors_flat(idx, Iflat, ncols, filt=filt)
        # make assertion
        assert np.all(idcs == [18, 19, 23])
        assert np.all(vals == [0., 1., 0.])

    def test_neighbors_bottomwrow(self):
        """Test 9."""
        I = np.zeros((20, 10))
        I[-1, :] = 2.0
        Iflat = np.ravel(I)
        ncols = 10
        idx = 190
        filt = 'none'
        idcs, vals = im_utils.neighbors_flat(idx, Iflat, ncols, filt=filt)
        # make assertion
        assert np.all(idcs == [180, 181, 191])
        assert np.all(vals == [0., 0., 2.])


class TestNeighbor_Vals:
    """Tests for conditions associated with the im_utils.neighbor_vals()."""

    def test_neighbor_zeros(self):
        """Test 1."""
        im = np.zeros((5, 5))
        vals = im_utils.neighbor_vals(im, 0, 0)
        # make assertion
        assert vals[4] == 0.
        assert vals[6] == 0.
        assert vals[7] == 0.

    def test_neighbor_midy(self):
        """Test 2."""
        im = np.zeros((5, 5))
        vals = im_utils.neighbor_vals(im, 0, 2)
        # make assertion
        assert vals[1] == 0.
        assert vals[2] == 0.
        assert vals[4] == 0.
        assert vals[6] == 0.
        assert vals[7] == 0.

    def test_neighbor_endy(self):
        """Test 3."""
        im = np.zeros((5, 5))
        vals = im_utils.neighbor_vals(im, 0, 4)
        # make assertion
        assert vals[1] == 0.
        assert vals[2] == 0.
        assert vals[4] == 0.

    def test_neighbor_endx(self):
        """Test 4."""
        im = np.zeros((5, 5))
        vals = im_utils.neighbor_vals(im, 4, 0)
        # make assertion
        assert vals[3] == 0.
        assert vals[5] == 0.
        assert vals[6] == 0.

    def test_neighbor_endxy(self):
        """Test 5."""
        im = np.zeros((5, 5))
        vals = im_utils.neighbor_vals(im, 4, 4)
        # make assertion
        assert vals[0] == 0.
        assert vals[1] == 0.
        assert vals[3] == 0.

    def test_neighbor_endxmid(self):
        """Test 6."""
        im = np.zeros((5, 5))
        vals = im_utils.neighbor_vals(im, 4, 2)
        # make assertion
        assert vals[0] == 0.
        assert vals[1] == 0.
        assert vals[3] == 0.
        assert vals[5] == 0.
        assert vals[6] == 0.

    def test_neighbor_endyzero(self):
        """Test 7."""
        im = np.zeros((5, 5))
        vals = im_utils.neighbor_vals(im, 2, 0)
        # make assertion
        assert vals[3] == 0.
        assert vals[4] == 0.
        assert vals[5] == 0.
        assert vals[6] == 0.
        assert vals[7] == 0.

    def test_neighbor_endymidx(self):
        """Test 8."""
        im = np.zeros((5, 5))
        vals = im_utils.neighbor_vals(im, 2, 4)
        # make assertion
        assert vals[0] == 0.
        assert vals[1] == 0.
        assert vals[2] == 0.
        assert vals[3] == 0.
        assert vals[4] == 0.


class TestRemoveBlobs:
    """Tests for im_utils.largest_blobs."""

    def test_remove_blobs(self):
        """Test 1."""
        I = np.zeros((5, 5))
        I[1:2, 1] = 1
        Ic = im_utils.remove_blobs(I, 3, connectivity=2)
        # returned array should be all 0s
        assert np.all(Ic == 0)

    def test_largest_blob_removal(self):
        """Test 2."""
        I = np.zeros((5, 5))
        I[1:4, 1:4] = 1
        Ic = im_utils.largest_blobs(I, nlargest=1, action='remove')
        # make assertion
        assert np.all(Ic) == 0

    def test_largest_blob_error(self):
        """Test 3."""
        I = np.zeros((5, 5))
        I[1:4, 1:4] = 1
        Ic = im_utils.largest_blobs(I, nlargest=1, action='invalid')
        # make assertion
        assert np.all(Ic == I)


class TestErode:
    """Tests for im_utils.erode."""

    def test_erode_no_n(self):
        """Test 1."""
        I = np.zeros((3, 3))
        Ie = im_utils.erode(I, n=0, strel='square')
        # make assertion
        assert np.all(I == Ie)

    def test_erode_square(self):
        """Test 2."""
        I = np.zeros((8, 8))
        I[1:7, 1:7] = 1
        Ie = im_utils.erode(I, n=1, strel='square')
        # make assertion
        assert np.all(Ie == np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 1., 1., 1., 1., 0., 0.],
                                      [0., 0., 1., 1., 1., 1., 0., 0.],
                                      [0., 0., 1., 1., 1., 1., 0., 0.],
                                      [0., 0., 1., 1., 1., 1., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0.]]))

    def test_erode_plus(self):
        """Test 3."""
        I = np.zeros((8, 8))
        I[0:7, 1:3] = 1
        I[2:5, 1:7] = 1
        Ie = im_utils.erode(I, n=1, strel='plus')
        # make assertion
        assert np.all(Ie == np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 1., 0., 0., 0., 0., 0.],
                                      [0., 0., 1., 1., 1., 1., 0., 0.],
                                      [0., 0., 1., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0.]]))

    def test_erode_disk(self):
        """Test 4."""
        I = np.zeros((8, 8))
        I[1:7, 1:7] = 1
        Ie = im_utils.erode(I, n=1, strel='disk')
        # make assertion
        assert np.all(Ie == np.zeros_like(Ie))


class TestDilate:
    """Tests for im_utils.dilate."""

    def test_dilate_no_n(self):
        """Test 1."""
        I = np.zeros((5, 5))
        Id = im_utils.dilate(I, n=0)
        # do assertion
        assert np.all(I == Id)

    def test_dilate_square(self):
        """Test 2."""
        I = np.zeros((5, 5))
        I[2, 2] = 1
        Id = im_utils.dilate(I, n=1, strel='square')
        # do assertion
        assert np.all(Id == np.array([[0., 0., 0., 0., 0.],
                                      [0., 1., 1., 1., 0.],
                                      [0., 1., 1., 1., 0.],
                                      [0., 1., 1., 1., 0.],
                                      [0., 0., 0., 0., 0.]]))

    def test_dilate_plus(self):
        """Test 3."""
        I = np.zeros((5, 5))
        I[2, 2] = 1
        Id = im_utils.dilate(I, n=1, strel='plus')
        # do assertion
        assert np.all(Id == np.array([[0., 0., 0., 0., 0.],
                                      [0., 0., 1., 0., 0.],
                                      [0., 1., 1., 1., 0.],
                                      [0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 0.]]))

    def test_dilate_disk(self):
        """Test 4."""
        I = np.zeros((8, 8))
        I[3, 3] = 1
        Id = im_utils.dilate(I, n=1, strel='disk')
        # do assertion
        assert np.all(Id == np.array([[0., 0., 0., 1., 0., 0., 0., 0.],
                                      [0., 1., 1., 1., 1., 1., 0., 0.],
                                      [0., 1., 1., 1., 1., 1., 0., 0.],
                                      [1., 1., 1., 1., 1., 1., 1., 0.],
                                      [0., 1., 1., 1., 1., 1., 0., 0.],
                                      [0., 1., 1., 1., 1., 1., 0., 0.],
                                      [0., 0., 0., 1., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0.]]))


def test_regionprops_small():
    """Small test of regionprops()."""
    I = np.zeros((3, 3))
    I[1, 1] = 1.0
    props = ['centroid', 'mean', 'perim_len', 'convex_area',
             'eccentricity', 'equivalent_diameter',
             'major_axis_length', 'minor_axis_length', 'invalidinput']
    info, _ = im_utils.regionprops(I, props)
    # make bunch of assertions
    assert np.all(info['centroid'] == [1., 1.])
    assert info['mean'] == 1.
    assert info['perim_len'] == 0.
    assert info['convex_area'] == 1
    assert info['eccentricity'] == 0
    assert info['equivalent_diameter'] == pytest.approx(1.12837917)
    assert info['major_axis_length'] == 0.
    assert info['minor_axis_length'] == 0.


def test_regionprops_big():
    """Bigger test of regionprops()."""
    I = np.zeros((20, 20))
    I[1:3, 1:3] = 1.0
    I[15:20, 15:20] = 1.0
    I[7:10, 4:12] = 1.0
    props = ['centroid', 'mean', 'perim_len', 'eccentricity']
    info, _ = im_utils.regionprops(I, props)
    # make bunch of assertions
    assert [*info] == props
    assert np.all(info['centroid'] == np.array([[1.5, 1.5],
                                                [8., 7.5],
                                                [17., 17.]])
                  )
    assert np.all(info['mean'] == np.array([1., 1., 1.]))
    assert np.all(info['perim_len'] == np.array([4., 18., 16.]))
    assert info['eccentricity'] == pytest.approx(np.array([0.,
                                                           0.93435318,
                                                           0.]))


def test_regionprops_perimeter():
    """Test of regionprops() 'perimeter' property."""
    I = np.zeros((20, 20))
    I[1:3, 1:3] = 1.0
    I[15:20, 15:20] = 1.0
    I[7:10, 4:12] = 1.0
    props = ['perimeter']
    info, _ = im_utils.regionprops(I, props)
    # make bunch of assertions
    assert [*info] == props
    assert len(info['perimeter']) == 3
    assert np.all(info['perimeter'][0] == np.array([[1, 1],
                                                    [2, 1],
                                                    [2, 2],
                                                    [1, 2]]))
    assert np.all(info['perimeter'][1] == np.array([[7, 4],
                                                    [8, 4],
                                                    [9, 4],
                                                    [9, 5],
                                                    [9, 6],
                                                    [9, 7],
                                                    [9, 8],
                                                    [9, 9],
                                                    [9, 10],
                                                    [9, 11],
                                                    [8, 11],
                                                    [7, 11],
                                                    [7, 10],
                                                    [7, 9],
                                                    [7, 8],
                                                    [7, 7],
                                                    [7, 6],
                                                    [7, 5]]))
    assert np.all(info['perimeter'][2] == np.array([[15, 15],
                                                    [16, 15],
                                                    [17, 15],
                                                    [18, 15],
                                                    [19, 15],
                                                    [19, 16],
                                                    [19, 17],
                                                    [19, 18],
                                                    [19, 19],
                                                    [18, 19],
                                                    [17, 19],
                                                    [16, 19],
                                                    [15, 19],
                                                    [15, 18],
                                                    [15, 17],
                                                    [15, 16]]))


def test_crop_binary_im():
    """Test crop_binary_im()."""
    I = np.zeros((5, 5))
    I[0, 0] = 1
    I[1, 2:5] = 1
    Icrop, pads = im_utils.crop_binary_im(I)
    # assertions
    assert np.shape(Icrop) == (2, 5)
    assert np.all(pads == [0, 0, 0, 3])


def test_fill_holes():
    """Test fill_holes()."""
    I = np.ones((3, 3))
    I[1, 1] = 0.
    I_filled = im_utils.fill_holes(I, maxholesize=0)
    # assertion that hole was filled
    assert I_filled[1, 1] == 1


def test_downsample():
    """Test downsample()."""
    I = np.zeros((10, 10))
    I[0:5, 0:5] = 1
    Iout = im_utils.downsample_binary_image(I, [2, 2], thresh=0.5)
    # expect to create downsampled 2x2 image with 3 zeros and a one value
    assert np.sum(Iout) == 1
    assert Iout[0, 0] == 1


class TestSkelKernels:
    """Checks for skel_kernels method."""

    def test_case_one(self):
        """Test case 1."""
        out = im_utils.skel_kernels(1)
        assert np.all(out == np.array([[0, 1, 0],
                                       [1, 1, 1],
                                       [0, 0, 0]]))

    def test_case_two(self):
        """Test case 2."""
        out = im_utils.skel_kernels(2)
        assert np.all(out == np.array([[0, 1, 0],
                                       [0, 1, 1],
                                       [0, 1, 0]]))

    def test_case_three(self):
        """Test case 3."""
        out = im_utils.skel_kernels(3)
        assert np.all(out == np.array([[0, 0, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]]))

    def test_case_four(self):
        """Test case 4."""
        out = im_utils.skel_kernels(4)
        assert np.all(out == np.array([[0, 1, 0],
                                       [1, 1, 0],
                                       [0, 1, 0]]))


class TestBpKernels:
    """Checks for bp_kernels method."""

    def test_case_one(self):
        """Test case 1."""
        out = im_utils.bp_kernels(1)
        assert np.all(out == np.array([[1, 1, 0],
                                       [0, 10, 0],
                                       [0, 0, 0]]))

    def test_case_two(self):
        """Test case 2."""
        out = im_utils.bp_kernels(2)
        assert np.all(out == np.array([[0, 1, 1],
                                       [0, 10, 0],
                                       [0, 0, 0]]))

    def test_case_three(self):
        """Test case 3."""
        out = im_utils.bp_kernels(3)
        assert np.all(out == np.array([[0, 0, 1],
                                       [0, 10, 1],
                                       [0, 0, 0]]))

    def test_case_four(self):
        """Test case 4."""
        out = im_utils.bp_kernels(4)
        assert np.all(out == np.array([[0, 0, 0],
                                       [0, 10, 1],
                                       [0, 0, 1]]))

    def test_case_five(self):
        """Test case 5."""
        out = im_utils.bp_kernels(5)
        assert np.all(out == np.array([[0, 0, 0],
                                       [0, 10, 0],
                                       [0, 1, 1]]))

    def test_case_six(self):
        """Test case 6."""
        out = im_utils.bp_kernels(6)
        assert np.all(out == np.array([[0, 0, 0],
                                       [0, 10, 0],
                                       [1, 1, 0]]))

    def test_case_seven(self):
        """Test case 7."""
        out = im_utils.bp_kernels(7)
        assert np.all(out == np.array([[0, 0, 0],
                                       [1, 10, 0],
                                       [1, 0, 0]]))

    def test_case_eight(self):
        """Test case 8."""
        out = im_utils.bp_kernels(8)
        assert np.all(out == np.array([[1, 0, 0],
                                       [1, 10, 0],
                                       [0, 0, 0]]))


def test_trim_idcs():
    """Tests the trim_idcs method."""
    idcs = np.array([[0, 1, 2, 5], [0, 0, 1, 5]]).T
    sizeI = [3, 3]
    new = im_utils.trim_idcs(sizeI, idcs)
    # make assertions
    assert np.all(new[0] == np.array([0, 0]))
    assert np.all(new[1] == np.array([1, 0]))
    assert np.all(new[2] == np.array([2, 1]))


class TestSkelBranchpts:
    """Tests for skel_branchpoints method."""

    def test_simple(self):
        """Simple branchpoint test."""
        I = np.zeros((5, 5))
        I[0:3, 2] = 1
        I[3, 1] = 1
        I[3, 3] = 1
        I[4, 0] = 1
        I[4, 4] = 1
        Ipbs = im_utils.skel_branchpoints(I)
        assert np.all(Ipbs == np.array([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 1]]))

    def test_plus(self):
        """Test a + configuration."""
        I = np.zeros((5, 5))
        I[:, 2] = 1
        I[2, :] = 1
        Ipbs = im_utils.skel_branchpoints(I)
        assert np.all(Ipbs == np.array([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]]))

    def test_cross(self):
        """Test a x configuration."""
        I = np.zeros((5, 5))
        for i in range(0, 5):
            I[i, i] = 1
            I[i, -(i+1)] = 1
        Ipbs = im_utils.skel_branchpoints(I)
        assert np.all(Ipbs == np.array([[1, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 1]]))

    def test_three_horiz(self):
        """Test 3 branchpoints in a row."""
        I = np.zeros((5, 10))
        I[2, :] = 1
        I[1:4, 3:6] = 1
        Ipbs = im_utils.skel_branchpoints(I)
        assert np.all(Ipbs == np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    def test_three_vert(self):
        """Test 3 branchpoints in a column."""
        I = np.zeros((10, 5))
        I[:, 2] = 1
        I[3:6, 1:4] = 1
        Ipbs = im_utils.skel_branchpoints(I)
        assert np.all(Ipbs == np.array([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 1, 1, 0],
                                        [0, 1, 1, 1, 0],
                                        [0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]]))


class TestSkelCurvature:
    """Tests for skel_pixel_curvature method."""

    def test_simplecurves(self):
        """Simple test of curvature."""
        I = np.zeros((5, 5))
        # 90 degree bend
        I[0:3, 2] = 1
        I[3, 0:3] = 1
        curv = im_utils.skel_pixel_curvature(I)
        assert curv[3, 2] == 90.
