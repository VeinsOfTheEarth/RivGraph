"""Tests for broader `rivgraph.classes.river` functions."""
import pytest
import sys
import os
import io
import numpy as np
from rivgraph.classes import river
from rivgraph import geo_utils
from rivgraph.rivers import river_utils as ru
from rivgraph.rivers import centerline_utils as cu


def test_compute_network(test_river, known_river):
    """Test compute_network()."""
    test_river.compute_network()

    # check that nodes and links are created
    assert len(test_river.nodes['id']) >= len(known_river.nodes['id'])
    assert len(test_river.links['id']) >= len(known_river.links['id'])


def test_prune_network(test_river, known_river):
    """Test prune_network()."""
    test_river.compute_mesh()
    test_river.prune_network()

    # check that nodes and links match known case
    assert len(test_river.nodes['id']) == len(known_river.nodes['id'])
    assert len(test_river.links['id']) == len(known_river.links['id'])

    # check that meshlines and meshpolys were created
    assert hasattr(test_river, 'meshlines') == True
    assert hasattr(test_river, 'meshpolys') == True
    assert hasattr(test_river, 'centerline_smooth') == True


def test_assign_flow_directions(test_river, known_river):
    """Test assigning flow directions."""
    test_river.assign_flow_directions()

    # make some simple assertions
    # node assertions
    assert len(test_river.nodes['conn']) == len(known_river.nodes['conn'])
    assert len(test_river.nodes['inlets']) == len(known_river.nodes['inlets'])
    assert len(test_river.nodes['outlets']) == len(known_river.nodes['outlets'])
    # link assertions
    assert len(test_river.links['conn']) == len(known_river.links['conn'])
    assert len(test_river.links['parallels']) == len(known_river.links['parallels'])

    # check that 90% of directions match known case
    # identify list of indices to check
    ind_list = range(0, len(known_river.nodes['idx']))

    # create list of connected idx values
    test_dirs = []
    known_dirs = []
    for j in ind_list:
        test_ind = test_river.nodes['idx'].index(known_river.nodes['idx'][j])
        # interrogate the 'conn' values to find corresponding 'idx' values
        t_inds = test_river.nodes['conn'][test_ind]
        t_idx = []
        for i in t_inds:
            t_idx.append(test_river.links['id'].index(i))

        k_inds = known_river.nodes['conn'][j]
        k_idx = []
        for i in k_inds:
            k_idx.append(known_river.links['id'].index(i))
        # add to the overall dirs lists
        test_dirs.append(test_river.links['idx'][t_idx[0]])
        known_dirs.append(known_river.links['idx'][k_idx[0]])

    # check how many sets of idx values match between the test and known case
    match_counter = 0
    for i in range(0, len(test_dirs)):
        if test_dirs[i] == known_dirs[i]:
            match_counter += 1

    # "soft" unit test -- check that over 90% of the idx values match
    assert match_counter / len(ind_list) > 0.9


# def test_assign_flow_directions_verbose(test_river):
#     """Test assigning flow directions verbosity."""
#     # set up capture string
#     capturedOutput = io.StringIO()
#     sys.stdout = capturedOutput
#
#     print(test_river)
#
#     test_river.verbose = True
#     test_river.assign_flow_directions()
#
#     # grab output
#     sys.stdout = sys.__stdout__
#     # assert output
#
#     properVerbosity = False
#     if 'Setting link directionality...' in capturedOutput.getvalue()[:-1]:
#         if 'to manually set flow directions.' in capturedOutput.getvalue()[:-1]:
#             if 'Attempting to fix' in capturedOutput.getvalue()[:-1]:
#                 properVerbosity = True
#
#     assert properVerbosity == True
    # assert capturedOutput.getvalue()[:-1] == 'Setting link directionality...Using tests/results/brahma/Brahmclip_fixlinks.csv to manually set flow directions.\nAttempting to fix 3 cycles.\nCould not fix cycle links: [[1472, 1471, 1452, 1455, 1476], [1604, 1634, 1635, 1605]].\nUse the csv file at tests/results/brahma/Brahmclip_fixlinks.csv to manually fix link directions.\ndone.'


# def test_chan_width(test_river):
#     """Test channel width function."""
#     width_channels, width_extent = ru.chan_width(test_river.centerline,
#                                                  test_river.Imask,
#                                                  pixarea=test_river.pixarea)

#     # make assertions
#     assert width_channels == pytest.approx(3495.076578549182)
#     assert width_extent == pytest.approx(8533.070813532931)

def test_max_valley_width(test_river):
    """Test max_valley_width function."""
    mvw = ru.max_valley_width(test_river.Imask)

    # make assertions
    assert mvw == pytest.approx(479.8541445064323)


def test_resample_line(test_river):
    """Test resample_line()."""
    resampled_coords, spline = cu.resample_line(test_river.centerline[0],
                                                test_river.centerline[1])

    # make assertions
    assert np.shape(resampled_coords) == np.shape(test_river.centerline)
    assert np.shape(spline) == (3,)


def test_evenly_space_line(test_river):
    """Test evenly_space_line()."""
    resampled_coords, spline = cu.evenly_space_line(test_river.centerline[0],
                                                    test_river.centerline[1])

    # make assertions
    assert np.shape(resampled_coords) == np.shape(test_river.centerline)
    assert np.shape(spline) == (3,)


def test_centerline_mesh(test_river):
    """Test centerline_mesh()."""

    avg_chan_width = np.sum(test_river.Imask) * test_river.pixarea / np.sum(test_river.links['len_adj'])
    mvw = ru.max_valley_width(test_river.Imask)

    perps_out, polys, cl_smoothed = ru.centerline_mesh(test_river.centerline,
                                            avg_chan_width,
                                            mvw*test_river.pixlen*1.1,
                                            mvw*test_river.pixlen*1.1/10,
                                            1)

    # make assertions
    assert np.shape(perps_out) == (86,)
    assert np.shape(polys) == (86,)
    assert len(cl_smoothed.xy[0]) == 4721
    assert cl_smoothed.length == pytest.approx(151134.13414783287)


def test_river_dirs(tmp_path):
    """Test river with exit sides 'ne' and 'sw'."""
    img_path = os.path.normpath(
        'tests/integration/data/Brahma/brahma_mask_clip.tif')
    out_path = os.path.join(tmp_path, 'cropped.tif')
    geo_utils.crop_geotif(img_path, npad=10, outpath=out_path)
    test_ne = river('Brahmclip', out_path,
                    os.path.join(tmp_path, 'brahma'),
                    exit_sides='ne')
    test_ne.compute_network()
    test_ne.compute_mesh()
    test_ne.prune_network()

    # make assertions
    assert len(test_ne.nodes['inlets']) == 1
    assert len(test_ne.nodes['outlets']) == 1
    assert test_ne.exit_sides == 'ne'

    # test sw
    test_sw = river('Brahmclip', out_path,
                    os.path.join(tmp_path, 'brahma'),
                    exit_sides='sw')
    test_sw.compute_network()
    test_sw.compute_mesh()
    test_sw.prune_network()

    # make assertions
    assert len(test_sw.nodes['inlets']) == 1
    assert len(test_sw.nodes['outlets']) == 1
    assert test_sw.exit_sides == 'sw'
