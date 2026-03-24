"""Unit tests for ln_utils.py."""
# import pytest
import os
try:
    from osgeo import gdal
except ImportError:
    import gdal
from rivgraph import ln_utils


def test_conn_links(test_net):
    """Test out the conn_links() function."""
    test_net.skeletonize()
    test_net.compute_network()
    # test some and make assertions
    link_pix_01 = ln_utils.conn_links(test_net.nodes, test_net.links, 139249)
    link_pix_02 = ln_utils.conn_links(test_net.nodes, test_net.links, 614511)
    link_pix_03 = ln_utils.conn_links(test_net.nodes, test_net.links, 762395)
    assert link_pix_01 == [139249, 139259, 129862, 139249, 151567, 139249]
    assert link_pix_02 == [744008, 740931, 1320184, 1411025, 2003991, 2000915]
    assert link_pix_03 == [2304222, 2301073]


def test_append_link_len(test_net):
    """Test append_link_lengths() function."""
    gobj = gdal.Open(
        os.path.normpath(
            'tests/integration/data/Colville/Colville_islands_filled.tif'))
    test_net.skeletonize()
    test_net.compute_network()
    # assert keys in links before getting lengths
    assert ('len' in test_net.links) is False
    # then run the function
    links = ln_utils.append_link_lengths(test_net.links, gobj)
    # then assert that lengths is in links
    assert ('len' in links) is True


def test_add_art_nodes(test_net):
    """Testing add_artificial_nodes() function."""
    gobj = gdal.Open(
        os.path.normpath(
            'tests/integration/data/Colville/Colville_islands_filled.tif'))
    test_net.skeletonize()
    test_net.compute_network()
    # assert no artificial links or nodes before function
    assert ('arts' in test_net.links) is False
    assert ('arts' in test_net.nodes) is False
    # then run the function
    links, nodes = ln_utils.add_artificial_nodes(test_net.links,
                                                 test_net.nodes,
                                                 gobj)
    # assert that arts is now in nodes and links
    assert ('arts' in links) is True
    assert ('arts' in nodes) is True


def test_junction_angles_exp(known_net):
    """Testing junction_angles with exp weighting."""
    nodes = known_net.nodes
    links = known_net.links
    imshape = known_net.imshape
    pixlen = known_net.pixlen
    # pop junction keys out of nodes before recreating them
    _ = nodes.pop('jtype')
    _ = nodes.pop('int_ang')
    _ = nodes.pop('width_ratio')
    nodes = ln_utils.junction_angles(links, nodes, imshape, pixlen,
                                     weight='exp')
    assert ('jtype' in nodes.keys()) is True
    assert ('int_ang' in nodes.keys()) is True
    assert ('width_ratio' in nodes.keys()) is True


def test_junction_angles_linear(known_net):
    """Testing junction_angles with linear weighting."""
    nodes = known_net.nodes
    links = known_net.links
    imshape = known_net.imshape
    pixlen = known_net.pixlen
    # pop junction keys out of nodes before recreating them
    _ = nodes.pop('jtype')
    _ = nodes.pop('int_ang')
    _ = nodes.pop('width_ratio')
    nodes = ln_utils.junction_angles(links, nodes, imshape, pixlen,
                                     weight='linear')
    assert ('jtype' in nodes.keys()) is True
    assert ('int_ang' in nodes.keys()) is True
    assert ('width_ratio' in nodes.keys()) is True


def test_artificial_nodes(synthetic_cycles):
    """Test adding artificial nodes."""
    synthetic_cycles.compute_link_width_and_length()
    links = synthetic_cycles.links
    nodes = synthetic_cycles.nodes
    gobj = gdal.Open(
        os.path.normpath('tests/integration/data/SyntheticCycle/skeleton.tif'))
    links, nodes = ln_utils.add_artificial_nodes(links, nodes, gobj)
    # make assertions
    assert ('arts' in links.keys()) is True
    assert links['arts'] == [[5, 6, 2]]
