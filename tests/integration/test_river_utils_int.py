"""Tests for `rivgraph.river.river_utils` functions."""
from rivgraph.rivers import river_utils as ru


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
