from __future__ import annotations

import rivgraph.ln_utils as lnu
from rivgraph.ordered_set import OrderedSet


def test_lookup_helpers_refresh_after_topology_changes():
    links = {"id": [10, 20], "idx": [[1, 2], [3, 4]], "conn": [[0, 1], [1, 2]]}
    nodes = {"id": [0, 1, 2], "idx": [1, 3, 4], "conn": [[10], [10, 20], [20]]}

    assert lnu.link_index(links, 20) == 1
    assert lnu.node_index(nodes, 1) == 1
    assert lnu.node_idx_index(nodes, 4) == 2

    links, nodes = lnu.delete_link(links, nodes, 10)

    assert lnu.link_index(links, 20) == 0
    assert lnu.node_index(nodes, 1) == 0
    assert lnu.node_idx_index(nodes, 4) == 1


def test_lookup_helpers_work_with_ordered_sets():
    links = {"id": OrderedSet([5, 7]), "idx": [[1], [2]], "conn": [[0], [1]]}
    nodes = {"id": OrderedSet([2, 4]), "idx": OrderedSet([9, 11]), "conn": [[5], [7]]}

    assert lnu.link_index(links, 7) == 1
    assert lnu.node_index(nodes, 4) == 1
    assert lnu.node_idx_index(nodes, 11) == 1
