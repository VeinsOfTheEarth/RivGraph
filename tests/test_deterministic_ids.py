from __future__ import annotations

import numpy as np

import rivgraph.ln_utils as lnu


def test_finalize_network_ids_relabels_consistently_and_marks_finalized():
    links = {
        "id": [9, 3],
        "idx": [np.array([8, 7, 6]), np.array([1, 2])],
        "conn": [[20, 10], [10, 30]],
        "parallels": [[9, 3]],
        "arts": [[9, 3]],
        "link_conn": [[3], [9]],
        "guess": [[20], [10]],
    }
    nodes = {
        "id": [30, 10, 20],
        "idx": [30, 10, 20],
        "conn": [[3], [9, 3], [9]],
        "inlets": [20],
        "outlets": [30],
        "arts": [10],
    }

    links, nodes = lnu.finalize_network_ids(links, nodes)

    assert lnu.network_ids_are_finalized(links, nodes) is True

    assert nodes["id"] == [2, 0, 1]
    assert links["id"] == [1, 0]
    assert nodes["conn"] == [[0], [1, 0], [1]]
    assert links["conn"] == [[1, 0], [0, 2]]
    assert nodes["inlets"] == [1]
    assert nodes["outlets"] == [2]
    assert nodes["arts"] == [0]
    assert links["parallels"] == [[1, 0]]
    assert links["arts"] == [[1, 0]]
    assert links["link_conn"] == [[0], [1]]
    assert links["guess"] == [[1], [0]]
