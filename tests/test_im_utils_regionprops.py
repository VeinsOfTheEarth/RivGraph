import numpy as np

from rivgraph import im_utils as iu


def _sample_blob():
    I = np.zeros((8, 8), dtype=np.uint8)
    I[2:6, 2:5] = 1
    return I


def test_regionprops_accepts_native_skimage_properties():
    I = _sample_blob()

    props, labeled = iu.regionprops(I, ['solidity', 'perimeter', 'label'])

    assert labeled.max() == 1
    assert 'solidity' in props
    assert 'perimeter' in props
    assert np.isclose(props['solidity'][0], 1.0)
    assert props['perimeter'][0] > 0


def test_regionprops_boundary_coords_is_custom_and_perimeter_is_scalar():
    I = _sample_blob()

    props, _ = iu.regionprops(I, ['boundary_coords', 'perimeter'])

    assert isinstance(props['boundary_coords'], list)
    assert len(props['boundary_coords']) == 1
    assert props['boundary_coords'][0].ndim == 2
    assert props['boundary_coords'][0].shape[1] == 2
    assert np.isscalar(props['perimeter'][0])



def test_regionprops_legacy_aliases_still_work():
    I = _sample_blob()

    props, _ = iu.regionprops(
        I,
        ['mean', 'perim_len', 'convex_area', 'major_axis_length',
         'minor_axis_length', 'equivalent_diameter'],
    )

    assert np.isclose(props['mean'][0], 1.0)
    assert props['perim_len'][0] > 0
    assert props['convex_area'][0] >= props['perim_len'][0] * 0 + 1  # sanity
    assert props['major_axis_length'][0] >= props['minor_axis_length'][0]
    assert props['equivalent_diameter'][0] > 0



def test_regionprops_coords_remains_blobwise_list():
    I = np.zeros((10, 10), dtype=np.uint8)
    I[1:3, 1:3] = 1
    I[6:9, 6:9] = 1

    props, _ = iu.regionprops(I, ['coords'])

    assert isinstance(props['coords'], list)
    assert len(props['coords']) == 2
    assert all(c.ndim == 2 and c.shape[1] == 2 for c in props['coords'])
