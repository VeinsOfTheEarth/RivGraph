import pytest
import sys, os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph import im_utils

def test_neighbor_idcs():
    x,y = im_utils.neighbor_idcs(2,6)
    assert x == [1, 2, 3, 1, 3, 1, 2, 3]
    assert y == [5, 5, 5, 6, 6, 7, 7, 7]

def test_neighbor_xy():
    x, y = im_utils.neighbor_xy(2, 6, 0)
    assert x == 1
    assert y == 5

    x1, y1 = im_utils.neighbor_xy(2, 6, 7)
    assert x1 == 3
    assert y1 == 7

def test_erode_square():
    i_sq = np.zeros((5,5))
    i_sq[1:4,1:4] = 1
    i_sq_ero = im_utils.erode(i_sq, n=1, strel='square')
    assert i_sq_ero[1,1] == 0.
    assert i_sq_ero[2,2] == 1.
    assert i_sq_ero[3,3] == 0.
