"""Unit tests for walk.py."""
import pytest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph import walk


class TestIdcs_No_Turnaround:
    """Tests for idcs_no_turnaround()."""

    def test_one(self):
        """Test first if."""
        Iskel = np.zeros((5, 5))
        idcs = [0, 6]
        poss_walk_idcs = walk.idcs_no_turnaround(idcs, Iskel)
        assert np.all(poss_walk_idcs == [7, 11, 12])

    def test_two(self):
        """Test second if."""
        Iskel = np.zeros((5, 5))
        idcs = [0, 5]
        poss_walk_idcs = walk.idcs_no_turnaround(idcs, Iskel)
        assert np.all(poss_walk_idcs == [9, 10, 11])

    def test_three(self):
        """Test third if."""
        Iskel = np.zeros((5, 5))
        idcs = [0, 4]
        poss_walk_idcs = walk.idcs_no_turnaround(idcs, Iskel)
        assert np.all(poss_walk_idcs == [3, 8, 9])

    def test_four(self):
        """Test fourth if."""
        Iskel = np.zeros((5, 5))
        idcs = [3, 4]
        poss_walk_idcs = walk.idcs_no_turnaround(idcs, Iskel)
        assert np.all(poss_walk_idcs == [0, 5, 10])

    def test_five(self):
        """Test fifth if."""
        Iskel = np.zeros((5, 5))
        idcs = [4, 0]
        poss_walk_idcs = walk.idcs_no_turnaround(idcs, Iskel)
        assert np.all(poss_walk_idcs == [-5, -4, 1])

    def test_six(self):
        """Test sixth if."""
        Iskel = np.zeros((5, 5))
        idcs = [5, 0]
        poss_walk_idcs = walk.idcs_no_turnaround(idcs, Iskel)
        assert np.all(poss_walk_idcs == [-6, -5, -4])

    def test_seven(self):
        """Test seventh if."""
        Iskel = np.zeros((5, 5))
        idcs = [6, 0]
        poss_walk_idcs = walk.idcs_no_turnaround(idcs, Iskel)
        assert np.all(poss_walk_idcs == [-1, -6, -5])
