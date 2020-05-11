import pytest

import sys, os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from rivgraph import directionality


def test_dummy():
    '''
    Dummy Unit Test
    '''
    assert 1+1 == 2
