# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 09:15:17 2020

@author: Jon
"""
import sys
import os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

print('with / - {}'.format(sys.path[-1]))

sys.path.insert(0,os.path.realpath(os.path.dirname(__file__)+".."))
print('without / - {}'.format(sys.path[-1]))

# from inspect import getsourcefile
# from os.path import abspath
# print('using inspect sourcefile - {}'.format(abspath(getsourcefile(lambda:0))))

# print(sys.path)

# sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)+"/.."))
# import rivgraph