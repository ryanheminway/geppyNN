# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:27:26 2023

@author: hemin
"""
from gepnnSandbox import *
from dill_utils import map_with_dill

if __name__ ==  '__main__':
    iters = 4
    successes = map_with_dill(runGEPNN, range(iters))
    success_total = sum(successes)

    print("Out of {} iterations, got {} successes".format(iters, success_total))