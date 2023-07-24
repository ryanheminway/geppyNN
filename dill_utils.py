# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:56:47 2023

@author: hemin
"""
import dill
from multiprocessing import Pool

## Dill helper functios. Necessary for multithreading
## Dill is used for serialization because standard multiprocessing 
## library has very limited serialization capabilities
def run_dill_fn(f, x, *args, **kwargs):
    fn = dill.loads(f)
    return fn(x, *args, **kwargs)
        
def map_with_dill(fn, inputs, *args, **kwargs):
    pool = Pool(8)
    # Byref required
    f = dill.dumps(fn, byref=True)
    results = [pool.apply_async(run_dill_fn, [f, x, *args], kwargs) for x in inputs]
    pool.close() # ATTENTION HERE
    pool.join()
    return [r.get() for r in results]

