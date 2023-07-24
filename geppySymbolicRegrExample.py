# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:17:39 2023

Example of using geppy library to solve a symbolic regression problem. 
This follows an example provided in the geppy repository: 
https://github.com/ShuhuaGao/geppy/blob/master/examples/sr/GEP_RNC_for_ML_with_UCI_Power_Plant_dataset.ipynb

@author: Ryan Heminway
"""
import geppy as gep
from deap import creator, base, tools
import numpy as np
import pandas as pd
import random

import operator 
import math
import datetime
import os 

# REQUIRED TO AVOID GRAPHVIZ ERRORS
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

# for reproducing results
s = 10
random.seed(s)
np.random.seed(s)

#### DATA INTAKE AND PREPROCESSING

#doublecheck the data is there
print(os.listdir("./data/."))

# read in the data to pandas
#   AT = Atmospheric Temperature in C
#   V = Exhaust Vacuum Speed
#   AP = Atmospheric Pressure
#   RH = Relative Humidity
#   PE = Power Output      (the target variable) 
PowerPlantData = pd.read_csv("./data/UCI_PowerPlant.csv")
print(PowerPlantData.describe())

# Split into 80/20 train/test chunks of data
msk = np.random.rand(len(PowerPlantData)) < 0.8
train = PowerPlantData[msk]
holdout = PowerPlantData[~msk]

# check the number of records we'll validate our MSE with
print(holdout.describe())

# check the number of records we'll train our algorithm with
print(train.describe())

# copy and convert our pandas dataframe into numpy variables.

# NOTE: I'm only feeding in the TRAIN values to the algorithms. Later I will independely check
# the MSE myself using a holdout test dataset

AT = train.AT.values
V  = train.V.values
AP = train.AP.values
RH = train.RH.values
Y = train.PE.values  # this is our target, now mapped to Y

print(AT)
print(V)
print(AP)
print(RH)
print(Y)


#### SETTING UP GEP 

# Need protected div to avoid divide by 0
# (TODO Ryan) Don't we want an error to be thrown??
def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

# Creating set of primitives that are allowed in the GEP
pset = gep.PrimitiveSet('Main', input_names=['AT','V','AP','RH'])
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
pset.add_function(math.sin, 1)        
pset.add_function(math.cos, 1)
pset.add_function(math.tan, 1)
pset.add_rnc_terminal()

# Weight -1 because its minimization
creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

h = 7          # head length
n_genes = 2    # number of genes in a chromosome
r = 10         # length of the RNC array
enable_ls = True # whether to apply the linear scaling technique

# Parameters for a GEP configured through a TOOLBOX
toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.randint, a=-10, b=10)   # each RNC is random integer within [-10, 10]
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)

# as a test I'm going to try and accelerate the fitness function
#from numba import jit

#@jit
def evaluate(individual):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)
    
    # below call the individual as a function over the inputs
    
    # Yp = np.array(list(map(func, X)))
    Yp = np.array(list(map(func, AT, V, AP, RH))) 
    
    # return the MSE as we are evaluating on it anyway - then the stats are more fun to watch...
    return np.mean((Y - Yp) ** 2),

#@jit
def evaluate_ls(individual):
    """
    First apply linear scaling (ls) to the individual 
    and then evaluate its fitness: MSE (mean squared error)
    """
    func = toolbox.compile(individual)
    Yp = np.array(list(map(func, AT, V, AP, RH)))
    
    # special cases which cannot be handled by np.linalg.lstsq: (1) individual has only a terminal 
    #  (2) individual returns the same value for all test cases, like 'x - x + 10'. np.linalg.lstsq will fail in such cases.
    # That is, the predicated value for all the examples remains identical, which may happen in the evolution.
    if isinstance(Yp, np.ndarray):
        Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))
        (individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, Y)   
        # residuals is the sum of squared errors
        if residuals.size > 0:
            return residuals[0] / len(Y),   # MSE
    
    # regarding the above special cases, the optimal linear scaling w.r.t LSM is just the mean of true target values
    individual.a = 0
    individual.b = np.mean(Y)
    return np.mean((Y - individual.b) ** 2),

if enable_ls:
    toolbox.register('evaluate', evaluate_ls)
else:
    toolbox.register('evaluate', evaluate)  


# Specifying Genetic Operators
toolbox.register('select', tools.selTournament, tournsize=3)
# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
# 2. Dc-specific operators
toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
# for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
toolbox.pbs['mut_rnc_array_dc'] = 1  # we can also give the probability via the pbs p


# track performance stats
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)



#### RUN EVOLUTION

# size of population and number of generations
n_pop = 120
n_gen = 50

champs = 3
pop = toolbox.population(n=n_pop)
hof = tools.HallOfFame(champs)   # only record the best three individuals ever found in all generations
startDT = datetime.datetime.now()
print(str(startDT))
# start evolution
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                          stats=stats, hall_of_fame=hof, verbose=True)
print ("Evolution times were:\n\nStarted:\t", startDT, "\nEnded:   \t", str(datetime.datetime.now()))


#### DISPLAY RESULTING INDIVIDUALS 

# print the best symbolic regression we found:
best_ind = hof[0]
symplified_best = gep.simplify(best_ind)

if enable_ls:
    symplified_best = best_ind.a * symplified_best + best_ind.b

key= '''
Given training examples of

    AT = Atmospheric Temperature (C)
    V = Exhaust Vacuum Speed
    AP = Atmospheric Pressure
    RH = Relative Humidity

we trained a computer using GEP to predict the 

    PE = Power Output

Our symbolic regression process found the following equation offers our best prediction:

'''

print('\n', key,'\t', str(symplified_best), '\n\nwhich formally is presented as:\n\n')

from sympy import *
init_printing()
print(symplified_best)

# we want to use symbol labels instead of words in the tree graph
rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}  
gep.export_expression_tree(best_ind, rename_labels, './data/numerical_expression_tree.png')



##### fixed - this works, and applis the actual function generated, with scaling option.
# accepts a pandas array, a list of data input Terminals, your best individual, and option to turn off linear scaling.

def CalculateGeppyModelOutput(testdata, finalTerminals, best_ind, enable_ls=true):
    
    # compile the best individual to a function
    finalfunc = toolbox.compile(best_ind)
    
    # Build a numpy arrays from pandas, with good tmp names 
    paramlist = []
    for term in finalTerminals:
        locals()["_holdout" + str(term)] = testdata[term].values
        paramlist = paramlist + ["_holdout" + str(term)]

    # use the tmp arrays names to complete the params part of the evaluation call
    ourparam_string = ", ".join(paramlist)
    ourfuncstring = 'np.array(list(map(finalfunc, ' + ourparam_string + ')))'
    
    # run the outputs over the data, and return the numpy array
    # this runs our core "discovered" geppy function, but doesn't include our linear scaling
    rawoutput = eval(ourfuncstring)
    
    # define a function to apply our linear scaling
    def lscaler(x, a=best_ind.a, b=best_ind.b):
        return a * x + b
    # build command to evaluate scaling
    correctionstring = 'np.array(list(map(lscaler, rawoutput)))'
    
    if enable_ls:
        # apply and return the linear scaled output
        return eval(correctionstring)
    else:
        return rawoutput   

# some previous example outputs
# (AT*(AP - 2*AT - V - 23) + 4*V)/(2*AT)       # MSE 23.5 is my best run.
# AP/2 - AT - V/2 - 8.0 + RH/(2*AT)            # MSE 26 # also a very good run.

# other results with worse performance on short/small runs were
# (AP*(3*AT + 1) + AT*V*(AP + AT + 2*RH)/3)/(AT*V)
# AP/2 - AT - V/2 - 6

print(holdout.describe())

predPE = CalculateGeppyModelOutput(holdout, ['AT','V','AP','RH'], best_ind)

print(predPE)

def colorful(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

