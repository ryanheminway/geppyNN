# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:15:12 2023

Sandbox to experiment with GEPNN 

@author: Ryan Heminway
"""
from geppy.core.entity import *
from geppy.core.symbol import *
from geppy.tools.toolbox import *
from geppy.tools.parser import *
from geppy.tools.mutation import *
from geppy.tools.crossover import *
from geppy.algorithms.basic import *
from geppy.support.visualization import *

# (NOTE Ryan) did not edit deap 
from deap import creator, base, tools

# # Make a folder for the run
from pathlib import Path
import time
import pickle 

from dill_utils import *

import pandas as pd
import numpy as np
import random
import os 
import torch

# REQUIRED TO AVOID GRAPHVIZ ERRORS
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

# -------------------- Function Set Definitions (start) --------------------- #
def binary_thresh(activation, threshold=1.0):
    return (activation > threshold).float()

def sigmoid(activation, threshold=1.0):
    return torch.sigmoid(activation)

def tanh(activation, threshold=1.0):
    return torch.tanh(activation)

def relu(activation, threshold=1.0):
    return torch.relu(activation)

def base_neuron_fn(in_tensor, w_tensor, threshold, activation="binary-thresh"):
    """
    Basic neuron functionality: dot product between inputs and weights, with
    activation function applied to the result. Expects inputs and weights 
    arguments to be given as PyTorch tensors.
    """
    activations = { 'binary-thresh' : binary_thresh,
                    'sigmoid' : sigmoid,
                    'tanh' : tanh,
                    'relu' : relu,
                 }
    act = torch.dot(in_tensor, w_tensor)
    act_fun = activations[activation]
    return act_fun(act, threshold)

def D_relu(in1, in2, weights=[1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return base_neuron_fn(in_tensor, weights_tensor, threshold, "relu")
    
def T_relu(in1, in2, in3, weights=[1.0,1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1), in3.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return base_neuron_fn(in_tensor, weights_tensor, threshold, "relu")

def Q_relu(in1, in2, in3, in4, weights=[1.0,1.0,1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1), in3.reshape(1), in4.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return base_neuron_fn(in_tensor, weights_tensor, threshold, "relu")

def D_tanh(in1, in2, weights=[1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return base_neuron_fn(in_tensor, weights_tensor, threshold, "tanh")
    
def T_tanh(in1, in2, in3, weights=[1.0,1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1), in3.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return base_neuron_fn(in_tensor, weights_tensor, threshold, "tanh")

def Q_tanh(in1, in2, in3, in4, weights=[1.0,1.0,1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1), in3.reshape(1), in4.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return base_neuron_fn(in_tensor, weights_tensor, threshold, "tanh")

def D_sigmoid(in1, in2, weights=[1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return base_neuron_fn(in_tensor, weights_tensor, threshold, "sigmoid")
    
def T_sigmoid(in1, in2, in3, weights=[1.0,1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1), in3.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return base_neuron_fn(in_tensor, weights_tensor, threshold, "sigmoid")

def Q_sigmoid(in1, in2, in3, in4, weights=[1.0,1.0,1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1), in3.reshape(1), in4.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return base_neuron_fn(in_tensor, weights_tensor, threshold, "sigmoid")

def nn_add(in1, in2, weights=[1.0, 1.0], threshold=1.0):
    a1 = in1 * weights[0]
    a2 = in2 * weights[1]
    return a1 + a2

def nn_sub(in1, in2, weights=[1.0, 1.0], threshold=1.0):
    a1 = in1 * weights[0]
    a2 = in2 * weights[1]
    return a1 - a2

def nn_mult(in1, in2, weights=[1.0, 1.0], threshold=1.0):
    a1 = in1 * weights[0]
    a2 = in2 * weights[1]
    return a1 * a2

def nn_div_protected(in1, in2, weights=[1.0, 1.0], threshold=1.0):
    a1 = in1 * weights[0]
    a2 = in2 * weights[1]
    if abs(a2) < 1e-6:
        return 1
    return a1 / a2

def T_link_softmax(*inputs):
    inputs_tuple = ()
    for inp in inputs:
        inputs_tuple = inputs_tuple + (inp.reshape(1),)
    in_tensor = torch.cat(inputs_tuple, 0)
    softmax = torch.nn.Softmax(dim=0)
    return softmax(in_tensor)
# -------------------- Function Set Definitions (end) --------------------- #


# ----------- Regression Problem (start) ----------------- #
"""
This block demonstrates solving a function-finding regression problem.
Specifically, I task GEPNN with modeling the function:
    Y = 2.718 * a^2 + 3.146 * a
    
As was published in the paper: "Gene Expression Programming Neural Network 
                                for Regression and Classification"
"""
# def evaluate(individual): 
#     func = toolbox.compile(individual)
    
#     def true_func(a):
#         return (2.718 * (a * a)) + (3.416 * a)
    
#     M = 1000
#     # (NOTE Ryan) Trying to follow fitness func definition according to paper
#     for i in range(10):
#         ours = func(i)
#         truth = true_func(i)
#         fitness_i = abs(ours - truth)
#         M = M - fitness_i
    
#     return M,

# # Create primitive set
# pset = PrimitiveSet('Main', input_names=['a'])
# pset.add_nn_function(nn_add, 2, name="add")
# pset.add_nn_function(nn_mult, 2, name="mult")
# pset.add_nn_function(nn_div_protected, 2, name="div")

# # Create Individual class and fitness measurement
# creator.create("FitnessMax", base.Fitness, weights=(1,)) # maximize fitness
# creator.create("Individual", Chromosome, fitness=creator.FitnessMax)

# # GEP NN parameters
# guided = True
# h = 7 # 8       # head length
# n_genes = 1 # number of genes in a chromosome
# r = 10      # length of RNC arrays

# toolbox = Toolbox()
# toolbox.register("weight_gen", random.uniform, -2, 2)
# toolbox.register("thresh_gen", random.randint, 1, 1)
# toolbox.register("gene_gen", GeneNN, pset=pset, head_length=h, 
#                   dw_rnc_gen=toolbox.weight_gen, dw_rnc_array_length=r, 
#                   dt_rnc_gen=toolbox.thresh_gen, dt_rnc_array_length=r, func_head=guided)
# toolbox.register("individual", creator.Individual, gene_gen=toolbox.gene_gen,
#                   n_genes=n_genes)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("compile", compile_, pset=pset)
# toolbox.register("evaluate", evaluate)
# toolbox.register("select", tools.selTournament, tournsize=3)

# toolbox.register("mut_dw", mutate_uniform_dw, ind_pb=0.044, pb=1)
# toolbox.register("mut_dt", mutate_uniform_dt, ind_pb=0.044, pb=1)
# toolbox.register("mut_tanspose_dw", transpose_dw, pb=0.1)
# toolbox.register("mut_transpose_dt", transpose_dt, pb=0.1)

# toolbox.register("mut_rncs_dw", mutate_rnc_array_dw, rnc_gen=toolbox.weight_gen, ind_pb='0.5p', pb=0.05)
# toolbox.register("mut_rncs_dt", mutate_rnc_array_dt, rnc_gen=toolbox.thresh_gen, ind_pb='0.5p', pb=0.05)
# toolbox.register("cx_1p", crossover_one_point, pb=0.6)

# # Certain operators only apply when HEAD can be mutated
# if guided:
    
#     toolbox.register("mut_uniform", mutate_uniform_save_head, pset=pset, ind_pb=0.044, pb=1)
# else:
#     toolbox.register("mut_uniform", mutate_uniform, pset=pset, ind_pb=0.044, pb=1)
#     toolbox.register("mut_is_transpose", is_transpose, pb=0.1)
#     toolbox.register("mut_ris_transpose", ris_transpose, pb=0.1)


# stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
# stats.register("avg", np.mean)
# stats.register("std", np.std)
# stats.register("min", np.min)
# stats.register("max", np.max)
# #stats.register("best_ind", lambda x: None)

# n_pop = 20
# n_gen = 5000

# champs = 3
# iters = 100

# today = time.strftime("%Y%m%d")
# run_dir = "runs"
# model_path = str(Path.cwd()) + "/" + run_dir + "/" + today + '_regr' 
# if guided:
#     model_path += "_guided"
# model_path += "/"
# Path(model_path).mkdir(parents=True, exist_ok=True)

# results_file = model_path + '/results.txt'
# def _write_to_file(file, content):
#     f = open(file, 'a')
#     f.write(content)  
#     f.close()


# _write_to_file(results_file, "Running GEPNN solver for a Regression problem\n")

# avg_fitness = 0
# for i in range(iters):
#     print("Running iteration: ", i)
    
#     pop = toolbox.population(n=n_pop)
#     hof = tools.HallOfFame(champs) 
#     # start evolution
#     pop, log = gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
#                               stats=stats, hall_of_fame=hof, verbose=False)
#     best_ind = hof[0]
    
#     print("got best: ", best_ind)
#     fitness_best = evaluate(best_ind)[0]
#     print("eval'd best: ", fitness_best)
#     _write_to_file(results_file, "Iter {} got best fitness: {} \n".format(i, fitness_best))
#     file_name = model_path + "regr_iter_{}_fit_{}".format(i, fitness_best) + ".png"
#     rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}  
#     export_expression_tree_nn(best_ind, rename_labels, file_name)
#     avg_fitness += fitness_best
    
#     pkl_file = open(model_path + "stats_iter_{}.pickle".format(i), 'wb')
#     pickle.dump(log, pkl_file)
    
    
    
# avg_fitness = avg_fitness / iters
# _write_to_file(results_file, "Finished. Got AVG fitness: {} \n".format(i, avg_fitness))

# ----------- Regression Problem (end) ----------------- #

# ----------- Glass Classificiation Problem (start) ----------------- #
"""
This block demonstrates solving multi-class classification using the
Glass dataset. GEPNN is formulated to accomplish this by creating 
multi-genic individuals where each gene is independently responsible
for classifying their respective class. A softmax is used as the 
linking function to produce the final output. 
"""
# Glass dataset from UCI MLR (added header manually)
# RI = Refractive Index
# Na = Sodium content
# Mg = Magnesium content
# Al = Aluminum content
# Si = Silicon content
# K = Potassium content
# Ca = Calcium content
# Ba = Barium content
# Fe = Iron content
# class_id = classification (1-7)
glass_dataset = pd.read_csv("./../glassDataset/glassDataset.csv")
print(glass_dataset.describe(include='all')) # include all for string class

msk = np.random.rand(len(glass_dataset)) < 0.8
train = glass_dataset[msk]
holdout = glass_dataset[~msk]
# check the number of records we'll validate with
print(holdout.describe())
# check the number of records we'll train our algorithm with
print(train.describe())

RI = torch.from_numpy(train.RI.values).float()
Na = torch.from_numpy(train.Na.values).float()
Mg = torch.from_numpy(train.Mg.values).float()
Al = torch.from_numpy(train.Al.values).float()
Si = torch.from_numpy(train.Si.values).float()
K = torch.from_numpy(train.K.values).float()
Ca = torch.from_numpy(train.Ca.values).float()
Ba = torch.from_numpy(train.Ba.values).float()
Fe = torch.from_numpy(train.Fe.values).float()
Y = train.class_id.values  # this is our target, now mapped to Y

def evaluate(individual): 
    func = toolbox.compile(individual)
    # Get predicted labels
    Yp_list = list(map(func, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe))
    Yp = torch.Tensor(len(Yp_list), 7).float()
    #print("Yp_list: ", Yp_list)
    torch.stack(Yp_list, dim=0, out=Yp) # Turn list of tensors into stacked tensor
    #print("Yp: ", Yp)
    # True labels (off by 1)
    labels = [torch.tensor(x-1) for x in Y]
    #print("labels: ", labels)
    labels_t = torch.Tensor(len(labels), 7).long()
    torch.stack(labels, dim=0, out=labels_t)
    #print("Yp: ", labels_t)
    # Cross entropy loss as fitness
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    #print("loss: ", loss(Yp, labels_t))
    return loss(Yp, labels_t).item(),
    
# Create primitive set
pset = PrimitiveSet('Main', input_names=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
pset.add_nn_function(D_relu, 2, name="D_r")
pset.add_nn_function(T_relu, 3, name="T_r")
pset.add_nn_function(Q_relu, 4, name="Q_r")
pset.add_nn_function(D_sigmoid, 2, name="D_s")
pset.add_nn_function(T_sigmoid, 3, name="T_s")
pset.add_nn_function(Q_sigmoid, 4, name="Q_s")
pset.add_nn_function(D_tanh, 2, name="D_t")
pset.add_nn_function(T_tanh, 3, name="T_t")
pset.add_nn_function(Q_tanh, 4, name="Q_t")
pset.add_nn_function(nn_add, 2, name="add")
pset.add_nn_function(nn_sub, 2, name="sub")
pset.add_nn_function(nn_mult, 2, name="mult")

# Create Individual class and fitness measurement
creator.create("FitnessMin", base.Fitness, weights=(-1,)) # maximize fitness
creator.create("Individual", Chromosome, fitness=creator.FitnessMin)

# GEP NN parameters
guided = True
h = 7 # 8       # head length
n_genes = 7 # number of genes in a chromosome
r = 10      # length of RNC arrays

toolbox = Toolbox()
toolbox.register("weight_gen", random.uniform, -2, 2)
toolbox.register("thresh_gen", random.randint, 1, 1)
toolbox.register("gene_gen", GeneNN, pset=pset, head_length=h, 
                  dw_rnc_gen=toolbox.weight_gen, dw_rnc_array_length=r, 
                  dt_rnc_gen=toolbox.thresh_gen, dt_rnc_array_length=r, func_head=guided)
toolbox.register("individual", creator.Individual, gene_gen=toolbox.gene_gen,
                  n_genes=n_genes, linker=T_link_softmax)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", compile_, pset=pset)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mut_dw", mutate_uniform_dw, ind_pb=0.044, pb=1)
toolbox.register("mut_dt", mutate_uniform_dt, ind_pb=0.044, pb=1)
toolbox.register("mut_tanspose_dw", transpose_dw, pb=0.1)
toolbox.register("mut_transpose_dt", transpose_dt, pb=0.1)
#toolbox.register("mut_rncs_dw", mutate_rnc_array_dw, rnc_gen=toolbox.weight_gen, ind_pb='0.5p', pb=0.05)
#toolbox.register("mut_rncs_dt", mutate_rnc_array_dt, rnc_gen=toolbox.thresh_gen, ind_pb='0.5p', pb=0.05)
toolbox.register("cx_1p", crossover_one_point, pb=0.6)

# Certain operators only apply when HEAD can be mutated
if guided:
    toolbox.register("mut_uniform", mutate_uniform_save_head, pset=pset, ind_pb=0.044, pb=1)
else:
    toolbox.register("mut_uniform", mutate_uniform, pset=pset, ind_pb=0.044, pb=1)
    toolbox.register("mut_is_transpose", is_transpose, pb=0.1)
    toolbox.register("mut_ris_transpose", ris_transpose, pb=0.1)

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
stats.register("train_acc", np.min)
stats.register("test_acc", np.min)

def test_accuracy(individual, dataset):
    func = toolbox.compile(individual)
    RI_t = torch.from_numpy(dataset.RI.values).float()
    Na_t = torch.from_numpy(dataset.Na.values).float()
    Mg_t = torch.from_numpy(dataset.Mg.values).float()
    Al_t = torch.from_numpy(dataset.Al.values).float()
    Si_t = torch.from_numpy(dataset.Si.values).float()
    K_t = torch.from_numpy(dataset.K.values).float()
    Ca_t = torch.from_numpy(dataset.Ca.values).float()
    Ba_t = torch.from_numpy(dataset.Ba.values).float()
    Fe_t = torch.from_numpy(dataset.Fe.values).float()
    # Get predicted labels
    Yp_list_tensors = list(map(func, RI_t, Na_t, Mg_t, Al_t, Si_t, K_t, Ca_t, Ba_t, Fe_t))
    Yp_list = [np.argmax(x.numpy()) for x in Yp_list_tensors]
    # True labels
    labels = [x-1 for x in dataset.class_id.values]
    matches = 0
    for i in range(len(labels)):
        if labels[i] == Yp_list[i]:
            matches += 1
    accuracy = matches / len(labels)
    return accuracy

n_pop = 20
n_gen = 1000

champs = 3

iters = 100
    
# Make a folder for the run
from pathlib import Path
import time
today = time.strftime("%Y%m%d")
run_dir = "runs"
model_path = str(Path.cwd()) + "/" + run_dir + "/" + today + '_regr' 
if guided:
    model_path += "_guided"
model_path += "/"
Path(model_path).mkdir(parents=True, exist_ok=True)

results_file = model_path + '/results.txt'
def _write_to_file(file, content):
    f = open(file, 'a')
    f.write(content)  
    f.close()
    
_write_to_file(results_file, "Running GEPNN solver for Glass Classification\n")
avg_test_acc = 0
avg_train_acc = 0
for i in range(iters):
    print("Running iteration: ", i)
    
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(champs) 
    # start evolution
    pop, log = gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                              stats=stats, hall_of_fame=hof, verbose=False)
    best_ind = hof[0]
    
    print("got best: ", best_ind)
    fitness_best = evaluate(best_ind)[0]
    print("eval'd best: ", fitness_best)
    train_acc = test_accuracy(best_ind, train)
    test_acc = test_accuracy(best_ind, holdout)
    print("Train accuracy: ", train_acc)
    print("Test accuracy: ", test_acc)
    # accumulating accuracies for averages
    avg_train_acc += train_acc
    avg_test_acc += test_acc
    _write_to_file(results_file, "Iter {} got train / test accuracies: {} / {} \n".format(i, train_acc, test_acc))
    
    # Save the best individual's network graph as a file 
    file_name = model_path + "glass_iter_tr_{:.4f}_te_{:.4f}".format(train_acc, test_acc) + str(i) + ".png"
    rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}  
    export_expression_tree_nn(best_ind, rename_labels, file_name)
    
    # Add final train / test accuracies as a non-series row
    record = { "train_acc" : train_acc, "test_acc" : test_acc }
    log.record(gen=n_gens + 1, nevals=0, **record)
    # Save statistics as a pickle object file on disk
    pkl_file = open(model_path + "stats_iter_{}.pickle".format(i), 'wb')
    pickle.dump(log, pkl_file)
    
avg_train_acc = avg_train_acc / iters
avg_test_acc = avg_test_acc / iters
_write_to_file(results_file, "Finished. Got AVG train / test accuracies: {} / {} \n".format(i, avg_train_acc, avg_test_acc))


# ----------- Glass Classificiation Problem (end) ------------------- #

# ----------- Iris Classificiation Problem (start) ----------------- #
"""
This block demonstrates solving multi-class classification using the
Iris dataset. GEPNN is formulated to accomplish this by creating 
multi-genic individuals where each gene is independently responsible
for classifying their respective class. A softmax is used as the 
linking function to produce the final output. 
"""
# def class_to_vec(sample_class):
#     """
#     Helper method to create a one-hot vector (PyTorch tensor) when given a 
#     string matching a class in the Iris dataset.
#     """
#     type_dict = { 'iris-setosa' : [1, 0, 0],
#                   'iris-virginica' : [0, 1, 0], 
#                   'iris-versicolor' : [0, 0, 1],
#                   }
#     s_c = sample_class.lower()
#     return torch.tensor(type_dict[s_c])

# def class_to_int(sample_class):
#     """
#     Helper method to create a integer based on a given class string.
#     """
#     type_dict = { 'iris-setosa' : 0,
#                   'iris-virginica' : 1, 
#                   'iris-versicolor' : 2,
#                   }
#     s_c = sample_class.lower()
#     return torch.tensor(type_dict[s_c])
    
# # Iris dataset from UCI MLR (added header manually)
# # s_len = Sepal Length (cm)
# # s_wid = Sepal Width (cm)
# # p_len = Petal Length (cm)
# # p_wid = Petal Width (cm)
# # c = classification (Iris-setosa', 'Iris-virginica', 'Iris-versicolour')
# iris_dataset = pd.read_csv("./../irisDataset/irisDataset.csv")
# print(iris_dataset.describe(include='all')) # include all for string class

# msk = np.random.rand(len(iris_dataset)) < 0.8
# train = iris_dataset[msk]
# holdout = iris_dataset[~msk]
# # check the number of records we'll validate our MSE with
# print(holdout.describe())
# # check the number of records we'll train our algorithm with
# print(train.describe())

# S_LEN = torch.from_numpy(train.s_len.values).float()
# S_WID  = torch.from_numpy(train.s_wid.values).float()
# P_LEN = torch.from_numpy(train.p_len.values).float()
# P_WID = torch.from_numpy(train.p_wid.values).float()
# Y = train.c.values  # this is our target, now mapped to Y

# # print(S_LEN)
# # print(S_WID)
# # print(P_LEN)
# # print(P_WID)
# # print(Y)

# def evaluate(individual): 
#     func = toolbox.compile(individual)
#     # Get predicted labels
#     Yp_list = list(map(func, S_LEN, S_WID, P_LEN, P_WID))
#     Yp = torch.Tensor(len(Yp_list), 3).float()
#     #print("Yp_list: ", Yp_list)
#     torch.stack(Yp_list, dim=0, out=Yp) # Turn list of tensors into stacked tensor
#     #print("Yp: ", Yp)
#     # True labels
#     labels = list(map(class_to_int, Y))
#     #print("labels: ", labels)
#     labels_t = torch.Tensor(len(labels), 3).long()
#     torch.stack(labels, dim=0, out=labels_t)
#     #print("Yp: ", labels_t)
#     # Cross entropy loss as fitness
#     loss = torch.nn.CrossEntropyLoss(reduction='sum')
#     #print("loss: ", loss(Yp, labels_t))
#     return loss(Yp, labels_t).item(),
    
# # Create primitive set
# pset = PrimitiveSet('Main', input_names=['sl', 'sw', 'pl', 'pw'])
# pset.add_nn_function(D_relu, 2, name="D_r")
# pset.add_nn_function(T_relu, 3, name="T_r")
# pset.add_nn_function(Q_relu, 4, name="Q_r")
# pset.add_nn_function(D_sigmoid, 2, name="D_s")
# pset.add_nn_function(T_sigmoid, 3, name="T_s")
# pset.add_nn_function(Q_sigmoid, 4, name="Q_s")
# pset.add_nn_function(D_tanh, 2, name="D_t")
# pset.add_nn_function(T_tanh, 3, name="T_t")
# pset.add_nn_function(Q_tanh, 4, name="Q_t")
# pset.add_nn_function(nn_add, 2, name="add")
# pset.add_nn_function(nn_sub, 2, name="sub")
# pset.add_nn_function(nn_mult, 2, name="mult")

# # Create Individual class and fitness measurement
# creator.create("FitnessMin", base.Fitness, weights=(-1,)) # maximize fitness
# creator.create("Individual", Chromosome, fitness=creator.FitnessMin)

# # GEP NN parameters
# guided = True
# h = 7 # 8       # head length
# n_genes = 3 # number of genes in a chromosome
# r = 10      # length of RNC arrays

# toolbox = Toolbox()
# toolbox.register("weight_gen", random.uniform, -2, 2)
# toolbox.register("thresh_gen", random.randint, 1, 1)
# toolbox.register("gene_gen", GeneNN, pset=pset, head_length=h, 
#                   dw_rnc_gen=toolbox.weight_gen, dw_rnc_array_length=r, 
#                   dt_rnc_gen=toolbox.thresh_gen, dt_rnc_array_length=r, func_head=guided)
# toolbox.register("individual", creator.Individual, gene_gen=toolbox.gene_gen,
#                   n_genes=n_genes, linker=T_link_softmax)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("compile", compile_, pset=pset)
# toolbox.register("evaluate", evaluate)
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("mut_dw", mutate_uniform_dw, ind_pb=0.044, pb=1)
# toolbox.register("mut_dt", mutate_uniform_dt, ind_pb=0.044, pb=1)
# toolbox.register("mut_tanspose_dw", transpose_dw, pb=0.1)
# toolbox.register("mut_transpose_dt", transpose_dt, pb=0.1)
# #toolbox.register("mut_rncs_dw", mutate_rnc_array_dw, rnc_gen=toolbox.weight_gen, ind_pb='0.5p', pb=0.05)
# #toolbox.register("mut_rncs_dt", mutate_rnc_array_dt, rnc_gen=toolbox.thresh_gen, ind_pb='0.5p', pb=0.05)
# toolbox.register("cx_1p", crossover_one_point, pb=0.6)

# # Certain operators only apply when HEAD can be mutated
# if guided:
#     toolbox.register("mut_uniform", mutate_uniform_save_head, pset=pset, ind_pb=0.044, pb=1)
# else:
#     toolbox.register("mut_uniform", mutate_uniform, pset=pset, ind_pb=0.044, pb=1)
#     toolbox.register("mut_is_transpose", is_transpose, pb=0.1)
#     toolbox.register("mut_ris_transpose", ris_transpose, pb=0.1)

# stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
# stats.register("avg", np.mean)
# stats.register("std", np.std)
# stats.register("min", np.min)
# stats.register("max", np.max)
# stats.register("train_acc", np.min)
# stats.register("test_acc", np.min)

# def test_accuracy(individual, dataset):
#     func = toolbox.compile(individual)
#     S_LEN_h = torch.from_numpy(dataset.s_len.values).float()
#     S_WID_h  = torch.from_numpy(dataset.s_wid.values).float()
#     P_LEN_h = torch.from_numpy(dataset.p_len.values).float()
#     P_WID_h = torch.from_numpy(dataset.p_wid.values).float() 
    
#     # Get predicted labels
#     Yp_list_tensors = list(map(func, S_LEN_h, S_WID_h, P_LEN_h, P_WID_h))
#     Yp_list = [np.argmax(x.numpy()) for x in Yp_list_tensors]
#     # True labels
#     labels_tensors = list(map(class_to_int, dataset.c.values))
#     labels = [x.item() for x in labels_tensors]
#     matches = 0
#     for i in range(len(labels)):
#         if labels[i] == Yp_list[i]:
#             matches += 1
#     accuracy = matches / len(labels)
#     return accuracy

# n_pop = 20
# n_gen = 1000

# champs = 3

# iters = 100
# #successes = 0

# # Make a folder for the run
# from pathlib import Path
# import time
# today = time.strftime("%Y%m%d")
# run_dir = "runs"
# model_path = str(Path.cwd()) + "/" + run_dir + "/" + today + '_regr' 
# if guided:
#     model_path += "_guided"
# model_path += "/"
# Path(model_path).mkdir(parents=True, exist_ok=True)

# results_file = model_path + '/results.txt'
# def _write_to_file(file, content):
#     f = open(file, 'a')
#     f.write(content)  
#     f.close()




# _write_to_file(results_file, "Running GEPNN solver for Iris Classification\n")
# avg_test_acc = 0
# avg_train_acc = 0
# for i in range(iters):
#     print("Running iteration: ", i)
    
#     pop = toolbox.population(n=n_pop)
#     hof = tools.HallOfFame(champs) 
#     # start evolution
#     pop, log = gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
#                               stats=stats, hall_of_fame=hof, verbose=False)
#     best_ind = hof[0]
    
#     print("got best: ", best_ind)
#     fitness_best = evaluate(best_ind)[0]
#     print("eval'd best: ", fitness_best)
#     train_acc = test_accuracy(best_ind, train)
#     test_acc = test_accuracy(best_ind, holdout)
#     print("Train accuracy: ", train_acc)
#     print("Test accuracy: ", test_acc)
#     # accumulating accuracies for averages
#     avg_train_acc += train_acc
#     avg_test_acc += test_acc
#     _write_to_file(results_file, "Iter {} got train / test accuracies: {} / {} \n".format(i, train_acc, test_acc))

#     # Save best network topology as a graph file
#     file_name = model_path + "iris_iter_tr_{:.4f}_te_{:.4f}".format(train_acc, test_acc) + str(i) + ".png"
#     rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}  
#     export_expression_tree_nn(best_ind, rename_labels, file_name)
    
#     # Add final train / test accuracies as a non-series row
#     record = { "train_acc" : train_acc, "test_acc" : test_acc }
#     log.record(gen=n_gens + 1, nevals=0, **record)
#     # Save statistics as a pickle object file on disk
#     pkl_file = open(model_path + "stats_iter_{}.pickle".format(i), 'wb')
#     pickle.dump(log, pkl_file)
    
# avg_train_acc = avg_train_acc / iters
# avg_test_acc = avg_test_acc / iters
# _write_to_file(results_file, "Finished. Got AVG train / test accuracies: {} / {} \n".format(i, avg_train_acc, avg_test_acc))
    
# ----------- Iris Classificiation Problem (end) ------------------- #


# ------------------ 6-Plexer Problem (start) ---------------------- #
"""
This block solves the 6-bit Multiplexer problem as was done in the 
original Candida Ferreira paper. The same setup is used as much as possible.
"""
# def U(in1, weights=[1.0], threshold=1.0):
#     in_tensor = in1.reshape(1)
#     weights_tensor = torch.tensor(weights, dtype=torch.float)
#     activation = torch.dot(in_tensor, weights_tensor)
#     return (activation > threshold).float()

# def D(in1, in2, weights=[1.0,1.0], threshold=1.0):
#     #in_tensor = torch.tensor([in1, in2], dtype=torch.float)
#     in_tensor = torch.cat((in1.reshape(1), in2.reshape(1)), 0)
#     weights_tensor = torch.tensor(weights, dtype=torch.float)
#     activation = torch.dot(in_tensor, weights_tensor)
#     return (activation > threshold).float()
    
# def T(in1, in2, in3, weights=[1.0,1.0,1.0], threshold=1.0):
#     in_tensor = torch.cat((in1.reshape(1), in2.reshape(1), in3.reshape(1)), 0)
#     weights_tensor = torch.tensor(weights, dtype=torch.float)
#     activation = torch.dot(in_tensor, weights_tensor)
#     return (activation > threshold).float()

# def Q(in1, in2, in3, in4, weights=[1.0,1.0,1.0,1.0], threshold=1.0):
#     in_tensor = torch.cat((in1.reshape(1), in2.reshape(1), in3.reshape(1), in4.reshape(1)), 0)
#     weights_tensor = torch.tensor(weights, dtype=torch.float)
#     activation = torch.dot(in_tensor, weights_tensor)
#     return (activation > threshold).float()

# def T_link_or(*inputs):
#     # Or function described by ferreira
#     inputs_tuple = ()
#     weights = []
#     for inp in inputs:
#         inputs_tuple = inputs_tuple + (inp.reshape(1),)
#         weights.append(1.0)
#     in_tensor = torch.cat(inputs_tuple, 0)
#     weights_tensor = torch.tensor(weights, dtype=torch.float)
#     activation = torch.dot(in_tensor, weights_tensor)
#     return (activation > 1.0).float()

# def six_plexer(x, y, a, b, c, d):
#     # Ideal six multiplexer, for testing
#     registers = [a.item(), b.item(), c.item(), d.item()]
#     index = x.item() + 2 * y.item()
#     return registers[int(index)]

# def evaluate(individual):
#     global toolbox
#     func = toolbox.compile(individual)
    
#     fitness = 0
#     # control signals
#     control_values = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
#     # register signals
#     register_values = [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 
#                         [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
#                         [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0],
#                         [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0],
#                         [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], 
#                         [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0],
#                         [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0],
#                         [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    
#     for control in control_values:
#         X = torch.tensor(control[0], dtype=torch.float)
#         Y = torch.tensor(control[1], dtype=torch.float)
        
#         for reg_val in register_values:
#             A = torch.tensor(reg_val[0], dtype=torch.float)
#             B = torch.tensor(reg_val[1], dtype=torch.float)
#             C = torch.tensor(reg_val[2], dtype=torch.float)
#             D = torch.tensor(reg_val[3], dtype=torch.float)
            
#             func_out = func(x=X, y=Y, a=A, b=B, c=C, d=D)
#             index = int((1*control[0] + 2*control[1]))
#             if (func_out == reg_val[index]):
#                 fitness += 1
        
#     return fitness,

# # Create primitive set with D, T, Q functions
# pset = PrimitiveSet('Main', input_names=['x', 'y', 'a', 'b', 'c', 'd'])
# pset.add_nn_function(U, 1, name="U")
# pset.add_nn_function(D, 2, name="D")
# pset.add_nn_function(T, 3, name="T")


# # Create Individual class and fitness measurement
# creator.create("FitnessMax", base.Fitness, weights=(1,)) # maximize fitness
# creator.create("Individual", Chromosome, fitness=creator.FitnessMax)

# # GEP NN parameters
# h = 5 # 17       # head length
# n_genes = 1 # number of genes in a chromosome
# r = 10      # length of RNC arrays

# toolbox = Toolbox()
# toolbox.register("weight_gen", random.uniform, -2, 2)
# toolbox.register("thresh_gen", random.randint, 1, 1)
# toolbox.register("gene_gen", GeneNN, pset=pset, head_length=h, 
#                   dw_rnc_gen=toolbox.weight_gen, dw_rnc_array_length=r, 
#                   dt_rnc_gen=toolbox.thresh_gen, dt_rnc_array_length=r)
# toolbox.register("individual", creator.Individual, gene_gen=toolbox.gene_gen,
#                   n_genes=n_genes) #, linker=T_link_or)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("compile", compile_, pset=pset)
# toolbox.register("evaluate", evaluate)
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("mut_uniform", mutate_uniform, pset=pset, ind_pb=0.044, pb=1)
# toolbox.register("mut_dw", mutate_uniform_dw, ind_pb=0.044, pb=1)
# toolbox.register("mut_dt", mutate_uniform_dt, ind_pb=0.044, pb=1)
# toolbox.register("mut_tanspose_dw", transpose_dw, pb=0.1)
# toolbox.register("mut_transpose_dt", transpose_dt, pb=0.1)
# toolbox.register("mut_is_transpose", is_transpose, pb=0.1)
# toolbox.register("mut_ris_transpose", ris_transpose, pb=0.1)
# toolbox.register("mut_rncs_dw", mutate_rnc_array_dw, rnc_gen=toolbox.weight_gen, ind_pb='0.002p', pb=1)
# toolbox.register("mut_rncs_dt", mutate_rnc_array_dt, rnc_gen=toolbox.thresh_gen, ind_pb='0.002p', pb=1)
# #toolbox.register("mut_gene_tranpose", gene_transpose, pb=0.1)
# toolbox.register("cx_1p", crossover_one_point, pb=0.6)
# #toolbox.register("cx_2p", crossover_two_point, pb=0.6)
# #toolbox.register("cx_gene", crossover_gene, pb=0.1)
    
# # (NOTE Ryan) To parallelize, had to run from another module. See `parRunner.py`
# # Ran for 100 iterations, got 0 successes and took many hours to run (even with 8 runs at a time)
# def runGEPNN(iteration):
#     global toolbox # , D, T, Q, evaluate

#     n_pop = 50
#     n_gen = 2 # 2000
#     champs = 3
#     pop = toolbox.population(n=n_pop)
#     hof = tools.HallOfFame(champs) 
#     # start evolution
#     pop, log = gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
#                               stats=None, hall_of_fame=hof, verbose=False)
#     best_ind = hof[0]
#     fitness_best = evaluate(best_ind)[0]
    
#     print("Iteration {} got best individual [ {} ] with best fitness of {} ".format(iteration, best_ind, fitness_best))
#     if fitness_best == 64:
#         return 1 # success
#     return 0
# ------------------ 6-Plexer Problem (end) ---------------------- #

# ------------------ XOR Problem (start) ---------------------- #
"""
This block tests whether GEP-NN can discover solutions to the classic
XOR boolean function. As was done in the original Canida Ferreira paper.
For the most part, the same genetic operators, inputs, and functions are 
provided as building blocks for evolution.
"""
"""
def D(in1, in2, weights=[1.0,1.0], threshold=1.0):
    #in_tensor = torch.tensor([in1, in2], dtype=torch.float)
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    activation = torch.dot(in_tensor, weights_tensor)
    return (activation > threshold).float()
    
def T(in1, in2, in3, weights=[1.0,1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1), in3.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    activation = torch.dot(in_tensor, weights_tensor)
    return (activation > threshold).float()

def Q(in1, in2, in3, in4, weights=[1.0,1.0,1.0,1.0], threshold=1.0):
    in_tensor = torch.cat((in1.reshape(1), in2.reshape(1), in3.reshape(1), in4.reshape(1)), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    activation = torch.dot(in_tensor, weights_tensor)
    return (activation > threshold).float()

def evaluate(individual):
    func = toolbox.compile(individual)
    fitness = 0
    if(func(a=torch.tensor([0.0], dtype=torch.float), b=torch.tensor([0.0], dtype=torch.float)) == 0.0):
        fitness += 1
    if(func(a=torch.tensor([1.0], dtype=torch.float), b=torch.tensor([0.0], dtype=torch.float)) == 1.0):
        fitness += 1
    if(func(a=torch.tensor([0.0], dtype=torch.float), b=torch.tensor([1.0], dtype=torch.float)) == 1.0):
        fitness += 1
    if(func(a=torch.tensor([1.0], dtype=torch.float), b=torch.tensor([1.0], dtype=torch.float)) == 0.0):
        fitness += 1
        
    return fitness,


# Create primitive set with D, T, Q functions
pset = PrimitiveSet('Main', input_names=['a', 'b'])
pset.add_nn_function(D, 2, name="D")
pset.add_nn_function(T, 3, name="T")
pset.add_nn_function(Q, 4, name="Q")

# Create Individual class and fitness measurement
creator.create("FitnessMax", base.Fitness, weights=(1,)) # maximize fitness
creator.create("Individual", Chromosome, fitness=creator.FitnessMax)

# GEP NN parameters
h = 4       # head length
n_genes = 1 # number of genes in a chromosome
r = 10      # length of RNC arrays

toolbox = Toolbox()
toolbox.register("weight_gen", random.uniform, -2, 2)
toolbox.register("thresh_gen", random.randint, 1, 1)
toolbox.register("gene_gen", GeneNN, pset=pset, head_length=h, 
                  dw_rnc_gen=toolbox.weight_gen, dw_rnc_array_length=r, 
                  dt_rnc_gen=toolbox.thresh_gen, dt_rnc_array_length=r)
toolbox.register("individual", creator.Individual, gene_gen=toolbox.gene_gen,
                  n_genes=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", compile_, pset=pset)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3) # Verify ferreira 
toolbox.register("mut_uniform", mutate_uniform, pset=pset, ind_pb=0.061, pb=1)
toolbox.register("mut_dw", mutate_uniform_dw, ind_pb=0.061, pb=1)
toolbox.register("mut_dt", mutate_uniform_dt, ind_pb=0.061, pb=1)
toolbox.register("mut_tanspose_dw", transpose_dw, pb=0.1)
toolbox.register("mut_transpose_dt", transpose_dt, pb=0.1)
toolbox.register("mut_is_transpose", is_transpose, pb=0.1)
toolbox.register("mut_ris_transpose", ris_transpose, pb=0.1)
#toolbox.register("mut_rncs_dw", mutate_rnc_array_dw, rnc_gen=toolbox.weight_gen, ind_pb='0.5p', pb=0.05)
#toolbox.register("mut_rncs_dt", mutate_rnc_array_dt, rnc_gen=toolbox.thresh_gen, ind_pb='0.5p', pb=0.05)
toolbox.register("cx_1p", crossover_one_point, pb=0.7)

n_pop = 30
n_gen = 50

champs = 3

iters = 100
successes = 0
for i in range(iters):
    print("Running iteration: ", i)
    
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(champs) 
    # start evolution
    pop, log = gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                              stats=None, hall_of_fame=hof, verbose=False)
    best_ind = hof[0]
    
    print("got best: ", best_ind)
    fitness_best = evaluate(best_ind)[0]
    print("eval'd best: ", fitness_best)
    if (fitness_best == 4):
        successes += 1
        
        if successes < 6:
            file_name = "./myXOR_iter" + str(i) + ".png"
            rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}  
            export_expression_tree_nn(best_ind, rename_labels, file_name)

# Using "Compact" settings, got 31/100 and 27/100
# Using "Redundant" settings, got 70/100 
print("Out of {} iterations, got {} successes".format(iters, successes))
"""
# ------------------ XOR Problem (end) ---------------------- #



# --------------- NN Generation Testing (start) ---------------#
"""
This block tests the creation of a individual with weights and thresholds,
and its expression into a NN function. To test, I setup a monoGenic 
individual to perfectly match the solution found by Candida Ferreira in
her original GEP-NN paper. 
"""
# def D(in1, in2, weights=[1.0,1.0], threshold=1.0):
#     in_tensor = torch.tensor([in1, in2], dtype=torch.float)
#     weights_tensor = torch.tensor(weights, dtype=torch.float)
#     activation = torch.dot(in_tensor, weights_tensor)
#     if activation.item() > threshold:
#         return 1.0
#     return 0.0
    

# # def T(in1, in2):
# #     # (NOTE Ryan) Just making this function as a placeholder... 
# #     return in1 + in2

# # def Q(in1, in2):
# #     # (NOTE Ryan) Just making this function as a placeholder... 
# #     return in1 + in2

# pset = PrimitiveSet('Main', input_names=['a', 'b'])
# pset.add_nn_function(D, 2, name="D")

# toolbox = Toolbox()
# toolbox.register("weight_gen", random.uniform, -2, 2)
# toolbox.register("thresh_gen", random.randint, -1, 1)

# # Chromosome : list(Genes)
# xorGene = GeneNN(pset, 3, toolbox.weight_gen, 10, toolbox.thresh_gen, 10)
# print(xorGene)
# print("Dw after init: ", xorGene.dw)
# print("Dt after init: ", xorGene.dt)
# print("Dw array after init: ", xorGene.dw_rnc_array)
# print("Dt array after init: ",  xorGene.dt_rnc_array)

# print(pset.functions) # [class version of 'D']
# print(pset.terminals) # [class version of 'a', class version of 'b']

# # Restructure the gene to be exactly like example from Candida Ferreira paper
# xorGene._dt_rnc_array = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # All 1 thresholds
# xorGene._dw_rnc_array = [-1.978, 0.514, -0.465, 1.22, -1.686, -1.797, 0.197, 1.606, 0, 1.753]
# xorGene[-9:-3] = [3, 9, 3, 2, 5, 7]
# xorGene[-3:] = [1, 1, 1]
# xorGene[0:2] = [pset.functions[0], pset.functions[0], pset.functions[0]]
# xorGene[3:6] = [pset.terminals[0], pset.terminals[1], pset.terminals[0], pset.terminals[1]]
# # DDDabab 

# print(xorGene)
# print("Dw after changes: ", xorGene.dw)
# print("Dt after changes: ", xorGene.dt)
# print("Dw array after changes: ", xorGene.dw_rnc_array)
# print("Dt array after changes: ",  xorGene.dt_rnc_array)
# print("KExpr: ", xorGene.kexpression)

# # Now need to make a NN out of this... it should work "perfect" on XOR 
# # compiled = _compile_gene(xorGene, pset)
# # print(compiled(a=1, b=1)) # 0
# # print(compiled(a=0, b=1)) # 1 
# # print(compiled(a=1, b=0)) # 1
# # print(compiled(a=0, b=0)) # 0 

# rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}  
# export_expression_tree_nn(xorGene, rename_labels, './ferreiraXOR.png')
# --------------- NN Generation Testing (end) -----------------#