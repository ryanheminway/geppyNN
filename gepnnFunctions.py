# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:12:56 2023

@author: hemin
"""

import torch

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

# def nn_add(in1, in2, weights=[1.0, 1.0], threshold=1.0):
#     a1 = in1 * weights[0]
#     a2 = in2 * weights[1]
#     return a1 + a2

# def nn_sub(in1, in2, weights=[1.0, 1.0], threshold=1.0):
#     a1 = in1 * weights[0]
#     a2 = in2 * weights[1]
#     return a1 - a2

# def nn_mult(in1, in2, weights=[1.0, 1.0], threshold=1.0):
#     a1 = in1 * weights[0]
#     a2 = in2 * weights[1]
#     return a1 * a2

def nn_add(*inputs):
    # Inputs must be a dynamica list where last entry is the threshold value,
    # second to last value is a list of weights, and remaining initial entries 
    # are the input values
    threshold = inputs[-1]
    weights = inputs[-2]
    input_vals = inputs[:-2]
    if len(weights) < len(inputs):
        weights = [1.0 for x in inputs]
        
    a = 0
    for i in range(len(input_vals)):
        a += (input_vals[i] * weights[i])
    
    return a

def nn_sub(*inputs):
    # Inputs must be a dynamica list where last entry is the threshold value,
    # second to last value is a list of weights, and remaining initial entries 
    # are the input values
    threshold = inputs[-1]
    weights = inputs[-2]
    input_vals = inputs[:-2]
    if len(weights) < len(inputs):
        weights = [1.0 for x in inputs]
        
    a = 0
    for i in range(len(input_vals)):
        a -= (input_vals[i] * weights[i])
    
    return a

def nn_mult(*inputs):
    # Inputs must be a dynamica list where last entry is the threshold value,
    # second to last value is a list of weights, and remaining initial entries 
    # are the input values
    threshold = inputs[-1]
    weights = inputs[-2]
    input_vals = inputs[:-2]
    if len(weights) < len(inputs):
        weights = [1.0 for x in inputs]
        
    a = 0
    for i in range(len(input_vals)):
        a = a * (input_vals[i] * weights[i])
    
    return a

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