
# coding: utf-8

# # Purity State Preparation for Qubits
# Quantum Neural Network (QNN) that prepare a dataset of density matrices with uniform purity

import os
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from pyquil import Program
from pyquil.gates import *
from pyquil.api import WavefunctionSimulator

import qutip

from scipy.optimize import minimize

wf_sim = WavefunctionSimulator()

# ==================== Network and Outputs ====================

def sp_network(params):
    circuit = Program()
    circuit += RZ(params[0], 0)
    circuit += RY(params[1], 0)
    circuit += RZ(params[2], 0)
    circuit += RZ(params[3], 1)
    circuit += RY(params[4], 1)
    circuit += RZ(params[5], 1)
    circuit += CNOT(1,0)
    circuit += RZ(params[6], 0)
    circuit += RY(params[7], 1)
    circuit += CNOT(0,1)
    circuit += RY(params[8], 1)
    circuit += CNOT(1,0)
    circuit += RZ(params[9], 0)
    circuit += RY(params[10], 0)
    circuit += RZ(params[11], 0)
    circuit += RZ(params[12], 1)
    circuit += RY(params[13], 1)
    circuit += RZ(params[14], 1)

    return circuit

def partial_trace(state):
    """ Take a state as a list of amplitudes and turn it into a density matrix, 
    with last qubit traced out"""

    qstate = state[0] * qutip.states.ket([0,0]) \
           + state[1] * qutip.states.ket([0,1]) \
           + state[2] * qutip.states.ket([1,0]) \
           + state[3] * qutip.states.ket([1,1]) 
    rho_output = np.array(qstate.ptrace(1).data.todense())
    return rho_output


def get_rho(params):
    """ Take parameters for the circuit and a wf simulator,
    and return the density matrix (as a np.array) corresponding to qubit 0"""

    circuit = sp_network(params)
    state = list(wf_sim.wavefunction(circuit))
    rho_output = partial_trace(state)
    
    return rho_output

# ==================== Cost function ====================

def purity_fct(rho):
    return np.real(np.trace(rho @ rho))

def entropy_fct(rho, eps=1e-8):
    return - np.real(np.trace(rho @ sp.linalg.logm(rho)))

def purity_mse(rho_output, rho_input):
    return np.mean((purity_fct(rho_output) - purity_fct(rho_input))**2)

def entropy_mse(rho_output, rho_input):
    return np.mean((entropy_fct(rho_output) - entropy_fct(rho_input))**2)

def trace_distance(rho1, rho2):
    return np.mean(np.real(np.linalg.eigvalsh(rho1 - rho2)**2))


def cost(params, property_mse, rho_input, reg):
    rho_output = get_rho(params)
    
    return trace_distance(rho_output, rho_input) + reg * property_mse(rho_output, rho_input)

# ==================== Training ====================

def prepare_states(rhos, property_name="purity", n_qumodes=1, cutoff=2, n_iters_min=None, n_iters_max=None, property_reg=1):
    if property_name == "purity":
        property_fct = purity_fct
        property_mse = purity_mse
    elif property_name == "entropy":
        property_fct = entropy_fct
        property_mse = entropy_mse

    params = np.random.normal(size=15, scale=0.01)
    params_list = []
    cost_list = []
    trace_distance_list = []
    property_mse_list = []
    property_list = []
    n_iters = n_iters_max

    for i, rho in enumerate(rhos):
        print("\n\n~~~~~~~~~~~~~~~~~~~~~~ Density Matrix {}/{} ~~~~~~~~~~~~~~~~~~~~~~\n".format(i+1, len(rhos)))
        print("Property: {:.7f}\n\n".format(property_fct(rho)))

        result = minimize(lambda params: cost(params, property_mse, rho, property_reg), params, method='L-BFGS-B')
        params_predict = result['x']
        curr_cost = result['fun']
        curr_rho = get_rho(params_predict)
        curr_property = property_fct(curr_rho)

        trace_distance_list.append(trace_distance(curr_rho, rho))
        property_mse_list.append(property_mse(curr_rho, rho))
        
        params_list.append(params_predict)
        cost_list.append(curr_cost)
        property_list.append(curr_property)
        
        print('\nCost: {: .7f} −− Property: {:.7f}'.format(cost_list[-1], property_list[-1]))
        
        params = params_predict # we change the initialization for the next step

    return params_list, trace_distance_list, property_mse_list

