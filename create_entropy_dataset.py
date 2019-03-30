import os
import argparse

import tensorflow as tf

import strawberryfields as sf
from strawberryfields.ops import *
from qutip import rand_dm

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import state_preparation as state_prep

def entropy_fct(rho):
    return - np.real(np.trace(rho @ sp.linalg.logm(rho)))

def generate_density_matrices(n_samples, size_hilbert, step_size=0.01):
    """Generate n_samples density matrices with a uniform distribution of entropy
    Input:
        - n_samples (int): number of density matrices to generate
        - size_hilbert (int): dimension of the hilbert space of the density matrices
        - step_size (float): for each entropy p, sample dm with entropy in [p, p+step_size]
    Output:
        - rhos: array of dimension (n_samples, size_hilbert, size_hilbert)
        - entropies: array of dimension (n_samples,)
    """

    print("Generate density matrices...")
    epsilon = 0.05
    list_entropies = np.arange(0, 1, epsilon)
    rhos = []
    entropies = []
    for i, s in enumerate(list_entropies): # for each entropy
        print(s)
        while len(rhos) <= (i+1)*(int(n_samples/len(list_entropies))):
            rho = rand_dm(size_hilbert, density=1).data.toarray()
            entropy = entropy_fct(rho)
            if s <= entropy <= s+epsilon:
                rhos.append(rho)
                entropies.append(entropy)
    rhos = np.array(rhos)
    entropies = np.array(entropies)

    return rhos, entropies

def create_dataset(n_qumodes, n_samples, cutoff, n_iters):
    size_hilbert = cutoff**n_qumodes

    rhos, entropies = generate_density_matrices(n_samples, size_hilbert)
    plt.hist(entropies)
    plt.show()

    list_params = state_prep.prepare_state(rhos, n_qumodes, cutoff, n_iters)
    
    return list_params, rhos, entropies

if __name__ == '__main__':
    data_dir = "data_entropies"
    
    parser = argparse.ArgumentParser(
    description='Create and save a dataset of prepared density matrices with a uniform distribution of entropy'
    )
    parser.add_argument('n_qumodes', type=int, help='Number of qumodes')
    parser.add_argument('n_samples', type=int, help='Number of density matrices to prepare')
    parser.add_argument('cutoff', type=int, default=3, help='Number of Fock state basis elements to consider')
    parser.add_argument('n_iter', type=int, default=3000, help='Number of iterations of the state preparation procedure')

    args = parser.parse_args()

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    list_params, rhos, entropies = create_dataset(args.n_qumodes, args.n_samples, args.cutoff, args.n_iter)
        
    np.save(os.path.join(data_dir, "list_params.npy"), list_params)
    np.save(os.path.join(data_dir, "rhos.npy"), rhos)
    np.save(os.path.join(data_dir, "entropies.npy"), entropies)


