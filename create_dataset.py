import os
import argparse

import tensorflow as tf

import strawberryfields as sf
from strawberryfields.ops import *

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import state_preparation as state_prep

def purity_fct(rho):
        return np.real(np.trace(rho @ rho))

def generate_density_matrices(n_samples, size_hilbert, step_size=0.01):
    """Generate n_samples density matrices with a uniform distribution of purity
    Input:
        - n_samples (int): number of density matrices to generate
        - size_hilbert (int): dimension of the hilbert space of the density matrices
        - step_size (float): for each purity p, sample dm with purity in [p, p+step_size]
    Output:
        - rhos: array of dimension (n_samples, size_hilbert, size_hilbert)
        - purities: array of dimension (n_samples,)
    """

    print("Generate density matrices...")
    rhos = []
    list_p = np.arange(1./size_hilbert, 1, step_size)
    for i, p in enumerate(list_p): # for each purity
        while len(rhos) < (i+1)*n_samples//len(list_p):
            # print(i, len(rhos), (i+1)*n_samples//len(list_p))
            # generate size_hilbert λi s.t. sum(λi)=1
            curr_max = 1
            lambda_list = []
            for j in range(size_hilbert-1):
                λ = np.random.uniform(0,curr_max)
                curr_max -= λ
                lambda_list.append(λ)
            lambda_list.append(1-sum(lambda_list)) 
            lambda_list = np.array(lambda_list)
            purity = sum(lambda_list**2)
            if p <= purity <= p+step_size:
                U = sp.stats.unitary_group.rvs(size_hilbert)
                rhos.append(U@np.diag(np.random.permutation(lambda_list))@np.conj(U).T)
    rhos = np.array(rhos)
    purities = np.array([purity_fct(rho) for rho in rhos])
    print(rhos.shape)
    return rhos, purities

def create_dataset(n_qumodes, n_samples, cutoff, n_iters):
    size_hilbert = cutoff**n_qumodes

    rhos, purities = generate_density_matrices(n_samples, size_hilbert)
    plt.hist(purities)
    plt.show()

    list_params = state_prep.prepare_state(rhos, n_qumodes, cutoff, n_iters)
    
    return list_params, rhos, purities

if __name__ == '__main__':
    data_dir = "data"
    
    parser = argparse.ArgumentParser(
    description='Create and save a dataset of prepared density matrices with a uniform distribution of purity'
    )
    parser.add_argument('n_qumodes', type=int, help='Number of qumodes')
    parser.add_argument('n_samples', type=int, help='Number of density matrices to prepare')
    parser.add_argument('cutoff', type=int, default=3, help='Number of Fock state basis elements to consider')
    parser.add_argument('n_iter', type=int, default=3000, help='Number of iterations of the state preparation procedure')

    args = parser.parse_args()

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    list_params, rhos, purities = create_dataset(args.n_qumodes, args.n_samples, args.cutoff, args.n_iter)
        
    np.save(os.path.join(data_dir, "list_params.npy"), list_params)
    np.save(os.path.join(data_dir, "rhos.npy"), rhos)
    np.save(os.path.join(data_dir, "purities.npy"), purities)


