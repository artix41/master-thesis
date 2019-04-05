import os
import argparse

from qutip import rand_dm

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

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
    list_entropies = np.arange(0, np.log(size_hilbert), step_size)
    rhos = []
    entropies = []
    for i, s in enumerate(list_entropies): # for each entropy
        print(s)
        while len(rhos) < (i+1)*n_samples//len(list_entropies):
            rho = rand_dm(size_hilbert, density=1).data.toarray()
            entropy = entropy_fct(rho)
            if s <= entropy <= s+step_size:
                rhos.append(rho)
                entropies.append(entropy)
    rhos = np.array(rhos)
    entropies = np.array(entropies)

    return rhos, entropies

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(
    description='Create and save a dataset of prepared density matrices with a uniform distribution of entropy'
    )
    parser.add_argument('type', type=str, default="cv", help='Continuous-variables ("cv") or Discrete ("discrete")')
    parser.add_argument('n_qumodes', type=int, help='Number of qumodes')
    parser.add_argument('n_samples', type=int, help='Number of density matrices to prepare')
    parser.add_argument('data_dir', type=str, help='Output directory for storing the generated data')
    parser.add_argument('cutoff', type=int, default=3, help='Number of Fock state basis elements to consider')
    parser.add_argument('n_iters_min', type=int, default=1000, help='Minimal number of iterations of the state preparation procedure')
    parser.add_argument('n_iters_max', type=int, default=1000, help='Maxima number of iterations of the state preparation procedure')

    args = parser.parse_args()

    data_dir = args.data_dir
    n_qumodes = args.n_qumodes
    n_samples = args.n_samples
    cutoff = args.cutoff
    n_iters_min = args.n_iters_min
    n_iters_max = args.n_iters_max

    size_hilbert = cutoff**n_qumodes

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if args.type == 'discrete':
        import discrete_state_preparation as state_prep
    if args.type == 'cv':
        import cv_state_preparation as state_prep

    rhos, entropies = generate_density_matrices(n_samples, size_hilbert)
    plt.hist(entropies)
    plt.show()

    list_params = state_prep.prepare_states(rhos, "entropy", n_qumodes, cutoff, n_iters_min, n_iters_max)
        
    np.save(os.path.join(data_dir, "list_params.npy"), list_params)
    np.save(os.path.join(data_dir, "rhos.npy"), rhos)
    np.save(os.path.join(data_dir, "entropies.npy"), entropies)


