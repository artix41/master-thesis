import tensorflow as tf

import strawberryfields as sf
from strawberryfields.ops import *

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

def purity_fct(rho, backend="tf"):
    if backend == "np":
        print("test")
        return np.real(np.trace(rho @ rho))
    if backend == "tf":
        return tf.real(tf.trace(rho @ rho))
    else:
        raise ValueError("Backend must be in ['tf', 'np']. Currently {}".format(backend))

def entropy_fct(rho, backend="tf", eps=1e-8):
    if backend == "np":
        return - np.real(np.trace(rho @ sp.linalg.logm(rho)))
    if backend == "tf":
        eig = tf.linalg.eigvalsh(rho)
        return - tf.real(tf.reduce_sum(eig * tf.log(eig + eps)))
    else:
        raise ValueError("Backend must be in ['tf', 'np']. Currently {}".format(backend))

def Interferometer(theta, phi, rphi, q):
	# parameterised interferometer acting on N qumodes
    # theta is a list of length N(N-1)/2
    # phi is a list of length N(N-1)/2
    # rphi is a list of length N-1
	# q is the list of qumodes the interferometer is to be applied to
    N = len(q)

    if N == 1:
        # the interferometer is a single rotation
        Rgate(rphi[0]) | q[0]
        return

    n = 0 # keep track of free parameters

    # Apply the Clements beamsplitter array
    # The array depth is N
    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            #skip even or odd pairs depending on layer
            if (l+k)%2 != 1:
                BSgate(theta[n], phi[n]) | (q1, q2)
                n += 1

    # apply the final local phase shifts to all modes except the last one
    for i in range(len(q)-1):
        Rgate(rphi[i]) | q[i]

def layer(i, q, params):
    sq_r, sq_phi, d_r, d_phi, inter_theta, inter_phi, inter_rphi, kappa = tuple(params)

    Interferometer(inter_theta[2*i], inter_phi[2*i], inter_rphi[2*i], q)
    
    for j in range(len(q)):
        Sgate(sq_r[i,j], sq_phi[i,j]) | q[j]
        
    Interferometer(inter_theta[2*i+1], inter_phi[2*i+1], inter_rphi[2*i+1], q)
    
    for j in range(len(q)):
        Dgate(d_r[i,j], d_phi[i,j]) | q[j]
        
    for j in range(len(q)):
        Kgate(kappa[i,j]) | q[j]

    return q

def state_preparation_network(q, n_layers, parameters):
    for i in range(n_layers):
        layer(i, q, parameters)

def trace_distance(rho1, rho2):
    return tf.reduce_mean(tf.square(tf.real(tf.linalg.eigvalsh(rho1 - rho2))))
    # return tf.reduce_mean(tf.square(tf.real(tf.norm(rho1 - rho2))))


def purity_mse(rho1, rho2, backend="tf"):
    return tf.square(purity_fct(rho1, backend) - purity_fct(rho2, backend))

def entropy_mse(rho1, rho2, backend="tf"):
    return tf.square(entropy_fct(rho1, backend) - entropy_fct(rho2, backend))

def prepare_states(rhos, property_name="purity", n_qumodes=1, cutoff=3, n_iters_min=1000, n_iters_max=3000, n_layers=20, property_reg=10, lambda_reg=0, lr=2e-3):
    if property_name == "purity":
        property_fct = purity_fct
        property_mse = purity_mse
    elif property_name == "entropy":
        property_fct = entropy_fct
        property_mse = entropy_mse

    size_system = n_qumodes*2
    size_hilbert = cutoff**n_qumodes
    # ================= Placeholders =================

    rho_input = tf.placeholder(tf.complex64, [size_hilbert, size_hilbert])
    lr_placeholder = tf.placeholder(tf.float32)

    # ================= Parameters ===================

    passive_std = 0.1
    active_std = 0.001

    # squeeze gate
    sq_r = tf.Variable(tf.random_normal(shape=[n_layers, size_system], stddev=active_std))
    sq_phi = tf.Variable(tf.random_normal(shape=[n_layers, size_system], stddev=passive_std))

    # displacement gate
    d_r = tf.Variable(tf.random_normal(shape=[n_layers, size_system], stddev=active_std))
    d_phi = tf.Variable(tf.random_normal(shape=[n_layers, size_system], stddev=passive_std))

    # interferometer
    inter_theta = tf.Variable(tf.random_normal(shape=[n_layers*2, int(size_system*(size_system-1)/2)], stddev=passive_std))
    inter_phi = tf.Variable(tf.random_normal(shape=[n_layers*2, int(size_system*(size_system-1)/2)], stddev=passive_std))
    inter_rphi = tf.Variable(tf.random_normal(shape=[n_layers*2, size_system-1], stddev=passive_std))

    # kerr gate
    kappa = tf.Variable(tf.random_normal(shape=[n_layers, size_system], stddev=active_std))

    parameters = [sq_r, sq_phi, d_r, d_phi, inter_theta, inter_phi, inter_rphi, kappa]

    # ================== Circuit ===================

    print("Prepare circuit...")
    engine, q = sf.Engine(n_qumodes*2)
    with engine:
        state_preparation_network(q, n_layers, parameters)

    state = engine.run('tf', cutoff_dim=cutoff, eval=False, modes=range(n_qumodes))

    if n_qumodes == 1:
        rho_output = state.dm()
    if n_qumodes == 2:
        rho_output = tf.reshape(tf.einsum('ijkl->ikjl', state.dm()), (size_hilbert, size_hilbert))
    elif n_qumodes > 2:
        raise ValueError("n_qumodes > 2 not yet supported")

    property_output = property_fct(rho_output)

    trace_output = tf.real(tf.trace(rho_output))

    # ============== Cost and optimizer =============

    print("Prepare cost and optimizer...")
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    cost = trace_distance(rho_output, rho_input) + property_reg * property_mse(rho_output, rho_input)

    optimiser = tf.train.AdamOptimizer(learning_rate=lr_placeholder)
    min_cost = optimiser.minimize(cost)

    # ================== Training ====================

    print("Prepare session...")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("trace before: ", sess.run(tf.trace(rho_output)))

    print("Start training...")

    params_list = []
    trace_distance_list = []
    property_mse_list = []
    n_iters = n_iters_max
    for i, rho in enumerate(rhos):
        print("\n\n~~~~~~~~~~~~~~~~~~~~~~ Density Matrix {}/{} ~~~~~~~~~~~~~~~~~~~~~~\n".format(i+1, len(rhos)))
        print("Property: {:.7f}\n\n".format(property_fct(rho, "np")))

        cost_list = []
        property_list = []

        if i == 1:
            n_iters = n_iters_min        
        for j in range(n_iters):
            _, curr_cost = sess.run([min_cost, cost], feed_dict={rho_input: rho, lr_placeholder: lr})
            curr_property, curr_rho, trace_dm = sess.run([property_output, rho_output, trace_output])

            cost_list.append(curr_cost)
            property_list.append(curr_property)
            print('Step {}/{} −− Cost: {: .7f} −− Property: {:.7f} −− Trace: {:.7f}'.format(j, n_iters, cost_list[-1], property_list[-1], trace_dm), end="\r")

        trace_distance_list.append(sess.run(trace_distance(curr_rho, rho)))
        property_mse_list.append(sess.run(property_mse(curr_rho, rho, "np")))
        print("Property MSE:", property_mse_list[-1])
        print("Trace distance:", trace_distance_list[-1])
        
        print('\nCost: {: .7f} −− Property: {:.7f}'.format(cost_list[-1], property_list[-1]))
        params_list.append(sess.run(parameters))

    return params_list, trace_distance_list, property_mse_list