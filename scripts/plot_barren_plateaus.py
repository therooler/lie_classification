import jax

jax.config.update('jax_enable_x64', True)
import pennylane as qml
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('science')

import os
import itertools as it

SHOW = True


### CIRCUITS ###
def layer_a14(params, nqubits: int):
    for i, wire in enumerate(range(0, nqubits, 2)):
        qml.PauliRot(params[0, i], pauli_word='XX', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i], pauli_word='YY', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i], pauli_word='XY', wires=(wire, wire + 1))

    for i, wire in enumerate(range(1, nqubits - 1, 2)):
        qml.PauliRot(params[0, i + nqubits // 2], pauli_word='XX', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i + nqubits // 2], pauli_word='YY', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i + nqubits // 2], pauli_word='XY', wires=(wire, wire + 1))


def circuit_a14(params, nqubits: int, depth: int, observable):
    for d in range(depth):
        layer_a14(params[d], nqubits)

    return qml.expval(observable)


def layer_a5_circ(params, nqubits: int):
    for i, wire in enumerate(range(0, nqubits, 2)):
        qml.PauliRot(params[0, i], pauli_word='XY', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i], pauli_word='YZ', wires=(wire, wire + 1))

    for i, wire in enumerate(range(1, nqubits - 1, 2)):
        qml.PauliRot(params[0, i + nqubits // 2], pauli_word='XY', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i + nqubits // 2], pauli_word='YZ', wires=(wire, wire + 1))
    qml.PauliRot(params[0, nqubits - 1], pauli_word='XY', wires=(nqubits - 1, 0))
    qml.PauliRot(params[1, nqubits - 1], pauli_word='YZ', wires=(nqubits - 1, 0))


def circuit_a5_circ(params, nqubits: int, depth: int, observable):
    for d in range(depth):
        layer_a5_circ(params[d], nqubits)

    return qml.expval(observable)


### DATA ###

def perform_runs(algebra: str, nqubits: int, depth: int, seed: int = 1234, nruns: int = 100):
    assert not nqubits % 2, 'number of qubits must be even'
    # set paths
    data_path = f'./data/barren_plateaus_nruns_{nruns}/'
    filename = f'grads_{algebra}_nq_{nqubits}_d_{depth}_seed_{seed}.csv'
    if not os.path.exists(data_path + filename):
        print(f"Getting data for N = {nqubits} - Depth = {depth} - Seed {seed}")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        # Set up devices and seed
        dev = qml.device("default.qubit", wires=nqubits)
        np.random.seed(seed)
        if algebra == 'a14':
            qnode = qml.QNode(circuit_a14, dev, interface='jax', diff_method="adjoint")
            parameters = np.random.rand(nruns, depth, 3, nqubits - 1) * np.pi

        elif algebra == 'a5_circ':
            qnode = qml.QNode(circuit_a5_circ, dev, interface='jax', diff_method="adjoint")
            parameters = np.random.rand(nruns, depth, 2, nqubits) * np.pi
        else:
            raise NotImplementedError
        # two local observable
        obs = qml.PauliZ(wires=0) @ qml.PauliZ(wires=1)

        grad_fn = jax.grad(lambda x: qnode(x, nqubits, depth, obs), argnums=(0,))
        # Calculate all runs in one vmap
        batched_grad_fn = jax.vmap(grad_fn)
        grads = np.array(batched_grad_fn(parameters))
        grads = grads.reshape((nruns, -1)).T
        df = pd.DataFrame(grads)
        df.to_csv(data_path + filename)
    else:
        print("Data already exists, skipping...")


def get_data(hyps, algebra: str, seed: int = 1233, nruns: int = 100):
    for (n, d) in hyps:
        perform_runs(algebra, n, d, seed, nruns=nruns)


### PLOTTING ###

def plot_data(hyps, algebra, seed: int = 1233, nruns: int = 100, y_axis = True):
    data_path = f'./data/barren_plateaus_nruns_{nruns}/'
    fig_path = f'./data/barren_plateaus_nruns_{nruns}/figures/'
    indices = np.stack(np.unravel_index(range(len(nqubit_list) * len(depth_list)),
                                        shape=(len(nqubit_list), len(depth_list)))).T
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(fig_path + f'{algebra}_grads_mean.npy'):

        grads_mean = np.zeros((len(nqubit_list), len(depth_list)))
        grads_var = np.zeros((len(nqubit_list), len(depth_list)))
        for i, (n, d) in enumerate(hyps):
            idx = indices[i]
            file_path = f'grads_{algebra}_nq_{n}_d_{d}_seed_{seed}.csv'
            try:
                grads = pd.read_csv(data_path + file_path)
                grads_mean[idx[0], idx[1]] = grads.mean(axis=1).iloc[0]
                grads_var[idx[0], idx[1]] = grads.var(axis=1).iloc[0]
            except FileNotFoundError:
                print("File not found, skipping for now...")

        np.save(fig_path + f'{algebra}_grads_mean.npy', grads_mean)
        np.save(fig_path + f'{algebra}_grads_var.npy', grads_var)
    else:
        print("Loading data...")
        grads_mean = np.load(fig_path + f'{algebra}_grads_mean.npy')
        grads_var = np.load(fig_path + f'{algebra}_grads_var.npy')

    fig, axs = plt.subplots(1, 1)
    if y_axis:
        fig.set_size_inches(2.7, 3.5)
        axs.set_ylabel(r'Var[$\partial_{\theta_1} C(\theta)$]')
        axs.legend()
    else:
        fig.set_size_inches(2.3, 3.5)
        axs.set_yticklabels([])
    for i, n, in enumerate(nqubit_list):
        axs.plot(depth_list, grads_var[i, :], label=f'N = {n}')
    axs.set_xlabel('depth')
    axs.set_yscale('log')
    axs.set_ylim([1e-6, 1e-1])


    plt.tight_layout()

    fig.savefig(fig_path + f'{algebra}_scaling_variance.pdf')

    if SHOW:
        plt.show()


def plot_data_combined(hyps, algebra_1, algebra_2, seed: int = 1233, nruns: int = 100):

    fig_path = f'./data/barren_plateaus_nruns_{nruns}/figures/'

    grads_var_1 = np.load(fig_path + f'{algebra_2}_grads_var.npy')
    grads_var_2 = np.load(fig_path + f'{algebra_1}_grads_var.npy')

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(4, 3.5)
    for i, n, in enumerate(nqubit_list):
        axs[0].plot(depth_list, grads_var_1[i, :], label=rf'$n = $ {n}')
    axs[0].set_xlabel('$L$')
    axs[0].set_ylabel(r'Var[$\partial_{\theta_1} C(\theta)$]')
    axs[0].set_yscale('log')
    axs[0].set_ylim([1e-6, 1e-1])
    axs[0].set_title(r'(a) $\mathfrak{a}_{5}^\circ$', y=-0.25)
    axs[0].tick_params(axis="x", which='both', top=False, right=False)
    axs[0].tick_params(axis="y", which='both', top=False, right=False)
    axs[0].legend(prop={'size': 8})
    for i, n, in enumerate(nqubit_list):
        axs[1].plot(depth_list, grads_var_2[i, :], label=rf'$n =$ {n}')
    fig.subplots_adjust(wspace=0.0)
    axs[1].set_xlabel('$L$')
    axs[1].set_yscale('log')
    axs[1].set_ylim([1e-6, 1e-1])
    axs[1].set_title(r'(b) $\mathfrak{a}_{14}$', y=-0.25)
    axs[1].set_yticks(())
    axs[1].tick_params(axis="x", which='both', top=False, right=False)
    axs[1].tick_params(axis="y", which='both', top=False, right=False, left=False)
    # axs[1].legend()
    # plt.tight_layout()
    fig.savefig(fig_path + f'{algebra_1}_{algebra_2}_scaling_variance.pdf')

    if SHOW:
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', default=None)
    args = parser.parse_args()
    nqubit_list = [4, 6, 8, 10, 12, 14, 16]
    depth_list = list(range(2, 26, 2))
    hyps = list(it.product(nqubit_list, depth_list))
    print(f"Number of hyps= {len(hyps)}")
    NRUNS = 1000
    if args.instance is None:
        # ALGEBRA = 'a5_circ'
        # get_data(hyps, algebra=ALGEBRA, nruns=NRUNS)
        # plot_data(hyps, algebra=ALGEBRA, nruns=NRUNS)

        # ALGEBRA = 'a14'
        # get_data(hyps, algebra=ALGEBRA, nruns=NRUNS)
        # plot_data(hyps, algebra=ALGEBRA, nruns=NRUNS, y_axis=False)

        plot_data_combined(hyps, algebra_1='a14', algebra_2='a5_circ', nruns=NRUNS)

    else:
        instance = int(args.instance)
        ALGEBRA = 'a14'
        get_data((hyps[instance],), algebra=ALGEBRA, nruns=NRUNS)

        ALGEBRA = 'a5_circ'
        get_data((hyps[instance],), algebra=ALGEBRA, nruns=NRUNS)
