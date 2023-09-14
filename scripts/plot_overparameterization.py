import jax
import optax
import scipy

import pennylane as qml
import numpy as np
import pandas as pd
import scipy.sparse.linalg as sspla
import matplotlib.pyplot as plt

import os
import itertools

plt.style.use('science')

SHOW = True
LEARNING_RATE = 5e-3
DATA = True
paulis = {'I': scipy.sparse.csr_matrix(np.eye(2).astype(np.complex64)),
          'x': scipy.sparse.csr_matrix(np.array([[0, 1], [1, 0]]).astype(np.complex64)),
          'y': scipy.sparse.csr_matrix(np.array([[0, -1j], [1j, 0]]).astype(np.complex64)),
          'z': scipy.sparse.csr_matrix(np.array([[1, 0], [0, -1]]).astype(np.complex64))}


def commutes(A: np.ndarray, B: np.ndarray):
    return np.allclose(A @ B, B @ A)


def commutator(A: np.ndarray, B: np.ndarray):
    return A @ B - B @ A


def exact_diagonalization_ising(nsites, J, g, interactions):
    H = scipy.sparse.csr_matrix((int(2 ** nsites), (int(2 ** nsites))), dtype=complex)
    for n, interac in enumerate(interactions):
        tprod = ["I" for _ in range(nsites)]
        for loc in interac:
            tprod[loc] = 'z'
        p = paulis[tprod[0]]
        for op in range(1, nsites):
            p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
        H += J * p
    for loc in range(nsites):
        tprod = ["I" for _ in range(nsites)]
        tprod[loc] = 'x'
        p = paulis[tprod[0]]
        for op in range(1, nsites):
            p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
        H += g * p
    return sspla.eigsh(H, which='SA', k=2)[0], H


def exact_symmetry_generator(nsites: int, p_sym: str):
    if p_sym in ['x', 'y', 'z']:
        tprod = [p_sym] * nsites
        p = paulis[tprod[0]]
        for op in range(1, nsites):
            p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
    elif p_sym in list("".join(x) for x in itertools.product('xyz', repeat=2)) and not nsites % 2:
        tprod = [p_sym[0], p_sym[1]] * nsites
        p = paulis[tprod[0]]
        for op in range(1, nsites):
            p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
    elif p_sym in list("".join(x) for x in itertools.product('xyz', repeat=3)) and not nsites % 2:
        tprod = [p_sym[0], p_sym[1], p_sym[2]] * nsites
        p = paulis[tprod[0]]
        for op in range(1, nsites):
            p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
    else:
        raise NotImplementedError

    return p


def exact_diagonalization_heisenberg(nsites, delta, interactions):
    H = scipy.sparse.csr_matrix((int(2 ** nsites), (int(2 ** nsites))), dtype=complex)
    for term in ['x', 'y', 'z']:
        for n, interac in enumerate(interactions):
            tprod = ["I" for _ in range(nsites)]
            for loc in interac:
                tprod[loc] = term
            p = paulis[tprod[0]]
            for op in range(1, nsites):
                p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
            if term == 'z':
                H += delta * p
            else:
                H += p

    return sspla.eigsh(H, which='SA', k=2)[0], H


def exact_diagonalization_random_H(nsites, complex=True):
    rng = np.random.RandomState(1234)

    H = rng.rand(int(2 ** nsites),
                 int(2 ** nsites)) \
        + 1j * rng.rand(int(2 ** nsites),
                        int(2 ** nsites)) * complex
    H = H + H.conj().T

    return np.linalg.eigh(H)[0], H


def exact_diagonalization_a6(nsites, delta, interactions):
    H = scipy.sparse.csr_matrix((int(2 ** nsites), (int(2 ** nsites))), dtype=complex)
    for term in ['xx', 'yz', 'zy']:
        for n, interac in enumerate(interactions):
            tprod = ["I" for _ in range(nsites)]
            tprod[interac[0]] = term[0]
            tprod[interac[1]] = term[1]
            p = paulis[tprod[0]]
            for op in range(1, nsites):
                p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
            if term == 'xx':
                H += delta * p
            else:
                H += p

    return sspla.eigsh(H, which='SA', k=2)[0], H


def exact_diagonalization_a5(nsites, delta, interactions):
    H = scipy.sparse.csr_matrix((int(2 ** nsites), (int(2 ** nsites))), dtype=complex)
    for term in ['xy', 'yz']:
        for n, interac in enumerate(interactions):
            tprod = ["I" for _ in range(nsites)]
            tprod[interac[0]] = term[0]
            tprod[interac[1]] = term[1]
            p = paulis[tprod[0]]
            for op in range(1, nsites):
                p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
            else:
                H += p

    return sspla.eigsh(H, which='SA', k=2)[0], H


def get_zero_state(nqubits):
    device = qml.device('default.qubit', wires=nqubits)

    @qml.qnode(device)
    def circuit():
        for i in range(nqubits):
            qml.Identity(wires=i)
        return qml.state()

    return circuit()


def get_plus_state(nqubits):
    device = qml.device('default.qubit', wires=nqubits)

    @qml.qnode(device)
    def circuit():
        for i in range(nqubits):
            qml.Hadamard(wires=i)
        return qml.state()

    return circuit()


def get_bell_state(nqubits):
    device = qml.device('default.qubit', wires=nqubits)

    @qml.qnode(device)
    def circuit():
        for i in range(0, nqubits, 2):
            qml.Hadamard(i)
            qml.CNOT(wires=(i, i + 1))
        return qml.state()

    return circuit()


def get_random_state(nqubits):
    state = np.random.rand(2 ** nqubits) + 1j * np.random.rand(2 ** nqubits)
    return state / np.linalg.norm(state)


### CIRCUITS ###
def layer_a8_circ(params, nqubits: int):
    for i, wire in enumerate(range(0, nqubits - (nqubits % 2), 2)):
        qml.PauliRot(params[0, i], pauli_word='YY', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i], pauli_word='IX', wires=(wire, wire + 1))

    for i, wire in enumerate(range(1, nqubits - 1, 2)):
        qml.PauliRot(params[0, i + nqubits // 2], pauli_word='YY', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i + nqubits // 2], pauli_word='IX', wires=(wire, wire + 1))
    qml.PauliRot(params[0, nqubits - 1], pauli_word='YY', wires=(nqubits - 1, 0))
    qml.PauliRot(params[1, nqubits - 1], pauli_word='IX', wires=(nqubits - 1, 0))


def circuit_a8_circ(params, nqubits: int, depth: int, observable, psi0=None):
    if psi0 is not None:
        qml.QubitStateVector(psi0, wires=list(range(nqubits)))
    for d in range(depth):
        layer_a8_circ(params[d], nqubits)

    return qml.expval(observable)


def layer_a7(params, nqubits: int):
    for i, wire in enumerate(range(0, nqubits - (nqubits % 2), 2)):
        qml.PauliRot(params[0, i], pauli_word='XX', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i], pauli_word='YY', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i], pauli_word='ZZ', wires=(wire, wire + 1))

    for i, wire in enumerate(range(1, nqubits - 1, 2)):
        qml.PauliRot(params[0, i + nqubits // 2], pauli_word='XX', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i + nqubits // 2], pauli_word='YY', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i + nqubits // 2], pauli_word='ZZ', wires=(wire, wire + 1))


def circuit_a7(params, nqubits: int, depth: int, observable, psi0=None):
    if psi0 is not None:
        qml.QubitStateVector(psi0, wires=list(range(nqubits)))
    for d in range(depth):
        layer_a7(params[d], nqubits)

    return qml.expval(observable)


def layer_a6(params, nqubits: int):
    for i, wire in enumerate(range(0, nqubits - (nqubits % 2), 2)):
        qml.PauliRot(params[0, i], pauli_word='XX', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i], pauli_word='YZ', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i], pauli_word='ZY', wires=(wire, wire + 1))

    for i, wire in enumerate(range(1, nqubits - 1, 2)):
        qml.PauliRot(params[0, i + nqubits // 2], pauli_word='XX', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i + nqubits // 2], pauli_word='YZ', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i + nqubits // 2], pauli_word='ZY', wires=(wire, wire + 1))


def circuit_a6(params, nqubits: int, depth: int, observable, psi0=None):
    if psi0 is not None:
        qml.QubitStateVector(psi0, wires=list(range(nqubits)))
    for d in range(depth):
        layer_a6(params[d], nqubits)

    return qml.expval(observable)


def layer_a5(params, nqubits: int):
    for i, wire in enumerate(range(0, nqubits - (nqubits % 2), 2)):
        qml.PauliRot(params[0, i], pauli_word='XY', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i], pauli_word='YZ', wires=(wire, wire + 1))

    for i, wire in enumerate(range(1, nqubits - 1, 2)):
        qml.PauliRot(params[0, i + nqubits // 2], pauli_word='XY', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i + nqubits // 2], pauli_word='YZ', wires=(wire, wire + 1))


def circuit_a5(params, nqubits: int, depth: int, observable, psi0=None):
    if psi0 is not None:
        qml.QubitStateVector(psi0, wires=list(range(nqubits)))
    for d in range(depth):
        layer_a5(params[d], nqubits)

    return qml.expval(observable)


# XX, XY, ZX
def layer_a17(params, nqubits: int):
    for i, wire in enumerate(range(0, nqubits - (nqubits % 2), 2)):
        qml.PauliRot(params[0, i], pauli_word='XX', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i], pauli_word='XY', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i], pauli_word='ZX', wires=(wire, wire + 1))

    for i, wire in enumerate(range(1, nqubits - 1, 2)):
        qml.PauliRot(params[0, i + nqubits // 2], pauli_word='XX', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i + nqubits // 2], pauli_word='XY', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i + nqubits // 2], pauli_word='ZX', wires=(wire, wire + 1))


def circuit_a17(params, nqubits: int, depth: int, observable, psi0=None):
    if psi0 is not None:
        qml.QubitStateVector(psi0, wires=list(range(nqubits)))
    for d in range(depth):
        layer_a17(params[d], nqubits)

    return qml.expval(observable)


# XY, YX, YZ
def layer_a11(params, nqubits: int):
    for i, wire in enumerate(range(0, nqubits - (nqubits % 2), 2)):
        qml.PauliRot(params[0, i], pauli_word='XY', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i], pauli_word='YX', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i], pauli_word='YZ', wires=(wire, wire + 1))

    for i, wire in enumerate(range(1, nqubits - 1, 2)):
        qml.PauliRot(params[0, i + nqubits // 2], pauli_word='XY', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i + nqubits // 2], pauli_word='YX', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i + nqubits // 2], pauli_word='YZ', wires=(wire, wire + 1))


def circuit_a11(params, nqubits: int, depth: int, observable, psi0=None):
    if psi0 is not None:
        qml.QubitStateVector(psi0, wires=list(range(nqubits)))
    for d in range(depth):
        layer_a11(params[d], nqubits)

    return qml.expval(observable)


### DATA ###
def get_circuit_and_cost(depth, nqubits, algebra):
    dev = qml.device("default.qubit", wires=nqubits)

    if algebra == 'a8_circ':
        qnode = qml.QNode(circuit_a8_circ, dev, interface="jax")
        parameter_shape = (depth, 2, nqubits - 1)

        # TFIM
        gs, H = exact_diagonalization_ising(nqubits, J=1.0, g=-1.0,
                                            interactions=[(i, i + 1) for i in range(nqubits - 1)] + [
                                                (nqubits - 1, 0)])

        H = H.toarray()
        symmetry_X = exact_symmetry_generator(nqubits, 'x').toarray()
        # |+>
        state = get_plus_state(nqubits)
        # state = get_zero_state(nqubits)
        # eigenstates = np.linalg.eigh(symmetry_X)[1]
        # state = eigenstates[:, 3]
        state_L = np.outer(state.conj(), state)

        print('[rho, H]]', commutes(state_L, H))
        print('[H, P_X]]', commutes(H, symmetry_X))
        print('[rho, P_X]', commutes(state_L, symmetry_X))
        print(gs)
        gs = gs[0]
    elif algebra == 'a5':
        qnode = qml.QNode(circuit_a5, dev, interface="jax")
        parameter_shape = (depth, 2, nqubits - 1)

        # Heisenberg
        gs, H = exact_diagonalization_a5(nqubits, delta=1.0,
                                         interactions=[(i, i + 1) for i in range(nqubits - 1)])

        H = H.toarray()
        if not (nqubits % 2):
            symmetry_XYZ = exact_symmetry_generator(nqubits, 'xyz').toarray()
            symmetry_YZX = exact_symmetry_generator(nqubits, 'yzx').toarray()
            symmetry_ZXY = exact_symmetry_generator(nqubits, 'zxy').toarray()

            # |sym>
            eigenstates = np.linalg.eigh(symmetry_XYZ)[1]
            mat = eigenstates.conj().T @ symmetry_YZX @ eigenstates
            print(np.diag(mat))
            print(np.allclose(np.diag(np.diag(mat)), mat))

            state = eigenstates[:, 3]
            state_L = np.outer(state.conj(), state)
            print(state.conj() @ symmetry_XYZ @ state)
            print(state.conj() @ symmetry_YZX @ state)
            print(state.conj() @ symmetry_ZXY @ state)

            print('[P_XYZ, P_YZX] = 0', commutes(symmetry_XYZ, symmetry_YZX))
            print('[P_YZX, P_ZYX] = 0', commutes(symmetry_YZX, symmetry_ZXY))
            print('[P_ZYX, P_XYZ] = 0', commutes(symmetry_ZXY, symmetry_ZXY))
            print('[rho, H]', commutes(state_L, H))
            print('[rho, P_XYZ]', commutes(state_L, symmetry_XYZ))
            print('[rho, P_YZX]', commutes(state_L, symmetry_YZX))
            print('[rho, P_ZXY]', commutes(state_L, symmetry_ZXY))
            raise ValueError
        else:
            # |sym>
            symmetry_I = np.eye(2 ** nqubits, dtype=complex)
            eigenstates = np.linalg.eigh(symmetry_I)[1]
            state = eigenstates[:, 0]
            print("Nothing to check!")

        print(gs)

        gs = gs[0]
    elif algebra == 'a6':
        qnode = qml.QNode(circuit_a6, dev, interface="jax")
        parameter_shape = (depth, 3, nqubits - 1)

        # Heisenberg
        gs, H = exact_diagonalization_a6(nqubits, delta=1.0,
                                         interactions=[(i, i + 1) for i in range(nqubits - 1)])

        H = H.toarray()
        state = get_plus_state(nqubits)

        # if not (nqubits % 2):
        symmetry_X = exact_symmetry_generator(nqubits, 'x').toarray()
        state_L = np.outer(state.conj(), state)

        print(commutes(symmetry_X, state_L))
        print(commutes(H, state_L))
        print(np.trace(state_L @ H))
        #     symmetry_YZ = exact_symmetry_generator(nqubits, 'yz').toarray()
        #     symmetry_ZY = exact_symmetry_generator(nqubits, 'zy').toarray()
        #     # |sym>
        #     eigenstates = np.linalg.eigh(symmetry_YZ)[1]

        print(gs)
        gs = gs[0]
    elif algebra == 'a7':
        qnode = qml.QNode(circuit_a7, dev, interface="jax")

        parameter_shape = (depth, 3, nqubits - 1)
        # Heisenberg
        gs, H = exact_diagonalization_heisenberg(nqubits, delta=1.0,
                                                 interactions=[(i, i + 1) for i in range(nqubits - 1)])

        H = H.toarray()
        symmetry_X = exact_symmetry_generator(nqubits, 'x').toarray()
        symmetry_Y = exact_symmetry_generator(nqubits, 'y').toarray()
        symmetry_Z = exact_symmetry_generator(nqubits, 'z').toarray()

        # |phi+>

        state = get_zero_state(nqubits)
        state_L = np.outer(state.conj(), state)

        print('commutes with H', commutes(state_L, H))
        print('commutes with P_X', commutes(state_L, symmetry_X))
        print('commutes with P_Y', commutes(state_L, symmetry_Y))
        print('commutes with P_Z', commutes(state_L, symmetry_Z))

        print(gs)
        gs = gs[0]
    elif algebra == 'a11':
        qnode = qml.QNode(circuit_a11, dev, interface="jax")

        parameter_shape = (depth, 3, nqubits - 1)
        # Heisenberg
        gs, H = exact_diagonalization_random_H(nqubits, complex=False)

        state = get_zero_state(nqubits)

        print(gs[:2])
        gs = gs[0]

    elif algebra == 'a17':
        qnode = qml.QNode(circuit_a17, dev, interface="jax")

        parameter_shape = (depth, 3, nqubits - 1)
        # Heisenberg
        gs, H = exact_diagonalization_random_H(nqubits)

        state = get_zero_state(nqubits)

        print(gs[:2])
        gs = gs[0]
    else:
        raise NotImplementedError

    return gs, H, state, qnode, parameter_shape


def optimization(params, opt_step, optimizer, max_steps):
    costs = []
    opt_state = optimizer.init(params)

    for i in range(1, max_steps + 1):
        params, opt_state, c = opt_step(params, opt_state)
        costs.append(c)
        if not i % 250:
            print(f"step {i} - cost={c}")
    return costs


def perform_runs(algebra: str, nqubits: int, depth: int, seeds, maxsteps: int = 100):
    # assert not nqubits % 2, 'number of qubits must be even'
    # set paths
    data_path = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra}/'
    gs, H, state, qnode, parameter_shape = get_circuit_and_cost(depth, nqubits, algebra)
    obs = qml.Hermitian(H, wires=list(range(nqubits)))
    cost_fn = lambda x: qnode(x, nqubits, depth, obs, psi0=state)
    optimizer = optax.adam(LEARNING_RATE)

    @jax.jit
    def opt_step(params, opt_state):
        print('TRACING')
        loss_value, grads = jax.value_and_grad(cost_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for seed in seeds:
        filename = f'costs_nq_{nqubits}_d_{depth}_seed_{seed}.csv'
        print(filename)
        if not os.path.exists(data_path + filename):
            print(f"Getting data for N = {nqubits} - Depth = {depth} - Seed {seed}")
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            # Set up devices and seed
            np.random.seed(seed)
            parameters = np.random.rand(*parameter_shape) * np.pi

            # Calculate all runs in one vmap
            costs = optimization(parameters,
                                 opt_step=opt_step,
                                 optimizer=optimizer,
                                 max_steps=maxsteps,
                                 )
            costs = np.array(costs)
            gs_array = np.ones_like(costs) * gs
            grads = np.stack([costs, gs_array], axis=1)

            df = pd.DataFrame(grads)
            df.to_csv(data_path + filename)
        else:
            print("Data already exists, skipping...")


def get_data(hyps, seeds, algebra: str, maxsteps: int = 100):
    for (n, d,) in hyps:
        perform_runs(algebra, n, d, seeds, maxsteps=maxsteps)


### PLOTTING ###

def plot_data(hyps, algebra, seed: int = 1233, maxsteps: int = 100):
    data_path = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra}/'
    fig_path = f'./data/overparameterization_maxsteps_{maxsteps}/figures/'
    indices = np.stack(np.unravel_index(range(len(nqubits) * len(depth_list)),
                                        shape=(len(nqubits), len(depth_list)))).T
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(fig_path + f'{algebra}_grads_mean.npy') & 0:
        # TODO: decide which run to get by criterion
        costs = np.zeros((len(nqubits), len(depth_list), maxsteps))

        for i, (n, d) in enumerate(hyps):
            idx = indices[i]
            file_path = f'grads_{algebra}_nq_{n}_d_{d}_seed_{seed}.csv'
            try:
                data = pd.read_csv(data_path + file_path, index_col=False).values[:, 1:]
                energies = data[:, 0]
                gs = data[0, -1]
                residuals = np.abs(energies - gs)
                costs[idx[0], idx[1]] = residuals
            except FileNotFoundError:
                print("File not found, skipping for now...")

        np.save(fig_path + f'{algebra}_costs.npy', costs)
    else:
        costs = np.load(fig_path + f'{algebra}_costs.npy')

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)
    for i, d, in enumerate(depth_list):
        axs.plot(list(range(1, maxsteps + 1)), costs[0, i], label=f'depth = {d}')
    axs.set_xlabel('steps')
    axs.set_ylabel(r'$|E_0 - E(\theta)|$')
    axs.set_yscale('log')
    axs.set_ylim([1e-4, 1e2])
    axs.legend()
    plt.tight_layout()

    fig.savefig(fig_path + f'{algebra}_overparameterization.pdf')
    if SHOW:
        plt.show()


def plot_combined_costs(hyps, algebra_1, algebra_2, maxsteps: int = 100):
    data_path_1 = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra_1}/'
    data_path_2 = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra_2}/'
    fig_path = f'./data/overparameterization_maxsteps_{maxsteps}/figures/'
    indices = np.stack(np.unravel_index(range(len(nqubits) * len(depth_list) * len(seeds)),
                                        shape=(len(nqubits), len(depth_list), len(seeds)))).T
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(nqubits)))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(fig_path + f'{algebra_1}_costs.npy') or not os.path.exists(
            fig_path + f'{algebra_2}_costs.npy') & DATA:
        costs_1 = np.ones((len(nqubits), len(depth_list), len(seeds)))
        costs_1_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        costs_2 = np.ones((len(nqubits), len(depth_list), len(seeds)))
        costs_2_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)

        for i, (n, d, s) in enumerate(hyps):
            idx = indices[i]
            file_path_1 = f'costs_{algebra_1}_nq_{n}_d_{d}_seed_{s}.csv'
            file_path_2 = f'costs_{algebra_2}_nq_{n}_d_{d}_seed_{s}.csv'
            print(file_path_1)
            print(file_path_2)
            try:
                data_1 = pd.read_csv(data_path_1 + file_path_1, index_col=False).values[:, 1:]

                final_energy_1 = data_1[:, 0][-1]
                gs_1 = data_1[0, -1]
                if s == 23552:
                    raise FileNotFoundError
                residuals_1 = np.abs(final_energy_1 - gs_1)
                costs_1[idx[0], idx[1], idx[2]] = residuals_1
                costs_1_mask[idx[0], idx[1], idx[2]] = True
            except FileNotFoundError:
                print("File not found, skipping for now...")
            try:
                data_2 = pd.read_csv(data_path_2 + file_path_2, index_col=False).values[:, 1:]
                final_energy_2 = data_2[:, 0][-1]
                gs_2 = data_2[0, -1]
                # if (n == 9) and (d == 16):
                #     plt.plot(np.abs(data_2[:, 0]-gs_2))
                #     plt.yscale('log')
                #     plt.show()
                residuals_2 = np.abs(final_energy_2 - gs_2)
                costs_2[idx[0], idx[1], idx[2]] = residuals_2
                costs_2_mask[idx[0], idx[1], idx[2]] = True
            except FileNotFoundError:
                print("File not found, skipping for now...")

        np.save(fig_path + f'{algebra_1}_costs.npy', costs_1)
        np.save(fig_path + f'{algebra_2}_costs.npy', costs_2)
        np.save(fig_path + f'{algebra_1}_mask.npy', costs_1_mask)
        np.save(fig_path + f'{algebra_2}_mask.npy', costs_2_mask)
    else:
        costs_1 = np.load(fig_path + f'{algebra_1}_costs.npy')
        costs_2 = np.load(fig_path + f'{algebra_2}_costs.npy')
        costs_1_mask = np.load(fig_path + f'{algebra_1}_mask.npy')
        costs_2_mask = np.load(fig_path + f'{algebra_2}_mask.npy')
    print(f"{algebra_1} files found - {np.sum(costs_1_mask)}/{len(hyps)}")
    print(f"{algebra_2} files found - {np.sum(costs_2_mask)}/{len(hyps)}")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(4, 3.5)
    costs_1_ma = np.ma.array(costs_1, mask=~costs_1_mask)
    costs_2_ma = np.ma.array(costs_2, mask=~costs_2_mask)
    for j, n in enumerate(nqubits):
        # axs[0].plot(depth_list, np.clip(np.mean(costs_1[j, :], axis=1), a_min=1e-10, a_max=np.inf), label=rf'$N$ = {n}',
        axs[0].plot(depth_list, np.clip(costs_1_ma[j, :].mean(axis=1), a_min=1e-10, a_max=np.inf), label=rf'$N$ = {n}',
                    color=colors[j])

    axs[0].set_ylabel(r'$|E_0 - E(\theta)|$')
    axs[0].set_xlabel('$L$')
    axs[0].set_yscale('log')
    axs[0].set_ylim([1e-12, 1e1])
    axs[0].set_title(r'(a) $\mathfrak{a}_{8}^\circ$', y=-0.25)
    axs[0].tick_params(axis="x", which='both', top=False, right=False)
    axs[0].tick_params(axis="y", which='both', top=False, right=False)
    axs[0].legend(prop={'size': 8})
    for j, n in enumerate(nqubits):
        axs[1].plot(depth_list, np.clip(costs_2_ma[j, :].mean(axis=1), a_min=1e-10, a_max=np.inf),
                    label=rf'$N$ = {n}', color=colors[j])

    fig.subplots_adjust(wspace=0.0)
    axs[1].set_xlabel('$L$')
    axs[1].set_yscale('log')
    axs[1].set_ylim([1e-12, 1e1])
    axs[1].set_title(r'(b) $\mathfrak{a}_{6}$', y=-0.25)
    axs[1].set_yticks(())
    axs[1].tick_params(axis="x", which='both', top=False, right=False)
    axs[1].tick_params(axis="y", which='both', top=False, right=False, left=False)
    axs[1].legend(prop={'size': 8})

    fig.savefig(fig_path + f'{algebra_1}_{algebra_2}_overparam.pdf')

    if SHOW:
        plt.show()


def reject_outliers(data: np.ma.MaskedArray, m=2):
    for d in range(data.shape[0]):
        mask_axis = np.abs(data[d] - data[d].mean()) < m * data[d].std()
        outlier_indices = np.where(~mask_axis)[0]
        for idx in outlier_indices:
            if not np.allclose(0.0, data[d] - data[d].mean()):
                data[d, idx] = np.ma.masked
    return data


def plot_combined_steps(hyps, algebra_1, algebra_2, algebra_3=None, maxsteps: int = 100, max_n=(-1, -1, -1)):
    data_path_1 = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra_1}/'
    data_path_2 = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra_2}/'
    data_path_3 = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra_3}/'
    fig_path = f'./data/overparameterization_maxsteps_{maxsteps}/figures/'
    indices = np.stack(np.unravel_index(range(len(nqubits) * len(depth_list) * len(seeds)),
                                        shape=(len(nqubits), len(depth_list), len(seeds)))).T
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(nqubits)))
    threshold = 5e-4
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if not os.path.exists(fig_path + f'{algebra_1}_steps.npy') or not os.path.exists(
            fig_path + f'{algebra_2}_steps.npy') & DATA:
        steps_1 = np.ones((len(nqubits), len(depth_list), len(seeds)))
        steps_1_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_2 = np.ones((len(nqubits), len(depth_list), len(seeds)))
        steps_2_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_3 = np.ones((len(nqubits), len(depth_list), len(seeds)))
        steps_3_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        for i, (n, d, s) in enumerate(hyps):
            idx = indices[i]
            file_path_1 = file_path_2 = file_path_3 = f'costs_nq_{n}_d_{d}_seed_{s}.csv'

            try:
                data_1 = pd.read_csv(data_path_1 + file_path_1, index_col=False).values[:, 1:]
                gs_1 = data_1[0, 1]
                energies_1 = np.abs(data_1[:, 0] - gs_1)
                loc = np.where(energies_1 < threshold)[0]
                if loc.size == 0:
                    loc = MAXSTEPS
                else:
                    loc = loc[0]
                if s == 23552:
                    raise FileNotFoundError
                steps_1[idx[0], idx[1],
                        idx[2]] = loc
                steps_1_mask[idx[0], idx[1], idx[2]] = True
            except FileNotFoundError:
                print("File not found, skipping for now...")
            try:
                data_2 = pd.read_csv(data_path_2 + file_path_2, index_col=False).values[:, 1:]
                gs_2 = data_2[0, 1]
                energies_2 = np.abs(data_2[:, 0] - gs_2)
                loc = np.where(energies_2 < threshold)[0]
                if loc.size == 0:
                    loc = MAXSTEPS
                else:
                    loc = loc[0]
                steps_2[idx[0], idx[1], idx[2]] = loc
                steps_2_mask[idx[0], idx[1], idx[2]] = True
            except FileNotFoundError:
                print("File not found, skipping for now...")
            if algebra_3 is not None:
                try:
                    data_3 = pd.read_csv(data_path_3 + file_path_3, index_col=False).values[:, 1:]
                    gs_3 = data_3[0, 1]
                    energies_3 = np.abs(data_3[:, 0] - gs_3)
                    loc = np.where(energies_3 < threshold)[0]
                    if loc.size == 0:
                        loc = MAXSTEPS
                    else:
                        loc = loc[0]
                    steps_3[idx[0], idx[1], idx[2]] = loc
                    steps_3_mask[idx[0], idx[1], idx[2]] = True

                except FileNotFoundError:
                    print(f"File not found for, {(n, d, s)}, skipping for now...")

        np.save(fig_path + f'{algebra_1}_steps.npy', steps_1)
        np.save(fig_path + f'{algebra_2}_steps.npy', steps_2)
        np.save(fig_path + f'{algebra_1}_steps_mask.npy', steps_1_mask)
        np.save(fig_path + f'{algebra_2}_steps_mask.npy', steps_2_mask)
        if algebra_3 is not None:
            np.save(fig_path + f'{algebra_3}_steps.npy', steps_3)
            np.save(fig_path + f'{algebra_3}_steps_mask.npy', steps_3_mask)
    else:
        steps_1 = np.load(fig_path + f'{algebra_1}_steps.npy')
        steps_2 = np.load(fig_path + f'{algebra_2}_steps.npy')
        steps_1_mask = np.load(fig_path + f'{algebra_1}_steps_mask.npy')
        steps_2_mask = np.load(fig_path + f'{algebra_2}_steps_mask.npy')
        if algebra_3 is not None:
            steps_3 = np.load(fig_path + f'{algebra_3}_steps.npy')
            steps_3_mask = np.load(fig_path + f'{algebra_2}_steps_mask.npy')

    print(f"{algebra_1} files found - {np.sum(steps_1_mask)}/{len(hyps)}")
    print(f"{algebra_2} files found - {np.sum(steps_2_mask)}/{len(hyps)}")
    costs_1_ma = np.ma.array(steps_1, mask=~steps_1_mask)
    costs_2_ma = np.ma.array(steps_2, mask=~steps_2_mask)
    if algebra_3 is not None:
        print(f"{algebra_3} files found - {np.sum(steps_3_mask)}/{len(hyps)}")
        costs_3_ma = np.ma.array(steps_3, mask=~steps_3_mask)
    if 0:
        for cost in [costs_1_ma, costs_2_ma]:
            costs_ma_mean = np.log(cost).mean(axis=2)
            costs_ma_std = np.log(cost).std(axis=2) + 1e-5
            fn = lambda x, a, b: a * 1 / x + b
            params = []
            depth_start = 3
            for i, n in enumerate(nqubits):
                res = scipy.optimize.curve_fit(fn, depth_list[depth_start:], costs_ma_mean[i, depth_start:],
                                               # sigma=costs_1_ma_std[i, depth_start:],
                                               p0=[1.0, 1.0])
                params.append(res[0])

            fig, ax = plt.subplots(1, 2)
            for i, n in enumerate(nqubits):
                line = ax[0].plot(depth_list[depth_start:], costs_ma_mean[i, depth_start:])
                ax[0].plot(depth_list[depth_start:], fn(np.array(depth_list[depth_start:]), *params[i]), linestyle='--',
                           color=line[0].get_color())

            ax[0].set_ylabel(r'$\log(\mathrm{Steps})$')
            ax[0].set_xlabel(r'$L$')
            scaling = np.stack(params)
            fn = lambda x, a, b: x ** a + b
            nstart = 2
            res = scipy.optimize.curve_fit(fn, nqubits[nstart:], scaling[nstart:, 0], p0=[1.0, 1.0])

            line = ax[1].plot(nqubits[nstart:], scaling[nstart:, 0], color='black')
            ax[1].plot(nqubits[nstart:], fn(np.array(nqubits[nstart:]), *res[0]), linestyle='--',
                       color='black', label=rf'$O(x^{{{res[0][0]:1.3f}}})$')
            ax[1].set_ylabel(r'$a$')
            ax[1].set_xlabel(r'$n$')
            ax[1].legend()
            plt.tight_layout()
            plt.show()

    if algebra_3 is not None:

        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(10, 3.5)
    else:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(4, 3.5)
    MARKERSIZE = 7
    for j, n in enumerate(nqubits[:max_n[0]]):
        data = reject_outliers(costs_1_ma[j, :], 5)
        axs[0].errorbar(depth_list,
                        np.clip(data.mean(axis=1), a_min=1e-10, a_max=np.inf),
                        yerr=np.clip(data.std(axis=1), a_min=1e-10, a_max=np.inf),
                        label=rf'$n$ = {n}',
                        color=colors[j], marker='.', markersize=MARKERSIZE)

    axs[0].set_ylabel(r'Steps')
    axs[0].set_yscale('log')
    # axs[0].set_xscale('log')
    axs[0].set_xlabel('$L$')
    axs[0].set_ylim([25, maxsteps + 0.1 * maxsteps])
    axs[0].set_title(r'(a) $\mathfrak{a}_{8}^\circ$', y=-0.25)
    axs[0].tick_params(axis="x", which='both', top=False, right=False)
    axs[0].tick_params(axis="y", which='both', top=False, right=False)
    axs[0].legend(prop={'size': 7})
    for j, n in enumerate(nqubits[:max_n[1]]):
        data = reject_outliers(costs_2_ma[j, :], 5)
        axs[1].errorbar(depth_list,
                        np.clip(data.mean(axis=1), a_min=1e-10, a_max=np.inf),
                        yerr=np.clip(data.std(axis=1), a_min=1e-10, a_max=np.inf),
                        label=rf'$n$ = {n}', color=colors[j], marker='.', markersize=MARKERSIZE)

    fig.subplots_adjust(wspace=0.0)
    axs[1].set_xlabel('$L$')
    axs[1].set_yscale('log')
    # axs[1].set_xscale('log')
    axs[1].set_ylim([25, maxsteps + 0.1 * maxsteps])
    axs[1].set_title(r'(b) $\mathfrak{a}_{11}$', y=-0.25)
    axs[1].set_yticks(())
    axs[1].tick_params(axis="x", which='both', top=False, right=False)
    axs[1].tick_params(axis="y", which='both', top=False, right=False, left=False)

    if algebra_3 is not None:
        for j, n in enumerate(nqubits[:max_n[2]]):
            if n % 2:
                data = reject_outliers(costs_3_ma[j, :], 5)
                axs[2].errorbar(np.array(depth_list),
                                np.clip(data.mean(axis=1), a_min=1e-10, a_max=np.inf),
                                yerr=np.clip(data.std(axis=1), a_min=1e-10, a_max=np.inf),
                                label=rf'$n$ = {n}', color=colors[j], marker='.', markersize=MARKERSIZE)
        axs[2].set_xlabel('$L$')
        axs[2].set_yscale('log')
        # axs[2].set_xscale('log')
        axs[2].set_ylim([25, maxsteps + 0.1 * maxsteps])
        axs[2].set_title(r'(c) $\mathfrak{a}_{7}$', y=-0.25)
        axs[2].set_yticks(())
        axs[2].tick_params(axis="x", which='both', top=False, right=False)
        axs[2].tick_params(axis="y", which='both', top=False, right=False, left=False)
        # axs[2].legend(prop={'size': 8})

        fig.savefig(fig_path + f'{algebra_1}_{algebra_2}_{algebra_3}_overparam_steps.pdf')
    else:
        fig.savefig(fig_path + f'{algebra_1}_{algebra_2}_overparam_steps.pdf')

    if SHOW:
        plt.show()


def plot_combined_success(hyps, algebra_1, algebra_2, algebra_3=None, maxsteps: int = 100, max_n=(-1, -1, -1)):
    data_path_1 = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra_1}/'
    data_path_2 = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra_2}/'
    data_path_3 = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra_3}/'
    fig_path = f'./data/overparameterization_maxsteps_{maxsteps}/figures/'
    indices = np.stack(np.unravel_index(range(len(nqubits) * len(depth_list) * len(seeds)),
                                        shape=(len(nqubits), len(depth_list), len(seeds)))).T
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(nqubits)))
    threshold = 5e-4
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(fig_path + f'{algebra_1}_success.npy') or not os.path.exists(
            fig_path + f'{algebra_2}_success.npy') or not os.path.exists(fig_path + f'{algebra_3}_success.npy') & DATA:
        steps_1 = np.ones((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_1_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_2 = np.ones((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_2_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_3 = np.ones((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_3_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)

        for i, (n, d, s) in enumerate(hyps):
            idx = indices[i]
            file_path_1 = file_path_2 = file_path_3 = f'costs_nq_{n}_d_{d}_seed_{s}.csv'
            try:
                data_1 = pd.read_csv(data_path_1 + file_path_1, index_col=False).values[:, 1:]
                gs_1 = data_1[0, -1]
                energies_1 = np.abs(data_1[:, 0] - gs_1)
                loc = np.where(energies_1 < threshold)[0]
                if loc.size == 0:
                    steps_1[idx[0], idx[1], idx[2]] = False
                else:
                    steps_1[idx[0], idx[1], idx[2]] = True

                steps_1_mask[idx[0], idx[1], idx[2]] = True
            except FileNotFoundError:
                print("File not found, skipping for now...")
            try:
                data_2 = pd.read_csv(data_path_2 + file_path_2, index_col=False).values[:, 1:]
                gs_2 = data_2[0, -1]
                energies_2 = np.abs(data_2[:, 0] - gs_2)
                loc = np.where(energies_2 < threshold)[0]
                if loc.size == 0:
                    steps_2[idx[0], idx[1], idx[2]] = False
                else:
                    steps_2[idx[0], idx[1], idx[2]] = True
                steps_2_mask[idx[0], idx[1], idx[2]] = True
            except FileNotFoundError:
                print("File not found, skipping for now...")
            if algebra_3 is not None:
                try:
                    data_3 = pd.read_csv(data_path_3 + file_path_3, index_col=False).values[:, 1:]
                    gs_3 = data_3[0, -1]
                    energies_3 = np.abs(data_3[:, 0] - gs_3)
                    loc = np.where(energies_3 < threshold)[0]
                    if loc.size == 0:
                        steps_3[idx[0], idx[1], idx[2]] = False
                    else:
                        steps_3[idx[0], idx[1], idx[2]] = True

                    steps_3_mask[idx[0], idx[1], idx[2]] = True
                except FileNotFoundError:
                    print("File not found, skipping for now...")

        np.save(fig_path + f'{algebra_1}_success.npy', steps_1)
        np.save(fig_path + f'{algebra_2}_success.npy', steps_2)
        np.save(fig_path + f'{algebra_3}_success.npy', steps_3)
        np.save(fig_path + f'{algebra_1}_success_mask.npy', steps_1_mask)
        np.save(fig_path + f'{algebra_2}_success_mask.npy', steps_2_mask)
        np.save(fig_path + f'{algebra_3}_success_mask.npy', steps_3_mask)
    else:
        steps_1 = np.load(fig_path + f'{algebra_1}_success.npy')
        steps_2 = np.load(fig_path + f'{algebra_2}_success.npy')
        steps_3 = np.load(fig_path + f'{algebra_3}_success.npy')
        steps_1_mask = np.load(fig_path + f'{algebra_1}_success_mask.npy')
        steps_2_mask = np.load(fig_path + f'{algebra_2}_success_mask.npy')
        steps_3_mask = np.load(fig_path + f'{algebra_3}_success_mask.npy')
    print(f"{algebra_1} files found - {np.sum(steps_1_mask)}/{len(hyps)}")
    print(f"{algebra_2} files found - {np.sum(steps_2_mask)}/{len(hyps)}")
    print(f"{algebra_3} files found - {np.sum(steps_3_mask)}/{len(hyps)}")
    costs_1_ma = np.ma.array(steps_1, mask=~steps_1_mask)
    costs_2_ma = np.ma.array(steps_2, mask=~steps_2_mask)
    costs_3_ma = np.ma.array(steps_3, mask=~steps_3_mask)
    costs = [costs_1_ma, costs_2_ma, costs_3_ma]
    masks = [steps_1_mask, steps_2_mask, steps_3_mask]
    if algebra_3 is not None:
        number_of_algs = 3
    else:
        number_of_algs = 2
    over_param_thr = np.zeros((number_of_algs, len(nqubits)))
    MARKERSIZE = 11
    LW = 2

    for i in range(number_of_algs):
        for j, n in enumerate(nqubits[:max_n[i]]):
            idx = np.where(np.isclose(costs[i][j].sum(axis=1) / masks[i][j].sum(axis=1), 1.0, atol=0.01))[0]
            if idx.size > 0:
                over_param_thr[i, j] = idx[0]
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(10. / 3., 3.5)
    algebra_names = [r'$\mathfrak{a}_8^\circ(n)$', r'$\mathfrak{a}_7(n)$', r'$\mathfrak{a}_{11}(n)$']
    scaling_fns = [lambda x, a, b: x ** a + b,
                   lambda x, a, b: a ** x + b,
                   lambda x, a, b: a ** x + b
                   ]
    mul = [4, 6, 6]
    for i in range(number_of_algs):
        if i == 2:
            odd_ns = list(range(len(nqubits[:max_n[i]])))
            odd_ns = [i for i in odd_ns if nqubits[i] % 2]
            print(odd_ns)
            nqubits_array = np.array(nqubits)[odd_ns]
            data = np.array([depth_list[int(k)] for k in over_param_thr[i, odd_ns]])
            multiplier = np.array([n for n in np.array(nqubits)[odd_ns]])
            data = data * multiplier * mul[i]
            res = scipy.optimize.curve_fit(scaling_fns[i], nqubits_array, data)
            line = axs.plot(nqubits_array, data,
                            label=algebra_names[i], marker='.', markersize=MARKERSIZE, linewidth=LW)
            # axs.plot(nqubits, scaling_fns[i](nqubits, *res[0]), color=line[0].get_color(), linestyle='dashed')

        elif i == 1:

            nqubits_array = np.array(nqubits[1:max_n[i]])
            data = np.array([depth_list[int(k)] for k in over_param_thr[i, 1:max_n[i]]])
            multiplier = np.array([n for n in nqubits[1:max_n[i]]])
            data = data * multiplier * mul[i]
            res = scipy.optimize.curve_fit(scaling_fns[i], nqubits_array, data)
            line = axs.plot(nqubits_array, data,
                            label=algebra_names[i], marker='.', markersize=MARKERSIZE, linewidth=LW)
            # axs.plot(nqubits, scaling_fns[i](nqubits, *res[0]), color=line[0].get_color(), linestyle='dashed')
        else:
            nqubits_array = np.array(nqubits[:max_n[i]])
            data = np.array([depth_list[int(k)] for k in over_param_thr[i, :max_n[i]]])
            multiplier = np.array([n for n in nqubits[:max_n[i]]])
            data = data * multiplier * mul[i]
            res = scipy.optimize.curve_fit(scaling_fns[i], nqubits_array, data)
            line = axs.plot(nqubits_array, data,
                            label=algebra_names[i], marker='.', markersize=MARKERSIZE, linewidth=LW)
            # axs.plot(nqubits, scaling_fns[i](nqubits, *res[0]), color=line[0].get_color(), linestyle='dashed')
        print(res[0])

    axs.plot(nqubits, [1.88 ** (n + 1.9) for n in nqubits], color='black', linestyle='dashed')
    # axs.plot(nqubits, [n**2 +10 for n in nqubits], label=r'$n^2/2$', color='gray', linestyle='dashed')
    axs.set_xlabel(r'$n$')
    axs.set_ylabel(r'\# Parameters')
    axs.set_yscale('log')
    axs.set_title(r'(d) Parameter scaling', y=-0.25)
    axs.tick_params(axis="x", which='both', top=False, right=False)
    axs.tick_params(axis="y", which='both', top=False, right=False)
    plt.legend(prop={'size': 10})
    plt.show()
    fig.savefig(fig_path + f'{algebra_1}_{algebra_2}_{algebra_3}_overparam_success_fit.pdf')
    if algebra_3 is not None:

        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(10, 3.5)
    else:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(4, 3.5)

    for j, n in enumerate(nqubits[:max_n[0]]):
        axs[0].plot(depth_list, costs_1_ma[j, :].sum(axis=1) / steps_1_mask[j, :].sum(axis=1), label=rf'$n$ = {n}',
                    color=colors[j], marker='.', markersize=MARKERSIZE)

    axs[0].set_ylabel(r'Success probability')
    axs[0].set_xlabel('$L$')
    axs[0].set_ylim([-0.05, 1.05])
    axs[0].set_title(r'(a) $\mathfrak{a}_{8}^\circ(n)$', y=-0.25)
    axs[0].tick_params(axis="x", which='both', top=False, right=False)
    axs[0].tick_params(axis="y", which='both', top=False, right=False)
    axs[0].legend(prop={'size': 10})
    for j, n in enumerate(nqubits[:max_n[1]]):
        if n == 3:
            pass
        else:
            axs[1].plot(depth_list, costs_2_ma[j, :].sum(axis=1) / steps_2_mask[j, :].sum(axis=1), label=rf'$n$ = {n}',
                        color=colors[j], marker='.', markersize=MARKERSIZE)

    fig.subplots_adjust(wspace=0.0)
    axs[1].set_xlabel('$L$')
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].set_title(r'(b) $\mathfrak{a}_{11}(n)$', y=-0.25)
    axs[1].set_yticks(())
    axs[1].tick_params(axis="x", which='both', top=False, right=False)
    axs[1].tick_params(axis="y", which='both', top=False, right=False, left=False)

    if algebra_3 is not None:
        for j, n in enumerate(nqubits[:max_n[2]]):
            if n % 2:
                axs[2].plot(depth_list, costs_3_ma[j, :].sum(axis=1) / steps_3_mask[j, :].sum(axis=1),
                            label=rf'$n$ = {n}',
                            color=colors[j], marker='.', markersize=MARKERSIZE)

        axs[2].set_xlabel('$L$')
        axs[2].set_ylim([-0.05, 1.05])
        axs[2].set_title(r'(c) $\mathfrak{a}_{7}(n)$', y=-0.25)
        axs[2].set_yticks(())
        axs[2].tick_params(axis="x", which='both', top=False, right=False)
        axs[2].tick_params(axis="y", which='both', top=False, right=False, left=False)

    fig.savefig(fig_path + f'{algebra_1}_{algebra_2}_{algebra_3}_overparam_success.pdf')

    if SHOW:
        plt.show()


def plot_first_eigenstate(hyps, algebra_1, maxsteps: int = 100):
    data_path_1 = f'./data/overparameterization_maxsteps_{maxsteps}/{algebra_1}/'
    fig_path = f'./data/overparameterization_maxsteps_{maxsteps}/figures/'
    indices = np.stack(np.unravel_index(range(len(nqubits) * len(depth_list) * len(seeds)),
                                        shape=(len(nqubits), len(depth_list), len(seeds)))).T
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(nqubits)))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(fig_path + f'{algebra_1}_costs.npy') or not os.path.exists(
            fig_path + f'{algebra_1}_costs.npy') & DATA:
        costs_1 = np.ones((len(nqubits), len(depth_list), len(seeds)))
        costs_1_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        costs_2 = np.ones((len(nqubits), len(depth_list), len(seeds)))
        costs_2_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        for i, (n, d, s) in enumerate(hyps):
            idx = indices[i]
            file_path_1 = f'costs_{algebra_1}_nq_{n}_d_{d}_seed_{s}.csv'
            print(file_path_1)
            try:
                data_1 = pd.read_csv(data_path_1 + file_path_1, index_col=False).values[:, 1:]

                final_energy_1 = data_1[:, 0][-1]
                gs_1 = data_1[0, -2]
                gs_2 = data_1[0, -1]
                residuals_1 = np.abs(final_energy_1 - gs_1)
                residuals_2 = np.abs(final_energy_1 - gs_2)
                print(n, final_energy_1, gs_1, gs_2)
                costs_1[idx[0], idx[1], idx[2]] = residuals_1
                costs_1_mask[idx[0], idx[1], idx[2]] = True
                costs_2[idx[0], idx[1], idx[2]] = residuals_2
                costs_2_mask[idx[0], idx[1], idx[2]] = True
            except FileNotFoundError:
                print("File not found, skipping for now...")

        np.save(fig_path + f'{algebra_1}_costs.npy', costs_1)
        np.save(fig_path + f'{algebra_1}_costs_es.npy', costs_2)
        np.save(fig_path + f'{algebra_1}_mask.npy', costs_1_mask)
        np.save(fig_path + f'{algebra_1}_mask_es.npy', costs_2_mask)
    else:
        costs_1 = np.load(fig_path + f'{algebra_1}_costs.npy')
        costs_2 = np.load(fig_path + f'{algebra_1}_costs_es.npy')
        costs_1_mask = np.load(fig_path + f'{algebra_1}_mask.npy')
        costs_2_mask = np.load(fig_path + f'{algebra_1}_mask_es.npy')

    print(f"{algebra_1} files found - {np.sum(costs_1_mask)}/{len(hyps)}")
    print(f"{algebra_1}_es files found - {np.sum(costs_2_mask)}/{len(hyps)}")
    MARKERSIZE = 7

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)
    costs_1_ma = np.ma.array(costs_1, mask=~costs_1_mask)
    costs_2_ma = np.ma.array(costs_2, mask=~costs_2_mask)
    for j, n in enumerate(nqubits):
        if (n % 2):
            axs.plot(depth_list, np.clip(costs_2_ma[j, :].mean(axis=1), a_min=1e-10, a_max=np.inf), label=rf'$N$ = {n}',
                     color=colors[j], markersize=MARKERSIZE, marker='.')
        else:
            axs.plot(depth_list, np.clip(costs_1_ma[j, :].mean(axis=1), a_min=1e-10, a_max=np.inf),
                     label=rf'$N$ = {n}',
                     color=colors[j], markersize=MARKERSIZE, marker='.')
    # else:
    #     axs.plot(depth_list, np.clip(costs_1_ma[j, :].mean(axis=1), a_min=1e-10, a_max=np.inf), label=rf'$N$ = {n}',
    #          color=colors[j])

    axs.set_ylabel(r'$|E_0 - E(\theta)|$')
    axs.set_yscale('log')
    axs.set_ylim([1e-12, 1e1])
    axs.set_title(r'(a) $\mathfrak{a}_{8}^\circ$', y=-0.25)
    axs.tick_params(axis="x", which='both', top=False, right=False)
    axs.tick_params(axis="y", which='both', top=False, right=False)
    axs.legend(prop={'size': 8})

    fig.savefig(fig_path + f'{algebra_1}_es_overparam.pdf')

    if SHOW:
        plt.show()


if __name__ == '__main__':

    import argparse

    number_of_runs = 100
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', default=0)
    args = parser.parse_args()
    nqubits = [n for n in range(3, 13)]
    # nqubits = [3, ]
    # depth_list = list(range(2, 18, 2)) + [20, 24, 28, 32]
    depth_list = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
    seeds = [2 ** 8 * i for i in range(number_of_runs)]
    hyps = list(itertools.product(nqubits, depth_list, seeds))
    hyps_multirun = list(itertools.product(nqubits, depth_list))
    print(f"Number of hyps= {len(hyps_multirun)}")
    print(f"Number of seeds = {len(seeds)}")
    MAXSTEPS = 3000
    # plot_first_eigenstate(hyps, 'a7', maxsteps=MAXSTEPS)
    # plot_combined_steps(hyps, 'a8_circ', 'a11', 'a7', maxsteps=MAXSTEPS, max_n=(12, 5, 6))
    plot_combined_success(hyps, 'a8_circ', 'a11', 'a7', maxsteps=MAXSTEPS, max_n=(12, 6, 7))

    if args.instance is None:
        # #
        ALGEBRA = 'a8_circ'
        get_data(hyps_multirun, seeds, algebra=ALGEBRA, maxsteps=MAXSTEPS)
        plot_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)

        # ALGEBRA = 'a5'
        # get_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)
        # plot_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)

        # ALGEBRA = 'a6'
        # get_data(hyps_multirun, seeds, algebra=ALGEBRA, maxsteps=MAXSTEPS)
        # plot_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)

        ALGEBRA = 'a7'
        get_data(hyps_multirun, seeds, algebra=ALGEBRA, maxsteps=MAXSTEPS)
        plot_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)

        ALGEBRA = 'a11'
        get_data(hyps_multirun, seeds, algebra=ALGEBRA, maxsteps=MAXSTEPS)
        plot_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)

        # ALGEBRA = 'a17'
        # get_data(hyps_multirun, seeds, algebra=ALGEBRA, maxsteps=MAXSTEPS)
        # plot_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)
    else:
        instance = int(args.instance)

        ALGEBRA = 'a8_circ'  # so(2n)⊕2
        get_data((hyps_multirun[instance],), seeds, algebra=ALGEBRA, maxsteps=MAXSTEPS)

        # ALGEBRA = 'a5'  # su(2^(n−2))⊕4,
        # get_data((hyps[instance],), algebra=ALGEBRA, maxsteps=MAXSTEPS)

        # ALGEBRA = 'a6'  # su(2^(n−2))⊕4,
        # get_data((hyps[instance_i],), algebra=ALGEBRA, maxsteps=MAXSTEPS)

        ALGEBRA = 'a7'  # su(2^n)
        get_data((hyps_multirun[instance],), seeds, algebra=ALGEBRA, maxsteps=MAXSTEPS)

        ALGEBRA = 'a11'  # so(2^(n−2))⊕4,
        get_data((hyps_multirun[instance],), seeds, algebra=ALGEBRA, maxsteps=MAXSTEPS)
        #
        # ALGEBRA = 'a17'  # su(2^(n−2))⊕4,
        # get_data((hyps_multirun[instance],), seeds, algebra=ALGEBRA, maxsteps=MAXSTEPS)
