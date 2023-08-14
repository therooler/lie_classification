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


def exact_diagonalization_random_H(nsites):
    rng = np.random.RandomState(1234)

    H = rng.rand(int(2 ** nsites),
                 int(2 ** nsites)) \
        + 1j * rng.rand(int(2 ** nsites),
                        int(2 ** nsites))
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


def get_random_state(nqubits):
    device = qml.device('default.qubit', wires=nqubits)

    @qml.qnode(device)
    def circuit():
        for i in range(0, nqubits):
            qml.RX(np.random.rand(1), i)
            qml.RY(np.random.rand(1), i)
            qml.RX(np.random.rand(1), i)
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
        qml.PauliRot(params[2, i], pauli_word='IX', wires=(wire, wire + 1))

    for i, wire in enumerate(range(1, nqubits - 1, 2)):
        qml.PauliRot(params[0, i + nqubits // 2], pauli_word='YY', wires=(wire, wire + 1))
        qml.PauliRot(params[2, i + nqubits // 2], pauli_word='IX', wires=(wire, wire + 1))
    qml.PauliRot(params[0, nqubits - 1], pauli_word='YY', wires=(nqubits - 1, 0))
    qml.PauliRot(params[2, nqubits - 1], pauli_word='IX', wires=(nqubits - 1, 0))


def circuit_a8_circ(params, nqubits: int, depth: int, observable, psi0=None):
    if psi0 is not None:
        qml.QubitStateVector(psi0, wires=list(range(nqubits)))
    for d in range(depth):
        layer_a8_circ(params[d], nqubits)

    return qml.expval(observable)


def layer_a7(params, nqubits: int):
    for i, wire in enumerate(range(0, nqubits, 2)):
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
        qml.PauliRot(params[1, i], pauli_word='ZX', wires=(wire, wire + 1))

    for i, wire in enumerate(range(1, nqubits - 1, 2)):
        qml.PauliRot(params[0, i + nqubits // 2], pauli_word='XX', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i + nqubits // 2], pauli_word='XY', wires=(wire, wire + 1))
        qml.PauliRot(params[1, i + nqubits // 2], pauli_word='ZX', wires=(wire, wire + 1))


def circuit_a17(params, nqubits: int, depth: int, observable, psi0=None):
    if psi0 is not None:
        qml.QubitStateVector(psi0, wires=list(range(nqubits)))
    for d in range(depth):
        layer_a17(params[d], nqubits)

    return qml.expval(observable)


### DATA ###

def perform_runs(algebra: str, nqubits: int, depth: int, seed: int = 1234, maxsteps: int = 100):
    # assert not nqubits % 2, 'number of qubits must be even'
    # set paths
    data_path = f'./data/overparameterization_maxsteps_{maxsteps}/'
    filename = f'costs_{algebra}_nq_{nqubits}_d_{depth}_seed_{seed}.csv'
    print(filename)
    if not os.path.exists(data_path + filename):
        print(f"Getting data for N = {nqubits} - Depth = {depth} - Seed {seed}")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        # Set up devices and seed
        dev = qml.device("default.qubit", wires=nqubits)
        np.random.seed(seed)
        state = None
        if algebra == 'a8_circ':
            qnode = qml.QNode(circuit_a8_circ, dev, interface="jax")
            parameters = np.random.rand(depth, 4, nqubits - 1) * np.pi

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
            parameters = np.random.rand(depth, 2, nqubits - 1) * np.pi
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
            parameters = np.random.rand(depth, 3, nqubits - 1) * np.pi
            # Heisenberg
            gs, H = exact_diagonalization_heisenberg(nqubits, delta=1.0,
                                                     interactions=[(i, i + 1) for i in range(nqubits - 1)])

            H = H.toarray()
            # if not (nqubits % 2):
            #     symmetry_X = exact_symmetry_generator(nqubits, 'x').toarray()
            #     symmetry_YZ = exact_symmetry_generator(nqubits, 'yz').toarray()
            #     symmetry_ZY = exact_symmetry_generator(nqubits, 'zy').toarray()
            #     # |sym>
            #     eigenstates = np.linalg.eigh(symmetry_YZ)[1]

            state = get_random_state(nqubits)

            print(gs)
            gs = gs[0]
        elif algebra == 'a7':
            qnode = qml.QNode(circuit_a7, dev, interface="jax")
            parameters = np.random.rand(depth, 3, nqubits - 1) * np.pi
            # Heisenberg
            gs, H = exact_diagonalization_heisenberg(nqubits, delta=1.0,
                                                     interactions=[(i, i + 1) for i in range(nqubits - 1)])

            H = H.toarray()
            symmetry_X = exact_symmetry_generator(nqubits, 'x').toarray()
            symmetry_Y = exact_symmetry_generator(nqubits, 'y').toarray()
            symmetry_Z = exact_symmetry_generator(nqubits, 'z').toarray()

            # |phi+>
            # state = get_bell_state(nqubits)
            eigenstates = np.linalg.eigh(symmetry_X)[1]
            state = eigenstates[:, 3]
            # state = get_random_state(nqubits)
            state_L = np.outer(state.conj(), state)

            print('commutes with H', commutes(state_L, H))
            print('commutes with P_X', commutes(state_L, symmetry_X))
            print('commutes with P_Y', commutes(state_L, symmetry_Y))
            print('commutes with P_Z', commutes(state_L, symmetry_Z))

            print(gs)
            gs = gs[0]
        elif algebra == 'a17':
            qnode = qml.QNode(circuit_a17, dev, interface="jax")
            parameters = np.random.rand(depth, 3, nqubits - 1) * np.pi
            # Heisenberg
            gs, H = exact_diagonalization_random_H(nqubits)

            state = get_zero_state(nqubits)

            print(gs[:2])
            gs = gs[0]
        else:
            raise NotImplementedError

        obs = qml.Hermitian(H, wires=list(range(nqubits)))
        cost_fn = lambda x: qnode(x, nqubits, depth, obs, psi0=state)

        # Calculate all runs in one vmap
        costs = optimization(parameters,
                             loss=cost_fn,
                             # grad_fn=grad_fn,
                             max_steps=maxsteps,
                             learning_rate=LEARNING_RATE)
        costs = np.array(costs)
        gs_array = np.ones_like(costs) * gs
        grads = np.stack([costs, gs_array], axis=1)

        df = pd.DataFrame(grads)
        df.to_csv(data_path + filename)
    else:
        print("Data already exists, skipping...")


def optimization(params, loss, max_steps, learning_rate):
    costs = []
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def opt_step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(1, max_steps + 1):
        params, opt_state, c = opt_step(params, opt_state)
        costs.append(c)
        if not i % 100:
            print(f"step {i} - cost={c}")
    return costs


def get_data(hyps, algebra: str, maxsteps: int = 100):
    for (n, d, s) in hyps:
        perform_runs(algebra, n, d, s, maxsteps=maxsteps)


### PLOTTING ###

def plot_data(hyps, algebra, seed: int = 1233, maxsteps: int = 100):
    data_path = f'./data/overparameterization_maxsteps_{maxsteps}/'
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
    data_path = f'./data/overparameterization_maxsteps_{maxsteps}/'
    fig_path = f'./data/overparameterization_maxsteps_{maxsteps}/figures/'
    indices = np.stack(np.unravel_index(range(len(nqubits) * len(depth_list) * len(seeds)),
                                        shape=(len(nqubits), len(depth_list), len(seeds)))).T
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(nqubits)))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(fig_path + f'{algebra_1}_costs.npy') or not os.path.exists(
            fig_path + f'{algebra_2}_costs.npy') & 1:
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
                data_1 = pd.read_csv(data_path + file_path_1, index_col=False).values[:, 1:]

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
                data_2 = pd.read_csv(data_path + file_path_2, index_col=False).values[:, 1:]
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
    axs[1].set_xlabel('$L$')
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


def plot_combined_steps(hyps, algebra_1, algebra_2, maxsteps: int = 100):
    data_path = f'./data/overparameterization_maxsteps_{maxsteps}/'
    fig_path = f'./data/overparameterization_maxsteps_{maxsteps}/figures/'
    indices = np.stack(np.unravel_index(range(len(nqubits) * len(depth_list) * len(seeds)),
                                        shape=(len(nqubits), len(depth_list), len(seeds)))).T
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(nqubits)))
    threshold = 5e-4
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(fig_path + f'{algebra_1}_steps.npy') or not os.path.exists(
            fig_path + f'{algebra_2}_steps.npy') & 1:
        steps_1 = np.ones((len(nqubits), len(depth_list), len(seeds)))
        steps_1_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_2 = np.ones((len(nqubits), len(depth_list), len(seeds)))
        steps_2_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)

        for i, (n, d, s) in enumerate(hyps):
            idx = indices[i]
            file_path_1 = f'costs_{algebra_1}_nq_{n}_d_{d}_seed_{s}.csv'
            file_path_2 = f'costs_{algebra_2}_nq_{n}_d_{d}_seed_{s}.csv'
            try:
                data_1 = pd.read_csv(data_path + file_path_1, index_col=False).values[:, 1:]
                gs_1 = data_1[0, -1]
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
                data_2 = pd.read_csv(data_path + file_path_2, index_col=False).values[:, 1:]
                gs_2 = data_2[0, -1]
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

        np.save(fig_path + f'{algebra_1}_steps.npy', steps_1)
        np.save(fig_path + f'{algebra_2}_steps.npy', steps_2)
        np.save(fig_path + f'{algebra_1}_steps_mask.npy', steps_1_mask)
        np.save(fig_path + f'{algebra_2}_steps_mask.npy', steps_2_mask)
    else:
        steps_1 = np.load(fig_path + f'{algebra_1}_steps.npy')
        steps_2 = np.load(fig_path + f'{algebra_2}_steps.npy')
        steps_1_mask = np.load(fig_path + f'{algebra_1}_steps_mask.npy')
        steps_2_mask = np.load(fig_path + f'{algebra_2}_steps_mask.npy')
    print(f"{algebra_1} files found - {np.sum(steps_1_mask)}/{len(hyps)}")
    print(f"{algebra_2} files found - {np.sum(steps_2_mask)}/{len(hyps)}")
    costs_1_ma = np.ma.array(steps_1, mask=~steps_1_mask)
    costs_2_ma = np.ma.array(steps_2, mask=~steps_2_mask)
    if 1:
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

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(4, 3.5)
    for j, n in enumerate(nqubits):
        axs[0].plot(depth_list, np.clip(costs_1_ma[j, :].mean(axis=1), a_min=1e-10, a_max=np.inf), label=rf'$n$ = {n}',
                    color=colors[j])

    axs[0].set_ylabel(r'Steps')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('$L$')
    axs[0].set_ylim([25, maxsteps + 0.1 * maxsteps])
    axs[0].set_title(r'(a) $\mathfrak{a}_{8}^\circ$', y=-0.25)
    axs[0].tick_params(axis="x", which='both', top=False, right=False)
    axs[0].tick_params(axis="y", which='both', top=False, right=False)
    axs[0].legend(prop={'size': 8})
    for j, n in enumerate(nqubits):
        axs[1].plot(depth_list, np.clip(costs_2_ma[j, :].mean(axis=1), a_min=1e-10, a_max=np.inf),
                    label=rf'$n$ = {n}', color=colors[j])

    fig.subplots_adjust(wspace=0.0)
    axs[1].set_xlabel('$L$')
    axs[1].set_yscale('log')
    axs[1].set_ylim([25, maxsteps + 0.1 * maxsteps])
    axs[1].set_title(r'(b) $\mathfrak{a}_{6}$', y=-0.25)
    axs[1].set_yticks(())
    axs[1].tick_params(axis="x", which='both', top=False, right=False)
    axs[1].tick_params(axis="y", which='both', top=False, right=False, left=False)
    axs[1].legend(prop={'size': 8})

    fig.savefig(fig_path + f'{algebra_1}_{algebra_2}_overparam_steps.pdf')

    if SHOW:
        plt.show()


def plot_combined_success(hyps, algebra_1, algebra_2, maxsteps: int = 100):
    data_path = f'./data/overparameterization_maxsteps_{maxsteps}/'
    fig_path = f'./data/overparameterization_maxsteps_{maxsteps}/figures/'
    indices = np.stack(np.unravel_index(range(len(nqubits) * len(depth_list) * len(seeds)),
                                        shape=(len(nqubits), len(depth_list), len(seeds)))).T
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(nqubits)))
    threshold = 5e-4
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(fig_path + f'{algebra_1}_success.npy') or not os.path.exists(
            fig_path + f'{algebra_2}_success.npy') & 1:
        steps_1 = np.ones((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_1_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_2 = np.ones((len(nqubits), len(depth_list), len(seeds)), dtype=bool)
        steps_2_mask = np.zeros((len(nqubits), len(depth_list), len(seeds)), dtype=bool)

        for i, (n, d, s) in enumerate(hyps):
            idx = indices[i]
            file_path_1 = f'costs_{algebra_1}_nq_{n}_d_{d}_seed_{s}.csv'
            file_path_2 = f'costs_{algebra_2}_nq_{n}_d_{d}_seed_{s}.csv'
            try:
                data_1 = pd.read_csv(data_path + file_path_1, index_col=False).values[:, 1:]
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
                data_2 = pd.read_csv(data_path + file_path_2, index_col=False).values[:, 1:]
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

        np.save(fig_path + f'{algebra_1}_success.npy', steps_1)
        np.save(fig_path + f'{algebra_2}_success.npy', steps_2)
        np.save(fig_path + f'{algebra_1}_success_mask.npy', steps_1_mask)
        np.save(fig_path + f'{algebra_2}_success_mask.npy', steps_2_mask)
    else:
        steps_1 = np.load(fig_path + f'{algebra_1}_success.npy')
        steps_2 = np.load(fig_path + f'{algebra_2}_success.npy')
        steps_1_mask = np.load(fig_path + f'{algebra_1}_success_mask.npy')
        steps_2_mask = np.load(fig_path + f'{algebra_2}_success_mask.npy')
    print(f"{algebra_1} files found - {np.sum(steps_1_mask)}/{len(hyps)}")
    print(f"{algebra_2} files found - {np.sum(steps_2_mask)}/{len(hyps)}")
    costs_1_ma = np.ma.array(steps_1, mask=~steps_1_mask)
    costs_2_ma = np.ma.array(steps_2, mask=~steps_2_mask)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(4, 3.5)
    for j, n in enumerate(nqubits):
        print(n, costs_1_ma[j, :].sum(axis=1) / steps_1_mask[j, :].sum(axis=1))
        print(n, steps_1_mask[j, :].sum(axis=1))
        axs[0].plot(depth_list, costs_1_ma[j, :].sum(axis=1) / steps_1_mask[j, :].sum(axis=1), label=rf'$n$ = {n}',
                    color=colors[j])

    axs[0].set_ylabel(r'Success probability')
    axs[0].set_xlabel('$L$')
    axs[0].set_ylim([-0.05, 1.05])
    axs[0].set_title(r'(a) $\mathfrak{a}_{8}^\circ$', y=-0.25)
    axs[0].tick_params(axis="x", which='both', top=False, right=False)
    axs[0].tick_params(axis="y", which='both', top=False, right=False)
    axs[0].legend(prop={'size': 8})
    for j, n in enumerate(nqubits):
        print(n, costs_2_ma[j, :].sum(axis=1) / steps_2_mask[j, :].sum(axis=1))
        print(n, steps_2_mask[j, :].sum(axis=1))
        axs[1].plot(depth_list, costs_2_ma[j, :].sum(axis=1) / steps_2_mask[j, :].sum(axis=1), label=rf'$n$ = {n}',
                    color=colors[j])

    fig.subplots_adjust(wspace=0.0)
    axs[1].set_xlabel('$L$')
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].set_title(r'(b) $\mathfrak{a}_{6}$', y=-0.25)
    axs[1].set_yticks(())
    axs[1].tick_params(axis="x", which='both', top=False, right=False)
    axs[1].tick_params(axis="y", which='both', top=False, right=False, left=False)
    axs[1].legend(prop={'size': 8})

    fig.savefig(fig_path + f'{algebra_1}_{algebra_2}_overparam_success.pdf')

    if SHOW:
        plt.show()


if __name__ == '__main__':

    import argparse

    number_of_runs = 100
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', default=None)
    args = parser.parse_args()
    nqubits = [n for n in range(4, 13)]
    depth_list = [2, 4, 8, 12, 16, 20, 24, 28, 32]
    seeds = [2 ** 8 * i for i in range(number_of_runs)]
    hyps = list(itertools.product(nqubits, depth_list, seeds))
    print(f"Number of hyps= {len(hyps)}")
    MAXSTEPS = 3000
    # plot_combined_costs(hyps, 'a8_circ', 'a17', maxsteps=MAXSTEPS)
    plot_combined_steps(hyps, 'a8_circ', 'a17', maxsteps=MAXSTEPS)
    # plot_combined_success(hyps, 'a8_circ', 'a17', maxsteps=MAXSTEPS)

    if args.instance is None:
        # #
        ALGEBRA = 'a8_circ'
        get_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)
        plot_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)

        # ALGEBRA = 'a5'
        # get_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)
        # plot_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)

        ALGEBRA = 'a6'
        get_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)
        plot_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS)

        # ALGEBRA = 'a7'
        # get_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS, seed=1233)
        # plot_data(hyps, algebra=ALGEBRA, maxsteps=MAXSTEPS, seed=1233)
    else:
        instance = int(args.instance)
        for ins in range(number_of_runs):
            instance_i = instance * number_of_runs + ins
            print(instance_i)
            ALGEBRA = 'a8_circ'  # so(2n)⊕2
            get_data((hyps[instance_i],), algebra=ALGEBRA, maxsteps=MAXSTEPS)

            # ALGEBRA = 'a5'  # su(2^(n−2))⊕4,
            # get_data((hyps[instance],), algebra=ALGEBRA, maxsteps=MAXSTEPS)

            # ALGEBRA = 'a6'  # su(2^(n−2))⊕4,
            # get_data((hyps[instance_i],), algebra=ALGEBRA, maxsteps=MAXSTEPS)

            # ALGEBRA = 'a7'  # su(2^n)
            # get_data((hyps[instance_i],), algebra=ALGEBRA, maxsteps=MAXSTEPS)

            ALGEBRA = 'a17'  # su(2^(n−2))⊕4,
            get_data((hyps[instance_i],), algebra=ALGEBRA, maxsteps=MAXSTEPS)
