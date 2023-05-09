import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.style.use('science')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
SHOW = False


def dim_su(x):
    return x ** 2 - 1


def dim_so(x):
    return x * (x - 1) // 2


def dim_sp(x):
    return x * (x + 1) // 2


def plot_dimensions_annotated(add_I=False):
    Nlist_2 = [4, 8, 16, 32, 64, 128]
    Nlist = [int(np.log2(x)) for x in Nlist_2]
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)
    NUM_ALGEBRAS = 23
    NUM_REDS = 7
    NUM_BLUES = 5
    reds_cm = plt.get_cmap('Reds')
    blues_cm = plt.get_cmap('Blues')
    colors_reds = [reds_cm(0.2 + 0.8 * i / NUM_REDS) for i in range(NUM_REDS)]
    colors_blues = [blues_cm(0.2 + 0.8 * i / NUM_BLUES) for i in range(NUM_BLUES)]
    for N in Nlist_2:
        if add_I:
            directory = f'./data/su{N}_I'
        else:
            directory = f'./data/su{N}'
        df = pd.read_csv(directory + '/' + 'meta.txt', delimiter=',')
        axs.scatter([N] * len(df['dim'].values), df['dim'].values, color='black')

    axs.plot(Nlist_2, list(map(lambda x: dim_su(2 ** x), Nlist)), label=r'$\mathfrak{su}(2^N)$', color=colors_reds[0],
             markersize=15,
             linewidth=2)
    axs.plot(Nlist_2, list(map(lambda x: 2 ** (2 * x - 1) - 2, Nlist)), label=r'$2^{2 N-1  }-2 $', color=colors_reds[1],
             markersize=15, linewidth=2)
    axs.plot(Nlist_2, list(map(lambda x: dim_su(2 ** (x - 1)), Nlist)), label=r'$\mathfrak{su}(2^{N-1})$',
             color=colors_reds[2],
             markersize=15, linewidth=2)
    axs.plot(Nlist_2[2:], list(map(lambda x: dim_su(2 ** (x - 1)) - 3, Nlist[2:])), label=r'Heisenberg',
             color=colors_reds[3], markersize=15, linewidth=2)
    axs.plot(Nlist_2, list(map(lambda x: dim_so(2 ** x), Nlist)), label=r'$\mathfrak{so}(2^N)$', color=colors_reds[4],
             markersize=15,
             linewidth=2)
    axs.plot(Nlist_2, list(map(lambda x: 2 ** (x - 2) * (2 ** (x - 1) + 1), Nlist)), label=r'$2^{N-2}(1+2^{N-1})$',
             color=colors_reds[NUM_BLUES], markersize=15, linewidth=2)
    axs.plot(Nlist_2, list(map(lambda x: dim_so(2 ** (x - 1)), Nlist)), label=r'$\mathfrak{so}(2^{N-1})$',
             color=colors_reds[6],
             markersize=15, linewidth=2)
    axs.plot(Nlist_2, list(map(lambda x: dim_so(2 * x), Nlist)), label=r'$\mathfrak{so}(2N)$', color=colors_blues[0],
             markersize=15,
             linewidth=2)
    axs.plot(Nlist_2, list(map(lambda x: dim_so(2 * x - 1), Nlist)), label=r'$\mathfrak{so}(2N - 1)$',
             color=colors_blues[1],
             markersize=15, linewidth=2)
    axs.plot(Nlist_2, list(map(lambda x: x * (x - 1), Nlist)), label=r'$XY$', color=colors_blues[2], markersize=15,
             linewidth=2)
    axs.plot(Nlist_2, list(map(lambda x: dim_so(x), Nlist)), label=r'$\mathfrak{so}(N)$', color=colors_blues[3],
             markersize=15,
             linewidth=2)
    axs.plot(Nlist_2, list(map(lambda x: x - 1, Nlist)), label=r'$U(1)^{\otimes N}$', color=colors_blues[4],
             markersize=15,
             linewidth=2)

    axs.set_xlabel(r'$2^N$')
    axs.set_ylabel('Dim')
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.legend(prop={'size': 6})
    # axs.set_title(
    #     f'Identification of subalgebras of 1D translational invariant '
    #     f'Hamiltonians {"(with I)" if add_I else ""}')
    fig.savefig(f'./figures/scaling_annotated{"_I" if add_I else ""}.pdf')
    fig.savefig(f'./figures/scaling_annotated{"_I" if add_I else ""}.png')

    if SHOW:
        plt.show()
    plt.close()


def plot_dimensions_per_set(add_I=False):
    Nlist_2 = [4, 8, 16, 32, 64, 128]
    Nlist = [int(np.log2(x)) for x in Nlist_2]
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)
    NUM_ALGEBRAS = 23
    dims = np.zeros((NUM_ALGEBRAS, len(Nlist_2)))
    cm_3 = plt.get_cmap('Dark2')
    cm = plt.get_cmap('tab20')
    colors = [cm(1. * i / 20) for i in range(20)] + [cm_3(1. * i / 8) for i in range(8)]
    for i in range(NUM_ALGEBRAS):
        for j, N in enumerate(Nlist_2):
            if add_I:
                directory = f'./data/su{N}_I'
            else:
                directory = f'./data/su{N}'
            with open(directory + '/' + f'pauliset_{i}.txt', 'r') as f:
                s = f.readline()
                s = s.strip('\n')
            dims[i, j] = int(s.split(" ")[-1])
        axs.plot(Nlist_2, dims[i, :], label=rf'$A_{{i}}$', marker='.', color=colors[i], markersize=15, linewidth=2)
    axs.legend()
    axs.set_xlabel(r'$2^N$')
    axs.set_ylabel('Dim')
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.grid()
    axs.legend(prop={'size': 6})
    # axs.set_title(
    #     f'Dimensions of all {NUM_ALGEBRAS} subalgebras of 1D translational invariant '
    #     f'Hamiltonians {"(with I)" if add_I else ""}')
    fig.savefig(f'./figures/scaling_traced{"_I" if add_I else ""}.pdf')
    fig.savefig(f'./figures/scaling_traced{"_I" if add_I else ""}.png')

    if SHOW:
        plt.show()
    plt.close()


if __name__ == "__main__":
    print("Creating figures...")
    plot_dimensions_annotated(False)
    plot_dimensions_per_set(False)
    plot_dimensions_annotated(True)
    plot_dimensions_per_set(True)
    print("Done!")
