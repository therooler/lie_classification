import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.style.use('science')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
SHOW = True


def dim_su(x):
    return x ** 2 - 1


def dim_so(x):
    return x * (x - 1) // 2


def dim_sp(x):
    return x * (x + 1) // 2


def add_algebra(ax, nlist, dims, name, color, linestyle='-'):
    ax.plot(nlist, dims, label=name, color=color, markersize=15, linewidth=2, linestyle=linestyle)


def dim_a3(x):
    if (x % 6) == 3:
        return dim_sp(2 ** (x - 1))
    elif (x % 6) == 1 or (x % 6) == 5:
        return dim_so(2 ** (x - 1))
    elif (x % 6) == 2 or (x % 6) == 4:
        return dim_su(2 ** (x - 2)) * 2
    elif not (x % 6):
        return dim_so(2 ** (x - 2)) * 4


def dim_a5(x):
    if (x % 8) == 3 or (x % 8) == 5:
        return dim_sp(2 ** (x - 1))
    elif (x % 8) == 4:
        return dim_sp(2 ** (x - 2)) * 4
    elif (x % 8) == 2 or (x % 8) == 6:
        return dim_su(2 ** (x - 2)) * 2
    elif (x % 8) == 1 or (x % 8) == 7:
        return dim_so(2 ** (x - 1))
    elif not (x % 8):
        return dim_so(2 ** (x - 2)) * 4


def dim_a6(x):
    if (x % 2) and (x >= 4):  # even
        return dim_su(2 ** (x - 2)) * 4
    else:  # odd
        return dim_su(2 ** (x - 1))


def plot_dimensions_annotated_closed(closed=False):
    MAXN = 8
    Nlist = list(i for i in range(3, MAXN + 1))
    Nlist_2 = list(2 ** i for i in range(3, MAXN + 1))
    Nlist_2 = Nlist
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    NUM_REDS = 8
    NUM_BLUES = 8
    reds_cm = plt.get_cmap('Reds')
    blues_cm = plt.get_cmap('Blues')
    colors_reds = [reds_cm(0.2 + 0.8 * i / NUM_REDS) for i in range(NUM_REDS)]
    colors_blues = [blues_cm(0.2 + 0.8 * i / NUM_BLUES) for i in range(NUM_BLUES)]

    add_algebra(axs, Nlist_2, list(map(lambda x: x - 1, Nlist)),
                r'$\mathfrak{a}_0$',
                color=colors_blues[0])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_so(x), Nlist)),
                r'$\mathfrak{a}_1$',
                color=colors_blues[1])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_so(x) * 2, Nlist)),
                r'$\mathfrak{a}_2\cong\mathfrak{a}_4$',
                color=colors_blues[2])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_so(2 * x - 1), Nlist)),
                r'$\mathfrak{a}_8$',
                color=colors_blues[3])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_so(2 * x), Nlist)),
                r'$\mathfrak{a}_{14}$',
                color=colors_blues[4])
    add_algebra(axs, Nlist_2, list(map(lambda x: x, Nlist)),
                r'$\mathfrak{b}_0$',
                color=colors_blues[5])
    add_algebra(axs, Nlist_2, list(map(lambda x: 2*x-1, Nlist)),
                r'$\mathfrak{b}_1$',
                color=colors_blues[6])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_su(2) * x, Nlist)),
                r'$\mathfrak{b}_3$',
                color=colors_blues[7])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_a3(x), Nlist)),
                r'$\mathfrak{a}_3$',
                color=colors_reds[0])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_a5(x), Nlist)),
                r'$\mathfrak{a}_5$',
                color=colors_reds[1])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_a6(x), Nlist)),
                r'$\mathfrak{a}_6\cong \mathfrak{a}_7\cong\mathfrak{a}_{10}$',
                color=colors_reds[2])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_sp(2 ** (x - 1)), Nlist)),
                r'$\mathfrak{a}_9$',
                color=colors_reds[3])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_so(2 ** x), Nlist)),
                r'$\mathfrak{a}_{11}=\mathfrak{a}_{16}$',
                color=colors_reds[4])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_su(2 ** (x - 1)) * 2, Nlist)),
                r'$\mathfrak{a}_{13}\cong\mathfrak{a}_{15}\cong\mathfrak{a}_{20}$',
                color=colors_reds[5])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_su(2 ** x), Nlist)),
                r'\begin{align*}&\mathfrak{a}_{12}=\mathfrak{a}_{17}=\mathfrak{a}_{18}=\\&\mathfrak{a}_{19}=\mathfrak{a}_{21}=\mathfrak{a}_{22}\end{align*}',
                color=colors_reds[5])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_sp(2 ** (x-1)) + 1, Nlist)),
                r'$\mathfrak{b}_2$',
                color=colors_reds[6])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_su(2 ** (x - 1))*2 + 1, Nlist)),
                r'$\mathfrak{b}_4$',
                color=colors_reds[7])
    list(map(lambda x: x - 1, Nlist))

    axs.set_xlabel(r'$N$')
    axs.set_ylabel('Dim')
    # axs.set_xscale('log')
    axs.set_yscale('log')
    axs.legend(prop={'size': 6}, frameon=True)
    fig.savefig(f'../figures/{"closed" if closed else "open"}/scaling_annotated_paper.pdf')

    if SHOW:
        plt.show()
    plt.close()


def plot_dimensions_annotated_closed_filled(closed=False):
    MAXN = 8
    Nlist = list(i for i in range(3, MAXN + 1))
    Nlist_2 = list(2 ** i for i in range(3, MAXN + 1))
    Nlist_2 = Nlist
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    val = 0.8
    alpha= 0.3
    reds_cm = plt.get_cmap('Reds')
    blues_cm = plt.get_cmap('Blues')
    greens_cm = plt.get_cmap('Greens')
    colors_reds = [reds_cm(val)]
    colors_blues = [blues_cm(val)]
    colors_greens = [greens_cm(val)]

    add_algebra(axs, Nlist_2, list(map(lambda x: x - 1, Nlist)),
                r'$\mathfrak{a}_0$',
                color=colors_greens[0])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_su(2) * x, Nlist)),
                r'$\mathfrak{b}_3$',
                color=colors_greens[0], linestyle='--')
    axs.fill_between(Nlist_2,
                     list(map(lambda x: x - 1, Nlist)),
                     list(map(lambda x: dim_su(2) * x, Nlist)),
                     alpha=alpha, color=colors_greens[0])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_so(x), Nlist)),
                r'$\mathfrak{a}_1$',
                color=colors_blues[0])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_so(2 * x), Nlist)),
                r'$\mathfrak{a}_{14}$',
                color=colors_blues[0], linestyle='--')
    axs.fill_between(Nlist_2,
                     list(map(lambda x: dim_so(x), Nlist)),
                     list(map(lambda x: dim_so(2 * x), Nlist)),
                     alpha=alpha, color=colors_blues[0])

    add_algebra(axs, Nlist_2, list(map(lambda x: dim_a3(x), Nlist)),
                r'$\mathfrak{a}_3$',
                color=colors_reds[0])
    add_algebra(axs, Nlist_2, list(map(lambda x: dim_su(2 ** x), Nlist)),
                r'$\mathfrak{a}_{12}$',
                color=colors_reds[0], linestyle='--')
    axs.fill_between(Nlist_2,
                     list(map(lambda x: dim_a3(x), Nlist)),
                     list(map(lambda x: dim_su(2 ** x), Nlist)),
                     alpha=alpha, color=colors_reds[0])

    axs.set_xlabel(r'$n$', fontsize=20)
    axs.set_ylabel(r'$\dim(\mathfrak{g})$', fontsize=20)
    # axs.set_xscale('log')
    axs.set_yscale('log')
    axs.legend(prop={'size': 12}, frameon=True)
    fig.savefig(f'../figures/{"closed" if closed else "open"}/scaling_annotated_paper.pdf')

    if SHOW:
        plt.show()
    plt.close()


if __name__ == "__main__":
    print("Creating figures...")
    print(os.getcwd())
    plot_dimensions_annotated_closed_filled()
