import matplotlib.pyplot as plt


COLUMN_WIDTH = 3.5
COLUMN_HIGHT = 2 * COLUMN_WIDTH / (1 + 5 ** 0.5)

PLOT_PARAMS = {
    'figure.figsize': (COLUMN_WIDTH, COLUMN_HIGHT),
    'figure.dpi': 72,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': 'cm',
    'axes.titlesize': 9,
    'axes.linewidth': 0.5,
    'grid.alpha': 0.42,
    'grid.linewidth': 0.5,
    'legend.title_fontsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 0.5,
    'text.usetex': True,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}


AGENT_NAMES = {
    'egreedy': r'$\varepsilon$-greedy',
    'exp3': 'EXP3',
    'softmax': 'Softmax',
    'ucb': 'UCB'
}


def set_style() -> None:
    plt.rcParams.update(PLOT_PARAMS)
