import matplotlib.pyplot as plt

from mapc_mab.envs.scenarios import StaticScenario
from mapc_mab.envs.scenarios.static import simple_scenario_5
from mapc_mab.plots.config import get_cmap


def plot(scenario: StaticScenario, filename: str) -> None:
    plt.rcParams['axes.linewidth'] = 0.0

    c = get_cmap(4)[1]
    _, ax = plt.subplots()

    for i, (ap, stations) in enumerate(scenario.associations.items()):
        ax.scatter(scenario.pos[ap, 0], scenario.pos[ap, 1], marker='x', color=c, s=15)
        ax.scatter(scenario.pos[stations, 0], scenario.pos[stations, 1], marker='.', color=c)
        ax.annotate(r'AP$_{0}$'.format(i + 1), (scenario.pos[ap, 0], scenario.pos[ap, 1] + 3), va='bottom', ha='center')

    for wall in scenario.walls_pos:
        ax.plot([wall[0], wall[2]], [wall[1], wall[3]], color='black', linewidth=2)

    ax.set_axisbelow(True)
    ax.set_xlim((-10, 30))
    ax.set_ylim((-10, 30))
    ax.set_xticks([-10, 0, 10, 20, 30])
    ax.set_yticks([-10, 0, 10, 20, 30])
    ax.set_xticklabels(['', '0', '', r'$d$', ''])
    ax.set_yticklabels(['', '0', '', r'$d$', ''])
    ax.tick_params(axis='both', which='both', labelsize=10)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_aspect('equal')
    ax.grid()

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    s = simple_scenario_5(d_ap=20, d_sta=2)
    plot(s, 'loc-simple_scenario_5.pdf')
