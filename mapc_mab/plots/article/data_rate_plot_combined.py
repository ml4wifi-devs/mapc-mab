import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from mapc_sim.constants import DATA_RATES, TAU

from mapc_mab.plots.config import AGENT_NAMES, COLUMN_WIDTH, COLUMN_HIGHT, get_cmap
from mapc_mab.plots.utils import confidence_interval


plt.rcParams.update({
    'figure.figsize': (3 * COLUMN_WIDTH, COLUMN_HIGHT + 0.7),
    'legend.fontsize': 9
})

AGGREGATE_STEPS = {
    "scenario_10m": 20,
    "scenario_20m": 20,
    "scenario_30m_long": 100,
}
TITLES = {
    "scenario_10m": r"(a) $d=10$ m",
    "scenario_20m": r"(b) $d=20$ m",
    "scenario_30m_long": r"(c) $d=30$ m",
}
CLASSIC_MAB = {
    "scenario_10m": "EGreedy",
    "scenario_20m": "NormalThompsonSampling",
    "scenario_30m_long": "Softmax",
}


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-f', '--file', type=str, default=f'../../envs/all_results.json')
    args = args.parse_args()

    with open(args.file, 'r') as file:
        results = json.load(file)

    fig, axes = plt.subplots(1, 3, sharey=True)
    fig.subplots_adjust(wspace=0.)

    for ax, scenario in zip(axes, results):
        scenario_name = scenario['scenario']['name']
        scenario_results = defaultdict(list)

        for agent in scenario['agents']:
            runs = [np.array(run).reshape((-1, AGGREGATE_STEPS[scenario_name])).mean(axis=-1) for run in agent['runs']]
            scenario_results[agent['agent']['name']].append((runs, agent['agent']['hierarchical']))

        colors = get_cmap(len(scenario_results))
        n_points = scenario['scenario']['n_steps'] // AGGREGATE_STEPS[scenario_name]
        xs = np.linspace(0, scenario['scenario']['n_steps'], n_points) * TAU

        if 'mcs' in scenario['scenario']['params']:
            ax.axhline(DATA_RATES[scenario['scenario']['params']['mcs']], linestyle='--', color='gray')

        if 'sec' in scenario['scenario']:
            for sec in scenario['scenario']['switch_steps']:
                ax.axvline(sec * TAU, linestyle='--', color='red')

        for c, (name, data) in zip(colors, scenario_results.items()):
            for run, hierarchical in data:
                mean, ci_low, ci_high = confidence_interval(np.asarray(run))

                if hierarchical:
                    ax.plot(xs, mean, c=c, marker='o')
                    ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color=c, linewidth=0.0)
                elif name == CLASSIC_MAB[scenario_name]:
                    ax.plot(xs, mean, linestyle='--', marker='^', c='gray', markersize=2)
                    ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color='gray', linewidth=0.0)

        ax.set_title(TITLES[scenario_name], y=-0.45, fontsize=12)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_xlim((xs[0], xs[-1]))
        ax.set_ylim(bottom=0, top=425)
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.grid()

        if scenario_name == 'scenario_10m':
            ax.set_ylabel('Effective data rate [Mb/s]', fontsize=12)
            ax.plot([], [], 'o', linestyle='-', c=colors[0], label=r'$\varepsilon$-greedy')
            ax.plot([], [], 'o', linestyle='-', c=colors[1], label='Softmax')
            ax.plot([], [], 'o', linestyle='-', c=colors[2], label='UCB')
            ax.plot([], [], 'o', linestyle='-', c=colors[3], label='TS')
            ax.legend(loc='upper left', title='Hierarchical MABs')

            ax2 = ax.twinx()
            ax2.axis('off')
            ax2.plot([], [], linestyle='--', c="gray", label='Single TX')
            ax2.plot([], [], '^', linestyle='--', c='gray', label='Best flat MAB')
            ax2.legend(loc='upper right', title='Baselines')

    plt.tight_layout()
    plt.savefig(f'data-rates.pdf', bbox_inches='tight')
    plt.show()
