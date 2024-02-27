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
    "scenario_4m": 15,
    "scenario_9m": 15,
    "scenario_20m_long": 75,
}
TITLES = {
    "scenario_4m": r"(a) $d=4$ m",
    "scenario_9m": r"(b) $d=9$ m",
    "scenario_20m_long": r"(c) $d=20$ m",
}
CLASSIC_MAB = {
    "scenario_4m": "Softmax",
    "scenario_9m": "Softmax",
    "scenario_20m_long": "Softmax",
}


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-f', '--file', type=str, required=True)
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

        ax.axhline(DATA_RATES[11], linestyle='--', color='gray', label='Single TX')

        for c, (name, data) in zip(colors, scenario_results.items()):
            for run, hierarchical in data:
                mean, ci_low, ci_high = confidence_interval(np.asarray(run))

                if hierarchical:
                    ax.plot(xs, mean, label=AGENT_NAMES.get(name, name), c=c, marker='o')
                    ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color=c, linewidth=0.0)
                elif name == CLASSIC_MAB[scenario_name]:
                    ax.plot(xs, mean, linestyle='--', marker='^', c='gray', markersize=2, label='Best classical MAB')
                    ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color='gray', linewidth=0.0)

        ax.set_title(TITLES[scenario_name], y=-0.45, fontsize=12)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_xlim((xs[0], xs[-1]))
        ax.set_ylim(bottom=0, top=400)
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.grid()

        if scenario_name == 'scenario_4m':
            ax.set_ylabel('Effective data rate [Mb/s]', fontsize=12)
            ax.legend(ncols=2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'data-rates.pdf', bbox_inches='tight')
    plt.clf()
