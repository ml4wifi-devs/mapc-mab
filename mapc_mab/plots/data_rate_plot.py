import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from mapc_sim.constants import DATA_RATES, TAU

from mapc_mab.plots.config import AGENT_NAMES, get_cmap
from mapc_mab.plots.utils import confidence_interval


def plot(scenario_results: dict, scenario_config: dict, aggregate_steps: int) -> None:
    colors = get_cmap(len(scenario_results))
    n_points = scenario['scenario']['n_steps'] // aggregate_steps
    xs = np.linspace(0, scenario['scenario']['n_steps'], n_points) * TAU

    if 'mcs' in scenario_config['params']:
        plt.axhline(DATA_RATES[scenario_config['params']['mcs']], linestyle='--', color='gray', label='Single TX')

    for c, (name, data) in zip(colors, scenario_results.items()):
        for run, hierarchical in data:
            mean, ci_low, ci_high = confidence_interval(np.asarray(run))

            if hierarchical:
                plt.plot(xs, mean, label=AGENT_NAMES.get(name, name), c=c)
            else:
                plt.plot(xs, mean, linestyle='--', c=c)

            plt.fill_between(xs, ci_low, ci_high, alpha=0.3, color=c, linewidth=0.0)

    plt.xlabel('Time [s]')
    plt.ylabel('Effective data rate [Mb/s]')
    plt.xlim((xs[0], xs[-1]))
    plt.ylim(bottom=0)
    plt.grid()
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f'rate-{scenario_config["name"]}.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-f', '--file', type=str, required=True)
    args.add_argument('-s', '--aggregate_steps', type=int, required=True)
    args = args.parse_args()

    with open(args.file, 'r') as file:
        results = json.load(file)

    for scenario in results:
        scenario_results = defaultdict(list)

        for agent in scenario['agents']:
            runs = [np.array(run).reshape((-1, args.aggregate_steps)).mean(axis=-1) for run in agent['runs']]
            scenario_results[agent['agent']['name']].append((runs, agent['agent']['hierarchical']))

        plot(scenario_results, scenario['scenario'], args.aggregate_steps)
