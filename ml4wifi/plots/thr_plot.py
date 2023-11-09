import json
from argparse import ArgumentParser
from typing import List

import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from chex import Array

from ml4wifi.envs.sim import DATA_RATES
from ml4wifi.plots.config import AGENT_NAMES
from ml4wifi.plots.utils import confidence_interval


def plot_thr(names: List, throughput: List, xs: Array, scenario_config: dict) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(names)))

    for i, (name, thr) in enumerate(zip(names, throughput)):
        mean, ci_low, ci_high = confidence_interval(thr)
        plt.plot(xs, mean, marker='o', markersize=1, label=AGENT_NAMES.get(name, name), c=colors[i])
        plt.fill_between(xs, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    if 'mcs' in scenario_config['params']:
        plt.axhline(DATA_RATES[scenario_config['params']['mcs']], linestyle='--', color='gray', label='Single TX')

    plt.xlabel('Step')
    plt.ylabel('Aggregate throughput [Mb/s]')
    plt.xlim((xs[0], xs[-1]))
    plt.ylim(bottom=0)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'thr-{scenario_config["name"]}.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-f', '--file', type=str, required=True)
    args.add_argument('-s', '--aggregate_steps', type=int, required=False)
    args = args.parse_args()

    with open(args.file, 'r') as file:
        results = json.load(file)

    for scenario in results:
        names, throughput = [], []
        aggregate_steps = args.aggregate_steps or int(scenario['scenario']['n_steps'] / 51)

        for agent in scenario['agents']:
            names.append(agent['agent']['name'])
            runs = [np.array(run).reshape((-1, aggregate_steps)).mean(axis=-1) for run in agent['runs']]
            throughput.append(np.array(runs))

        xs = np.arange(throughput[0].shape[-1]) * aggregate_steps
        plot_thr(names, throughput, xs, scenario['scenario'])
