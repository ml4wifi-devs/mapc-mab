import json
from argparse import ArgumentParser
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ml4wifi.envs.sim import DATA_RATES, TAU
from ml4wifi.plots.config import AGENT_NAMES, COLUMN_WIDTH, COLUMN_HIGHT, get_cmap
from ml4wifi.plots.utils import confidence_interval

plt.rcParams.update({
    'figure.figsize': (3 * COLUMN_WIDTH, COLUMN_HIGHT + 0.4),
    'legend.fontsize': 7
})

AGGREGATE_STEPS = {
    "scenario_10m": 15,
    "scenario_20m": 15,
    "scenario_25m_long": 75,
}
TITLES = {
    "scenario_10m": r"(a) $d=10$ m",
    "scenario_20m": r"(b) $d=20$ m",
    "scenario_25m_long": r"(c) $d=25$ m",
}


def plot(names: List, data_rate: List, scenario_config: dict) -> None:
    colors = get_cmap(len(names))
    xs = np.linspace(0, scenario['scenario']['n_steps'], data_rate[0].shape[-1]) * TAU

    if 'mcs' in scenario_config['params']:
        plt.axhline(DATA_RATES[scenario_config['params']['mcs']], linestyle='--', color='gray', label='Single TX')

    for i, (name, rate) in enumerate(zip(names, data_rate)):
        mean, ci_low, ci_high = confidence_interval(rate)
        plt.plot(xs, mean, marker='o', label=AGENT_NAMES.get(name, name), c=colors[i])
        plt.fill_between(xs, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    plt.xlabel('Time [s]')
    plt.ylabel('Effective data rate [Mb/s]')
    plt.xlim((xs[0], xs[-1]))
    plt.ylim(bottom=0)
    plt.grid()
    plt.legend(ncol=2, loc='lower right')
    plt.tight_layout()
    plt.savefig(f'rate-{scenario_config["name"]}.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-f', '--file', type=str, required=True)
    args = args.parse_args()

    with open(args.file, 'r') as file:
        results = json.load(file)

    fig, axes = plt.subplots(1, 3, sharey=True)
    fig.subplots_adjust(wspace=0.)


    for ax, scenario in zip(axes, results):
        names, data_rate = [], []
        scenario_name = scenario['scenario']['name']
        scenario_config = scenario['scenario']

        for agent in scenario['agents']:
            names.append(agent['agent']['name'])
            runs = [np.array(run).reshape((-1, AGGREGATE_STEPS[scenario_name])).mean(axis=-1) for run in agent['runs']]
            data_rate.append(np.array(runs))
        
        colors = get_cmap(len(names))
        xs = np.linspace(0, scenario['scenario']['n_steps'], data_rate[0].shape[-1]) * TAU

        for i, (name, rate) in enumerate(zip(names, data_rate)):
            mean, ci_low, ci_high = confidence_interval(rate)
            ax.plot(xs, mean, marker='o', label=AGENT_NAMES.get(name, name), c=colors[i])
            ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)
        
        if 'mcs' in scenario_config['params']:
            ax.axhline(DATA_RATES[scenario_config['params']['mcs']], linestyle='--', color='gray', label='Single TX')

        ax.set_title(TITLES[scenario_name])
        ax.set_xlabel('Time [s]')
        ax.set_xlim((xs[0], xs[-1]))
        ax.set_ylim(bottom=0, top=325)
        ax.grid()

        if scenario_name == 'scenario_10m':
            ax.set_ylabel('Effective data rate [Mb/s]')
            ax.legend(ncol=3, loc='upper left')


    plt.tight_layout()
    plt.savefig(f'data-rates.pdf', bbox_inches='tight')
    plt.clf()
