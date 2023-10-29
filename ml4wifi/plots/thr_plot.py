import json
from argparse import ArgumentParser
from typing import List

import jax.numpy as jnp
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from chex import Array

from ml4wifi.plots.config import AGENT_NAMES
from ml4wifi.plots.utils import confidence_interval


def plot_thr(names: List, throughput: List, xs: Array, scenario: str) -> None:
    colors = pl.cm.viridis(jnp.linspace(0., 1., len(names)))

    for i, (name, thr) in enumerate(zip(names, throughput)):
        mean, ci_low, ci_high = confidence_interval(thr)
        plt.plot(xs, mean, marker='o', markersize=2, label=AGENT_NAMES[name], c=colors[i])
        plt.fill_between(xs, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    plt.xlabel('Step')
    plt.ylabel('Aggregate throughput [Mb/s]')
    plt.xlim((xs[0], xs[-1]))
    plt.ylim((0, 125))
    plt.grid()
    plt.legend()
    plt.savefig(f'thr-{scenario}.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--file', type=str, required=True)
    args.add_argument('--steps', type=int, required=True)
    args = args.parse_args()

    with open(args.file, 'r') as file:
        results = json.load(file)

    for scenario in results:
        names, throughput = [], []

        for agent in scenario['agents']:
            names.append(agent['agent'])
            runs = [jnp.array(run).reshape((-1, args.steps)).mean(axis=-1) for run in agent['runs']]
            throughput.append(jnp.array(runs))

        xs = jnp.arange(throughput[0].shape[-1]) * args.steps
        plot_thr(names, throughput, xs, scenario['scenario'])
