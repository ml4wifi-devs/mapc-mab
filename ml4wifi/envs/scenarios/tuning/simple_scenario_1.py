import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from argparse import ArgumentParser

from ml4wifi.envs.scenarios.static import *
import ml4wifi.plots

COLORS = pl.cm.viridis(jnp.linspace(0., 1., 3))
plt.rcParams.update({'figure.figsize': (4, 3)})

def run(distance: int, mcs: int = 11, seed: int = 42, plot: bool = False):

    # Define test-case key and scenario
    key = jax.random.PRNGKey(seed)
    scenario = simple_scenario_1(d=distance, mcs=mcs)

    # Transmission matrices indicating which node is transmitting to which node:
    # - in this example, AP 1 is transmitting to STA 1, and AP 2 is transmitting to STA 4
    tx_optimal = jnp.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])

    # - in this example, AP 1 is transmitting to STA 1, and AP 2 is transmitting to STA 3
    tx_suboptimal = jnp.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    # - in this example, AP 1 is transmitting to STA 2, and AP 2 is transmitting to STA 3
    tx_wasteful = jnp.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    # - this is a benchmark example with single transmission from AP 1 to STA 1
    tx_single = jnp.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    # Simulate the network
    thr_optimal, thr_suboptimal, thr_wasteful, thr_single = [], [], [], []
    n_steps = 200
    for _ in range(n_steps):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        thr_optimal.append(scenario.thr_fn(k1, tx_optimal))
        thr_suboptimal.append(scenario.thr_fn(k2, tx_suboptimal))
        thr_wasteful.append(scenario.thr_fn(k3, tx_wasteful))
        thr_single.append(scenario.thr_fn(k4, tx_single))
    thr_optimal = jnp.array(thr_optimal)
    thr_suboptimal = jnp.array(thr_suboptimal)
    thr_wasteful = jnp.array(thr_wasteful)
    thr_single = jnp.array(thr_single)
    
    # Plot the approximate throughput
    if plot:
        xs = jnp.arange(n_steps)
        plt.plot(xs, thr_optimal, label='o-AP1 o o AP2-o (optimal)', color=COLORS[0])
        plt.plot(xs, thr_suboptimal, label='o-AP1 o o-AP2 o (suboptimal)', color=COLORS[1])
        plt.plot(xs, thr_wasteful, label='o AP1-o o-AP2 o (wasteful)', color=COLORS[2])
        plt.plot(xs, thr_single, label='single transmission', color='black', linestyle='--')
        plt.xlim(0, n_steps)
        plt.ylim(0, 150)
        plt.xlabel('Timestep')
        plt.ylabel('Approximated throughput [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'scenario_1_thr_d{distance:.4f}.pdf', bbox_inches='tight')
        plt.clf()
    
    return (
        jnp.mean(thr_optimal),
        jnp.mean(thr_suboptimal),
        jnp.mean(thr_wasteful),
        jnp.mean(thr_single)
    )


def plot_cumulative():

    plt.plot(distances, mean_optimal, label='o-AP1 o o AP2-o (optimal)', color=COLORS[0])
    plt.plot(distances, mean_suboptimal, label='o-AP1 o o-AP2 o (suboptimal)', color=COLORS[1])
    plt.plot(distances, mean_wasteful, label='o AP1-o o-AP2 o (wasteful)', color=COLORS[2])
    plt.plot(distances, mean_single, label='single transmission', color='black', linestyle='--')
    plt.xscale('log')
    plt.ylim(0, 150)
    plt.xlabel('Distance gap [m]')
    plt.ylabel('Approximated throughput [Mb/s]')
    plt.title(f'MCS {mcs}')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'scenario_1_cum_mcs{mcs}.pdf', bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":

    # Define argument parser
    parser = ArgumentParser()
    parser.add_argument("-m", "--mcs", type=int, help="MCS index")
    parser.add_argument("-r", "--resolution", type=int, default=50, help="The distance space resolution to search")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot throughput in time")

    # Parse arguments
    args = parser.parse_args()
    mcs = args.mcs
    res = args.resolution
    plot_flag = args.plot

    # Run the simulation
    print(f"=== MCS {mcs} ===")
    mean_optimal, mean_suboptimal, mean_wasteful, mean_single = [], [], [], []
    distances = jnp.logspace(0, 3, res, base=10)
    for d in distances:
        thr_optimal, thr_suboptimal, thr_wasteful, thr_single = run(distance=d, mcs=int(mcs), plot=plot_flag)
        mean_optimal.append(thr_optimal)
        mean_suboptimal.append(thr_suboptimal)
        mean_wasteful.append(thr_wasteful)
        mean_single.append(thr_single)
        print(f"Distance {d:.3f}m: {thr_optimal:.2f} > {thr_suboptimal:.2f} > {thr_wasteful:.2f} > {thr_single:.2f}")
    
    # Plot the approximate throughput
    plot_cumulative()

    # Plot the scenario topology
    d = 10
    tmp_scenario = simple_scenario_1(d=d, mcs=mcs)
    tmp_scenario.plot(f'scenario_1_top_dap_{d}.pdf')

# o-AP1 o o AP2-o (optimal)
# o-AP1 o o-AP2 o (suboptimal)
# o AP1-o o-AP2 o (wasteful)

