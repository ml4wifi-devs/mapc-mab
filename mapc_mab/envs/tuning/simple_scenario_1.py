import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from mapc_mab.envs.static_scenarios import simple_scenario_1
from mapc_mab.plots.config import get_cmap


COLORS = get_cmap(3)
plt.rcParams.update({'figure.figsize': (4, 3)})
plt.rcParams.update({'lines.linewidth': 0.8})


def run(distance: int, mcs: int = 4, seed: int = 42, plot: bool = False):

    # Define test-case key and scenario
    key = jax.random.PRNGKey(seed)
    scenario = simple_scenario_1(d=distance, mcs=mcs)

    # Transmission matrices indicating which node is transmitting to which node:
    # - in this example, AP A is transmitting to STA 1, and AP B is transmitting to STA 4
    tx_optimal = jnp.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])

    # - in this example, AP A is transmitting to STA 1, and AP B is transmitting to STA 3
    tx_suboptimal = jnp.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    # - in this example, AP A is transmitting to STA 2, and AP B is transmitting to STA 3
    tx_wasteful = jnp.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    # - this is a benchmark example with single transmission from AP A to STA 1
    tx_single = jnp.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    # Simulate the network
    rate_optimal, rate_suboptimal, rate_wasteful, rate_single = [], [], [], []
    n_steps = 200

    for _ in range(n_steps):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        rate_optimal.append(scenario(k1, tx_optimal))
        rate_suboptimal.append(scenario(k2, tx_suboptimal))
        rate_wasteful.append(scenario(k3, tx_wasteful))
        rate_single.append(scenario(k4, tx_single))

    rate_optimal = jnp.array(rate_optimal)
    rate_suboptimal = jnp.array(rate_suboptimal)
    rate_wasteful = jnp.array(rate_wasteful)
    rate_single = jnp.array(rate_single)
    
    # Plot effective data rate
    if plot:
        xs = jnp.arange(n_steps)
        plt.plot(xs, rate_optimal, label='o-A o o B-o (optimal)', color=COLORS[0])
        plt.plot(xs, rate_suboptimal, label='o-A o o-B o (suboptimal)', color=COLORS[1])
        plt.plot(xs, rate_wasteful, label='o A-o o-B o (wasteful)', color=COLORS[2])
        plt.plot(xs, rate_single, label='single transmission', color='black', linestyle='--')
        plt.xlim(0, n_steps)
        plt.ylim(0, 150)
        plt.xlabel('Timestep')
        plt.ylabel('Effective data rate [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'scenario_1_d{distance:.4f}.pdf', bbox_inches='tight')
        plt.clf()
    
    return (
        jnp.mean(rate_optimal),
        jnp.mean(rate_suboptimal),
        jnp.mean(rate_wasteful),
        jnp.mean(rate_single)
    )


def plot_cumulative():

    plt.plot(distances, mean_optimal, label='o-A o o B-o (optimal)', color=COLORS[0])
    plt.plot(distances, mean_suboptimal, label='o-A o o-B o (suboptimal)', color=COLORS[1])
    plt.plot(distances, mean_wasteful, label='o A-o o-B o (wasteful)', color=COLORS[2])
    plt.plot(distances, mean_single, label='single transmission', color='black', linestyle='--')
    plt.xscale('log')
    plt.ylim(0, 150)
    plt.xlabel('Distance gap [m]')
    plt.ylabel('Effective data rate [Mb/s]')
    plt.title(f'MCS {mcs}')
    plt.legend(loc='upper left')
    plt.grid(which='major')
    plt.tight_layout()
    plt.savefig(f'scenario_1_cum_mcs{mcs}.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":

    # Define argument parser
    parser = ArgumentParser()
    parser.add_argument("-m", "--mcs", type=int, help="MCS index")
    parser.add_argument("-r", "--resolution", type=int, default=50, help="The distance space resolution to search")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot effective data rate")

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
        rate_optimal, rate_suboptimal, rate_wasteful, rate_single = run(distance=d, mcs=int(mcs), plot=plot_flag)
        mean_optimal.append(rate_optimal)
        mean_suboptimal.append(rate_suboptimal)
        mean_wasteful.append(rate_wasteful)
        mean_single.append(rate_single)
        print(f"Distance {d:.3f}m: {rate_optimal:.2f} > {rate_suboptimal:.2f} > {rate_wasteful:.2f} > {rate_single:.2f}")
    
    # Plot the effective data rate
    plot_cumulative()

    # Plot the scenario topology
    d = 10
    tmp_scenario = simple_scenario_1(d=d, mcs=mcs)
    tmp_scenario.plot(f'scenario_1_top_d_{d}.pdf')
