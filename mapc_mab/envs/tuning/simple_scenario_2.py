import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from mapc_mab.envs.static_scenarios import simple_scenario_2
from mapc_mab.plots.config import get_cmap


COLORS = get_cmap(4)
plt.rcParams.update({'figure.figsize': (4, 3)})
plt.rcParams.update({'lines.linewidth': 0.8})


def run(distance_ap: int, distance_sta: int, mcs: int = 11, seed: int = 42, plot: bool = False):

    # Define test-case key and scenario
    key = jax.random.PRNGKey(seed)
    scenario = simple_scenario_2(d_ap=distance_ap, d_sta=distance_sta, mcs=mcs)

    # Transmission matrices indicating which node is transmitting to which node:
    # - in this example, AP A is transmitting to STA 1, AP B is transmitting to STA 6,
    #   AP C is transmitting to STA 11, and AP D is transmitting to STA 16
    tx_external_4 = jnp.zeros((20, 20))
    tx_external_4 = tx_external_4.at[0, 4].set(1)
    tx_external_4 = tx_external_4.at[1, 9].set(1)
    tx_external_4 = tx_external_4.at[2, 14].set(1)
    tx_external_4 = tx_external_4.at[3, 19].set(1)

    # - in this example, AP A is transmitting to STA 3, AP B is transmitting to STA 8,
    #   AP C is transmitting to STA 9, and AP D is transmitting to STA 14
    tx_internal_4 = jnp.zeros((20, 20))
    tx_internal_4 = tx_internal_4.at[0, 6].set(1)
    tx_internal_4 = tx_internal_4.at[1, 11].set(1)
    tx_internal_4 = tx_internal_4.at[2, 12].set(1)
    tx_internal_4 = tx_internal_4.at[3, 17].set(1)

    # - in this example, AP A is transmitting to STA 1, and AP C is transmitting to STA 11
    tx_external_2 = jnp.zeros((20, 20))
    tx_external_2 = tx_external_2.at[0, 4].set(1)
    tx_external_2 = tx_external_2.at[2, 14].set(1)

    # - in this example, AP A is transmitting to STA 3, and AP C is transmitting to STA 9
    tx_internal_2 = jnp.zeros((20, 20))
    tx_internal_2 = tx_internal_2.at[0, 6].set(1)
    tx_internal_2 = tx_internal_2.at[2, 12].set(1)

    # - this is a benchmark example with single transmission from AP A to STA 1
    tx_single = jnp.zeros((20, 20))
    tx_single = tx_single.at[0, 4].set(1)

    # Simulate the network
    rate_external_4, rate_internal_4, rate_external_2, rate_internal_2, rate_single = [], [], [], [], []
    n_steps = 200

    for _ in range(n_steps):
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        rate_external_4.append(scenario(k1, tx_external_4))
        rate_internal_4.append(scenario(k2, tx_internal_4))
        rate_external_2.append(scenario(k3, tx_external_2))
        rate_internal_2.append(scenario(k4, tx_internal_2))
        rate_single.append(scenario(k5, tx_single))

    rate_external_4 = jnp.array(rate_external_4)
    rate_internal_4 = jnp.array(rate_internal_4)
    rate_external_2 = jnp.array(rate_external_2)
    rate_internal_2 = jnp.array(rate_internal_2)
    rate_single = jnp.array(rate_single)
    
    # Plot effective data rate
    if plot:
        xs = jnp.arange(n_steps)
        plt.plot(xs, rate_external_4, label='external 4', color=COLORS[0])
        plt.plot(xs, rate_internal_4, label='internal 4', color=COLORS[1])
        plt.plot(xs, rate_external_2, label='external 2 (diagonal)', color=COLORS[2])
        plt.plot(xs, rate_internal_2, label='internal 2 (diagonal)', color=COLORS[3])
        plt.plot(xs, rate_single, label='single transmission', color='black', linestyle='--')
        plt.xlim(0, n_steps)
        plt.ylim(0, 600)
        plt.xlabel('Timestep')
        plt.ylabel('Effective data rate [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'scenario_2_dap{distance_ap:.4f}_dsta{distance_sta:.4f}.pdf', bbox_inches='tight')
        plt.clf()
    
    return (
        jnp.mean(rate_external_4),
        jnp.mean(rate_internal_4),
        jnp.mean(rate_external_2),
        jnp.mean(rate_internal_2),
        jnp.mean(rate_single)
    )


def plot_cumulative():
    plt.plot(distances_ap, mean_external_4, label='external 4', color=COLORS[0])
    plt.plot(distances_ap, mean_internal_4, label='internal 4', color=COLORS[1])
    plt.plot(distances_ap, mean_external_2, label='external 2 (diagonal)', color=COLORS[2])
    plt.plot(distances_ap, mean_internal_2, label='internal 2 (diagonal)', color=COLORS[3])
    plt.plot(distances_ap, mean_single, label='single transmission', color='black', linestyle='--')
    plt.xscale('log')
    plt.ylim(0, 600)
    plt.xlabel('Inter-AP distance [m]')
    plt.ylabel('Effective data rate [Mb/s]')
    plt.title(f'MCS {mcs}, AP-STA distance {distance_sta} m')
    plt.legend(loc='upper left')
    plt.grid(which='major')
    plt.tight_layout()
    plt.savefig(f'scenario_2_cum_mcs{mcs}_dsta{distance_sta}.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":

    # Define argument parser
    parser = ArgumentParser()
    parser.add_argument("-m", "--mcs", type=int, help="MCS index")
    parser.add_argument("-s", "--distance_sta", type=float, help="Distance between AP and STA [m]")
    parser.add_argument("-r", "--resolution", type=int, default=50, help="The distance space resolution to search")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot effective data rate")

    # Parse arguments
    args = parser.parse_args()
    mcs = args.mcs
    distance_sta = args.distance_sta
    res = args.resolution
    plot_flag = args.plot

    # Run the simulation
    print(f"=== MCS {mcs}, d_sta {distance_sta} m ===")
    mean_external_4, mean_internal_4, mean_external_2, mean_internal_2, mean_single = [], [], [], [], []
    distances_ap = jnp.logspace(1, 2.5, res, base=10)

    for d in distances_ap:
        rate_external_4, rate_internal_4, rate_external_2, rate_internal_2, rate_single = run(
            distance_ap=d, distance_sta=distance_sta, mcs=mcs, plot=plot_flag, seed=42
        )
        mean_external_4.append(rate_external_4)
        mean_internal_4.append(rate_internal_4)
        mean_external_2.append(rate_external_2)
        mean_internal_2.append(rate_internal_2)
        mean_single.append(rate_single)
        print(f"Distance {d:.3f}m: ", end="")
        print(f"{rate_external_4:.2f} > {rate_internal_4:.2f} > {rate_external_2:.2f} > {rate_internal_2:.2f} > {rate_single:.2f}")
    
    # Plot effective data rate
    plot_cumulative()

    # Plot the scenario topology
    d_ap = 40.
    tmp_scenario = simple_scenario_2(d_ap=d_ap, d_sta=distance_sta, mcs=mcs)
    tmp_scenario.plot(f'scenario_2_top_dap_{d_ap}_dsta{distance_sta}.pdf')
