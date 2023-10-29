import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ml4wifi.envs.scenarios.static import *

def run(distance_ap: int, distance_sta: int, mcs: int = 11, seed: int = 42, plot: bool = False):

    # Define test-case key and scenario
    key = jax.random.PRNGKey(seed)
    scenario = simple_scenario_2(d_ap=distance_ap, d_sta=distance_sta, mcs=mcs)

    # Transmission matrices indicating which node is transmitting to which node:
    # - in this example, AP 1 is transmitting to STA 1, and AP 2 is transmitting to STA 4
    tx_optimal = jnp.zeros((20, 20))
    tx_optimal = tx_optimal.at[0, 4].set(1)
    tx_optimal = tx_optimal.at[2, 14].set(1)

    # - in this example, AP 1 is transmitting to STA 1, and AP 2 is transmitting to STA 3
    tx_suboptimal = jnp.zeros((20, 20))
    tx_suboptimal = tx_suboptimal.at[0, 6].set(1)
    tx_suboptimal = tx_suboptimal.at[2, 12].set(1)

    # Simulate the network
    thr_optimal, thr_suboptimal = [], []
    n_steps = 200
    for _ in range(n_steps):
        key, k1, k2 = jax.random.split(key, 3)
        thr_optimal.append(scenario.thr_fn(k1, tx_optimal))
        thr_suboptimal.append(scenario.thr_fn(k2, tx_suboptimal))
    thr_optimal = jnp.array(thr_optimal)
    thr_suboptimal = jnp.array(thr_suboptimal)
    
    # Plot the approximate throughput
    if plot:
        xs = jnp.arange(n_steps)
        plt.plot(xs, thr_optimal, label='optimal')
        plt.plot(xs, thr_suboptimal, label='suboptimal')
        plt.xlim(0, n_steps)
        plt.ylim(0, 150)
        plt.xlabel('Timestep')
        plt.ylabel('Approximated throughput [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'scenario_1_thr_dap{distance_ap:.4f}_dsta{distance_sta:.4f}.pdf', bbox_inches='tight')
        plt.clf()
    
    return jnp.mean(thr_optimal), jnp.mean(thr_suboptimal)


if __name__ == "__main__":

    mcs = int(sys.argv[1])
    distance_sta = float(sys.argv[2])
    print(f"=== MCS {mcs}, d_sta {distance_sta} m ===")

    mean_optimal, mean_suboptimal = [], []
    distances_ap = jnp.logspace(0.66, 2., 50, base=10)
    for d in distances_ap:
        thr_optimal, thr_suboptimal = run(distance_ap=d, distance_sta=distance_sta, mcs=mcs, plot=False)
        mean_optimal.append(thr_optimal)
        mean_suboptimal.append(thr_suboptimal)
        print(f"Distance {d:.3f}m: {thr_optimal:.2f} > {thr_suboptimal:.2f}")
    
    plt.plot(distances_ap, mean_optimal, label='optimal')
    plt.plot(distances_ap, mean_suboptimal, label='suboptimal')
    plt.xscale('log')
    plt.xlabel('Distance [m]')
    plt.ylabel('Approximated throughput [Mb/s]')
    plt.title(f'MCS {mcs}, d_STA {distance_sta} m')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'scenario_1_cum_mcs{mcs}_dsta{distance_sta}.pdf', bbox_inches='tight')
    plt.clf()

