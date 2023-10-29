import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ml4wifi.envs.scenarios.static import *

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

    # Simulate the network
    thr_optimal, thr_suboptimal, thr_wasteful = [], [], []
    n_steps = 200
    for _ in range(n_steps):
        key, k1, k2, k3 = jax.random.split(key, 4)
        thr_optimal.append(scenario.thr_fn(k1, tx_optimal))
        thr_suboptimal.append(scenario.thr_fn(k2, tx_suboptimal))
        thr_wasteful.append(scenario.thr_fn(k3, tx_wasteful))
    thr_optimal = jnp.array(thr_optimal)
    thr_suboptimal = jnp.array(thr_suboptimal)
    thr_wasteful = jnp.array(thr_wasteful)
    
    # Plot the approximate throughput
    if plot:
        xs = jnp.arange(n_steps)
        plt.plot(xs, thr_optimal, label='optimal')
        plt.plot(xs, thr_suboptimal, label='suboptimal')
        plt.plot(xs, thr_wasteful, label='wasteful')
        plt.xlim(0, n_steps)
        plt.ylim(0, 150)
        plt.xlabel('Timestep')
        plt.ylabel('Approximated throughput [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'scenario_1_thr_d{distance:.4f}.pdf', bbox_inches='tight')
        plt.clf()
    
    return jnp.mean(thr_optimal), jnp.mean(thr_suboptimal), jnp.mean(thr_wasteful)


if __name__ == "__main__":

    mcs = sys.argv[1]
    print(f"=== MCS {mcs} ===")

    mean_optimal, mean_suboptimal, mean_wasteful = [], [], []
    distances = jnp.logspace(-3, 3, 10, base=10)
    for d in distances:
        thr_optimal, thr_suboptimal, thr_wasteful = run(distance=d, mcs=int(mcs), plot=True)
        mean_optimal.append(thr_optimal)
        mean_suboptimal.append(thr_suboptimal)
        mean_wasteful.append(thr_wasteful)
        print(f"Distance {d:.3f}m: {thr_optimal:.2f} > {thr_suboptimal:.2f} > {thr_wasteful:.2f}")
    
    plt.plot(distances, mean_optimal, label='optimal')
    plt.plot(distances, mean_suboptimal, label='suboptimal')
    plt.plot(distances, mean_wasteful, label='wasteful')
    plt.xscale('log')
    plt.xlabel('Distance [m]')
    plt.ylabel('Approximated throughput [Mb/s]')
    plt.title(f'MCS {mcs}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'scenario_1_cum_mcs{mcs}.pdf', bbox_inches='tight')
    plt.clf()

