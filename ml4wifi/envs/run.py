from argparse import ArgumentParser
from typing import List

import pandas as pd
from chex import PRNGKey
from reinforced_lib.agents.mab import EGreedy, Exp3, Softmax, UCB

from ml4wifi.agents import AgentFactory
from ml4wifi.envs.scenarios import Scenario
from ml4wifi.envs.scenarios.static import *


ALL_SCENARIOS = [
    simple_scenario_1(),
    simple_scenario_2(),
    simple_scenario_3(),
    random_scenario_1()
]

AGENTS = {
    'egreedy': (EGreedy, {'e': 0.1, 'optimistic_start': 1e3}),
    'exp3': (Exp3, {'gamma': 0.1, 'min_reward': 0., 'max_reward': 1e3}),
    'softmax': (Softmax, {'lr': 1., 'tau': 1., 'multiplier': 1e-3}),
    'ucb': (UCB, {'c': 0.1})
}


def run_scenario(agent_factory: AgentFactory, scenario: Scenario, n_reps: int, n_steps: int, key: PRNGKey) -> List:
    results = []

    for i in range(n_reps):
        agent = agent_factory.hierarchical_agent()

        for _ in range(n_steps):
            key, subkey = jax.random.split(key)
            tx = agent.sample()
            thr = scenario(subkey, tx)
            agent.update(thr)
            results.append(thr)

    return results


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--n_reps', type=int, required=True)
    args.add_argument('--n_steps', type=int, required=True)
    args.add_argument('--output', type=str, default='all_results.csv')
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    df = pd.DataFrame(columns=['scenario', 'agent', 'rep', 'step', 'thr'])

    rep_column = [i for i in range(args.n_reps) for _ in range(args.n_steps)]
    step_column = [i for _ in range(args.n_reps) for i in range(args.n_steps)]

    for agent_name, agent_params in AGENTS.items():
        for scenario in ALL_SCENARIOS:
            key = jax.random.PRNGKey(args.seed)
            associations = scenario.get_associations()
            agent_factory = AgentFactory(associations, *agent_params)
            results = run_scenario(agent_factory, scenario, args.n_reps, args.n_steps, key)

            scenario_column = [scenario.name] * args.n_reps * args.n_steps
            agent_column = [agent_name] * args.n_reps * args.n_steps

            new_df = pd.DataFrame({
                'scenario': scenario_column,
                'agent': agent_column,
                'rep': rep_column,
                'step': step_column,
                'thr': results
            })
            df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(args.output, index=False)
