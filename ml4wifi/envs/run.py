import json
from argparse import ArgumentParser
from typing import List

import jax
from chex import PRNGKey
from reinforced_lib.agents.mab import EGreedy, Exp3, Softmax, UCB

from ml4wifi.agents import AgentFactory
from ml4wifi.envs.scenarios.static import StaticScenario, get_all_scenarios


AGENTS = [
    (EGreedy, {'e': 0.1, 'optimistic_start': 1e3}),
    (Exp3, {'gamma': 0.1, 'min_reward': 0., 'max_reward': 1e3}),
    (Softmax, {'lr': 1., 'tau': 1., 'multiplier': 1e-3}),
    (UCB, {'c': 0.1})
]


def run_scenario(agent_factory: AgentFactory, scenario: StaticScenario, n_reps: int, n_steps: int, key: PRNGKey) -> List:
    runs = []

    for i in range(n_reps):
        agent = agent_factory.hierarchical_agent()
        runs.append([])

        for j in range(n_steps):
            key, subkey = jax.random.split(key)
            tx = agent.sample()
            thr = scenario(subkey, tx)
            agent.update(thr)

    return runs


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--n_reps', type=int, required=True)
    args.add_argument('--n_steps', type=int, required=True)
    args.add_argument('--output', type=str, default='all_results.json')
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    results = []

    for scenario in get_all_scenarios():
        associations = scenario.get_associations()
        agents = []

        for agent_type, agent_params in AGENTS:
            key = jax.random.PRNGKey(args.seed)
            agent_factory = AgentFactory(associations, agent_type, agent_params)

            agents.append({
                'agent': agent_type.__name__.lower(),
                'runs': run_scenario(agent_factory, scenario, args.n_reps, args.n_steps, key)
            })

        results.append({
            'scenario': scenario.name,
            'agents': agents
        })

    with open(args.output, 'w') as file:
        json.dump(results, file)
