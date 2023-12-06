import json
from argparse import ArgumentParser
from typing import List

import jax
import numpy as np
from reinforced_lib.agents.mab import *
from tqdm import tqdm

from ml4wifi.agents import MapcAgentFactory
from ml4wifi.agents.thompson_sampling import NormalThompsonSampling
from ml4wifi.envs.scenarios.static import *


def run_scenario(
        agent_factory: MapcAgentFactory,
        scenario: StaticScenario,
        n_reps: int,
        n_steps: int,
        seed: int
) -> List:
    key = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    runs = []
    actions = []

    for i in range(n_reps):
        agent = agent_factory.create_mapc_agent()
        runs.append([])
        actions.append([])
        data_rate = 0.

        for j in range(n_steps):
            key, scenario_key = jax.random.split(key)
            tx = agent.sample(data_rate)
            data_rate = scenario(scenario_key, tx)
            runs[-1].append(data_rate)
            actions[-1].append(scenario.tx_matrix_to_action(tx))

    return jax.tree_map(lambda x: x.tolist(), runs), actions


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-c', '--config', type=str, default='default_config.json')
    args.add_argument('-o', '--output', type=str, default='all_results.json')
    args = args.parse_args()

    with open(args.config, 'r') as file:
        config = json.load(file)

    all_results = []

    for scenario_config in tqdm(config['scenarios'], desc='Scenarios'):
        scenario = globals()[scenario_config['scenario']](**scenario_config['params'])

        associations = scenario.get_associations()
        scenario_results = []

        for agent_config in tqdm(config['agents'], desc='Agents', leave=False):
            agent_factory = MapcAgentFactory(associations, globals()[agent_config['name']], agent_config['params'])

            runs, actions = run_scenario(agent_factory, scenario, config['n_reps'], scenario_config['n_steps'], config['seed'])
            scenario_results.append({
                'agent': {
                    'name': agent_config['name'],
                    'params': agent_config['params']
                },
                'runs': runs,
                'actions': actions
            })

        all_results.append({
            'scenario': scenario_config,
            'agents': scenario_results
        })

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
