import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from copy import deepcopy
from argparse import ArgumentParser

import jax
from reinforced_lib.agents.mab import *
from tqdm import tqdm

from mapc_mab.agents import MapcAgentFactory
from mapc_mab.envs.dynamic_scenarios import *
from mapc_mab.envs.static_scenarios import *


def run_scenario(
        agent_factory: MapcAgentFactory,
        scenario: Scenario,
        n_reps: int,
        n_steps: int,
        seed: int
) -> tuple[list, list]:
    key = jax.random.PRNGKey(seed)
    agent_copy = agent_factory.create_mapc_agent()

    runs = []
    actions = []

    for i in range(n_reps):
        agent = deepcopy(agent_copy)
        scenario.reset()

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

        if scenario_config['insert_wall']:
            n_nodes = len(scenario.associations) + sum(len(stas) for stas in scenario.associations.values())
            switch_steps = [0, scenario_config['n_steps'] // 2]
            scenario = DynamicScenario.from_static(scenario, walls_sec=jnp.zeros((n_nodes, n_nodes)), switch_steps=switch_steps)

        scenario_results = []

        for agent_config in tqdm(config['agents'], desc='Agents', leave=False):
            agent_factory = MapcAgentFactory(
                scenario.associations, globals()[agent_config['name']], agent_config['params'], agent_config['hierarchical'], config['seed']
            )

            runs, actions = run_scenario(agent_factory, scenario, config['n_reps'], scenario_config['n_steps'], config['seed'])
            scenario_results.append({
                'agent': {
                    'name': agent_config['name'],
                    'params': agent_config['params'],
                    'hierarchical': agent_config['hierarchical']
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
