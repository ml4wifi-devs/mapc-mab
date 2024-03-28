import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from copy import deepcopy
from argparse import ArgumentParser

import jax
from reinforced_lib.agents.mab import *
from tqdm import tqdm

from mapc_mab.agents import MapcAgent, MapcAgentFactory
from mapc_mab.envs.dynamic_scenarios import *
from mapc_mab.envs.static_scenarios import *


def mapc_agent_with_new_association(agent: MapcAgent, agent_sec: MapcAgent) -> MapcAgent:
    for ap in agent_sec.associations:
        for sta in agent_sec.associations[ap]:
            if sta in agent.associations[ap]:
                if agent.hierarchical:
                    agent_sec.find_groups_dict[sta] = agent.find_groups_dict[sta]
                else:
                    raise ValueError('Flat agents are not supported')

    return agent_sec


def new_pos_and_association(scenario: DynamicScenario):
    pos_sec = scenario.pos.copy().at[[5, 6, 17, 18], 0].set(scenario.pos[1, 0])
    associations_sec = scenario.associations.copy()
    associations_sec[0].remove(5)
    associations_sec[0].remove(6)
    associations_sec[3].remove(17)
    associations_sec[3].remove(18)
    associations_sec[1] += [5, 6]
    associations_sec[2] += [17, 18]
    return associations_sec, pos_sec


def run_scenario(
        agent_factory: MapcAgentFactory,
        agent_factory_sec: MapcAgentFactory,
        scenario: DynamicScenario,
        n_reps: int,
        n_steps: int,
        seed: int
) -> tuple[list, list]:
    key = jax.random.PRNGKey(seed)
    agent_copy = agent_factory.create_mapc_agent()
    agent_sec_copy = agent_factory_sec.create_mapc_agent() if agent_factory_sec is not None else None

    runs = []
    actions = []

    for i in range(n_reps):
        agent = deepcopy(agent_copy)
        agent_sec = deepcopy(agent_sec_copy)
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

            if agent_factory_sec is not None and j == n_steps // 2:
                agent = mapc_agent_with_new_association(agent, agent_sec)

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

        if scenario_config['move_stations']:
            associations_sec, pos_sec = new_pos_and_association(scenario)
            switch_steps = [0, scenario_config['n_steps'] // 2]
            scenario = DynamicScenario.from_static(scenario, pos_sec=pos_sec, switch_steps=switch_steps)

        scenario_results = []

        for agent_config in tqdm(config['agents'], desc='Agents', leave=False):
            agent_factory = MapcAgentFactory(
                scenario.associations, globals()[agent_config['name']], agent_config['params'], agent_config['hierarchical'], config['seed']
            )

            if scenario_config['move_stations']:
                agent_factory_sec = MapcAgentFactory(
                    associations_sec, globals()[agent_config['name']], agent_config['params'], agent_config['hierarchical'], config['seed']
                )
            else:
                agent_factory_sec = None

            runs, actions = run_scenario(agent_factory, agent_factory_sec, scenario, config['n_reps'], scenario_config['n_steps'], config['seed'])
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
