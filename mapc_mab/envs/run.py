import os
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '-1'
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

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
        slots_ahead: int,
        seed: int
) -> tuple[list, list]:
    key = jax.random.PRNGKey(seed)
    runs = []
    actions = []

    for _ in range(n_reps):
        agent = agent_factory.create_mapc_agent()
        scenario.reset()
        runs.append([])
        actions.append([])

        step = 0
        while step < n_steps:
            key, scenario_key = jax.random.split(key)
            scenario_schedule_keys = jax.random.split(scenario_key, slots_ahead)

            # Schedule the transmissions for the next n slots ahead
            tx_schedule = [agent.sample() for _ in range(slots_ahead)]

            # Get the data rates and rewards for the scheduled transmissions
            results = [scenario(k, *tx_tuple) for k, tx_tuple in zip(scenario_schedule_keys, tx_schedule)]
            data_rates, rewards = zip(*results)

            # Update the agent with the data rates as rewards
            agent.update(jnp.array(rewards))

            # Save the data rates and actions, increment the step
            runs[-1] += data_rates
            actions[-1] += [scenario.tx_matrix_to_action(tx_tuple[0]) for tx_tuple in tx_schedule]
            step += slots_ahead

    return jax.tree.map(lambda x: x.tolist(), runs), actions


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

        if 'sec' in scenario_config:
            scenario_params_sec = deepcopy(scenario_config['params'])
            scenario_params_sec.update(scenario_config['sec'])
            scenario_sec = globals()[scenario_config['scenario']](**scenario_params_sec)
            scenario = DynamicScenario.from_static_scenarios(scenario, scenario_sec, scenario_config['switch_steps'])

        scenario_results = []

        for agent_config in tqdm(config['agents'], desc='Agents', leave=False):
            agent_factory = MapcAgentFactory(
                associations=scenario.associations,
                agent_type=globals()[agent_config['name']],
                agent_params_lvl1=agent_config['params1'],
                agent_params_lvl2=agent_config['params2'] if agent_config['hierarchical'] else None,
                agent_params_lvl3=agent_config['params3'] if agent_config['hierarchical'] else None,
                hierarchical=agent_config['hierarchical'],
                seed=config['seed']
            )

            runs, actions = run_scenario(agent_factory, scenario, config['n_reps'], scenario_config['n_steps'], scenario_config['slots_ahead'], config['seed'])
            scenario_results.append({
                'agent': {
                    'name': agent_config['name'],
                    'params1': agent_config['params1'],
                    'params2': agent_config['params2'] if agent_config['hierarchical'] else None,
                    'params3': agent_config['params3'] if agent_config['hierarchical'] else None,
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
