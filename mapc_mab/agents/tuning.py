import os
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '-1'
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

from argparse import ArgumentParser
from functools import partial

import numpy as np
import optuna
from reinforced_lib.agents.mab import *

from mapc_mab.agents import MapcAgentFactory
from mapc_mab.envs.run import run_scenario
from mapc_mab.envs.dynamic_scenarios import random_scenario


TRAINING_SCENARIOS = [
    (random_scenario(seed=1, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, max_steps=500), 500),
    (random_scenario(seed=2, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, max_steps=1000), 1000),
    (random_scenario(seed=3, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, max_steps=500), 500),
    (random_scenario(seed=4, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, max_steps=2000), 2000),
    (random_scenario(seed=5, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, max_steps=500), 500),
    (random_scenario(seed=6, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, max_steps=3000), 3000),
    (random_scenario(seed=7, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, max_steps=500), 500),
    (random_scenario(seed=8, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, max_steps=1000), 1000),
    (random_scenario(seed=9, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, max_steps=500), 500),
    (random_scenario(seed=10, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, max_steps=2000), 2000),
    (random_scenario(seed=11, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, max_steps=500), 500),
    (random_scenario(seed=12, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, max_steps=3000), 3000),
    (random_scenario(seed=13, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, max_steps=500), 500),
    (random_scenario(seed=14, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, max_steps=1000), 1000),
    (random_scenario(seed=15, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, max_steps=500), 500),
    (random_scenario(seed=16, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, max_steps=2000), 2000),
    (random_scenario(seed=17, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, max_steps=500), 500),
    (random_scenario(seed=18, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, max_steps=3000), 3000),
    (random_scenario(seed=19, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, max_steps=500), 500),
    (random_scenario(seed=20, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, max_steps=1000), 1000),
    (random_scenario(seed=21, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, max_steps=500), 500),
    (random_scenario(seed=22, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, max_steps=2000), 2000),
    (random_scenario(seed=23, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, max_steps=500), 500),
    (random_scenario(seed=24, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, max_steps=3000), 3000)
]
SLOTS_AHEAD = 1


def objective(trial: optuna.Trial, agent: str, hierarchical: bool, seed: int) -> float:
    if agent == 'EGreedy':
        def suggest_params(level):
            return {
                'e': trial.suggest_float(f'e_{level}', 0.01, 0.1, log=True),
                'optimistic_start': trial.suggest_float(f'optimistic_start_{level}', 0., 100.),
                'alpha': trial.suggest_float(f'alpha_{level}', 0., 1.)
            }

        agent_type = EGreedy
        agent_params_lvl1 = suggest_params(1)
        if hierarchical:
            agent_params_lvl2 = suggest_params(2)
            agent_params_lvl3 = suggest_params(3)

    elif agent == 'Softmax':
        def suggest_params(level):
            return {
                'lr': trial.suggest_float(f'lr_{level}', 0.01, 10., log=True),
                'tau': trial.suggest_float(f'tau_{level}', 0.1, 10., log=True),
                'alpha': trial.suggest_float(f'alpha_{level}', 0., 1.)
            }

        agent_type = Softmax
        agent_params_lvl1 = suggest_params(1)
        if hierarchical:
            agent_params_lvl2 = suggest_params(2)
            agent_params_lvl3 = suggest_params(3)

    elif agent == 'UCB':
        def suggest_params(level):
            return {
                'c': trial.suggest_float(f'c_{level}', 0., 5.),
                'gamma': trial.suggest_float(f'gamma_{level}', 0., 1.)
            }

        agent_type = UCB
        agent_params_lvl1 = suggest_params(1)
        if hierarchical:
            agent_params_lvl2 = suggest_params(2)
            agent_params_lvl3 = suggest_params(3)

    elif agent == 'NormalThompsonSampling':
        def suggest_params(level):
            return {
                'alpha': trial.suggest_float(f'alpha_{level}', 0., 10.),
                'beta': trial.suggest_float(f'beta_{level}', 0., 10.),
                'lam': 1.,
                'mu': trial.suggest_float(f'mu_{level}', 0., 5.)
            }

        agent_type = NormalThompsonSampling
        agent_params_lvl1 = suggest_params(1)
        if hierarchical:
            agent_params_lvl2 = suggest_params(2)
            agent_params_lvl3 = suggest_params(3)

    else:
        raise ValueError(f'Unknown agent {agent}')

    runs = []

    for step, (scenario, n_steps) in enumerate(TRAINING_SCENARIOS):
        if hierarchical:
            agent_factory = MapcAgentFactory(scenario.associations, agent_type, agent_params_lvl1, agent_params_lvl2, agent_params_lvl3, hierarchical=True, seed=seed)
        else:
            agent_factory = MapcAgentFactory(scenario.associations, agent_type, agent_params_lvl1, hierarchical=False, seed=seed)

        results = np.mean(run_scenario(agent_factory, scenario, n_reps=1, n_steps=n_steps, seed=seed)[0])
        runs.append(results)

        trial.report(results, step)

    return np.mean(runs)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-a', '--agent', type=str, required=True)
    args.add_argument('-d', '--database', type=str, required=True)
    args.add_argument('-f', '--flat', action='store_true', default=False)
    args.add_argument('-n', '--n_trials', type=int, default=200)
    args.add_argument('-s', '--seed', type=int, default=42)
    args = args.parse_args()

    study = optuna.create_study(
        storage=f'sqlite:///{args.database}',
        study_name=args.agent,
        load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )

    study.optimize(
        partial(objective, agent=args.agent, hierarchical=not args.flat, seed=args.seed),
        n_trials=args.n_trials,
        n_jobs=-1,
        show_progress_bar=True
    )
