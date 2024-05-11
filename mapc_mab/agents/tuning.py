import os
os.environ['JAX_ENABLE_X64'] = 'True'

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
    (random_scenario(seed=6, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, max_steps=3000), 3000)
]
SLOTS_AHEAD = 1


def objective(trial: optuna.Trial, agent: str, hierarchical: bool) -> float:
    if agent == 'EGreedy':
        agent_type = EGreedy
        agent_params = {
            'e': trial.suggest_float('e', 0.001, 0.1, log=True),
            'optimistic_start': trial.suggest_float('optimistic_start', 0., 1000.),
            'alpha': trial.suggest_float('alpha', 0., 1.)
        }
    elif agent == 'Softmax':
        agent_type = Softmax
        agent_params = {
            'lr': trial.suggest_float('lr', 0.01, 100., log=True),
            'tau': trial.suggest_float('tau', 0.001, 100., log=True),
            'multiplier': trial.suggest_float('multiplier', 0.0001, 1., log=True),
            'alpha': trial.suggest_float('alpha', 0., 1.)
        }
    elif agent == 'UCB':
        agent_type = UCB
        agent_params = {
            'c': trial.suggest_float('c', 0., 100.),
            'gamma': trial.suggest_float('gamma', 0., 1.)
        }
    elif agent == 'NormalThompsonSampling':
        agent_type = NormalThompsonSampling
        agent_params = {
            'alpha': trial.suggest_float('alpha', 0., 100.),
            'beta': trial.suggest_float('beta', 0., 100.),
            'lam': 0.,
            'mu': trial.suggest_float('mu', 0., 1000.)
        }
    else:
        raise ValueError(f'Unknown agent {agent}')

    runs = []

    for step, (scenario, n_steps) in enumerate(TRAINING_SCENARIOS):
        agent_factory = MapcAgentFactory(scenario.associations, agent_type, agent_params, hierarchical, seed=42)
        results = np.mean(run_scenario(agent_factory, scenario, n_reps=1, n_steps=n_steps, slots_ahead=SLOTS_AHEAD, seed=42)[0])
        runs.append(results)

        trial.report(results, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(runs)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-a', '--agent', type=str, required=True)
    args.add_argument('-d', '--database', type=str, required=True)
    args.add_argument('-f', '--flat', action='store_true', default=False)
    args.add_argument('-n', '--n_trials', type=int, default=200)
    args = args.parse_args()

    study = optuna.create_study(
        storage=f'sqlite:///{args.database}',
        study_name=args.agent,
        load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner()
    )

    study.optimize(
        partial(objective, agent=args.agent, hierarchical=not args.flat),
        n_trials=args.n_trials,
        n_jobs=-1,
        show_progress_bar=True
    )
