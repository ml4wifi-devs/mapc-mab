import os
os.environ['JAX_ENABLE_X64'] = 'True'

from argparse import ArgumentParser
from functools import partial

import numpy as np
import optuna
from reinforced_lib.agents.mab import *

from ml4wifi.agents import MapcAgentFactory
from ml4wifi.envs.run import run_scenario
from ml4wifi.envs.scenarios.static import random_scenario


TRAINING_SCENARIOS = [
    random_scenario(seed=7, d_ap=100., n_ap=2, d_sta=5., n_sta_per_ap=5),
    random_scenario(seed=18, d_ap=100., n_ap=3, d_sta=3., n_sta_per_ap=3),
    random_scenario(seed=16, d_ap=100., n_ap=4, d_sta=3., n_sta_per_ap=4),
    random_scenario(seed=19, d_ap=100., n_ap=5, d_sta=5., n_sta_per_ap=4),
    random_scenario(seed=6, d_ap=100., n_ap=2, d_sta=1., n_sta_per_ap=3),
    random_scenario(seed=9, d_ap=100., n_ap=3, d_sta=3., n_sta_per_ap=5),
    random_scenario(seed=3, d_ap=100., n_ap=4, d_sta=1., n_sta_per_ap=5),
    random_scenario(seed=19, d_ap=100., n_ap=5, d_sta=1., n_sta_per_ap=4)
]


def objective(trial: optuna.Trial, agent: str, n_steps: int) -> float:
    if agent == 'EGreedy':
        agent_type = EGreedy
        agent_params = {
            'e': trial.suggest_float('e', 1e-7, 1e-1, log=True),
            'optimistic_start': trial.suggest_float('optimistic_start', 1., 1e5, log=True)
        }
    elif agent == 'Softmax':
        agent_type = Softmax
        agent_params = {
            'lr': trial.suggest_float('lr', 1e-2, 1e2, log=True),
            'tau': trial.suggest_float('tau', 1e-3, 1e2, log=True),
            'multiplier': trial.suggest_float('multiplier', 1e-4, 10., log=True)
        }
    elif agent == 'UCB':
        agent_type = UCB
        agent_params = {
            'c': trial.suggest_float('c', 0., 1e3)
        }
    elif agent == 'NormalThompsonSampling':
        agent_type = NormalThompsonSampling
        agent_params = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1e3, log=True),
            'beta': trial.suggest_float('beta', 1e-3, 1e3, log=True),
            'lam': trial.suggest_float('lam', 1e-4, 1e2, log=True),
            'mu': trial.suggest_float('mu', 1e-1, 5e4, log=True)
        }
    else:
        raise ValueError(f'Unknown agent {agent}')

    runs = []

    for step, scenario in enumerate(TRAINING_SCENARIOS):
        associations = scenario.get_associations()
        agent_factory = MapcAgentFactory(associations, agent_type, agent_params)

        results = np.mean(run_scenario(agent_factory, scenario, n_reps=1, n_steps=n_steps, seed=42))
        runs.append(results)

        trial.report(results, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(runs)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-a', '--agent', type=str, required=True)
    args.add_argument('-d', '--database', type=str, default='optuna.db')
    args.add_argument('-p', '--plot', action='store_true', default=False)
    args.add_argument('-s', '--n_steps', type=int, default=700)
    args.add_argument('-n', '--n_trials', type=int, required=True)
    args = args.parse_args()

    if args.plot:
        for i, scenario in enumerate(TRAINING_SCENARIOS, start=1):
            scenario.plot(f'training_scenario_{i}.pdf')

    study = optuna.create_study(
        storage=f'sqlite:///{args.database}',
        study_name=args.agent,
        load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner()
    )

    study.optimize(
        partial(objective, agent=args.agent, n_steps=args.n_steps),
        n_trials=args.n_trials,
        n_jobs=-1,
        show_progress_bar=True
    )
