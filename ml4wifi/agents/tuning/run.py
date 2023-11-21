from argparse import ArgumentParser
from functools import partial

import numpy as np
import optuna
from reinforced_lib.agents.mab import *

from ml4wifi.agents import MapcAgentFactory
from ml4wifi.envs.run import run_scenario
from ml4wifi.envs.scenarios.static import random_scenario_1


TRAINING_SCENARIOS = [
    random_scenario_1(seed=1, d_ap=200., n_ap=2, d_sta=40., n_sta_per_ap=2, mcs=4),
    random_scenario_1(seed=7, d_ap=200., n_ap=2, d_sta=40., n_sta_per_ap=3, mcs=4),
    random_scenario_1(seed=7, d_ap=200., n_ap=3, d_sta=40., n_sta_per_ap=2, mcs=4),
    random_scenario_1(seed=7, d_ap=200., n_ap=3, d_sta=40., n_sta_per_ap=3, mcs=6),
    random_scenario_1(seed=10, d_ap=200., n_ap=3, d_sta=40., n_sta_per_ap=4, mcs=6),
    random_scenario_1(seed=9, d_ap=100., n_ap=4, d_sta=5., n_sta_per_ap=2, mcs=11),
    random_scenario_1(seed=10, d_ap=100., n_ap=4, d_sta=5., n_sta_per_ap=3, mcs=11),
    random_scenario_1(seed=20, d_ap=200., n_ap=4, d_sta=5., n_sta_per_ap=4, mcs=11)
]


def objective(trial: optuna.Trial, agent: str, n_steps: int) -> float:
    if agent == 'EGreedy':
        agent_type = EGreedy
        agent_params = {
            'e': trial.suggest_float('e', 1e-6, 1., log=True),
            'optimistic_start': trial.suggest_float('optimistic_start', 0., 1e4)
        }
    elif agent == 'Exp3':
        agent_type = Exp3
        agent_params = {
            'gamma': trial.suggest_float('gamma', 1e-6, 1., log=True),
            'min_reward': 0.,
            'max_reward': 1000.
        }
    elif agent == 'Softmax':
        agent_type = Softmax
        agent_params = {
            'lr': trial.suggest_float('lr', 1e-2, 1e2, log=True),
            'tau': trial.suggest_float('tau', 1e-2, 1e2, log=True),
            'multiplier': trial.suggest_float('multiplier', 1e-4, 1., log=True)
        }
    elif agent == 'UCB':
        agent_type = UCB
        agent_params = {
            'c': trial.suggest_float('c', 0., 1e3)
        }
    else:
        raise ValueError(f'Unknown agent {agent}')

    runs = []

    for step, scenario in enumerate(TRAINING_SCENARIOS):
        associations = scenario.get_associations()
        agent_factory = MapcAgentFactory(associations, agent_type, agent_params)

        results = run_scenario(agent_factory, scenario, n_reps=1, n_steps=n_steps, seed=42)
        runs.append(results)

        trial.report(np.mean(results), step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(runs)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-a', '--agent', type=str, required=True)
    args.add_argument('-d', '--database', type=str, default='optuna.db')
    args.add_argument('-p', '--plot', action='store_true', default=False)
    args.add_argument('-s', '--n_steps', type=int, default=2000)
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
