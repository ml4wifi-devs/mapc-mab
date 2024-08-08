import json
import string
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict

from mapc_mab import plots

APS_NAMES = string.ascii_uppercase


def plot_histogram(actions: list, txops_slots: int, save_name: str) -> None:
    _, ax = plt.subplots(figsize=(8, 4))
    ax.bar([x[0] for x in actions], [x[1] for x in actions])
    ax.set_xlabel("APs' groups")
    ax.set_ylabel("Transmission opportunities")
    ax.set_title(
        "Histogram of simultaneously transmitting APs' groups" +
        f"\nTotal slots: {txops_slots}"
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-f', '--file', type=str, required=True)
    args.add_argument('-w', "--warmup", type=int, required=False, default=0)
    args.add_argument('-s', '--scenario', type=str, required=False)
    args.add_argument('-a', '--agent', type=str, required=False)
    args = args.parse_args()

    with open(args.file, 'r') as file:
        results = json.load(file)
    
    scenario = list(filter(lambda x: x['scenario']['name'] == args.scenario, results))[0] if args.scenario else results[0]
    agent = list(filter(lambda x: x['agent']['name'] == args.agent, scenario['agents']))[0] if args.agent else scenario['agents'][0]
    n_runs = len(agent['runs'])
    txops_slots = scenario['scenario']['n_steps'] - args.warmup

    # Define frequency dict of actions
    actions_dict = defaultdict(lambda: 0)

    # Count actions
    for run in agent["actions"]:
        for action in run[args.warmup:]:
            # Cast action to immutable type
            action = tuple(action.keys())
            actions_dict[action] += 1
    
    # Sort actions by frequency
    actions = sorted(actions_dict.items(), key=lambda x: x[1], reverse=True)

    # Scale actions by number of runs
    actions = [(action, freq / n_runs) for action, freq in actions]

    # Convert actions to APs' names
    aps = tuple(set([ap for action in actions_dict.keys() for ap in action]))
    aps = {ap: i for i, ap in enumerate(aps)}
    action_to_names = lambda tx: "".join([APS_NAMES[aps[t]] for t in tx])
    actions_aps = [(action_to_names(tx), freq) for tx, freq in actions]

    # Aggregate actions by tx APs
    actions_aps_aggregated = defaultdict(lambda: 0)
    for action, freq in actions_aps:
        actions_aps_aggregated[action] += freq

    # Re-run sorting
    actions_aps_aggregated = sorted(actions_aps_aggregated.items(), key=lambda x: x[1], reverse=True)

    # Plot histogram of actions_aps_aggregated
    save_name = f"aggregating-by-APs-{scenario['scenario']['name']}-{agent['agent']['name']}.pdf"
    plot_histogram(actions_aps_aggregated, txops_slots, save_name)