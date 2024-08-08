import json
import string
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict

from mapc_mab import plots

APS_NAMES = string.ascii_uppercase


def plot_histogram(actions: list, txops_slots: int, fairness: float, save_name: str) -> None:
    n_stas = len(actions)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([x[0] for x in actions], [x[1] for x in actions])
    ax.plot([-0.5, n_stas - 0.5], [txops_slots / n_stas] * 2, color="black", linestyle="--", label="round-robin\nsingle transmission")
    ax.set_xlabel("Stations")
    ax.set_ylabel("Transmission opportunities")
    ax.set_title(
        f"Fairness of transmission opportunities" +
        f"\nJain's fairness index: {fairness:.3f}" +
        f"\nTotal slots: {txops_slots}"
    )
    ax.legend()
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
        for action in run[args.warmup:].values():
            actions_dict[action] += 1
    
    # Sort actions by frequency
    actions = sorted(actions_dict.items(), key=lambda x: x[1], reverse=True)

    # Scale actions by number of runs
    actions = [(action, freq / n_runs) for action, freq in actions]

    # Aggregate actions by tx APs
    actions_stas_aggregated = defaultdict(lambda: 0)
    for action, freq in actions:
        actions_stas_aggregated[f"STA_{action}"] += freq

    # Re-run sorting
    actions_stas_aggregated = sorted(actions_stas_aggregated.items(), key=lambda x: x[1], reverse=True)

    # Calculate Jain's fairness index
    txops_values = [x[1] for x in actions_stas_aggregated]
    sta_fairness = np.power(np.sum(txops_values), 2) / (len(txops_values) * np.sum(np.power(txops_values, 2)))
    print(f"Jain's fairness index: {sta_fairness}")

    # Plot histogram of actions_stas_aggregated
    save_name = f"aggregating-by-STAs-{scenario['scenario']['name']}-{agent['agent']['name']}.pdf"
    plot_histogram(actions_stas_aggregated, txops_slots, sta_fairness, save_name)