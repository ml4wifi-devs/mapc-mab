#!/bin/bash

RESOLUTION=50

align_scenario_1() {
    echo "Aligning scenario 1..."
    python mapc_mab/envs/scenarios/tuning/simple_scenario_1.py -m "4" -r "8" -p
    for (( mcs = 0; mcs < 12; mcs++ )); do
        echo "...MCS $mcs"
        python mapc_mab/envs/scenarios/tuning/simple_scenario_1.py -m "$mcs" -r "$1"
    done
    echo "Done."
}

align_scenario_2() {
    echo "Aligning scenario 2..."
    python mapc_mab/envs/scenarios/tuning/simple_scenario_2.py -m "11" -s "1" -r "8" -p
    python mapc_mab/envs/scenarios/tuning/simple_scenario_2.py -m "11" -s "1" -r "$1"
    echo "Done."
}

align_scenario_3() {
    echo "Aligning scenario 3..."
    python mapc_mab/envs/scenarios/tuning/simple_scenario_3.py -m "4" -r "8" -p
    for (( mcs = 0; mcs < 12; mcs++ )); do
        echo "...MCS $mcs"
        python mapc_mab/envs/scenarios/tuning/simple_scenario_3.py -m "$mcs" -r "$1"
    done
    echo "Done."
}

align_scenario_5() {
    echo "Aligning scenario 5..."
    python mapc_mab/envs/scenarios/tuning/simple_scenario_5.py -m "11" -s "2" -r "8" -p
    python mapc_mab/envs/scenarios/tuning/simple_scenario_5.py -m "11" -s "2" -r "$1"
    echo "Done."
}

align_scenario_1 $RESOLUTION
align_scenario_2 $RESOLUTION
align_scenario_3 $RESOLUTION
align_scenario_5 $RESOLUTION
