#!/bin/bash

RESOLUTION=50

align_scenario_1() {
    echo "Aligning scenario 1..."
    python ml4wifi/params_fit/scenario_1_alignment.py -m "11" -r "8" -p
    for (( mcs = 0; mcs < 12; mcs++ )); do
        echo "...MCS $mcs"
        python ml4wifi/params_fit/scenario_1_alignment.py -m "$mcs" -r "$1"
    done
    echo "Done."
}

align_scenario_2() {
    echo "Aligning scenario 2..."
    python ml4wifi/params_fit/scenario_2_alignment.py -m "11" -s "1" -r "8" -p
    python ml4wifi/params_fit/scenario_2_alignment.py -m "11" -s "1" -r "$1"
    echo "Done."
}

align_scenario_1 $RESOLUTION
align_scenario_2 $RESOLUTION
