#!/bin/bash

python tuning.py -a EGreedy -d egreedy.db -n 300
python tuning.py -a EGreedy -d flat_egreedy.db -f -n 200

python tuning.py -a Softmax -d softmax.db -n 300
python tuning.py -a Softmax -d flat_softmax.db -f -n 200

python tuning.py -a UCB -d ucb.db -n 300
python tuning.py -a UCB -d flat_ucb.db -f -n 200

python tuning.py -a NormalThompsonSampling -d ts.db -n 300
python tuning.py -a NormalThompsonSampling -d flat_ts.db -f -n 200
