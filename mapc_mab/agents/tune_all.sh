#!/bin/bash

python tuning.py -a EGreedy -d egreedy.db
python tuning.py -a EGreedy -d flat_egreedy.db -f

python tuning.py -a Softmax -d softmax.db
python tuning.py -a Softmax -d flat_softmax.db -f

python tuning.py -a UCB -d ucb.db
python tuning.py -a UCB -d flat_ucb.db -f

python tuning.py -a NormalThompsonSampling -d ts.db
python tuning.py -a NormalThompsonSampling -d flat_ts.db -f
