#!/bin/bash

if [[ $# -eq 0 ]]; then
    DBNAME="optuna.db"
else
    DBNAME=$1
fi

optuna-dashboard "sqlite:///$DBNAME"
