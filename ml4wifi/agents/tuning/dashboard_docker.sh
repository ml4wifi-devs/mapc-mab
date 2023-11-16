#!/bin/bash

if [[ $# -eq 0 ]]; then
    DBNAME="optuna.db"
else
    DBNAME=$1
fi

docker run -it --rm -p 8080:8080 -v "$(pwd):/app" -w /app  ghcr.io/optuna/optuna-dashboard "sqlite:///$DBNAME"
