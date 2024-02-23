#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path-to-database-file>"
    exit 1
fi

optuna-dashboard "sqlite:///$1"
