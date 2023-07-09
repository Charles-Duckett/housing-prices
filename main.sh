#!/bin/bash

# clear the terminal
clear

# initialize conda
source ~/opt/anaconda3/etc/profile.d/conda.sh

# activate your conda environment
conda activate my-gpt

echo "Running feature-engineering.py"
python feature-engineering.py