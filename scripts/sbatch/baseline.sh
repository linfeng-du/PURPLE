#!/bin/bash

source /home/linfeng/.bashrc
source /home/linfeng/projects/def-cpsmcgil/linfeng/environment/bandit_pr/bin/activate
python src/baseline.py $@
