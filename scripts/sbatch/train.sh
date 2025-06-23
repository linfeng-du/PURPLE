#!/bin/bash

source /home/linfeng/.bashrc
source /home/linfeng/projects/def-cpsmcgil/linfeng/envs/bandit/bin/activate
python src/train.py $@
