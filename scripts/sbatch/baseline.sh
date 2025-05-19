#!/bin/bash

source /home/linfeng/.bashrc
source /home/linfeng/projects/def-cpsmcgil/linfeng/environment/bandit/bin/activate
python src/baseline.py $@
