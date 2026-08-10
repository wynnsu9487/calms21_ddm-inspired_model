#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:26:23 2026

@author: wynnsu
"""

#loads data from calms21 json file
#distributes to feature_extract.py & feature_selection.py
#avoids repeatedly reading json file each time both files execute

import json
from pathlib import Path

DATA_DIR = Path('/Users/wynnsu/Downloads/DeepLabCut/notebook_h5_csv')

with open(DATA_DIR / 'calms21_task1_train.json', 'r') as file:
    data_train = json.load(file)

with open(DATA_DIR / 'calms21_task1_test.json', 'r') as file:
    data_test = json.load(file)