#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Translates NVM file to MVSNet friendly format.

Given NVM file, parses into camera / points array, and translates into
MVS input format, for testing purpose
"""
from nvm.nvm import NVM
from mvs.mvs import MVS

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

DATASET_NAME='castle'

DEPTH_MIN = 425
DEPTH_INT = 2.5

DATASET_DIR = '/Users/jae/Research/dataset'
OUTPUT_DIR = '/Users/jae/Research/output/test'

INPUT_NVM_FILE = '{}/{}/reconstruction0.nvm'.format(DATASET_DIR, DATASET_NAME)
INPUT_IMAGE_PATH = '{}/{}/images/'.format(DATASET_DIR, DATASET_NAME)
OUTPUT_MVS_PATH = '{}/{}_{}_{}/'.format(OUTPUT_DIR, DATASET_NAME, DEPTH_MIN, DEPTH_INT)

NUM_PAIR_LIMIT = 5

# parse NVM
nvm_object = NVM().from_file(INPUT_NVM_FILE, INPUT_IMAGE_PATH)

# translate to MVS
mvs_object = MVS(DEPTH_MIN, DEPTH_INT)
mvs_object.from_nvm(nvm_object.models[0], INPUT_IMAGE_PATH)

# write output to MVS PATH
mvs_object.write_to_path(OUTPUT_MVS_PATH, NUM_PAIR_LIMIT)
