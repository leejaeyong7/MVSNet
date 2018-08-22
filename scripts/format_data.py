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

INPUT_NVM_FILE = './test_data/test.nvm'
INPUT_IMAGE_PATH = './test_data/test_images/'
OUTPUT_MVS_PATH = './test_output/'

DEPTH_MIN = 450
DEPTH_INT = 0.75

NUM_PAIR_LIMIT = 10

# parse NVM
nvm_object = NVM().from_file(INPUT_NVM_FILE)

# translate to MVS
mvs_object = MVS(DEPTH_MIN, DEPTH_INT)
mvs_object.from_nvm(nvm_object.models[0], INPUT_IMAGE_PATH)

# write output to MVS PATH
mvs_object.write_to_path(OUTPUT_MVS_PATH, NUM_PAIR_LIMIT)
