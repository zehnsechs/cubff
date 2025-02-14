# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import sys
import re
import string
import struct

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '/bin/')
import cubff
from collections import defaultdict

NUM_RUNS = 100 

INIT_SEED = 0
THRESHOLD_ENTROPY = 3
THRESHOLD_HEADS_COUNT = 100000

MAX_EPOCHS = 10000


def StandardParams(f):
    params = cubff.SimulationParams()
    params.seed = INIT_SEED
    if f:
        params.load_from = f
    return params


def head_distances(soup):
    d = defaultdict(int)
    for i in range(len(soup)//64):
        p = soup[i*64:(i+1)*64]
        d[(p[0]-p[1])%128] += 1
    return d


def find_threshold_epoch(params):
    initial_epoch = 3700
    epochs = 0
    h_epoch = 0
    e_epoch = 0

    def callback(state):
        nonlocal initial_epoch
        if not initial_epoch:
            initial_epoch = state.epoch
        nonlocal epochs
        epochs = state.epoch
        d = head_distances(state.soup)
        m = max(d,key = d.get)

        nonlocal e_epoch
        nonlocal h_epoch
        if not e_epoch and state.higher_entropy > THRESHOLD_ENTROPY:
            e_epoch = epochs
        if not h_epoch and d[m] > THRESHOLD_HEADS_COUNT:
            h_epoch = epochs
        return state.epoch > MAX_EPOCHS + initial_epoch

    cubff.RunSimulation("bff", params, None, callback)
    return h_epoch,e_epoch

seeds = list(range(NUM_RUNS))
with open("".join(random.choice(string.ascii_lowercase) for _ in range(8)), "a") as rs:
    for s in seeds:

      params = StandardParams(None)
      params.seed = s
      print(s,find_threshold_epoch(params))
