# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from scripts import ranges as R

range_seed = [
    [['seed'], [x * 100 for x in range(8)]],
]

range_dim = [
    [['environment', 'dim'], [2, 10, 20, 100]],
]

range_fixed = [
    [['environment', 'fixed'], [False, True]],
]

# use reduce
range_maml = R.merge_ranges(range_seed, range_dim)
range_maml = R.merge_ranges(range_maml, range_fixed)

