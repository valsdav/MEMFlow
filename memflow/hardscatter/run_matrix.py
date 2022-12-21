from __future__ import annotations

import time

import hardscatter
import numpy as np
import vector

p1 = np.array(
    [
        [0.5000000e03, 0.0000000e00, 0.0000000e00, 0.5000000e03],
        [0.5000000e03, 0.0000000e00, 0.0000000e00, -0.5000000e03],
        [0.2500000e03, 0.0240730e03, 0.4173556e03, -0.1872274e03],
        [0.2500000e03, 0.0840730e03, 0.4173556e03, -0.1872274e03],
        [0.5000000e03, -0.1040730e03, -0.4173556e03, 0.1872274e03],
    ]
)


start = time.time()
pdg = [2, 2, -6, 6, 25]
p1T = p1.T
print(p1T)
for i in range(100000):
    hardscatter.smatrix_ttH(pdg, p1T)
end = time.time()

print("Elapsed (loop) = %s" % (end - start))


p1 = np.random

start = time.time()
for i in range(100000):
    hardscatter.smatrix_ttH(pdg, p1T)
end = time.time()

print("Elapsed (2nd loop) = %s" % (end - start))


p2 = np.tile(p1T, (10000, 1, 1))
pdgs2 = np.tile([2, 2, -6, 6, 25], (10000, 1, 1))
start = time.time()
hardscatter.smatrix_ttH_many(pdgs2, p2)
end = time.time()

print("Elapsed (no compilation) = %s" % (end - start))
