import numpy as np
from mem_eval import compute_mem 

# Create a 3D NumPy array
arr = np.random.rand(10000, 6, 4).astype(np.float64)

print(arr)
# Process the array
out = compute_mem(arr)

