import cuda_filters
import numpy as np
from scipy.ndimage import median_filter
from timeit import default_timer
class Timer():
    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.end = default_timer()
        print(self.end - self.start)

arr = np.random.rand(9, 512, 512).astype(np.float32)
size = 3
with Timer() as t:
    filtered = cuda_filters.medfilt3(arr, size, 'clamp')
with Timer() as t:
    out2 = median_filter(arr, size=(size, size, size), mode='nearest')
print(np.array_equal(filtered, out2))
