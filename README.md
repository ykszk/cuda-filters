# Image filters boosted by CUDA
* 3D median filter
  * Using forgetful selection.
  * Only filter size 3, 5, and 7 are available.
* basic morphological filters

## Build
Edit `TARGET_CUDA_ARCHS` in `cmake/SetCompilerFlags.cmake`

# Python

## Build module
```shell
cd python
git clone --depth 1 https://github.com/pybind/pybind11.git
cd ..
pip install .
```

### Bulid in Windows + Anaconda
- Open `x64 Native Tools Command Prompt for VS`
- Activate conda environment by `%UserProfile%\anaconda3\Scripts\activate`
- Build python module (described above)

## Example
```python
import numpy as np
import cuda_filters
arr3d = np.random.rand(512, 512, 512).astype(np.float32)
filter_size = 5 # 3, 5, or 7
mode = 'mirror' # clamp, border, mirror, or wrap
result = cuda_filters.medfilt3(arr3d, filter_size, mode)
```

In a test environment with Intel Core i7-4790K and NVIDIA TITAN X(Pascal), `cuda_filters.medfilt3` (shown above) took about 0.88 secs and `scipy.ndimage.median_filter` (code shown below) took about 154 secs.

```python
from scipy.ndimage import median_filter
result = median_filter(arr3d, size=filter_size, mode=mode)
```

# MATLAB
Execute static.bat in the build directory.
