import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit

with open ("cudaconvnet_kernels.cu", "r") as f:
    kernel_code = f.read() #.replace('\n', '')

template_params = {
    'B_Y': 1,
    'B_X': 1,
    'imgsPerThread': 1,
    'filtersPerThread': 1,
    'numColors': 3,
    'pixelCache': 4,
    'scale': 'false',
    'checkImgBounds': 'true',
    'colorCache': 4
    }

for key, val in template_params.items():
    kernel_code = kernel_code.replace(key, str(val))
SourceModule(kernel_code)