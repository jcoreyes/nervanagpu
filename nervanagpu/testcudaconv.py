import logging
import numpy as np
from nervanagpu import NervanaGPU
import pycuda.autoinit
import math
logging.basicConfig(level=20)
logger = logging.getLogger()



def run():
    ng = NervanaGPU(stochastic_round=False)

    dt = np.float32
    # N: Number of images in mini-batch
    # C: Number of input feature maps
    # K: Number of output feature maps

    # D: Depth  of input image
    # H: Height of input image
    # W: Width  of input image

    # T: Depth  of filter kernel
    # R: Height of filter kernel
    # S: Width  of filter kernel
    # 
    # * images:      (numColors, imgSizeY, imgSizeX, numImages) with stride given
    # * filters:     (numColors, filterPixels, numFilters) if conv
    # *              (numModules, numColors, filterPixels, numFilters) otherwise
    # *
    # * targets:     (numFilters, numModulesY, numModulesX, numImages)

    N = 128
    C = 3
    K = 64

    D = 1
    H = 64
    W = 64

    T = 1
    R = 8
    S = 8

    pad_h = pad_w = 0
    str_h = str_w = 4

    layer = ng.conv_layer(dt, N, C, K,
            D=D, H=H, W=W,
            T=T, R=R, S=S,
            pad_d=0, pad_h=pad_h, pad_w=pad_w,
            str_d=1, str_h=str_h, str_w=str_w,
            grid_P=0, grid_Q=0, update_size=None)

    numImages = N 
    numFilters = K

    numModulesY = int(math.ceil(float(H - R + 1 + 2*pad_h) / str_h))
    numModulesX = int(math.ceil(float(W - S + 1 + 2*pad_w) / str_w))

    print "Num Modules ", numModulesX, numModulesY

    I = ng.ones((C, H, W, N), dtype=dt)
    F = ng.ones((C, S*R, numFilters), dtype=dt)
    O = ng.zeros((numFilters, numModulesY, numModulesX, N), dtype=dt)
    layer.sizeI = I.size
    layer.sizeF = F.size
    layer.sizeO = O.size

    ng.fprop_cuda_conv(layer, I, F, O)



if __name__ == '__main__':
    run()
