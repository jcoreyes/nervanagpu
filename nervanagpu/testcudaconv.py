import logging
import numpy as np
from nervanagpu import NervanaGPU
import pycuda.autoinit
logging.basicConfig(level=20)
logger = logging.getLogger()



def run():
    ng = NervanaGPU(stochastic_round=False)

    bt = np.float32
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
    C = 8
    K = 8

    D = 3
    H = 64
    W = 64

    T = 3
    R = 5
    S = 5

    numFilters = 1

    layer = ng.conv_layer(bt, N, C, K,
            D=D, H=H, W=W,
            T=T, R=R, S=S,
            pad_d=0, pad_h=0, pad_w=0,
            str_d=1, str_h=1, str_w=1,
            grid_P=0, grid_Q=0, update_size=None)
    I = ng.ones((D, H, W, N))
    F = ng.ones((T, S*R, numFilters))
    O = ng.zeros((numFilters, 13, 13, 128))
    layer.sizeI = I.size
    layer.sizeF = F.size
    layer.sizeO = O.size

    #kwargs = {'backend': be, 'batch_size': 1, 'nofm':192}
    #layer.initialize(kwargs)
    inputs = np.zeros((5, 5))
    ng.fprop_cuda_conv(layer, I, F, O)



if __name__ == '__main__':
    run()
