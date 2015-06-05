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


    # Set up images, filters, and outputs
    # imgd = np.loadtxt("im1.txt")
    # img = np.zeros((64, 64, 3))
    # print imgd.shape
    # for i in range(3):
    #     img[:, :, i] = imgd[i*64:(i+1)*64, :]
    # hostImages = np.tile(img)

    hostImages = np.random.rand(C, H, W, N)
    hostFilters = np.random.uniform(low=0.0, high=1.0, size=(C, S*R, numFilters)) #np.ones((C, S*R, numFilters)) #
    hostOutputs = np.zeros((numFilters, numModulesY, numModulesX, N))

    print "Input sum", np.sum(hostImages)

    # Run cc2 kernel    
    devI = ng.array(hostImages, dtype=dt)
    devF = ng.array(hostFilters, dtype=dt)
    devO = ng.array(hostOutputs, dtype=dt)

    ng.fprop_cuda_conv(layer, devI, devF, devO)

    print "CC2 input sum: ", np.sum(devI.asnumpyarray())
    print "CC2 output sum: ", np.sum(devO.asnumpyarray())

    # Run maxwel kernel
    # images: (C * H * W, N)
    # filters:  (C * S * R , numFilters)
    # outputs:  (numFilters * numModulesX * numModulesY, N)
    devI = ng.array(hostImages.reshape((C*H*W, N)), dtype=dt)
    devF = ng.array(hostFilters.reshape((C*S*R, numFilters)), dtype=dt)
    devO2 = ng.array(hostOutputs.reshape(numFilters*numModulesX*numModulesY, N), dtype=dt)

    ng.fprop_conv(layer, devI, devF, devO2)
    print "NG input sum: ", np.sum(devI.asnumpyarray())
    print "NG output sum: ", np.sum(devO2.asnumpyarray())

    hostOutputs1 = np.reshape(devO.asnumpyarray(), devO2.shape)
    hostOutputs2 = devO2.asnumpyarray()

    for i in xrange(hostOutputs1.shape[0]):
       for j in xrange(hostOutputs1.shape[1]):
           assert(abs(hostOutputs1[i, j] - hostOutputs2[i, j]) < 1e-4)

    # I = ng.ones((C, H, W, N), dtype=dt)
    # F = ng.ones((C, S*R, numFilters), dtype=dt)
    # O = ng.zeros((numFilters, numModulesY, numModulesX, N), dtype=dt)
    # layer.sizeI = I.size
    # layer.sizeF = F.size
    # layer.sizeO = O.size

if __name__ == '__main__':
    run()
