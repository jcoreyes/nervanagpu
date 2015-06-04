#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import atexit
import math

# Textures aren't properly configured yet with pycuda
TEXTURE_SIZE_MAX = 0

def divup(a, b):
    if (a % b):
        return a / b + 1
    else:
        return a / b

def get_kernel_func(A, B, C,
                       imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride,
                       numImgColors, numGroups,
                       scaleTargets, scaleOutput, conv):
    kernel_args = []
    numFilterColors = numImgColors / numGroups
    numFilters = B.shape[2]
    numModules = numModulesY * numModulesX
    numImages = A.shape[3]
    imgPixels = A.shape[1] * A.shape[2]
    imgSizeX = imgPixels / imgSizeY
    filterModuleMult = 1 if conv else numModules #conv ? 1 : numModules
   
    assert(numGroups > 1 or (numImgColors > 0 and (numImgColors <= 3 or numImgColors % 4 == 0)))
    assert(numGroups == 1 or numFilterColors % 4 == 0)
    assert(numFilters % (16 * numGroups) == 0)
    assert(numImgColors % numGroups == 0)
    assert(A.size/A.shape[3] == imgPixels * numImgColors)
    assert(imgSizeY * imgSizeX == imgPixels)
    numFiltersPerGroup = numFilters / numGroups
    imgStride = A.shape[3] # images does not need to be a contiguous matri
    filterPixels = B.shape[1] # (filterModuleMult * numFilterColors)
    filterSize = int(math.sqrt(filterPixels))
    assert(filterSize * filterSize == filterPixels)
    #print B.shape
    #print numFilterColors, filterPixels, filterModuleMult
    #assert(B.size == filterModuleMult * numFilterColors * filterPixels)
    
    # These routines don't handle the case when only part of the image is visited in the convolutio
    assert(paddingStart <= 0)
    print paddingStart + (numModulesX-1)*moduleStride + filterSize
    assert(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX)
    assert(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY)
    assert(moduleStride <= filterSize)

    imgsPerThread = 4 if numImages % 128 == 0 else 2 if numImages % 64 == 0 else 1
    filtersPerThread = threadsY = 4;
    if (numImgColors <= 3):
        filtersPerThread = 16 if numFiltersPerGroup % 64 == 0 else 12 if numFiltersPerGroup % 48 == 0 else 8 if numFiltersPerGroup % 32 == 0 else 4
    else:
        filtersPerThread = 16 if numFiltersPerGroup % 64 == 0 else 8 if numFiltersPerGroup % 32 == 0 else 4
        threadsY = 8 if numFiltersPerGroup % 128 == 0 and numFilterColors % 8 == 0  and imgsPerThread != 4 else 4
    threadsX = 32;

    threads = (threadsX, threadsY, 1)
    blocks = (divup(numImages, threads[0] * imgsPerThread), (numModules * numFilters) / (threads[1] * filtersPerThread))
    
    checkImgBounds = numImages % (threads[0]*imgsPerThread) != 0
    scale = scaleTargets != 0
    if (scaleTargets == 0):
        C = C.reshape((numFilters * numModules, numImages))
    else:
        assert(C.size/C.shape[3] == numFilters * numModules)
        assert(C.shape[3] == numImages)

    if (scale == False):
        if (checkImgBounds == False):
            if (numFilterColors % 8 == 0):
                if (numImages % 128 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        if (A.nbytes<TEXTURE_SIZE_MAX and B.nbytes<TEXTURE_SIZE_MAX):
                            #cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, False, False >, cudaFuncCachePreferL1)
                            func = 'filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex'
                            template_param = (4, 32, 4, 16, 4, False, False)
                            kernel_args = (blocks, threads, A.getTextureObject(), B.getTextureObject(), C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                        else:
                            #cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, False, False >, cudaFuncCachePreferL1)
                            func = 'filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4'
                            template_param = (4, 32, 4, 16, 4, False, False)
                            kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        if (A.nbytes<TEXTURE_SIZE_MAX and B.nbytes<TEXTURE_SIZE_MAX):
                            #cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, False, False >, cudaFuncCachePreferL1)
                            func = 'filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex'
                            template_param = (4, 32, 4, 16, 4, False, False)
                            kernel_args = (blocks, threads, A.getTextureObject(), B.getTextureObject(), C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                        else:
                            #cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, False, False >, cudaFuncCachePreferL1)
                            func = 'filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4'
                            template_param = (4, 32, 4, 16, 4, False, False)
                            kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 8, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 8, 8, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 4, 8, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                elif (numImages % 64 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 2, 16, 8, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (8, 32, 2, 16, 8, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 8, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 16, 8, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 8, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 8, 8, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 4, 8, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                elif (numImages % 32 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (8, 32, 1, 16, 8, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 8, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 8, 8, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 4, 8, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
            elif (numFilterColors % 4 == 0):
                if (numImages % 128 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 16, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 16, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 8, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 4, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                elif (numImages % 64 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 16, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 16, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 8, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 4, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                elif (numImages % 32 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 8, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 4, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 3):
                if (numImages % 128 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        if (A.nbytes<TEXTURE_SIZE_MAX and B.nbytes<TEXTURE_SIZE_MAX):
                            #cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, False, False >, cudaFuncCachePreferShared)
                            func = 'filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex'
                            template_param = (4, 32, 4, 16, 3, 4, False, False)
                            kernel_args = (blocks, threads, A.getTextureObject(), B.getTextureObject(), C.gpudata,numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                        else:
                            #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 3, 4, False, False >, cudaFuncCachePreferShared)
                            func = 'filterActs_YxX_color'
                            template_param = (4, 32, 4, 16, 3, 4, False, False)
                            kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata,numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        if (A.nbytes<TEXTURE_SIZE_MAX and B.nbytes<TEXTURE_SIZE_MAX):
                            #cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, False, False >, cudaFuncCachePreferShared)
                            func = 'filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex'
                            template_param = (4, 32, 4, 12, 3, 4, False, False)
                            kernel_args = (blocks, threads, A.getTextureObject(), B.getTextureObject(), C.gpudata,numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                        else:
                            #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 3, 4, False, False >, cudaFuncCachePreferShared)
                            func = 'filterActs_YxX_color'
                            template_param = (4, 32, 4, 12, 3, 4, False, False)
                            kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata,numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 3, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 8, 3, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 3, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 4, 3, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 64 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 3, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 16, 3, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 3, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 12, 3, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 3, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 8, 3, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 3, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 4, 3, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 32 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 3, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 3, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 3, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 3, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 2):
                if (numImages % 128 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 16, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 12, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 8, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 4, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 64 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 16, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 12, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 8, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 4, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 32 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 2, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 1):
                if (numImages % 128 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 16, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 12, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 8, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 4, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 64 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 16, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 12, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 8, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 4, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 32 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, False, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 1, 4, False, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
        elif (checkImgBounds == True):
            if (numFilterColors % 8 == 0):
                if (numImages % 1 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (8, 32, 1, 16, 8, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 8, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 8, 8, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 4, 8, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
            elif (numFilterColors % 4 == 0):
                if (numImages % 1 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 8, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 4, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 3):
                if (numImages % 1 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 3, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 3, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 3, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 3, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 2):
                if (numImages % 1 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 2, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 2, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 2, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 2, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 1):
                if (numImages % 1 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 1, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 1, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 1, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, False, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 1, 4, False, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
    elif (scale == True):
        if (checkImgBounds == False):
            if (numFilterColors % 8 == 0):
                if (numImages % 128 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        if (A.nbytes<TEXTURE_SIZE_MAX and B.nbytes<TEXTURE_SIZE_MAX):
                            #cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, True, False >, cudaFuncCachePreferL1)
                            func = 'filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex'
                            template_param = (4, 32, 4, 16, 4, True, False)
                            kernel_args = (blocks, threads, A.getTextureObject(), B.getTextureObject(), C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                        else:
                            #cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, True, False >, cudaFuncCachePreferL1)
                            func = 'filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4'
                            template_param = (4, 32, 4, 16, 4, True, False)
                            kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        if (A.nbytes<TEXTURE_SIZE_MAX and B.nbytes<TEXTURE_SIZE_MAX):
                            #cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, True, False >, cudaFuncCachePreferL1)
                            func = 'filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex'
                            template_param = (4, 32, 4, 16, 4, True, False)
                            kernel_args = (blocks, threads, A.getTextureObject(), B.getTextureObject(), C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                        else:
                            #cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, True, False >, cudaFuncCachePreferL1)
                            func = 'filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4'
                            template_param = (4, 32, 4, 16, 4, True, False)
                            kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 8, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 8, 8, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 4, 8, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                elif (numImages % 64 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 2, 16, 8, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (8, 32, 2, 16, 8, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 8, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 16, 8, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 8, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 8, 8, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 4, 8, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                elif (numImages % 32 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (8, 32, 1, 16, 8, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 8, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 8, 8, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 4, 8, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
            elif (numFilterColors % 4 == 0):
                if (numImages % 128 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 16, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 16, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 8, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 4, 4, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                elif (numImages % 64 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 16, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 16, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 8, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 2, 4, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                elif (numImages % 32 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 8, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 4, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 3):
                if (numImages % 128 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        if (A.nbytes<TEXTURE_SIZE_MAX and B.nbytes<TEXTURE_SIZE_MAX):
                            #cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, True, False >, cudaFuncCachePreferShared)
                            func = 'filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex'
                            template_param = (4, 32, 4, 16, 3, 4, True, False)
                            kernel_args = (blocks, threads, A.getTextureObject(), B.getTextureObject(), C.gpudata,numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                        else:
                            #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 3, 4, True, False >, cudaFuncCachePreferShared)
                            func = 'filterActs_YxX_color'
                            template_param = (4, 32, 4, 16, 3, 4, True, False)
                            kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata,numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        if (A.nbytes<TEXTURE_SIZE_MAX and B.nbytes<TEXTURE_SIZE_MAX):
                            #cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, True, False >, cudaFuncCachePreferShared)
                            func = 'filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex'
                            template_param = (4, 32, 4, 12, 3, 4, True, False)
                            kernel_args = (blocks, threads, A.getTextureObject(), B.getTextureObject(), C.gpudata,numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                        else:
                            #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 3, 4, True, False >, cudaFuncCachePreferShared)
                            func = 'filterActs_YxX_color'
                            template_param = (4, 32, 4, 12, 3, 4, True, False)
                            kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata,numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 3, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 8, 3, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 3, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 4, 3, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 64 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 3, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 16, 3, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 3, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 12, 3, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 3, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 8, 3, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 3, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 4, 3, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 32 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 3, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 3, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 3, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 3, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 2):
                if (numImages % 128 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 16, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 12, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 8, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 4, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 64 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 16, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 12, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 8, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 4, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 32 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 2, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 1):
                if (numImages % 128 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 16, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 12, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 8, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 4, 4, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 64 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 16, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 12, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 8, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 2, 4, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                elif (numImages % 32 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, True, False >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 1, 4, True, False)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
        elif (checkImgBounds == True):
            if (numFilterColors % 8 == 0):
                if (numImages % 1 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (8, 32, 1, 16, 8, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 8, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 8, 8, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 4, 8, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
            elif (numFilterColors % 4 == 0):
                if (numImages % 1 == 0):
                    if (numFiltersPerGroup % 128 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 16, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 8, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_sparse2'
                        template_param = (4, 32, 1, 4, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 3):
                if (numImages % 1 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 3, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 3, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 3, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 3, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 2):
                if (numImages % 1 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 2, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 2, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 2, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 2, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
            elif (numFilterColors == 1):
                if (numImages % 1 == 0):
                    if (numFiltersPerGroup % 64 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 16, 1, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 48 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 12, 1, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 32 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 8, 1, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)
                    elif (numFiltersPerGroup % 1 == 0):
                        #cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, True, True >, cudaFuncCachePreferShared)
                        func = 'filterActs_YxX_color'
                        template_param = (4, 32, 1, 4, 1, 4, True, True)
                        kernel_args = (blocks, threads, A.gpudata, B.gpudata, C.gpudata, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv)

    with open (os.path.dirname(__file__) + "/cudaconvnet_kernels.cu", "r") as f:
        kernel_code = f.read() #.replace('\n', '')
    t_param_name1 = ['B_Y', 'B_X', 'imgsPerThread', 'filtersPerThread', 'colorCache', 'scale', 'checkImgBounds']
    t_param_name2 = ['B_Y', 'B_X', 'imgsPerThread', 'filtersPerThread', 'numColors', 'pixelCache', 'scale', 'checkImgBounds']

    # test = {
    #     'B_Y': 1,
    #     'B_X': 1,
    #     'imgsPerThread': 1,
    #     'filtersPerThread': 1,
    #     'numColors': 3,
    #     'pixelCache': 4,
    #     'scale': 'false',
    #     'checkImgBounds': 'true',
    #     'colorCache': 4
    #     }
    
    template_name = t_param_name1 if len(template_param) == len(t_param_name1) else t_param_name2
    print func
    print template_param
    # for (key, val) in zip(template_name, template_param):
    #     kernel_code = kernel_code.replace(key, str(val).lower())

    # unused_params = ['numColors', 'pixelCache', 'colorCache']
    # for unused_key in unused_params:
    #     if unused_key not in template_name:
    #         kernel_code = kernel_code.replace(unused_key, str(1))

    template_params = {
        'B_Y': 4,
        'B_X': 32,
        'imgsPerThread': 4,
        'filtersPerThread': 16,
        'numColors': 3,
        'pixelCache': 4,
        'scale': 'false',
        'checkImgBounds': 'false',
        'colorCache': 4
        }

    for key, val in template_params.items():
        kernel_code = kernel_code.replace(key, str(val))

    module = SourceModule(kernel_code)
    kernel_func = module.get_function(func)
    if len(template_param) == 16:
        kernel_func.prepare("PPPIIIIIIIIIIff?")
    else:
        kernel_func.prepare("PPPIIIIIIIIIIff?")
    return kernel_func, kernel_args

def get_module(template_params):
    kernel_code = "" 
    #template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache,
    #          bool scale, bool checkImgBounds>
    #4, 32, 4, 16, 3, 4
    for key, val in template_params.items():
        kernel_code = kernel_code.replace(key, str(val))
    return SourceModule(kernel_code)

