#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import atexit

def get_module(template_params):
    kernel_code = """
    #define FA_COLOR3_IMPRELOAD(c,i) imPreload[c][i] = iPidxNext < 0 || (checkImgBounds && myImgIdx + i * B_X >= numImages) ? 0 : mm[c * imgPixels * imgStride + i * B_X];
    #define FA_COLOR3_IMPRELOAD_TX(c,i) imPreload[c][i] = iPidxNext < 0 || (checkImgBounds && myImgIdx + i * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imagesOffset2 + c * imgPixels * imgStride + i * B_X);
    #define DIVUP(x, y) (((x) + (y) - 1) / (y))
    __device__ __forceinline__ void filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(int fPidx, int imgLoadModPosY, int imgLoadModPosX,
                                                                                            int imgSizeX, int filterSize, int& iPidx) {
        int x = imgLoadModPosX + (fPidx) % filterSize;
        int y = imgLoadModPosY + (fPidx) / filterSize;
        iPidx = y >= 0 && y < imgSizeX && x >= 0 && x < imgSizeX ? y * imgSizeX + x : -1;
    }
    __global__ void filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
                                           const int numImages, const int numFilters,
                                           const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                           const int moduleStride,
                                           const int numModulesY, const int numModulesX, const int imgStride,
                                           const float scaleTargets, const float scaleOutputs,
                                           const bool conv/*, const bool noloads*/) {
        __shared__ float shFilters[numColors][pixelCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
        __shared__ float shImages[numColors][pixelCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
        const int imgPixels = imgSizeY * imgSizeX;
        const int filterPixels = filterSize * filterSize;
        const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
        const int moduleIdx = blockIdx.y / blocksPerModule;
        const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);

        const int numModules = numModulesX * numModulesY;
        // Another fun insanity: the % B_X makes things faster, even though threadIdx.x is
        // in the range 0..31. It appears that this allows the compiler to optimize?
        const int tx = threadIdx.x % B_X;
        const int ty = threadIdx.y % B_Y;
        const int tidx = ty * B_X + threadIdx.x;
        const int warp = tidx / 32;

        const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
        const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

        const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
        const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
        const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    //    images += myImgIdx;
    //    filters += blockFilterIdx
    //            + shFilterLoadY * numFilters + shFilterLoadX;
    //    if (!conv) { // NOTE: UNTESTED!
    //        filters += moduleIdx * numColors * filterPixels * numFilters;
    //    }

        const int imagesOffset = myImgIdx;
        const int filtersOffset = blockFilterIdx + shFilterLoadY * numFilters + shFilterLoadX
                                + (conv ? 0 : moduleIdx * numColors * filterPixels * numFilters);

        targets += moduleIdx * numImages
                + (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
                + myImgIdx;

        float prod[imgsPerThread][filtersPerThread];
        #pragma unroll
        for(int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
                prod[i][f] = 0;
            }
        }

        int iPidxNext;
        float imPreload[numColors][imgsPerThread];
        float fPreload[numColors][DIVUP(pixelCache*filtersPerThread,B_X)];

        if (warp < 3) {
            #pragma unroll
            for (int c = 0; c < numColors; ++c) {
                #pragma unroll
                for (int p = 0; p < pixelCache; p += 2) {
                    if (p + shFilterLoadY < filterPixels) {
                        fPreload[c][p/2] = tex1Dfetch<float>(filters, filtersOffset + p * numFilters + c * numFilters * filterPixels);
                    } else {
                        fPreload[c][p/2] = 0;
                    }
                }
            }
        }

        filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

        #pragma unroll
        for (int c = 0; c < numColors; ++c) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (iPidxNext >= 0 && (!checkImgBounds || myImgIdx + i * B_X < numImages)) {
                    imPreload[c][i] = tex1Dfetch<float>(images, imagesOffset + (c * imgPixels + iPidxNext) * imgStride + i * B_X);
                } else {
                    imPreload[c][i] =  0;
                }
            }
        }

        for (int p = 0; p < filterPixels; p += pixelCache) {
            const int fPidxNext = p + pixelCache >= filterPixels ? 0 : p + pixelCache;
            filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(fPidxNext + ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

            #pragma unroll
            for (int c = 0; c < numColors; ++c) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    // NOTE: bank conflicts here!
                    shImages[c][ty][tx * imgsPerThread + i] = imPreload[c][i];
                }
            }

            if (warp < 3) {
                #pragma unroll
                for (int c = 0; c < numColors; ++c) {
                    #pragma unroll
                    for (int pp = 0; pp < pixelCache; pp += 2) {
                        shFilters[c][pp + shFilterLoadY][shFilterLoadX] = fPreload[c][pp/2];
                    }
                }
            }

            __syncthreads();
    //        const float* ff = &filters[numFilters * fPidxNext];
    //        const float* mm = &images[imgStride * iPidxNext];
            const int filtersOffset2 = filtersOffset + numFilters * fPidxNext;
            const int imagesOffset2 = imagesOffset + imgStride * iPidxNext;

            #pragma unroll
            for (int i = 0; i < imgsPerThread; ++i) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    FA_COLOR3_IMPRELOAD_TX(c,i);
                }
            }

            #pragma unroll
            for (int c = 0; c < numColors; c++) {
                #pragma unroll
                for (int pp = 0; pp < 2; pp++) {
                    fPreload[c][pp] = warp >= 3 || fPidxNext + pp*2 + shFilterLoadY >= filterPixels ? 0 : tex1Dfetch<float>(filters, filtersOffset2 +  c * numFilters* filterPixels + pp*2 * numFilters);
                }
                #pragma unroll
                for (int pp = 0; pp < pixelCache; pp++) {
                    #pragma unroll
                    for(int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for(int f = 0; f < filtersPerThread; f++) {
                            prod[i][f] += shImages[c][pp][tx * imgsPerThread + i] * shFilters[c][pp][ty * filtersPerThread + f];
                        }
                    }
                }

            }
            __syncthreads();
        }

        if (scaleFlag) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                        targets[i * B_X + f * numImages * numModules] = scaleTargets * targets[i * B_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
                    }
                }
            }
        } else {
            // Note: reversing order of these loops costs 2 registers, but saves time
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                        targets[i * B_X + f * numImages * numModules] = scaleOutputs * prod[i][f];
                    }
                }
            }
        }
    }
    __global__ void filterActs_YxX_color(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv) {
        __shared__ float shFilters[pixelCache*numColors][B_Y * filtersPerThread]; // pre-load pixelCache pixels from B_Y*filtersPerThread filters
        __shared__ float shImages[pixelCache*numColors][B_X * imgsPerThread]; // pre-load pixelCache pixels from B_X*imgsPerThread images
        const int imgPixels = imgSizeY * imgSizeX;
        const int filterPixels = filterSize * filterSize;

        const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
        const int moduleIdx = blockIdx.y / blocksPerModule;
        const int blockFilterIdx = blockIdx.y % blocksPerModule;

        const int tidx = threadIdx.y * B_X + threadIdx.x;

        const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
        const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;
        const int numModules = numModulesY * numModulesX;
        const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
        const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
        const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
        images += myImgIdx;
        filters += filtersPerThread * B_Y * blockFilterIdx
                 + shFilterLoadY * numFilters + shFilterLoadX;
        if (!conv) {
            filters += moduleIdx * numColors * filterPixels * numFilters;
        }

        targets += moduleIdx * numImages
                + (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y*filtersPerThread) * numImages * numModulesY * numModulesX
                + myImgIdx;


        float prod[filtersPerThread][imgsPerThread];
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for(int g = 0; g < imgsPerThread; g++) {
                prod[f][g] = 0;
            }
        }
        //float* shImgLoad = &shImages[0][threadIdx.x]; 
        for (int p = 0; p < filterPixels; p += pixelCache) {
            /*
             * Load pixelCache pixels from B_Y*filtersPerThread filters
             * This condition covers the case when B_X is not divisible by filtersPerThread.
             * In this case, not all of the threads will participate in the loading operation.
             * This ensures that in each loop iteration, an integer number of rows of shFilters
             * are filled, which makes indexing simple.
             */
            if (B_X % filtersPerThread == 0 || shFilterLoadY < B_X/filtersPerThread) {
                #pragma unroll
                for (int p2 = 0; p2 < pixelCache; p2 += B_X/filtersPerThread) {
                    const bool omit = pixelCache % (B_X / filtersPerThread) == 0;
                    const int preloadPx = shFilterLoadY + p2;
                    if (omit || preloadPx < pixelCache) {
                        if (p + preloadPx < filterPixels) {
                            #pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shFilters[shFilterLoadY + p2 + c * pixelCache][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shFilters[shFilterLoadY + p2 + c * pixelCache][shFilterLoadX] = 0;
                            }
                        }
                    }
                }
            }
            
            /*
             * Load pixelCache pixels from B_X*imgsPerThread images.
             */
            #pragma unroll
            for (int ly = 0; ly < pixelCache; ly += B_Y) {
                const int preloadPx = ly + threadIdx.y;
                const int pixIdx = p + preloadPx;
                const bool omit = pixelCache % B_Y == 0; // Compile-time condition
                /*
                 * Don't load any image pixels corresponding to filter pixels that don't exist.
                 */
                if (pixIdx < filterPixels && (omit || preloadPx < pixelCache)) {
                    const int x = imgLoadModPosX + pixIdx % filterSize;
                    const int y = imgLoadModPosY + pixIdx / filterSize;
                     
                    if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                        float* m = &images[imgStride * (y * imgSizeX + x)];
                        
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread; i++) {
                                if (!checkImgBounds || myImgIdx + i * B_X < numImages) { 
                                    shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = m[c * imgStride * imgPixels + i * B_X];
                                } else {
                                    shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = 0;
                                }
                            }
                        }
                    } else { // Padding
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            #pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = 0;
                            }
                        }
                    }
                }
            }
            
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < pixelCache*numColors; i++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        prod[f][g] += shImages[i][g + threadIdx.x * imgsPerThread] * shFilters[i][threadIdx.y * filtersPerThread + f];
                    }
                }
            }
            __syncthreads();
        }

        if (scaleFlag) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for (int g = 0; g < imgsPerThread; g++) {
                    if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                        targets[g * B_X + f * numImages * numModules] = scaleTargets * targets[g * B_X + f * numImages * numModules] + scaleOutputs * prod[f][g];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[g * B_X + f * numImages * numModules] = scaleOutputs * prod[f][g];
                    }
                }
            }
        }
    }
    """
    #template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache,
    #          bool scale, bool checkImgBounds>
    #4, 32, 4, 16, 3, 4
    for key, val in template_params.items():
        kernel_code = kernel_code.replace(key, str(val))
    return SourceModule(kernel_code)

