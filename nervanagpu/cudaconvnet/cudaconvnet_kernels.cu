__device__ __forceinline__ void filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(int fPidx, int imgLoadModPosY, int imgLoadModPosX,
                                                                                        int imgSizeX, int filterSize, int& iPidx) {
    int x = imgLoadModPosX + (fPidx) % filterSize;
    int y = imgLoadModPosY + (fPidx) / filterSize;
    iPidx = y >= 0 && y < imgSizeX && x >= 0 && x < imgSizeX ? y * imgSizeX + x : -1;
}

#define FA_COLOR3_IMPRELOAD(c,i) imPreload[c][i] = iPidxNext < 0 || (checkImgBounds && myImgIdx + i * B_X >= numImages) ? 0 : mm[c * imgPixels * imgStride + i * B_X];
#define FA_COLOR3_IMPRELOAD_TX(c,i) imPreload[c][i] = iPidxNext < 0 || (checkImgBounds && myImgIdx + i * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imagesOffset2 + c * imgPixels * imgStride + i * B_X);
#define DIVUP(x, y) (((x) + (y) - 1) / (y))

#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 */
//__launch_bounds__(128,3)
__global__ void filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
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
    // Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
    // in the range 0..31. It appears that this allows the compiler to optimize?
    const int tx = threadIdx.x % B_X;
    const int ty = threadIdx.y % B_Y;
    const int tidx = ty * B_X + threadIdx.x;

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
    float fPreload[numColors][pixelCache*filtersPerThread/B_X];

    #pragma unroll
    for (int c = 0; c < numColors; ++c) {
        #pragma unroll
        for (int p = 0; p < pixelCache; p += B_X/filtersPerThread) {
            if (p + shFilterLoadY < filterPixels) {
                fPreload[c][p*filtersPerThread/B_X] = tex1Dfetch<float>(filters, filtersOffset + p * numFilters + c * numFilters * filterPixels);
            } else{
                fPreload[c][p*filtersPerThread/B_X] = 0;
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
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int c = 0; c < numColors; ++c) {
                // NOTE: bank conflicts here!
                shImages[c][ty][tx * imgsPerThread + i] = imPreload[c][i];
            }
        }

        const int fPidxNext = p + pixelCache >= filterPixels ? 0 : p + pixelCache;
        filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(fPidxNext + ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

//        const float* ff = &filters[numFilters * fPidxNext];
//        const float* mm = &images[imgStride * iPidxNext];
        const int filtersOffset2 = filtersOffset + numFilters * fPidxNext;
        const int imagesOffset2 = imagesOffset + imgStride * iPidxNext;

        FA_COLOR3_IMPRELOAD_TX(0,0);
        FA_COLOR3_IMPRELOAD_TX(0,1);
        FA_COLOR3_IMPRELOAD_TX(0,2);
        FA_COLOR3_IMPRELOAD_TX(0,3);


        #pragma unroll
        for (int c = 0; c < numColors; ++c) {
            #pragma unroll
            for (int pp = 0; pp < pixelCache; pp += B_X/filtersPerThread) {
                shFilters[c][pp + shFilterLoadY][shFilterLoadX] = fPreload[c][pp*filtersPerThread/B_X];
            }
        }

        __syncthreads();
        FA_COLOR3_IMPRELOAD_TX(1,0);
        FA_COLOR3_IMPRELOAD_TX(1,1);
        FA_COLOR3_IMPRELOAD_TX(1,2);
        FA_COLOR3_IMPRELOAD_TX(1,3);
        FA_COLOR3_IMPRELOAD_TX(2,0);
        FA_COLOR3_IMPRELOAD_TX(2,1);
        FA_COLOR3_IMPRELOAD_TX(2,2);
        FA_COLOR3_IMPRELOAD_TX(2,3);
        #pragma unroll
        for (int c = 0; c < numColors; c++) {
            #pragma unroll
            for (int pp = 0; pp < 2; pp++) {
                fPreload[c][pp] = fPidxNext + pp*(B_X/filtersPerThread) + shFilterLoadY >= filterPixels ? 0 : tex1Dfetch<float>(filters, filtersOffset2 + c * numFilters* filterPixels + pp*(B_X/filtersPerThread) * numFilters);
            }
        }
        #pragma unroll
        for (int pp = 0; pp < pixelCache; pp++) {
            #pragma unroll
            for (int c = 0; c < numColors; c++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int i = 0; i < imgsPerThread; i++) {
                        prod[i][f] += shImages[c][pp][tx * imgsPerThread + i] * shFilters[c][pp][ty * filtersPerThread + f];
                    }
                }
            }
        }

        __syncthreads();
    }

    if (scale) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleTargets * targets[i * B_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
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

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * This won't be pretty.
 */
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

    if (scale) {
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

__device__ inline void filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(int filterSize, int imgSizeX, int imgLoadModPosY, int imgLoadModPosX, int imgY, int imgX, int& fPidx, int& iPidx) {
    int filterPxY = imgY - imgLoadModPosY;
    int filterPxX = imgX - imgLoadModPosX;
    fPidx = filterPxY * filterSize + filterPxX;
    iPidx = imgY * imgSizeX + imgX; // Pixel index in img
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * Note: in git there's a 1.5% faster version of this which sues 167 registers instead of 154...
 * it's basically the same thing, but it doesn't do the next-pixel computation. It just avoids
 * pre-loading when it rolls over to the next pixel.
 */
__global__ void filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv/*, const bool noloads*/) {
    __shared__ float shFilters[colorCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[colorCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;
    // Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
    // in the range 0..31. It appears that this allows the compiler to optimize?
    const int tx = threadIdx.x % B_X;
    const int ty = threadIdx.y % B_Y;
    const int tidx = ty * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
            + myImgIdx;

    float prod[imgsPerThread][filtersPerThread];
//    float fCache[filtersPerThread];
    #pragma unroll
    for(int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
            prod[i][f] = 0;
        }
    }
    // NOTE: these max/min functions increase register usage as compared to my macros
    const int imgStartX = max(0, imgLoadModPosX);
    const int imgStartY = max(0, imgLoadModPosY);
    const int imgEndX = min(imgLoadModPosX + filterSize, imgSizeX);
    const int imgEndY = min(imgLoadModPosY + filterSize, imgSizeY);
//    __shared__ int imgPos[]

    int fPidx, iPidx;
    float imPreload[imgsPerThread];
    float fPreload[colorCache*filtersPerThread/B_X];
//    float fCache[filtersPerThread];

    filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgStartY, imgStartX, fPidx, iPidx);

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
            imPreload[i] = images[imgStride * iPidx + i * B_X];
        } else {
            imPreload[i] = 0;
        }
    }
    if (/*B_X % filtersPerThread == 0 ||*/ shFilterLoadY < B_X/filtersPerThread) { // This if statement reduces reg usage..
        #pragma unroll
        for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
            fPreload[c*filtersPerThread/B_X] = filters[(c * filterPixels + fPidx) * numFilters];
        }
    }
    for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
//        const int filterPxY = imgY - imgLoadModPosY;
        for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
//            const int filterPxX = imgX - imgLoadModPosX;
//            const int p = filterPxY * filterSize + filterPxX;
//            const int pixIdx = imgY * imgSizeX + imgX;// Pixel index in img
//            setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgY, imgX, &p, &pixIdx);
//            float* m = &images[imgStride * pixIdx];
            const bool lastPixel = imgY == imgEndY - 1 && imgX == imgEndX - 1;
            int imgYNext = imgY;
            int imgXNext = imgX;
            int fPidxNext, iPidxNext;
            if (!lastPixel) {
                imgYNext = imgY + (imgX + 1 == imgEndX);
                imgXNext = imgX + 1 == imgEndX ? imgStartX : imgX + 1;
            }
            filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgYNext, imgXNext, fPidxNext, iPidxNext);
            for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
                const float* ff = &filters[numFilters * ((oc + colorCache) * filterPixels + fPidx)];
                const float* mm = &images[imgStride * ((oc + colorCache) * imgPixels + iPidx)];
                if (oc == numFilterColors - colorCache) {
                    ff = &filters[fPidxNext * numFilters];
                    mm = &images[iPidxNext * imgStride];
                    fPidx = fPidxNext;
                    iPidx = iPidxNext;
                }

                #pragma unroll
                for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
                    shFilters[c + shFilterLoadY][shFilterLoadX] = fPreload[c*filtersPerThread/B_X];
                }

                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    // NOTE: bank conflicts here!
                    shImages[ty][tx * imgsPerThread + i] = imPreload[i];
                }
                imPreload[0] = (checkImgBounds && myImgIdx + 0 * B_X >= numImages) ? 0 : mm[0 * B_X];
                imPreload[1] = (checkImgBounds && myImgIdx + 1 * B_X >= numImages) ? 0 : mm[1 * B_X];
                imPreload[2] = (checkImgBounds && myImgIdx + 2 * B_X >= numImages) ? 0 : mm[2 * B_X];

                __syncthreads();

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[0][threadIdx.x * imgsPerThread + i] * shFilters[0][threadIdx.y * filtersPerThread + f];
                    }
                }

                fPreload[0] = ff[0];

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[1][threadIdx.x * imgsPerThread + i] * shFilters[1][threadIdx.y * filtersPerThread + f];
                    }
                }

                fPreload[1] = ff[(B_X/filtersPerThread * filterPixels) * numFilters];

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[2][threadIdx.x * imgsPerThread + i] * shFilters[2][threadIdx.y * filtersPerThread + f];
                    }
                }

                imPreload[3] = (checkImgBounds && myImgIdx + 3 * B_X >= numImages) ? 0 : mm[3 * B_X];

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[3][threadIdx.x * imgsPerThread + i] * shFilters[3][threadIdx.y * filtersPerThread + f];
                    }
                }
                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleTargets * targets[i * B_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
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

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 */
__global__ void filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv/*, const bool noloads*/) {
    __shared__ float shFilters[colorCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[colorCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;
    // Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
    // in the range 0..31. It appears that this allows the compiler to optimize?
    const int tx = threadIdx.x % B_X;
    const int ty = threadIdx.y % B_Y;
    const int tidx = ty * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
    const int imgOffset = (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;

//    images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
    const int filterOffset = blockFilterIdx
            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX + (conv ? 0 : moduleIdx * numFilterColors * filterPixels * numFilters);
//    filters +=blockFilterIdx
//            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX;
//    if (!conv) {
//        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
//    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
            + myImgIdx;

    float prod[imgsPerThread][filtersPerThread];
//    float fCache[filtersPerThread];
    #pragma unroll
    for(int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
            prod[i][f] = 0;
        }
    }
    // NOTE: these max/min functions increase register usage as compared to my macros
    const int imgStartX = max(0, imgLoadModPosX);
    const int imgStartY = max(0, imgLoadModPosY);
    const int imgEndX = min(imgLoadModPosX + filterSize, imgSizeX);
    const int imgEndY = min(imgLoadModPosY + filterSize, imgSizeY);
//    __shared__ int imgPos[]

    int fPidx, iPidx;
    float imPreload[imgsPerThread]; // [4]
    float fPreload[colorCache*filtersPerThread/B_X]; // [2]
//    float fCache[filtersPerThread];

    filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgStartY, imgStartX, fPidx, iPidx);

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
            imPreload[i] = tex1Dfetch<float>(images, imgOffset + imgStride * iPidx + i * B_X);
        } else {
            imPreload[i] = 0;
        }
    }
    if (/*B_X % filtersPerThread == 0 ||*/ shFilterLoadY < B_X/filtersPerThread) { // This if statement reduces reg usage..
        #pragma unroll
        for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
            fPreload[c*filtersPerThread/B_X] = tex1Dfetch<float>(filters, filterOffset + (c * filterPixels + fPidx) * numFilters);
        }
    }
    for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
//        const int filterPxY = imgY - imgLoadModPosY;
        for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
//            const int filterPxX = imgX - imgLoadModPosX;
//            const int p = filterPxY * filterSize + filterPxX;
//            const int pixIdx = imgY * imgSizeX + imgX;// Pixel index in img
//            setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgY, imgX, &p, &pixIdx);
//            float* m = &images[imgStride * pixIdx];
            const bool lastPixel = imgY == imgEndY - 1 && imgX == imgEndX - 1;
            int imgYNext = imgY;
            int imgXNext = imgX;
            int fPidxNext, iPidxNext;
            if (!lastPixel) {
                imgYNext = imgY + (imgX + 1 == imgEndX);
                imgXNext = imgX + 1 == imgEndX ? imgStartX : imgX + 1;
            }
            filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgYNext, imgXNext, fPidxNext, iPidxNext);
            for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
//                const float* ff = &filters[numFilters * ((oc + colorCache) * filterPixels + fPidx)];
//                const float* mm = &images[imgStride * ((oc + colorCache) * imgPixels + iPidx)];
                int imgOffset2 = imgOffset + imgStride * ((oc + colorCache) * imgPixels + iPidx);
                int filterOffset2 = filterOffset + numFilters * ((oc + colorCache) * filterPixels + fPidx);
                if (oc == numFilterColors - colorCache) {
                    filterOffset2 = filterOffset + fPidxNext * numFilters;
                    imgOffset2 = imgOffset + iPidxNext * imgStride;
                    fPidx = fPidxNext;
                    iPidx = iPidxNext;
                }

                #pragma unroll
                for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
                    shFilters[c + shFilterLoadY][shFilterLoadX] = fPreload[c*filtersPerThread/B_X];
                }

                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    // NOTE: bank conflicts here!
                    shImages[ty][tx * imgsPerThread + i] = imPreload[i];
                }
                imPreload[0] = (checkImgBounds && myImgIdx + 0 * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imgOffset2 + 0 * B_X);
                imPreload[1] = (checkImgBounds && myImgIdx + 1 * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imgOffset2 + 1 * B_X);
                imPreload[2] = (checkImgBounds && myImgIdx + 2 * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imgOffset2 + 2 * B_X);

                __syncthreads();

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[0][threadIdx.x * imgsPerThread + i] * shFilters[0][threadIdx.y * filtersPerThread + f];
                    }
                }

                fPreload[0] = tex1Dfetch<float>(filters, filterOffset2 + 0);

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[1][threadIdx.x * imgsPerThread + i] * shFilters[1][threadIdx.y * filtersPerThread + f];
                    }
                }

                fPreload[1] = tex1Dfetch<float>(filters, filterOffset2 + (B_X/filtersPerThread * filterPixels) * numFilters);

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[2][threadIdx.x * imgsPerThread + i] * shFilters[2][threadIdx.y * filtersPerThread + f];
                    }
                }

                imPreload[3] = (checkImgBounds && myImgIdx + 3 * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imgOffset2 + 3 * B_X);

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[3][threadIdx.x * imgsPerThread + i] * shFilters[3][threadIdx.y * filtersPerThread + f];
                    }
                }
                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleTargets * targets[i * B_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
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

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of module and B_Y * filtersPerThread
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numColors, filterPixels, numFilters) if conv
 *              (numModules, numColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * 
 * Number of filters per module should be divisible by B_Y * filtersPerThread
 * checkImgBounds indicates whether number of images is divisible by B_X * imgsPerThread
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
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

    if (scale) {
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

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of B_Y * filtersPerThread
 *
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 * colorCache: how many colors to put into shmem
 *
 * numFilters should be divisible by B_Y * filtersPerThread
 * numImages be divisible by B_X * imgsPerThread
 * numFilterColors should be divisible by colorCache.
 * numImgColors must be even.
 * numFilters must be divisible by numGroups.
 * no restrictions on pixelCache
 * The imgSize here is the size of the actual image without the padding.
 * As always, try to make B_X * imgsPerThread == B_Y * filtersPerThread for maximum efficiency.
 *
 */
__global__ void filterActs_YxX_sparse2(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups, 
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv) {
    __shared__ float shFilters[colorCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[colorCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }
    const int imgStartX = MAX(0, imgLoadModPosX);
    const int imgStartY = MAX(0, imgLoadModPosY);
    const int imgEndX = MIN(imgLoadModPosX + filterSize, imgSizeX);
    const int imgEndY = MIN(imgLoadModPosY + filterSize, imgSizeY);
//    __shared__ int imgPos[]

    for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
        const int filterPxY = imgY - imgLoadModPosY;
        for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
            const int filterPxX = imgX - imgLoadModPosX;
            const int p = filterPxY * filterSize + filterPxX;
            for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
                
                /*
                 * Load a pixel from B_Y*filtersPerThread filters
                 * This condition covers the case when B_X is not divisible by filtersPerThread.
                 * In this case, not all of the threads will participate in the loading operation.
                 * This ensures that in each loop iteration, an integer number of rows of shFilters
                 * are filled, which makes indexing simple.
                 
                 * nvcc is behaving in a completely insane way: removing this condition under
                 * 
                 */
                if (/*B_X % filtersPerThread == 0 ||*/ shFilterLoadY < B_X/filtersPerThread) {
                    #pragma unroll
                    for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
                        if (colorCache % (B_X/filtersPerThread) == 0 || c + shFilterLoadY < colorCache) {
                            shFilters[c + shFilterLoadY][shFilterLoadX] = filters[((oc+c) * filterPixels + p) * numFilters];
                        }
                    }
                }
 
                /*
                 * Load a pixel from B_X*imgsPerThread images.
                 */
                const int pixIdx = imgY * imgSizeX + imgX;// Pixel index in img

                float* m = &images[imgStride * (oc * imgPixels + pixIdx)];
                #pragma unroll
                for (int c = 0; c < colorCache; c += B_Y) {
                    if (colorCache % B_Y == 0 || threadIdx.y + c < colorCache) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                                shImages[c + threadIdx.y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
                            } else {
                                shImages[c + threadIdx.y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                }
                
                __syncthreads();

                for (int c = 0; c < colorCache; c++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        #pragma unroll
                        for(int f = 0; f < filtersPerThread; f++) {
                            prod[f][g] += shImages[c][g * B_X + threadIdx.x] * shFilters[c][threadIdx.y + f * B_Y];
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
                }
            }
        }
    }
}