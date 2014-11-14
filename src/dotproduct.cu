/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
* \file kernels.cu
*  This file contains the kernel implementation that will be used to acelerate the algorithm PAHOFCUDA
*
* @author Fernando Cobo Aguilera
* @date October 2014
*/


#include "dotproduct.cuh"

#define IMUL(a, b) (int)a*b
#define ACCUM_N 1024


/**
* This kernel executes several dot products. Due to the fact that d_A is
* bigger than d_B, that is to say, size(d_B*n) = size(d_A),
* every subdivision of d_A is multiplied with d_B
* \param d_A First matrix 
* \param d_B Second matrix
* \param d_C Matrix result
*/
__global__ void DotProductKernel(cv::gpu::PtrStepSz<unsigned char> d_a,cv::gpu::PtrStepSz<unsigned char> d_b,cv::gpu::PtrStepSz<short int> d_c){

    //Accumulators cache
    __shared__ int accumResult[ACCUM_N];


    ////////////////////////////////////////////////////////////////////////////
    // Cycle through every pair of vectors,
    // taking into account that vector counts can be different
    // from total number of thread blocks
    ////////////////////////////////////////////////////////////////////////////
    for (int vec = blockIdx.x; vec < VECTOR_N; vec += gridDim.x)
    {
        int vectorBase = IMUL(ELEMENT_N, vec);
        int vectorEnd  = vectorBase + ELEMENT_N;

        ////////////////////////////////////////////////////////////////////////
        // Each accumulator cycles through vectors with
        // stride equal to number of total number of accumulators ACCUM_N
        // At this stage ACCUM_N is only preferred be a multiple of warp size
        // to meet memory coalescing alignment constraints.
        ////////////////////////////////////////////////////////////////////////
        for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x)
        {
            int sum = 0;
            for (int pos = vectorBase + iAccum; pos < vectorEnd; pos += ACCUM_N){
				sum += d_a.ptr(0)[pos] * d_b.ptr((int)(pos % ELEMENT_N)/ SIZE_ROW)[pos % ELEMENT_N % SIZE_ROW];
			}

            accumResult[iAccum] = sum;
        }

        ////////////////////////////////////////////////////////////////////////
        // Perform tree-like reduction of accumulators' results.
        // ACCUM_N has to be power of two at this stage
        ////////////////////////////////////////////////////////////////////////
        for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();

            for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x)
                accumResult[iAccum] += accumResult[stride + iAccum];
        }

        if (threadIdx.x == 0) d_c[vec] = accumResult[0];
    }
}


void DotProduct(cv::gpu::GpuMat d_matrix_a,cv::gpu::GpuMat d_matrix_b, cudaStream_t stream,cv::gpu::GpuMat d_dot_result){

	DotProductKernel<<<128,256,0,stream>>>(d_matrix_a, d_matrix_b, d_dot_result);
}


