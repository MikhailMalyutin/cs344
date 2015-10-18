/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include <thrust/extrema.h>

__global__ void reduceMinSerial(float * d_reduce_result,
                                const float * d_in, size_t size) {
    float res = d_in[0];
    for (size_t i=1;i<size; ++i) {
         res = min(d_in[i], res);
    }
    *d_reduce_result = res;
}

__global__ void reduceMaxSerial(float * d_reduce_result,
                                const float * d_in, size_t size) {
    float res = d_in[0];
    for (size_t i=1;i<size; ++i) {
        res = max(d_in[i], res);
    }
    *d_reduce_result = res;
}

__global__ void reduceMin(float * d_reduce_result,
                          const float * d_in) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t blockSize = blockDim.x;
    size_t blockId = blockIdx.x;
    size_t myId = tid + blockSize * blockId;
    sdata[tid] = d_in[myId];
    __syncthreads();

    int oldS = blockDim.x;
    for (unsigned int s = blockSize/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = thrust::min(sdata[tid], sdata[tid+s]);
        }
        // if currently used s does not cover all the numbers computed by previous s.
        if (tid == 0 && 2*s < oldS) {
            sdata[tid] = thrust::min(sdata[tid], sdata[tid+2*s]);
        }
        __syncthreads();
        oldS = s;
    }

    if (tid == 0) {
        d_reduce_result[blockId] = sdata[0];
    }
}

__global__ void reduceMax(float * d_reduce_result,
                          const float * d_in) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t blockSize = blockDim.x;
    size_t blockId = blockIdx.x;
    size_t myId = tid + blockSize * blockId;
    sdata[tid] = d_in[myId];
    __syncthreads();

    unsigned int oldS = blockSize;
    for (unsigned int s = blockSize/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = thrust::max(sdata[tid], sdata[tid+s]);
        }
        // if currently used s does not cover all the numbers computed by previous s.
        if (tid == 0 && 2*s < oldS) {
            sdata[tid] = thrust::max(sdata[tid], sdata[tid+2*s]);
        }
        __syncthreads();
        oldS = s;
    }

    if (tid == 0) {
        d_reduce_result[blockId] = sdata[0];
    }
}

__global__ void histogram(const float* const d_in,
                          unsigned int* const d_res,
                          const float lumMin,
                          const float lumRange,
                          const int numBins) {
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + blockDim.x * blockIdx.x;
    unsigned int binId = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((d_in[myId] - lumMin) / lumRange * numBins));
    d_res[myId] = 0;
    __syncthreads();
    atomicAdd(&(d_res[binId]), 1);
}

__global__ void histogramSerial(const float* const d_in,
                          unsigned int* const d_res,
                          const float lumMin,
                          const float lumRange,
                          const int numBins,
                          const int size) {
  for (size_t i = 0; i < numBins; ++i) d_res[i] = 0;

  for (size_t i = 0; i < size; ++i) {
    unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((d_in[i] - lumMin) / lumRange * numBins));
    d_res[bin]++;
  }
}

__global__ void scan(const unsigned int* const d_in,
                          unsigned int* const d_res,
                          const int size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + blockDim.x * blockIdx.x;
    sdata[myId] = d_in[myId];

    for (unsigned int s = 1; s <= size/2; s *= 2) {
        if (myId - s > 0 && myId - s < size) {
            __syncthreads();
            int prev = sdata[myId-s];
            __syncthreads();

            sdata[myId] += prev;
        }

    }
    __syncthreads();
    if (tid == 0) {
        d_res[0] = 0;
    } else {
        d_res[myId] = sdata[myId-1];
    }
}

__global__ void scanSerial(const unsigned int* const d_in,
                          unsigned int* const d_res,
                          const int numBins) {
  d_res[0] = 0;
  for (size_t i = 1; i < numBins; ++i) {
    d_res[i] = d_res[i - 1] + d_in[i - 1];
  }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum

    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    float* d_inter;
    float* d_out;

    int maxThreads = 1024;
    int numBlocks = (numCols*numRows)/maxThreads;

    //for intermediate array
    int arrayBytes = numBlocks*sizeof(float);

    //allocate memory for device
    checkCudaErrors(cudaMalloc((void **) &d_inter, arrayBytes));
    checkCudaErrors(cudaMalloc((void **) &d_out, sizeof(float)));

    //for kernel call
    const dim3 gridSize = numBlocks;
    const dim3 blockSize = maxThreads;

    //call kernel for minimum
    reduceMin<<<gridSize, blockSize, maxThreads*sizeof(float)>>>(d_inter, d_logLuminance);
    reduceMin<<<1, gridSize, numBlocks*sizeof(float)>>>(d_out, d_inter);
    //reduceMinSerial<<<1, 1>>>(d_out, d_logLuminance, numCols*numRows);
    checkCudaErrors(cudaMemcpy(&min_logLum, d_out, 1 * sizeof(float), cudaMemcpyDeviceToHost));

    reduceMax<<<gridSize, blockSize, maxThreads*sizeof(float)>>>(d_inter, d_logLuminance);
    reduceMax<<<1, gridSize, numBlocks*sizeof(float)>>>(d_out, d_inter);
    //reduceMaxSerial<<<1, 1>>>(d_out, d_logLuminance, numCols*numRows);
    checkCudaErrors(cudaMemcpy(&max_logLum, d_out, 1 * sizeof(float), cudaMemcpyDeviceToHost));

    //2) subtract them to find the range
    const float range = max_logLum - min_logLum;

    unsigned int* d_histo;
    checkCudaErrors(cudaMalloc((void **) &d_histo, numBins*sizeof(float)));

    histogram<<<gridSize, blockSize>>>(d_logLuminance,
                          d_histo,
                          min_logLum,
                          range,
                          numBins);
    //histogramSerial<<<1, 1>>>(d_logLuminance,
    //                      d_histo,
    //                      min_logLum,
    //                      range,
    //                      numBins,
    //                     numCols*numRows);

    scan<<<1, numBins, numBins*sizeof(int)>>>(d_histo,
                          d_cdf,
                          numBins);
    //scanSerial<<<1, 1>>>(d_histo, d_cdf, numBins);


  //next perform the actual tone-mapping
  //we map each luminance value to its new value
  //and then transform back to RGB space

}