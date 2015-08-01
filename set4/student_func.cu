//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
const unsigned int maxThreads = 1024;
const unsigned int numBits = 1;
const unsigned int numBins = 1 << numBits;

void displayCudaBufferWindow(unsigned int* const d_buf, const size_t numElems, const size_t from, const size_t to) {
  unsigned int *buf = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(buf,  d_buf,  sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  for (int i=from ; i< to; ++i) {
      std::cout << std::hex << buf[i] << " " << std::endl;
  }
  std::cout << std::endl;

  delete[] buf;
}

void displayCudaBuffer(unsigned int* const d_buf, const size_t numElems) {
  displayCudaBufferWindow(d_buf, numElems, 0, numElems);
}


unsigned int displayCudaBufferMax(unsigned int* const d_buf, const size_t numElems) {
  unsigned int *buf = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(buf,  d_buf,  sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  unsigned int max = buf[0];
  unsigned int idx = 0;
  for (int i=0 ; i< numElems; ++i) {
      if (max < buf[i]) {
          max = buf[i];
          idx = i;
      }
  }
  std::cout << "max " << std::dec << max << " idx " << idx << std::endl;
  int begin = idx - 10;
  if (begin < 0) begin = 0;

  int lastIndex = idx + 10;
  if (lastIndex > numElems) {
      lastIndex = idx + 1;
  }

  displayCudaBufferWindow(d_buf, numElems, begin, lastIndex);
  if (numElems > 10) {
      std::cout << "last " << std::endl;
      displayCudaBufferWindow(d_buf, numElems, numElems - 10, numElems);
  }

  delete[] buf;
  return max;
}

__global__ void histogram(const unsigned int* const d_in,
                          unsigned int* const d_res,
                          const unsigned int mask,
                          const unsigned int i,const size_t numElems, const unsigned int numBins) {
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + blockDim.x * blockIdx.x;
    if (myId < numBins) { //������� ������ ����������
       d_res[myId] = 0;
    }
    if (myId >=numElems) {
        return;
    }
    unsigned int binId = (d_in[myId] & mask) >> i;
    __syncthreads();
    atomicAdd(&(d_res[binId]), 1);
}

__device__ void scanReduceForBlock(unsigned int* const d_res,
                          const size_t size, unsigned int myId) {
    unsigned int prevId;
    unsigned int prevValue;
    unsigned int myValue;

    for (unsigned int s = 2; s <= size; s *= 2) {
        __syncthreads();
        prevId    = myId - s/2;
        prevValue = (myId >= s/2) ? d_res[prevId] : 0;
        myValue   = d_res[myId];
        __syncthreads();
        if (((myId+1) % s) == 0 && (myId >= s/2)) {
            d_res[myId] = myValue + prevValue;
        }
    }

}

__device__  void scanDownStepForBlock(unsigned int* const d_res,
                          const unsigned int initialS, unsigned int myId1) {
    unsigned int prevId;
    unsigned int prevValue;
    unsigned int myValue;

    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + (blockDim.x) * blockIdx.x;
    if (myId >=initialS || tid != myId) {
        return;
    }

    for (unsigned int s = initialS; s >= 2; s /= 2) {
        __syncthreads();
        prevId = myId - s / 2;
        prevValue = (myId >= s/2) ? d_res[prevId] : 0;
        myValue = d_res[myId];
        __syncthreads();
        if (((myId+1) % s)  == 0 && myId >= s/2) {
            d_res[prevId] = myValue;
            d_res[myId] = myValue + prevValue;
        }
    }

}

__device__  void scanDownStepDevice(unsigned int* const d_res,
                          const unsigned int initialS, unsigned int myId) {
    unsigned int prevId;
    unsigned int prevValue;
    unsigned int myValue;

    for (unsigned int s = initialS; s >= 2; s /= 2) {
        __syncthreads();
        prevId = myId - s / 2;
        prevValue = (myId >= s/2) ? d_res[prevId] : 0;
        myValue = d_res[myId];
        __syncthreads();
        if (((myId+1) % s)  == 0 && myId >= s/2) {
            d_res[prevId] = myValue;
            d_res[myId] = myValue + prevValue;
        }
    }

}


__device__ void scanReduce(const unsigned int* const d_in,
                          unsigned int* const d_res,
                          const size_t size, unsigned int myId) {
    d_res[myId] = d_in[myId];

    unsigned int prevId;
    unsigned int prevValue;
    unsigned int myValue;

    scanReduceForBlock(d_res, maxThreads, myId);

    for (unsigned int s = maxThreads * 2; s <= size; s *= 2) {
        if (s > maxThreads) {
        }
        __syncthreads();
        prevId    = myId - s/2;
        prevValue = (myId >= s/2) ? d_res[prevId] : 0;
        myValue   = d_res[myId];
        __syncthreads();
        if (((myId+1) % s) == 0 && (myId >= s/2)) {
            d_res[myId] = myValue + prevValue;
        }
    }

}

__device__  void scanDownStep(unsigned int* const d_res,
                          const size_t size, unsigned int myId) {
    d_res[size-1] = 0;

    unsigned int prevId;
    unsigned int prevValue;
    unsigned int myValue;

    for (unsigned int s = size; s >= 2; s /= 2) {
        __syncthreads();
        prevId = myId - s / 2;
        prevValue = (myId >= s/2) ? d_res[prevId] : 0;
        myValue = d_res[myId];
        __syncthreads();
        if (((myId+1) % s)  == 0 && myId >= s/2) {
            d_res[prevId] = myValue;
            d_res[myId] = myValue + prevValue;
        }
    }

}

__device__ unsigned int myMin(const unsigned int a, const unsigned int b) {
    if (a < b) return a;
    return b;
}

__global__  void blellochScan(const unsigned int* const d_in,
                          unsigned int* const d_res,
                          const size_t size) {
    extern __shared__ unsigned int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + (blockDim.x) * blockIdx.x;
    if (myId >=size) {
        return;
    }
    d_res[myId] = d_in[myId];
    scanReduceForBlock(d_res, myMin(maxThreads, size), myId);
    d_res[size-1] = 0;
    __syncthreads();

    unsigned int ssize = size / maxThreads; //������� ��������� ����� � ����������� �������
    if (ssize > 1) {
        unsigned int interval = size/ ssize;
        if (myId == tid && myId < ssize) { //��������� ������ ������ ������ �����
            sdata[myId] = d_res[myId * interval + interval - 1];
            scanReduceForBlock(sdata, ssize, myId);
            __syncthreads();
            //scanReduceForBlock(sdata, maxThreads/2, compactedMyId);
            __syncthreads();
            sdata[ssize-1] = 0;
            __syncthreads();
            scanDownStepForBlock(sdata, ssize, myId);
            d_res[myId * interval + interval - 1] = sdata[myId];
        }
    }
}

__global__  void blellochScanDownstep(const unsigned int* const d_in,
                          unsigned int* const d_res,
                          const size_t size) {
    extern __shared__ unsigned int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + (blockDim.x) * blockIdx.x;
    if (myId >=size) {
        return;
    }
    unsigned int initialS = myMin(maxThreads, size);
    scanDownStepDevice(d_res, initialS, myId);
}


/**
d_binScan - ��� ������� �������� �������,
�������� ��������, ���� ����� �������� ���������, � ������ ���� �� ����� � ��� �������
d_vals_dst - �������� �������� ��� ������� id ��� ����������� �������
d_vals_dst ����� ����� ��������� ���������
**/
__global__ void gather(const unsigned int* const d_vals_src,
                       const unsigned int* const d_pos_src,
                       const unsigned int* const d_new_index_src,
                       unsigned int* const d_vals_dst,
                       unsigned int* const d_pos_dst,
                       const unsigned int numElems) {
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + (blockDim.x) * blockIdx.x;

    if (myId >= numElems) {
        return;
    }
    __syncthreads();
    unsigned int newIndex = d_new_index_src[myId];
    __syncthreads();
    d_vals_dst[newIndex] = d_vals_src[myId];
    d_pos_dst[newIndex]  = d_pos_src[myId];
}

/**
d_binScan - ��� ������� �������� �������,
�������� ��������, ���� ����� �������� ���������, � ������ ���� �� ����� � ��� �������
d_disp_src - �������� �������� ��� ������� id ��� ����������� �������
**/
__global__ void getNewIndexes(const unsigned int* const d_vals_src,
                       const unsigned int* const d_disp_src,
                       const unsigned int* const d_binScan,
                       unsigned int* const d_new_index_dst,
                       const unsigned int mask,
                       const unsigned int i,
                       const unsigned int numElems) {
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + (blockDim.x) * blockIdx.x;

    if (myId >= numElems) {
        return;
    }
    unsigned int myIdOffset = d_disp_src[myId]; //� ��� ���� ��, ��� ������ �� ���������� ��������,
    //������ � ��������� ����������
    unsigned int binId = (d_vals_src[myId] & mask) >> i;
    __syncthreads();
    unsigned int offset = d_binScan[binId];
    unsigned int newIndex = offset + myIdOffset;
    d_new_index_dst[myId] = newIndex;
}

__global__ void mapToBin(const unsigned int* const d_vals_src,
                       unsigned int* const d_vals_dst,
                       const unsigned int mask,
                       const unsigned int i,
                       const unsigned int mappedBean,
                       const unsigned int numElems) {
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + (blockDim.x) * blockIdx.x;

    if (myId >= numElems) {
        return;
    }
    unsigned int beanId = (d_vals_src[myId] & mask) >> i;
    d_vals_dst[myId] = (beanId == mappedBean) ? 1 : 0;
}

__global__ void resetMapToBin(const unsigned int* const d_vals_src,
                       unsigned int* const d_vals_dst,
                       const unsigned int mask,
                       const unsigned int i,
                       const unsigned int mappedBean,
                       const unsigned int numElems) {
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + (blockDim.x) * blockIdx.x;

    if (myId >= numElems) {
        return;
    }
    int beanId = (d_vals_src[myId] & mask) >> i;
    if (beanId != mappedBean) {
        d_vals_dst[myId] = 0;
    }
}

__global__ void clear(unsigned int* const d_vals_dst,
                       const unsigned int numElems) {
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + (blockDim.x) * blockIdx.x;

    if (myId >= numElems) {
        return;
    }
    d_vals_dst[myId] = 0;
}

__global__ void copy(const unsigned int* const d_src,
                     unsigned int* const d_dst,
                     const unsigned int numElems) {
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + blockDim.x * blockIdx.x;
    if (myId >= numElems) {
        return;
    }
    d_dst[myId] = d_src[myId];
}

__global__ void sum(const unsigned int* const d_src,
                     unsigned int* const d_dst,
                     const unsigned int numElems) {
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + blockDim.x * blockIdx.x;
    if (myId >= numElems) {
        return;
    }
    d_dst[myId] = d_src[myId] + d_dst[myId];
}

unsigned int getNearest(unsigned int const number) {
    unsigned int result = 1;
    while( result < number ) {
        result <<= 1;
    }
    return result;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               size_t numElems)
{
  unsigned int* d_binScan;
  unsigned int* d_binHistogram;
  unsigned int* d_temp;
  unsigned int* d_temp1;
  unsigned int* d_iv = d_inputVals;
  unsigned int* d_ip = d_inputPos;
  unsigned int* d_ov = d_outputVals;
  unsigned int* d_op = d_outputPos;

  numElems = 15000;//32;//16;//18000;
  int elemstoDisplay = 16;

  int alignedBuferElems = getNearest(numElems);

  checkCudaErrors(cudaMalloc((void **) &d_binScan, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_binHistogram, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_temp, alignedBuferElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_temp1, alignedBuferElems * sizeof(unsigned int)));

  std::cout << "numElems " << numElems << std::endl;
  std::cout << "numBins " << numBins << std::endl;

  std::cout << "d_inputVals " << std::endl;
  displayCudaBuffer(d_inputVals, elemstoDisplay);

  //a simple radix sort - only guaranteed to work for numBits that are multiples of 2
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
      unsigned int mask = (numBins - 1) << i;

      clear<<<(numElems+maxThreads-1)/maxThreads, maxThreads>>>(d_ov,numElems);

      for (unsigned int j = 0; j < numBins; ++j) {
          //checkCudaErrors(cudaMemset(d_temp, 0,  sizeof(unsigned int) * alignedBuferElems));
          clear<<<(alignedBuferElems+maxThreads-1)/maxThreads, maxThreads>>>(d_temp,alignedBuferElems);
          clear<<<(alignedBuferElems+maxThreads-1)/maxThreads, maxThreads>>>(d_temp1,alignedBuferElems);
          std::cout << "after clear" << std::endl;
          displayCudaBufferMax(d_temp, alignedBuferElems);

          mapToBin<<<(numElems+maxThreads-1)/maxThreads, maxThreads>>>(d_iv,d_temp,mask,i,j,numElems);
          std::cout << "mapToBin" << j << " " <<  mask << " " << i << std::endl;
          displayCudaBuffer(d_temp, elemstoDisplay);
          //std::cout << "DEEP " << std::endl;
          //displayCudaBufferWindow(d_temp, numElems, 2000, 2010);
          displayCudaBufferMax(d_temp, alignedBuferElems);

          blellochScan<<<(alignedBuferElems+maxThreads-1)/maxThreads, maxThreads, maxThreads * sizeof(unsigned int)>>>(d_temp, d_temp1, alignedBuferElems);
          blellochScanDownstep<<<(alignedBuferElems+maxThreads-1)/maxThreads, maxThreads, maxThreads * sizeof(unsigned int)>>>(d_temp, d_temp1, alignedBuferElems);

          std::cout << "scan " << std::endl;
          displayCudaBuffer(d_temp1, elemstoDisplay);
          unsigned int max = displayCudaBufferMax(d_temp1, numElems);
          if (max > numElems) {
              std::cout << "ERROR " << std::endl;
              displayCudaBufferWindow(d_temp, numElems,  6391, 6403);
              displayCudaBufferWindow(d_temp1, numElems, 6391, 6403);

          }

          resetMapToBin<<<(numElems+maxThreads-1)/maxThreads, maxThreads>>>(d_iv,d_temp1,mask,i,j,numElems);
          std::cout << "resetMapToBin " << std::endl;
          displayCudaBuffer(d_temp1, elemstoDisplay);
          displayCudaBufferMax(d_temp1, numElems);

          sum<<<(numElems+maxThreads-1)/maxThreads, maxThreads>>>(d_temp1,d_ov,numElems);
          std::cout << "sum " << std::endl;
          displayCudaBufferMax(d_ov, numElems);
          displayCudaBuffer(d_ov, elemstoDisplay);
      }

      histogram<<<(numElems+maxThreads-1)/maxThreads, maxThreads>>>(d_iv, d_binHistogram, mask, i, numElems, numBins);
      //histogram<<<1, numElems>>>(d_iv, d_binHistogram, mask, i, numElems, numBins);
      std::cout << "d_binHistogram " << std::endl;
      displayCudaBuffer(d_binHistogram, numBins);

      //perform exclusive prefix sum (scan) on binHistogram to get starting
      //location for each bin
      //scan<<<1, numBins, numBins*sizeof(unsigned int)>>>(d_binHistogram, d_binScan, numBins);
      //scan<<<1, numBins, numBins*sizeof(unsigned int)>>>(d_binHistogram, d_binHistogram, numBins);
      blellochScan<<<1, numBins, numBins * sizeof(unsigned int)>>>(d_binHistogram, d_binScan, numBins);
      blellochScanDownstep<<<1, numBins, numBins * sizeof(unsigned int)>>>(d_binScan, d_binScan, numBins);
      std::cout << "d_binScan " << std::endl;
      displayCudaBuffer(d_binScan, numBins);

      //Gather everything into the correct location
      //need to move vals and positions
      unsigned int* d_disp_src = d_ov;
      unsigned int* d_new_index = d_op;
      displayCudaBufferMax(d_disp_src, numElems);
      getNewIndexes<<<(numElems+maxThreads-1)/maxThreads, maxThreads>>>(d_iv, d_disp_src, d_binScan, d_new_index, mask, i, numElems);
      std::cout << "after getNewIndexes " << std::endl;
      displayCudaBuffer(d_new_index, elemstoDisplay);
      gather<<<(numElems+maxThreads-1)/maxThreads, maxThreads>>>(d_iv, d_ip, d_new_index, d_ov, d_op, numElems);
      //gather<<<1, numElems>>>(d_iv, d_ip, d_ov, d_op, d_binScan, mask, i, numElems);
      std::cout << "after gather " << std::endl;
      displayCudaBuffer(d_ov, elemstoDisplay);

      //swap the buffers (pointers only)
      std::swap(d_ov, d_iv);
      std::swap(d_op, d_ip);
  }

  //we did an even number of iterations, need to copy from input buffer into output
  copy<<<(numElems+maxThreads-1)/maxThreads, maxThreads>>>(d_iv, d_ov, numElems);
  copy<<<(numElems+maxThreads-1)/maxThreads, maxThreads>>>(d_ip, d_op, numElems);
  //copy<<<1, numElems>>>(d_iv, d_ov, numElems);
  //copy<<<1, numElems>>>(d_ip, d_op, numElems);

  std::cout << "d_outputVals " << std::endl;
  displayCudaBuffer(d_outputVals, elemstoDisplay);
  displayCudaBufferMax(d_outputVals, numElems);
  std::cout << "d_inputVals " << std::endl;
  displayCudaBuffer(d_inputVals, elemstoDisplay);
  displayCudaBufferMax(d_inputVals, numElems);

  checkCudaErrors(cudaFree(d_binScan));
  checkCudaErrors(cudaFree(d_binHistogram));
  checkCudaErrors(cudaFree(d_temp));
  checkCudaErrors(cudaFree(d_temp1));
}



