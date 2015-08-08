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
const unsigned int MAX_THREADS = 1024;
const unsigned int NUM_BITS    = 2;
const unsigned int NUM_BINS    = 1 << NUM_BITS;

//HELPERS----------------------------------------------------------------------

__device__ unsigned int getMyId() {
    unsigned int tid  = threadIdx.x;
    return tid + blockDim.x * blockIdx.x;
}

void displayCudaBufferWindow(const unsigned int* const d_buf,
                             const size_t              numElems,
                             const size_t              from,
                             const size_t              to) {
    unsigned int *buf = new unsigned int[numElems];
    checkCudaErrors(cudaMemcpy(buf,  d_buf,  sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
    for (int i = from ; i < to; ++i) {
        std::cout << std::hex << buf[i] << " " << std::endl;
    }
    std::cout << std::endl;

    delete[] buf;
}

void displayReducedArray(const unsigned int* const d_buf,
                         const size_t              size) {
    int ssize = size / MAX_THREADS;
    unsigned int *buf = new unsigned int[size];
    checkCudaErrors(cudaMemcpy(buf,  d_buf,  sizeof(unsigned int) * size, cudaMemcpyDeviceToHost));
    if (ssize > 1) {
        int interval = size / ssize;

        std::cout << std::hex << "REDUCED" << std::endl;
        for (int myId = 0; myId < ssize; ++myId) {
            std::cout << std::dec << buf[myId * interval + interval - 1] << " " << std::endl;
        }
    }
}

void displayCudaBuffer(const unsigned int* const d_buf,
                       const size_t              numElems) {
  displayCudaBufferWindow(d_buf, numElems, 0, numElems);
}

void displayCheckSum(const unsigned int* const d_buf,
                             const size_t              numElems) {
  unsigned int *buf = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(buf,  d_buf,  sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  int checksum = 0;
  for (int i = 0 ; i < numElems; ++i) {
      checksum += buf[i];
  }
  std::cout << "checksum " << std::dec << checksum << std::endl;

  delete[] buf;
}

unsigned int displayCudaBufferMax(const unsigned int* const d_buf,
                                  const size_t              numElems) {
  unsigned int *buf = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(buf,  d_buf,  sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  unsigned int max = buf[0];
  unsigned int idx = 0;
  for (int i = 0 ; i < numElems; ++i) {
      if (max <= buf[i]) {
          max = buf[i];
          idx = i;
      }
  }
  displayCheckSum(d_buf, numElems);
  std::cout << "max " << std::dec << max << " idx " << idx << std::endl;
  int begin = idx - 10;
  if (begin < 0) begin = 0;

  int lastIndex = idx + 10;
  if (lastIndex > numElems) {
      lastIndex = idx + 1;
  }

  displayCudaBufferWindow(d_buf, numElems, begin, lastIndex);
  std::cout << "last " << std::endl;
  if (numElems > 50) {
      displayCudaBufferWindow(d_buf, numElems, numElems - 50, numElems);
  } else {
      displayCudaBufferWindow(d_buf, numElems, 0,             numElems);
  }

  delete[] buf;
  return max;
}

__device__ unsigned int myMin(const unsigned int a,
                              const unsigned int b) {
    if (a < b) return a;
    return b;
}

__global__ void fill10 (      unsigned int* const d_dst,
                        const unsigned int numElems) {
    const unsigned int myId = getMyId();
    if (myId >= numElems) {
        return;
    }
    d_dst[myId] = numElems - myId;
}

//ALGORITHMS-------------------------------------------------------------------

__global__ void histogram(const unsigned int* const d_in,
                                unsigned int* const d_res,
                          const unsigned int        mask,
                          const unsigned int        i,
                          const size_t              numElems) {
    const unsigned int myId = getMyId();
    if (myId < NUM_BINS) { //очистка буфера результата
       d_res[myId] = 0;
    }
    if (myId >= numElems) {
        return;
    }
    unsigned int binId = (d_in[myId] & mask) >> i;

    atomicAdd(&(d_res[binId]), 1);
}

__device__ void scanReduceForBlock(      unsigned int* const d_res,
                                   const size_t              maxDisplacement,
                                   const unsigned int        size,
                                   const unsigned int        myId) {
    unsigned int nextId;
    unsigned int prevValue;
    unsigned int nextValue;

    for (unsigned int s = 1; s <= maxDisplacement / 2; s *= 2) {

        __syncthreads();
        prevValue = d_res[myId];
        nextId    = myId + s;
        nextValue = nextId < size ? d_res[nextId] : 0;

        __syncthreads();
        if (((nextId + 1) % (s * 2)) == 0 && (nextId < size)) {
            d_res[nextId] = prevValue + nextValue;
        }
    }
}

__device__  void scanDownStepDevice(      unsigned int* const d_res,
                                    const unsigned int        initialS,
                                    const unsigned int        myId) {
    int prevId;
    unsigned int prevValue;
    unsigned int myValue;

    for (unsigned int s = initialS; s >= 2; s /= 2) {

        __syncthreads();
        prevId    = myId - s / 2;
        prevValue = (prevId >= 0) ? d_res[prevId] : 0;
        myValue   =                 d_res[myId];

        __syncthreads();
        if (((myId + 1) % s) == 0 && prevId >= 0) {
            d_res[prevId] = myValue;
            d_res[myId]   = myValue + prevValue;
        }
    }
}

__global__  void compact(const unsigned int* const d_in,
                               unsigned int* const d_res,
                         const size_t              size,
                         const int                 ssize) {
    const unsigned int myId = getMyId();
    if (myId >=size) {
        return;
    }
    unsigned int interval = size / ssize;
    const unsigned int reducedId = myId / interval;
    int myCurrentIndex = reducedId * interval + interval - 1;
    if (myId > 0 && myId % myCurrentIndex == 0) { //исполняем только внутри одного блока
        d_res[reducedId] = d_in[myCurrentIndex];
    }
}

__global__  void enlarge(const unsigned int* const d_in,
                               unsigned int* const d_res,
                         const size_t              size,
                         const int                 ssize) {
    const unsigned int myId = getMyId();
    if (myId >=size) {
        return;
    }

    unsigned int interval = size/ ssize;
    const unsigned int reducedId = myId / interval;
    int myCurrentIndex = reducedId * interval + interval - 1;
    if (myId > 0 && myId % myCurrentIndex == 0) { //исполняем только внутри одного блока
        d_res[myCurrentIndex] = d_in[reducedId];
    }
}

__global__  void blellochBlockScan(const unsigned int* const d_in,
                                         unsigned int* const d_res,
                                   const size_t              size) {
    const unsigned int myId = getMyId();
    if (myId >=size) {
        return;
    }
    d_res[myId] = d_in[myId];
    scanReduceForBlock(d_res, size, size, myId);

    d_res[size-1] = 0;

    __syncthreads();
     scanDownStepDevice(d_res, size, myId);
}

__global__  void blellochBigScan(const unsigned int* const d_in,
                                       unsigned int* const d_res,
                                 const size_t              size) {
    extern __shared__ unsigned int sdata[];
    const unsigned int myId = getMyId();
    if (myId >=size) {
        return;
    }
    d_res[myId] = d_in[myId];
    scanReduceForBlock(d_res, myMin(MAX_THREADS, size), size, myId);
    d_res[size - 1] = 0;
}

__global__  void blellochBigScanDownstep(      unsigned int* const d_res,
                                         const size_t              size) {
    const unsigned int myId = getMyId();
    if (myId >= size) {
        return;
    }
    unsigned int initialS = myMin(MAX_THREADS, size);
    scanDownStepDevice(d_res, initialS, myId);
}


/**
d_binScan - для каждого элемента корзины,
содержит смещение, куда нужно положить результат, в случае если он попал в эту корзину
d_vals_dst - содержит смещение для данного id для конкретногй корзины
d_vals_dst также будет содержать результат
**/
__global__ void gather(const unsigned int* const d_vals_src,
                       const unsigned int* const d_pos_src,
                       const unsigned int* const d_new_index_src,
                             unsigned int* const d_vals_dst,
                             unsigned int* const d_pos_dst,
                       const unsigned int        numElems) {
    const unsigned int myId = getMyId();

    if (myId >= numElems) {
        return;
    }

    unsigned int newIndex = d_new_index_src[myId];

    d_vals_dst[newIndex] = d_vals_src[myId];
    d_pos_dst[newIndex]  = d_pos_src[myId];
}

/**
d_binScan - для каждого элемента корзины,
содержит смещение, куда нужно положить результат, в случае если он попал в эту корзину
d_disp_src - содержит смещение для данного id для конкретногй корзины
**/
__global__ void getNewIndexes(const unsigned int* const d_vals_src,
                              const unsigned int* const d_disp_src,
                              const unsigned int* const d_binScan,
                                    unsigned int* const d_new_index_dst,
                              const unsigned int        mask,
                              const unsigned int        i,
                              const unsigned int        numElems) {
    const unsigned int myId = getMyId();

    if (myId >= numElems) {
        return;
    }
    unsigned int myIdOffset = d_disp_src[myId]; //у нас свой ид, для нашего ид определяем смещение,
    //кладем в локальную переменную
    unsigned int binId = (d_vals_src[myId] & mask) >> i;

    unsigned int offset   = d_binScan[binId];
    unsigned int newIndex = offset + myIdOffset;
    d_new_index_dst[myId] = newIndex;
}

__global__ void mapToBin(const unsigned int* const d_vals_src,
                               unsigned int* const d_vals_dst,
                         const unsigned int        mask,
                         const unsigned int        i,
                         const unsigned int        mappedBean,
                         const unsigned int        numElems) {
    unsigned int tid  = threadIdx.x;
    unsigned int myId = tid + (blockDim.x) * blockIdx.x;

    if (myId >= numElems) {
        return;
    }
    unsigned int beanId = (d_vals_src[myId] & mask) >> i;
    d_vals_dst[myId] = (beanId == mappedBean) ? 1 : 0;
}

__global__ void resetMapToBin(const unsigned int* const d_vals_src,
                                    unsigned int* const d_vals_dst,
                              const unsigned int        mask,
                              const unsigned int        i,
                              const unsigned int        mappedBean,
                              const unsigned int        numElems) {
    const unsigned int myId = getMyId();

    if (myId >= numElems) {
        return;
    }
    int beanId = (d_vals_src[myId] & mask) >> i;
    if (beanId != mappedBean) {
        d_vals_dst[myId] = 0;
    }
}

__global__ void clear(      unsigned int* const d_vals_dst,
                      const unsigned int        numElems) {
    const unsigned int myId = getMyId();

    if (myId >= numElems) {
        return;
    }
    d_vals_dst[myId] = 0;
}

__global__ void copy(const unsigned int* const d_src,
                           unsigned int* const d_dst,
                     const unsigned int        numElems) {
    const unsigned int myId = getMyId();
    if (myId >= numElems) {
        return;
    }
    d_dst[myId] = d_src[myId];
}

__global__ void sum(const unsigned int* const d_src,
                          unsigned int* const d_dst,
                    const unsigned int numElems) {
    const unsigned int myId = getMyId();
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

//MAIN--------------------------------------------------------------------

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               size_t              numElems)
{
  unsigned int* d_binScan;
  unsigned int* d_binHistogram;
  unsigned int* d_temp;
  unsigned int* d_temp1;
  unsigned int* d_iv = d_inputVals;
  unsigned int* d_ip = d_inputPos;
  unsigned int* d_ov = d_outputVals;
  unsigned int* d_op = d_outputPos;
  unsigned int* sdata;

  //numElems = 16383;//32;//16;//18000;
  int elemstoDisplay = 16;

  int alignedBuferElems = getNearest(numElems);
  int ssize             = alignedBuferElems / MAX_THREADS; //сколько элементов будет в прореженном массиве

  checkCudaErrors(cudaMalloc((void **) &d_binScan,      NUM_BINS          * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_binHistogram, NUM_BINS          * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_temp,         alignedBuferElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_temp1,        alignedBuferElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &sdata,          ssize             * sizeof(unsigned int)));

  const unsigned int numBlocksForAligned  = (alignedBuferElems + MAX_THREADS - 1) / MAX_THREADS;
  const unsigned int numBlocksForElements = (numElems          + MAX_THREADS - 1) / MAX_THREADS;

  //fill10 <<<numBlocksForElements, MAX_THREADS>>> (d_iv, numElems);

  //std::cout << "numElems " << numElems << std::endl;
  //std::cout << "NUM_BINS " << NUM_BINS << std::endl;

  //std::cout << "d_inputVals " << std::endl;
  //displayCudaBuffer(d_inputVals, elemstoDisplay);

  //a simple radix sort - only guaranteed to work for NUM_BITS that are multiples of 2
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += NUM_BITS) {
      unsigned int mask = (NUM_BINS - 1) << i;

      clear <<<numBlocksForElements, MAX_THREADS>>> (d_ov,numElems);

      for (unsigned int j = 0; j < NUM_BINS; ++j) {
          //checkCudaErrors(cudaMemset(d_temp, 0,  sizeof(unsigned int) * alignedBuferElems));
          clear <<<numBlocksForAligned, MAX_THREADS>>> (d_temp,  alignedBuferElems);
          clear <<<numBlocksForAligned, MAX_THREADS>>> (d_temp1, alignedBuferElems);

          mapToBin <<<numBlocksForElements, MAX_THREADS>>> (d_iv, d_temp, mask, i, j, numElems);
          //std::cout << "mapToBin" << j << " " <<  mask << " " << i << std::endl;
          //displayCudaBuffer(d_temp, elemstoDisplay);
          //std::cout << "DEEP " << std::endl;
          //displayCudaBufferWindow(d_temp, numElems, 2000, 2010);
          //displayCudaBufferMax(d_temp, alignedBuferElems);

          blellochBigScan <<<numBlocksForAligned, MAX_THREADS>>>
                       (d_temp, d_temp1, alignedBuferElems);
          if (ssize > 1) {
              compact <<<numBlocksForAligned, MAX_THREADS>>>
                       (d_temp1, sdata, alignedBuferElems, ssize);
              //std::cout << "compact " << std::endl;
              //displayCudaBuffer(sdata,  ssize);
              blellochBlockScan <<<1, ssize>>> (sdata, sdata, ssize);
              //std::cout << "blellochBlockScan " << std::endl;
              //displayCudaBuffer(sdata,  ssize);
              enlarge <<<numBlocksForAligned, MAX_THREADS>>>
                       (sdata, d_temp1, alignedBuferElems, ssize);
          }
          blellochBigScanDownstep <<<numBlocksForAligned, MAX_THREADS>>>
                       (d_temp1, alignedBuferElems);

          //std::cout << "scan " << std::endl;
          //displayCudaBuffer(d_temp1,   elemstoDisplay);
          //displayReducedArray(d_temp1, alignedBuferElems);
          //unsigned int max = displayCudaBufferMax(d_temp1, alignedBuferElems);

          resetMapToBin <<<numBlocksForElements, MAX_THREADS>>>
                        (d_iv, d_temp1, mask, i, j, numElems);
          //std::cout << "resetMapToBin " << std::endl;
          //displayCudaBuffer(d_temp1, elemstoDisplay);
          //displayCudaBufferMax(d_temp1, numElems);

          sum <<<numBlocksForElements, MAX_THREADS>>>
              (d_temp1,d_ov,numElems);
          //std::cout << "sum " << std::endl;
          //displayCheckSum(d_ov, numElems);
          //displayCudaBufferMax(d_ov, numElems);
          //displayCudaBuffer(d_ov, elemstoDisplay);
      }

      histogram <<<numBlocksForElements, MAX_THREADS>>> (d_iv, d_binHistogram, mask, i, numElems);
      //histogram<<<1, numElems>>>(d_iv, d_binHistogram, mask, i, numElems, NUM_BINS);
      //std::cout << "d_binHistogram " << std::endl;
      //displayCudaBuffer(d_binHistogram, NUM_BINS);

      //perform exclusive prefix sum (scan) on binHistogram to get starting
      //location for each bin
      blellochBlockScan <<<1, NUM_BINS>>>
                           (d_binHistogram, d_binScan, NUM_BINS);
      //std::cout << "d_binScan " << std::endl;
      //displayCudaBuffer(d_binScan, NUM_BINS);

      //Gather everything into the correct location
      //need to move vals and positions
      unsigned int* d_disp_src  = d_ov;
      unsigned int* d_new_index = d_temp;
      //displayCudaBufferMax(d_disp_src, numElems);
      getNewIndexes <<<numBlocksForElements, MAX_THREADS>>>
                    (d_iv, d_disp_src, d_binScan, d_new_index, mask, i, numElems);
      //std::cout << "after getNewIndexes " << std::endl;
      //displayCudaBuffer(d_new_index, elemstoDisplay);
      //std::cout << "before gather " << std::endl;
      //displayCudaBuffer(d_ov, elemstoDisplay);
      //displayCheckSum(d_iv, numElems);
      gather <<<numBlocksForElements, MAX_THREADS>>>
             (d_iv, d_ip, d_new_index, d_ov, d_op, numElems);
      //std::cout << "after gather " << std::endl;
      //displayCudaBuffer(d_ov, elemstoDisplay);
      //displayCheckSum(d_ov, numElems);

      //swap the buffers (pointers only)
      std::swap(d_ov, d_iv);
      std::swap(d_op, d_ip);
  }
  //std::cout << "FINISHED " << std::endl;
  //std::cout << "COPY " << std::endl;
  //we did an even number of iterations, need to copy from input buffer into output
  copy <<<numBlocksForElements, MAX_THREADS>>> (d_iv, d_ov, numElems);
  copy <<<numBlocksForElements, MAX_THREADS>>> (d_ip, d_op, numElems);

  //std::cout << "d_outputVals " << std::endl;
  //displayCudaBuffer(d_outputVals, elemstoDisplay);
  //displayCudaBufferMax(d_outputVals, numElems);
  //std::cout << "d_inputVals " << std::endl;
  //displayCudaBuffer(d_inputVals, elemstoDisplay);
  //displayCudaBufferMax(d_inputVals, numElems);

  checkCudaErrors(cudaFree(d_binScan));
  checkCudaErrors(cudaFree(d_binHistogram));
  checkCudaErrors(cudaFree(d_temp));
  checkCudaErrors(cudaFree(d_temp1));
  checkCudaErrors(cudaFree(sdata));
}



