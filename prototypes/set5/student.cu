/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "reference.cpp"

const unsigned int MAX_THREADS = 1024;
const unsigned int SIMD_THREADS = 32;
const unsigned int MAX_BLOCKS = MAX_THREADS / SIMD_THREADS;

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
        std::cout << std::dec << buf[i] << " " << std::endl;
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

__device__ void blockHisto(const unsigned int* const vals,    //INPUT
                                 unsigned int* const histo) { //OUPUT)
    const unsigned int myId = getMyId();
    const unsigned int binId = vals[myId];

    atomicAdd(&(histo[binId]), 1);
}

//MAIN--------------------------------------------------------------------

__global__
void yourHistoSlow(const unsigned int* const vals,    //INPUT
                         unsigned int* const histo,   //OUPUT
                                  int        numVals)
{
    const unsigned int myId = getMyId();
    if (myId <= numVals) {
        const unsigned int binId = vals[myId];

        atomicAdd(&(histo[binId]), 1);
    }
}

__global__
void yourHisto(const unsigned int* const vals,       //INPUT
                     unsigned int* const histo,      //OUPUT
               const unsigned int        numBins)
{
    extern __shared__ unsigned char sdata[];

    const unsigned int tidX           = threadIdx.x;
    const unsigned int blockId        = blockDim.x * blockIdx.x;
    const unsigned int myId           = tidX + blockId;
    const unsigned int simdTid        = tidX % SIMD_THREADS;
    const unsigned int simdBlock      = tidX / SIMD_THREADS;
    const unsigned int simdBlockStart = simdBlock * SIMD_THREADS;
    const unsigned int tid            = simdBlockStart + simdTid;
    const unsigned int s1idx          = numBins;

    //for (unsigned int i = 0; i < MAX_BLOCKS; ++i) {
        data[simdTid * numBins + tid] = 0;
    //}

    __syncthreads();
    const unsigned int curVal     = vals[myId];
    const unsigned int smAdr = simdTid * numBins + curVal;
    sdata[smAdr] = sdata[smAdr] + 1;
    //atomicAdd(&(sdata[curVal]), 1);

    __syncthreads();
    unsigned int blockHistoVal = 0;
   // if (blockId == 0) {
        for (unsigned int i = 0; i < MAX_BLOCKS; ++i) {
            blockHistoVal += sdata[i * numBins + tid] = 0;
        }
    //}

     //sdata[tid];
    if (blockHistoVal != 0) {
        atomicAdd(&(histo[tid]), blockHistoVal);
    }
}

void computeHistogram(const unsigned int* const d_vals,  //INPUT
                            unsigned int* const d_histo, //OUTPUT
                      const unsigned int        numBins,
                      const unsigned int        numElems)
{
  const unsigned int MAX_BLOCKS = 16384;

  const unsigned int LAYER_SIZE = MAX_BLOCKS * MAX_THREADS;
  const unsigned int numBlocksForElements = (numElems + LAYER_SIZE - 1) / LAYER_SIZE;
  std::cout << "numElems " << numElems << std::endl;
  std::cout << "LAYER_SIZE " << LAYER_SIZE << std::endl;
  std::cout << "numBlocksForElements " << numBlocksForElements << std::endl;
  //displayCudaBufferMax(d_vals, numElems);
  //std::cout << "numBins "  << numBins << std::endl;
  //displayCudaBufferMax(d_histo, numBins);

  for (unsigned int i=0; i < numBlocksForElements; ++i) {
      const unsigned int elems = (LAYER_SIZE * i + LAYER_SIZE < numElems)
          ? LAYER_SIZE
          : numElems - LAYER_SIZE * i;
      const unsigned int numBlocks = elems / MAX_THREADS;
      //std::cout << "elems "  << elems << std::endl;
      //std::cout << "lastIndex "  << LAYER_SIZE * i + elems << std::endl;
      yourHisto<<<numBlocks, MAX_THREADS, numBins*8*sizeof(unsigned int)>>> (d_vals + LAYER_SIZE * i, d_histo, numBins);
      //yourHistoSlow<<<numBlocks, MAX_THREADS>>> (d_vals + LAYER_SIZE * i, d_histo, elems);
  }

  //if you want to use/launch more than one kernel,
  //feel free
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //std::cout << "RESULT "  << numBins << std::endl;
  //displayCudaBufferMax(d_histo, numBins);

 // delete[] h_vals;
 // delete[] h_histo;
 // delete[] your_histo;*/
}
