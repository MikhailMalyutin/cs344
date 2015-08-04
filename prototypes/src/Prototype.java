public class Prototype {
    private static final int MAX_THREADS = 256;
    private static final int NUM_BITS = 1;
    private static final int NUM_BINS = 1 << NUM_BITS;

    private static void scanReduceForBlock(int d_res[],
                                           int initialS, int size) {
        int prevId;
        int prevValue;
        int nextId;
        int nextValue;

        for (int s = 1; s <= initialS/2; s *= 2) {
            for (int myId = 0; myId < size; ++ myId) {
                prevId = myId;
                prevValue = d_res[prevId];
                nextId = myId + s;
                nextValue = nextId < size ? d_res[nextId] : 0;

                if (((nextId + 1) % (s*2)) == 0 && (nextId < size)) {
                    d_res[nextId] = prevValue + nextValue;
                }
            }
            System.out.print(d_res);
        }
    }

    private static void scanDownStepForBlock(int d_res[], int initialS, int maxSize) {
        int prevId;
        int prevValue;
        int myValue;

        for (int s = initialS; s >= 2; s /= 2) {
            for (int myId = 0; myId < maxSize; ++myId) {
                prevId = myId - s / 2;
                prevValue = (myId >= s / 2) ? d_res[prevId] : 0;
                myValue = d_res[myId];

                if (((myId + 1) % s) == 0 && myId >= s / 2) {
                    d_res[prevId] = myValue;
                    d_res[myId] = myValue + prevValue;
                }
            }
        }
    }

    public static void blellochScan(int d_in[],
                               int d_res[],
                               int size) {
        for (int myId = 0; myId < size; ++ myId) {
            if (myId >= size) {
                return;
            }
            d_res[myId] = d_in[myId];
        }

        scanReduceForBlock(d_res, Math.min(MAX_THREADS, size), size);
        d_res[size - 1] = 0;

        int ssize = size / MAX_THREADS;
        if (ssize > 1) {
            int interval = size/ ssize;
            int sdata[] = new int[ssize];
            for (int myId = 0; myId < ssize; ++myId) {
                sdata[myId] = d_res[myId * interval + interval - 1];
            }
            scanReduceForBlock(sdata, MAX_THREADS, ssize);
            sdata[ssize - 1] = 0;
            scanDownStepForBlock(sdata, MAX_THREADS, ssize);
            for (int myId = 0; myId < ssize; ++myId) {
                d_res[myId * interval + interval - 1] = sdata[myId];
            }
        }
    }

    static void histogram(int[] d_in, int[] d_res,
                              int mask,
                              int i,
                   int numElems, int NUM_BINS) {
        for (int myId = 0; myId < numElems; ++myId) {
            if (myId < NUM_BINS) { //очистка буфера результата
                d_res[myId] = 0;
            }
            int binId = (d_in[myId] & mask) >> i;
            d_res[binId]+=1;
        }
    }

    static int getNearest(int number) {
        int result = 1;
        while( result < number ) {
            result <<= 1;
        }
        return result;
    }

    static void mapToBin(int[] d_vals_src, int[] d_vals_dst,
                              int mask,
                              int i,
                              int mappedBean,
                              int numElems) {

        for (int myId =0; myId < numElems; ++myId) {
            int beanId = (d_vals_src[myId] & mask) >> i;
            d_vals_dst[myId] = (beanId == mappedBean) ? 1 : 0;
        }
    }

    static void resetMapToBin(int[] d_vals_src,
                                  int[] d_vals_dst,
                                  int mask,
                                  int i,
                                  int mappedBean,
                                  int numElems) {
        for (int myId =0; myId < numElems; ++myId) {
            int beanId = (d_vals_src[myId] & mask) >> i;
            if (beanId != mappedBean) {
                d_vals_dst[myId] = 0;
            }
        }
    }

    static void sum(int[] d_src,
             int[] d_dst,
             int numElems) {
        for (int myId =0; myId < numElems; ++myId) {
            d_dst[myId] = d_src[myId] + d_dst[myId];
        }
    }

    static void gather(int[] d_vals_src,
                int[] d_pos_src,
                int[] d_vals_dst,
                int[] d_pos_dst,
                int[] d_binScan,
                int mask,
                int i,
                int numElems) {

        int[] pos = new int[d_vals_dst.length];
        copy(d_vals_dst, pos, numElems);

        for (int myId = 0 ;myId < numElems; ++myId) {
            int myIdOffset = pos[myId];

            int binId = (d_vals_src[myId] & mask) >> i;
            int offset = d_binScan[binId];
            int newIndex = offset + myIdOffset;
            d_vals_dst[newIndex] = d_vals_src[myId];
            d_pos_dst[newIndex] = d_pos_src[myId];
        }
        System.out.print(d_vals_dst);
    }

    static void your_sort(int d_inputVals[],
                   int d_inputPos[],
                   int d_outputVals[],
                   int d_outputPos[],
                   int numElems)
    {
        int[] d_binScan;
        int[] d_binHistogram;
        int[] d_temp;
        int[] d_temp1;
        int[] d_iv = d_inputVals;
        int[] d_ip = d_inputPos;
        int[] d_ov = d_outputVals;
        int[] d_op = d_outputPos;

        int alignedBuferElems = getNearest(numElems);

        d_binScan = new int[NUM_BINS];
        d_binHistogram = new int[NUM_BINS];
        d_temp = new int[alignedBuferElems];
        d_temp1 = new int[alignedBuferElems];

        //a simple radix sort - only guaranteed to work for NUM_BITS that are multiples of 2
        for (int i = 0; i < 8 * 4; i += NUM_BITS) {
        int mask = (NUM_BINS - 1) << i;

        clear(d_ov,numElems);

        for (int j = 0; j < NUM_BINS; ++j) {
            clear(d_temp, alignedBuferElems);
            clear(d_temp1, alignedBuferElems);

            mapToBin(d_iv, d_temp, mask, i, j, numElems);

            blellochScan(d_temp, d_temp1, alignedBuferElems);
            blellochScanDownstep(d_temp, d_temp1, alignedBuferElems);

            resetMapToBin(d_iv, d_temp1, mask, i, j, numElems);

            sum(d_temp1,d_ov,numElems);
        }

        histogram(d_iv, d_binHistogram, mask, i, numElems, NUM_BINS);

        blellochScan(d_binHistogram, d_binScan, NUM_BINS);
        blellochScanDownstep(d_binScan, d_binScan, NUM_BINS);

        gather(d_iv, d_ip, d_ov, d_op, d_binScan, mask, i, numElems);

        //swap the buffers (pointers only)
        swap(d_ov, d_iv);
        swap(d_op, d_ip);
    }

        //we did an even number of iterations, need to copy from input buffer into output
        copy(d_iv, d_ov, numElems);
        copy(d_ip, d_op, numElems);
    }

    private static void copy(int[] d_ip, int[] d_op, int numElems) {
        System.arraycopy(d_ip, 0, d_op, 0, numElems);
    }

    private static void swap(int[] d_ov, int[] d_iv) {
        int[] temp = new int[d_ov.length];
        copy(d_ov, temp, d_ov.length);//temp=d_ov
        copy(d_iv, d_ov, d_ov.length);//d_ov=d_iv
        copy(temp, d_iv, d_ov.length);//d_iv=temp
    }

    private static void blellochScanDownstep(int[] d_temp, int[] d_temp1, int alignedBuferElems) {
        scanDownStepForBlock(d_temp1, MAX_THREADS, alignedBuferElems);
    }

    private static void clear(int[] d_ov, int numElems) {
        for (int i = 0; i<numElems; ++i) {
            d_ov[i] = 0;
        }
    }

    public static void main(String[] args) {
        int smallData[] = {1,2};
        int smallResult[] = {0, 0};
        //blellochScan(smallData, smallResult, smallResult.length);
        //scanDownStepForBlock(smallResult, MAX_THREADS, smallResult.length);

        int inDataUnwr[] = {1,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
                /**0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                1**/
        };
        int alignedBuferElems = getNearest(inDataUnwr.length);
        int inData[] = new int[alignedBuferElems];
        for (int i=0; i<inDataUnwr.length; ++i) {
            inData[i] = inDataUnwr[i];
        }
        int outData[] = new int[alignedBuferElems];
        //blellochScan(inData, outData, alignedBuferElems);
        //scanDownStepForBlock(outData, MAX_THREADS, alignedBuferElems);
        System.out.println(outData);
        int sortData[] = getUnsortedSeq(1500);
        int sortVal[] = getUnsortedSeq(1500);
        int resData[] = new int[sortData.length];
        int resVal[] = new int[sortData.length];
       // clear(sortData, sortData.length);
        your_sort(sortData, sortVal, resData, resVal, sortData.length);
    }

    private static int[] getUnsortedSeq(int numElems) {
        int[] result = new int[numElems];
        for (int i = 0; i< numElems; ++i) {
            result[i] = numElems - i;
        }
        return result;
    }
}
