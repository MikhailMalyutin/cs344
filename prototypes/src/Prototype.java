public class Prototype {
    private static int maxThreads = 8;

    private static void scanReduceForBlock(int d_res[],
                                           int initialS, int size) {
        int prevId;
        int prevValue;
        int myValue;

        for (int s = 2; s <= initialS; s *= 2) {
            for (int myId = 0; myId < size; ++ myId) {
                prevId = myId - s / 2;
                prevValue = (myId >= s / 2) ? d_res[prevId] : 0;
                myValue = d_res[myId];

                if (((myId + 1) % s) == 0 && (myId >= s / 2)) {
                    d_res[myId] = myValue + prevValue;
                }
            }
            System.out.print(d_res);
        }
    }

    private static void scanDownStepForBlock(int d_res[], int initialS, int maxSize) {
        int prevId;
        int prevValue;
        int myValue;

        for (int myId = 0; myId < maxSize; ++ myId) {

            for (int s = initialS; s >= 2; s /= 2) {
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
        int sdata[] = new int[maxThreads];
        for (int myId = 0; myId < size; ++ myId) {
            if (myId >= size) {
                return;
            }
            d_res[myId] = d_in[myId];
        }

        scanReduceForBlock(d_res, Math.min(maxThreads, size), size);
        d_res[size - 1] = 0;

        int compactRatio = size / maxThreads;
        if (compactRatio > 1) {
            for (int myId = 0; myId < compactRatio; ++myId) {
                sdata[myId] = d_res[myId * maxThreads + maxThreads - 1];
            }
            scanReduceForBlock(sdata, maxThreads, maxThreads);
            //scanReduceForBlock(sdata, maxThreads/2, compactedMyId);
            sdata[compactRatio - 1] = 0;
            scanDownStepForBlock(sdata, maxThreads, maxThreads);
            for (int myId = 0; myId < compactRatio; ++myId) {
                d_res[myId * maxThreads + maxThreads - 1] = sdata[myId];
            }
        }
    }

    public static void main(String[] args) {
        int inData[] = {1,
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
                1/**,
                0,
                0,
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
        int outData[] = new int[inData.length];
        blellochScan(inData, outData, inData.length);
        scanDownStepForBlock(outData, maxThreads, inData.length);
        System.out.println(outData);
    }
}
