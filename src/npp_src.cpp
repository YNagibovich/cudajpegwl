#include "driver_types.h"

typedef unsigned char   uint8;
typedef unsigned int    uint32;
typedef int             int32;

namespace cuda_common
{
    __device__ unsigned char clip_value(unsigned char x, unsigned char min_val, unsigned char  max_val) {
        if (x > max_val) {
            return max_val;
        }
        else if (x < min_val) {
            return min_val;
        }
        else {
            return x;
        }
    }

    extern "C"
        __global__ void kernel_rgb2yuv(float *src_img, unsigned char* Y, unsigned char* u, unsigned char* v,
            int src_width, int src_height, size_t yPitch)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= src_width)
            return; //x = width - 1;

        if (y >= src_height)
            return; // y = height - 1;

        float B = src_img[y * src_width + x];
        float G = src_img[src_width * src_height + y * src_width + x];
        float R = src_img[src_width * src_height * 2 + y * src_width + x];

        Y[y * yPitch + x] = clip_value((unsigned char)(0.299 * R + 0.587 * G + 0.114 * B), 0, 255);
        u[y * src_width + x] = clip_value((unsigned char)(-0.147 * R - 0.289 * G + 0.436 * B + 128), 0, 255);
        v[y * src_width + x] = clip_value((unsigned char)(0.615 * R - 0.515 * G - 0.100 * B + 128), 0, 255);

        //Y[y * yPitch + x] = clip_value((unsigned char)(0.257 * R + 0.504 * G + 0.098 * B + 16), 0, 255);
        //u[y * src_width + x] = clip_value((unsigned char)(-0.148 * R - 0.291 * G + 0.439 * B + 128), 0, 255);
        //v[y * src_width + x] = clip_value((unsigned char)(0.439 * R - 0.368 * G - 0.071 * B + 128), 0, 255);
    }

    extern "C"
        __global__ void kernel_resize_UV(unsigned char* src_img, unsigned char *dst_img,
            int src_width, int src_height, int dst_width, int dst_height, int nPitch)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= dst_width)
            return; //x = width - 1;

        if (y >= dst_height)
            return; // y = height - 1;

        float fx = (x + 0.5)*src_width / (float)dst_width - 0.5;
        float fy = (y + 0.5)*src_height / (float)dst_height - 0.5;
        int ax = floor(fx);
        int ay = floor(fy);
        if (ax < 0)
        {
            ax = 0;
        }
        else if (ax > src_width - 2)
        {
            ax = src_width - 2;
        }

        if (ay < 0) {
            ay = 0;
        }
        else if (ay > src_height - 2)
        {
            ay = src_height - 2;
        }

        int A = ax + ay*src_width;
        int B = ax + ay*src_width + 1;
        int C = ax + ay*src_width + src_width;
        int D = ax + ay*src_width + src_width + 1;

        float w1, w2, w3, w4;
        w1 = fx - ax;
        w2 = 1 - w1;
        w3 = fy - ay;
        w4 = 1 - w3;

        unsigned char val = src_img[A] * w2*w4 + src_img[B] * w1*w4 + src_img[C] * w2*w3 + src_img[D] * w1*w3;

        dst_img[y * nPitch + x] = clip_value(val, 0, 255);
    }

    cudaError_t RGB2YUV(float* d_srcRGB, int src_width, int src_height,
        unsigned char* Y, size_t yPitch, int yWidth, int yHeight,
        unsigned char* U, size_t uPitch, int uWidth, int uHeight,
        unsigned char* V, size_t vPitch, int vWidth, int vHeight)
    {
        unsigned char * u;
        unsigned char * v;

        cudaError_t cudaStatus;

        cudaStatus = cudaMalloc((void**)&u, src_width * src_height * sizeof(unsigned char));
        cudaStatus = cudaMalloc((void**)&v, src_width * src_height * sizeof(unsigned char));

        dim3 block(32, 16, 1);
        dim3 grid((src_width + (block.x - 1)) / block.x, (src_height + (block.y - 1)) / block.y, 1);
        dim3 grid1((uWidth + (block.x - 1)) / block.x, (uHeight + (block.y - 1)) / block.y, 1);
        dim3 grid2((vWidth + (block.x - 1)) / block.x, (vHeight + (block.y - 1)) / block.y, 1);

        kernel_rgb2yuv << < grid, block >> > (d_srcRGB, Y, u, v, src_width, src_height, yPitch);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernel_rgb2yuv launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_rgb2yuv!\n", cudaStatus);
            goto Error;
        }

        kernel_resize_UV << < grid1, block >> > (u, U, src_width, src_height, uWidth, uHeight, uPitch);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernel_resize_UV launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_resize_UV!\n", cudaStatus);
            goto Error;
        }

        kernel_resize_UV << < grid2, block >> > (v, V, src_width, src_height, vWidth, vHeight, vPitch);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernel_resize_UV launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_resize_UV!\n", cudaStatus);
            goto Error;
        }

    Error:
        cudaFree(u);
        cudaFree(v);

        return cudaStatus;
    }
}


int jpegNPP(const char *szOutputFile, float* d_srcRGB, int img_width, int img_height)
{
    NppiDCTState *pDCTState;
    NPP_CHECK_NPP(nppiDCTInitAlloc(&pDCTState));

    // Parsing and Huffman Decoding (on host)
    FrameHeader oFrameHeader;
    QuantizationTable aQuantizationTables[4];
    Npp8u *pdQuantizationTables;
    cudaMalloc(&pdQuantizationTables, 64 * 4);

    HuffmanTable aHuffmanTables[4];
    HuffmanTable *pHuffmanDCTables = aHuffmanTables;
    HuffmanTable *pHuffmanACTables = &aHuffmanTables[2];
    ScanHeader oScanHeader;
    memset(&oFrameHeader, 0, sizeof(FrameHeader));
    memset(aQuantizationTables, 0, 4 * sizeof(QuantizationTable));
    memset(aHuffmanTables, 0, 4 * sizeof(HuffmanTable));
    int nMCUBlocksH = 0;
    int nMCUBlocksV = 0;

    int nRestartInterval = -1;

    NppiSize aSrcSize[3];
    Npp16s *apdDCT[3] = { 0,0,0 };
    Npp32s aDCTStep[3];

    Npp8u *apSrcImage[3] = { 0,0,0 };
    Npp32s aSrcImageStep[3];
    size_t aSrcPitch[3];

    /***************************
    *
    *   Output    編碼部分
    *
    ***************************/

    unsigned char STD_DC_Y_NRCODES[16] = { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
    unsigned char STD_DC_Y_VALUES[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

    unsigned char STD_DC_UV_NRCODES[16] = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
    unsigned char STD_DC_UV_VALUES[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

    unsigned char STD_AC_Y_NRCODES[16] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0X7D };
    unsigned char STD_AC_Y_VALUES[162] =
    {
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
        0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
        0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
        0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa
    };

    unsigned char STD_AC_UV_NRCODES[16] = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0X77 };
    unsigned char STD_AC_UV_VALUES[162] =
    {
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
        0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
        0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
        0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
        0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
        0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa
    };

    //填充Huffman表
    aHuffmanTables[0].nClassAndIdentifier = 0;
    memcpy(aHuffmanTables[0].aCodes, STD_DC_Y_NRCODES, 16);
    memcpy(aHuffmanTables[0].aTable, STD_DC_Y_VALUES, 12);

    aHuffmanTables[1].nClassAndIdentifier = 1;
    memcpy(aHuffmanTables[1].aCodes, STD_DC_UV_NRCODES, 16);
    memcpy(aHuffmanTables[1].aTable, STD_DC_UV_VALUES, 12);

    aHuffmanTables[2].nClassAndIdentifier = 16;
    memcpy(aHuffmanTables[2].aCodes, STD_AC_Y_NRCODES, 16);
    memcpy(aHuffmanTables[2].aTable, STD_AC_Y_VALUES, 162);

    aHuffmanTables[3].nClassAndIdentifier = 17;
    memcpy(aHuffmanTables[3].aCodes, STD_AC_UV_NRCODES, 16);
    memcpy(aHuffmanTables[3].aTable, STD_AC_UV_VALUES, 162);

    //標準亮度信號量化模板
    unsigned char std_Y_QT[64] =
    {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };

    //標準色差信號量化模板
    unsigned char std_UV_QT[64] =
    {
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99
    };

    //填充量化表
    aQuantizationTables[0].nPrecisionAndIdentifier = 0;
    memcpy(aQuantizationTables[0].aTable, std_Y_QT, 64);
    aQuantizationTables[1].nPrecisionAndIdentifier = 1;
    memcpy(aQuantizationTables[1].aTable, std_UV_QT, 64);

    NPP_CHECK_CUDA(cudaMemcpyAsync(pdQuantizationTables, aQuantizationTables[0].aTable, 64, cudaMemcpyHostToDevice));
    NPP_CHECK_CUDA(cudaMemcpyAsync(pdQuantizationTables + 64, aQuantizationTables[1].aTable, 64, cudaMemcpyHostToDevice));

    //填充幀頭
    oFrameHeader.nSamplePrecision = 8;
    oFrameHeader.nComponents = 3;
    oFrameHeader.aComponentIdentifier[0] = 1;
    oFrameHeader.aComponentIdentifier[1] = 2;
    oFrameHeader.aComponentIdentifier[2] = 3;
    oFrameHeader.aSamplingFactors[0] = 34;
    oFrameHeader.aSamplingFactors[1] = 17;
    oFrameHeader.aSamplingFactors[2] = 17;
    oFrameHeader.aQuantizationTableSelector[0] = 0;
    oFrameHeader.aQuantizationTableSelector[1] = 1;
    oFrameHeader.aQuantizationTableSelector[2] = 1;
    oFrameHeader.nWidth = img_width;
    oFrameHeader.nHeight = img_height;

    for (int i = 0; i < oFrameHeader.nComponents; ++i)
    {
        nMCUBlocksV = max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f);
        nMCUBlocksH = max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4);
    }

    for (int i = 0; i < oFrameHeader.nComponents; ++i)
    {
        NppiSize oBlocks;
        NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f };

        oBlocks.width = (int)ceil((oFrameHeader.nWidth + 7) / 8 *
            static_cast<float>(oBlocksPerMCU.width) / nMCUBlocksH);
        oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

        oBlocks.height = (int)ceil((oFrameHeader.nHeight + 7) / 8 *
            static_cast<float>(oBlocksPerMCU.height) / nMCUBlocksV);
        oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

        aSrcSize[i].width = oBlocks.width * 8;
        aSrcSize[i].height = oBlocks.height * 8;

        // Allocate Memory
        size_t nPitch;
        NPP_CHECK_CUDA(cudaMallocPitch(&apdDCT[i], &nPitch, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height));
        aDCTStep[i] = static_cast<Npp32s>(nPitch);

        NPP_CHECK_CUDA(cudaMallocPitch(&apSrcImage[i], &nPitch, aSrcSize[i].width, aSrcSize[i].height));

        aSrcPitch[i] = nPitch;
        aSrcImageStep[i] = static_cast<Npp32s>(nPitch);
    }

    //RGB2YUV
    cudaError_t cudaStatus;
    cudaStatus = cuda_common::RGB2YUV(d_srcRGB, img_width, img_height,
        apSrcImage[0], aSrcPitch[0], aSrcSize[0].width, aSrcSize[0].height,
        apSrcImage[1], aSrcPitch[1], aSrcSize[1].width, aSrcSize[1].height,
        apSrcImage[2], aSrcPitch[2], aSrcSize[2].width, aSrcSize[2].height);

    /**
    * Forward DCT, quantization and level shift part of the JPEG encoding.
    * Input is expected in 8x8 macro blocks and output is expected to be in 64x1
    * macro blocks. The new version of the primitive takes the ROI in image pixel size and
    * works with DCT coefficients that are in zig-zag order.
    */
    int k = 0;
    NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apSrcImage[0], aSrcImageStep[0],
        apdDCT[0], aDCTStep[0],
        pdQuantizationTables + k * 64,
        aSrcSize[0],
        pDCTState));
    k = 1;
    NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apSrcImage[1], aSrcImageStep[1],
        apdDCT[1], aDCTStep[1],
        pdQuantizationTables + k * 64,
        aSrcSize[1],
        pDCTState));

    NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apSrcImage[2], aSrcImageStep[2],
        apdDCT[2], aDCTStep[2],
        pdQuantizationTables + k * 64,
        aSrcSize[2],
        pDCTState));


    // Huffman Encoding
    Npp8u *pdScan;
    Npp32s nScanLength;
    NPP_CHECK_CUDA(cudaMalloc(&pdScan, 4 << 20));

    Npp8u *pJpegEncoderTemp;
    Npp32s nTempSize;
    NPP_CHECK_NPP(nppiEncodeHuffmanGetSize(aSrcSize[0], 3, &nTempSize));
    NPP_CHECK_CUDA(cudaMalloc(&pJpegEncoderTemp, nTempSize));

    NppiEncodeHuffmanSpec *apHuffmanDCTable[3];
    NppiEncodeHuffmanSpec *apHuffmanACTable[3];

    /**
    * Allocates memory and creates a Huffman table in a format that is suitable for the encoder.
    */
    NppStatus t_status;
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[0].aCodes, nppiDCTable, &apHuffmanDCTable[0]);
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[0].aCodes, nppiACTable, &apHuffmanACTable[0]);
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[1].aCodes, nppiDCTable, &apHuffmanDCTable[1]);
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[1].aCodes, nppiACTable, &apHuffmanACTable[1]);
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[1].aCodes, nppiDCTable, &apHuffmanDCTable[2]);
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[1].aCodes, nppiACTable, &apHuffmanACTable[2]);

    /**
    * Huffman Encoding of the JPEG Encoding.
    * Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
    */
    Npp32s nSs = 0;
    Npp32s nSe = 63;
    Npp32s nH = 0;
    Npp32s nL = 0;
    NPP_CHECK_NPP(nppiEncodeHuffmanScan_JPEG_8u16s_P3R(apdDCT, aDCTStep,
        0, nSs, nSe, nH, nL,
        pdScan, &nScanLength,
        apHuffmanDCTable,
        apHuffmanACTable,
        aSrcSize,
        pJpegEncoderTemp));

    for (int i = 0; i < 3; ++i)
    {
        nppiEncodeHuffmanSpecFree_JPEG(apHuffmanDCTable[i]);
        nppiEncodeHuffmanSpecFree_JPEG(apHuffmanACTable[i]);
    }

    // Write JPEG
    unsigned char *pDstJpeg = new unsigned char[4 << 20];
    unsigned char *pDstOutput = pDstJpeg;

    writeMarker(0x0D8, pDstOutput);
    writeJFIFTag(pDstOutput);
    writeQuantizationTable(aQuantizationTables[0], pDstOutput);
    writeQuantizationTable(aQuantizationTables[1], pDstOutput);

    writeFrameHeader(oFrameHeader, pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[1], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[1], pDstOutput);

    oScanHeader.nComponents = 3;
    oScanHeader.aComponentSelector[0] = 1;
    oScanHeader.aComponentSelector[1] = 2;
    oScanHeader.aComponentSelector[2] = 3;
    oScanHeader.aHuffmanTablesSelector[0] = 0;
    oScanHeader.aHuffmanTablesSelector[1] = 17;
    oScanHeader.aHuffmanTablesSelector[2] = 17;
    oScanHeader.nSs = 0;
    oScanHeader.nSe = 63;
    oScanHeader.nA = 0;

    writeScanHeader(oScanHeader, pDstOutput);
    NPP_CHECK_CUDA(cudaMemcpy(pDstOutput, pdScan, nScanLength, cudaMemcpyDeviceToHost));
    pDstOutput += nScanLength;
    writeMarker(0x0D9, pDstOutput);

    {
        // Write result to file.
        std::ofstream outputFile(szOutputFile, ios::out | ios::binary);
        outputFile.write(reinterpret_cast<const char *>(pDstJpeg), static_cast<int>(pDstOutput - pDstJpeg));
    }

    // Cleanup
    delete[] pDstJpeg;

    cudaFree(pJpegEncoderTemp);
    cudaFree(pdQuantizationTables);
    cudaFree(pdScan);

    nppiDCTFree(pDCTState);

    for (int i = 0; i < 3; ++i)
    {
        cudaFree(apdDCT[i]);
        cudaFree(apSrcImage[i]);
    }

    return EXIT_SUCCESS;
}

