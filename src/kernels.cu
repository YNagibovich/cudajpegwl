#include <cuda.h>
#include <stdio.h>

#include "kernels.cuh"

#include "device_launch_parameters.h"

typedef unsigned char   uint8;
typedef unsigned int    uint32;
typedef int             int32;

//namespace cuda_common
//{
    extern "C"
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
        __global__ void kernel_rgb2yuv(unsigned short* src_img, unsigned char* Y, unsigned char* u, unsigned char* v,
            int src_width, int src_height, size_t yPitch)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= src_width)
            return; //x = width - 1;

        if (y >= src_height)
            return; // y = height - 1;

        float B = (float)(src_img[y * src_width + x]>>2);
        //float G = src_img[src_width * src_height + y * src_width + x];
        //float R = src_img[src_width * src_height * 2 + y * src_width + x];
        float G = B;
        float R = B;


        Y[y * yPitch + x] = clip_value((unsigned char)(0.299 * R + 0.587 * G + 0.114 * B), 0, 255);
        u[y * src_width + x] = clip_value((unsigned char)(-0.147 * R - 0.289 * G + 0.436 * B + 128), 0, 255);
        v[y * src_width + x] = clip_value((unsigned char)(0.615 * R - 0.515 * G - 0.100 * B + 128), 0, 255);

        // old
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
//}

