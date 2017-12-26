#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" __global__ void kernel_rgb2yuv(short* src_img, unsigned char* Y, unsigned char* u, unsigned char* v,
    int src_width, int src_height, size_t yPitch, int rIntercept, int rSlope, short minWindowValue, short windowWidth);
extern "C" __global__ void kernel_resize_UV(unsigned char* src_img, unsigned char *dst_img,
    int src_width, int src_height, int dst_width, int dst_height, int nPitch);
