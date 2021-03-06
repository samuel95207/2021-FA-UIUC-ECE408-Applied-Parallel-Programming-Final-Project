#include <cmath>
#include <iostream>

#include "gpu-new-forward.h"

#define KERNEL_WIDTH 7

#define TILE_WIDTH_C1 16
#define TILE_WIDTH_C4 16

#define STREAM_NUM 50

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cout << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
    }
    return result;
}

__constant__ float Kernel[4096];


__global__ void conv_forward_kernel_C1(float *__restrict__ y, const float *__restrict__ x,
                                       const int B, const int M, const int C, const int H,
                                       const int W, const int K, const int offset) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire
    mini-batch The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) \
    y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d_constant(i3, i2, i1, i0) Kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(W_out / float(TILE_WIDTH_C1));
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int b = blockIdx.x + offset;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH_C1 + ty;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH_C1 + tx;

    // Calculate Convolution
    if ((w < W_out) && (h < H_out)) {
        float acc = 0.0f;
        // Unroll for loop
        acc += x4d(b, 0, h + 0, w + 0) * k4d_constant(m, 0, 0, 0) +
               x4d(b, 0, h + 0, w + 1) * k4d_constant(m, 0, 0, 1) +
               x4d(b, 0, h + 0, w + 2) * k4d_constant(m, 0, 0, 2) +
               x4d(b, 0, h + 0, w + 3) * k4d_constant(m, 0, 0, 3) +
               x4d(b, 0, h + 0, w + 4) * k4d_constant(m, 0, 0, 4) +
               x4d(b, 0, h + 0, w + 5) * k4d_constant(m, 0, 0, 5) +
               x4d(b, 0, h + 0, w + 6) * k4d_constant(m, 0, 0, 6) +
               x4d(b, 0, h + 1, w + 0) * k4d_constant(m, 0, 1, 0) +
               x4d(b, 0, h + 1, w + 1) * k4d_constant(m, 0, 1, 1) +
               x4d(b, 0, h + 1, w + 2) * k4d_constant(m, 0, 1, 2) +
               x4d(b, 0, h + 1, w + 3) * k4d_constant(m, 0, 1, 3) +
               x4d(b, 0, h + 1, w + 4) * k4d_constant(m, 0, 1, 4) +
               x4d(b, 0, h + 1, w + 5) * k4d_constant(m, 0, 1, 5) +
               x4d(b, 0, h + 1, w + 6) * k4d_constant(m, 0, 1, 6) +
               x4d(b, 0, h + 2, w + 0) * k4d_constant(m, 0, 2, 0) +
               x4d(b, 0, h + 2, w + 1) * k4d_constant(m, 0, 2, 1) +
               x4d(b, 0, h + 2, w + 2) * k4d_constant(m, 0, 2, 2) +
               x4d(b, 0, h + 2, w + 3) * k4d_constant(m, 0, 2, 3) +
               x4d(b, 0, h + 2, w + 4) * k4d_constant(m, 0, 2, 4) +
               x4d(b, 0, h + 2, w + 5) * k4d_constant(m, 0, 2, 5) +
               x4d(b, 0, h + 2, w + 6) * k4d_constant(m, 0, 2, 6) +
               x4d(b, 0, h + 3, w + 0) * k4d_constant(m, 0, 3, 0) +
               x4d(b, 0, h + 3, w + 1) * k4d_constant(m, 0, 3, 1) +
               x4d(b, 0, h + 3, w + 2) * k4d_constant(m, 0, 3, 2) +
               x4d(b, 0, h + 3, w + 3) * k4d_constant(m, 0, 3, 3) +
               x4d(b, 0, h + 3, w + 4) * k4d_constant(m, 0, 3, 4) +
               x4d(b, 0, h + 3, w + 5) * k4d_constant(m, 0, 3, 5) +
               x4d(b, 0, h + 3, w + 6) * k4d_constant(m, 0, 3, 6) +
               x4d(b, 0, h + 4, w + 0) * k4d_constant(m, 0, 4, 0) +
               x4d(b, 0, h + 4, w + 1) * k4d_constant(m, 0, 4, 1) +
               x4d(b, 0, h + 4, w + 2) * k4d_constant(m, 0, 4, 2) +
               x4d(b, 0, h + 4, w + 3) * k4d_constant(m, 0, 4, 3) +
               x4d(b, 0, h + 4, w + 4) * k4d_constant(m, 0, 4, 4) +
               x4d(b, 0, h + 4, w + 5) * k4d_constant(m, 0, 4, 5) +
               x4d(b, 0, h + 4, w + 6) * k4d_constant(m, 0, 4, 6) +
               x4d(b, 0, h + 5, w + 0) * k4d_constant(m, 0, 5, 0) +
               x4d(b, 0, h + 5, w + 1) * k4d_constant(m, 0, 5, 1) +
               x4d(b, 0, h + 5, w + 2) * k4d_constant(m, 0, 5, 2) +
               x4d(b, 0, h + 5, w + 3) * k4d_constant(m, 0, 5, 3) +
               x4d(b, 0, h + 5, w + 4) * k4d_constant(m, 0, 5, 4) +
               x4d(b, 0, h + 5, w + 5) * k4d_constant(m, 0, 5, 5) +
               x4d(b, 0, h + 5, w + 6) * k4d_constant(m, 0, 5, 6) +
               x4d(b, 0, h + 6, w + 0) * k4d_constant(m, 0, 6, 0) +
               x4d(b, 0, h + 6, w + 1) * k4d_constant(m, 0, 6, 1) +
               x4d(b, 0, h + 6, w + 2) * k4d_constant(m, 0, 6, 2) +
               x4d(b, 0, h + 6, w + 3) * k4d_constant(m, 0, 6, 3) +
               x4d(b, 0, h + 6, w + 4) * k4d_constant(m, 0, 6, 4) +
               x4d(b, 0, h + 6, w + 5) * k4d_constant(m, 0, 6, 5) +
               x4d(b, 0, h + 6, w + 6) * k4d_constant(m, 0, 6, 6);
        y4d(b, m, h, w) = acc;
    }


#undef y4d
#undef x4d
#undef x_shared3d
#undef k4d_constant
}


__global__ void conv_forward_kernel_C4(float *__restrict__ y, const float *__restrict__ x,
                                       const int B, const int M, const int C, const int H,
                                       const int W, const int K, const int offset) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire
    mini-batch The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) \
    y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d_constant(i3, i2, i1, i0) Kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(W_out / float(TILE_WIDTH_C4));
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int b = blockIdx.x + offset;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH_C4 + ty;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH_C4 + tx;

    // Calculate Convolution
    if ((w < W_out) && (h < H_out)) {
        float acc = 0.0f;
// Unroll for loop
#pragma unroll 4
        for (int c = 0; c < 4; c++) {
            acc += x4d(b, c, h + 0, w + 0) * k4d_constant(m, c, 0, 0) +
                   x4d(b, c, h + 0, w + 1) * k4d_constant(m, c, 0, 1) +
                   x4d(b, c, h + 0, w + 2) * k4d_constant(m, c, 0, 2) +
                   x4d(b, c, h + 0, w + 3) * k4d_constant(m, c, 0, 3) +
                   x4d(b, c, h + 0, w + 4) * k4d_constant(m, c, 0, 4) +
                   x4d(b, c, h + 0, w + 5) * k4d_constant(m, c, 0, 5) +
                   x4d(b, c, h + 0, w + 6) * k4d_constant(m, c, 0, 6) +
                   x4d(b, c, h + 1, w + 0) * k4d_constant(m, c, 1, 0) +
                   x4d(b, c, h + 1, w + 1) * k4d_constant(m, c, 1, 1) +
                   x4d(b, c, h + 1, w + 2) * k4d_constant(m, c, 1, 2) +
                   x4d(b, c, h + 1, w + 3) * k4d_constant(m, c, 1, 3) +
                   x4d(b, c, h + 1, w + 4) * k4d_constant(m, c, 1, 4) +
                   x4d(b, c, h + 1, w + 5) * k4d_constant(m, c, 1, 5) +
                   x4d(b, c, h + 1, w + 6) * k4d_constant(m, c, 1, 6) +
                   x4d(b, c, h + 2, w + 0) * k4d_constant(m, c, 2, 0) +
                   x4d(b, c, h + 2, w + 1) * k4d_constant(m, c, 2, 1) +
                   x4d(b, c, h + 2, w + 2) * k4d_constant(m, c, 2, 2) +
                   x4d(b, c, h + 2, w + 3) * k4d_constant(m, c, 2, 3) +
                   x4d(b, c, h + 2, w + 4) * k4d_constant(m, c, 2, 4) +
                   x4d(b, c, h + 2, w + 5) * k4d_constant(m, c, 2, 5) +
                   x4d(b, c, h + 2, w + 6) * k4d_constant(m, c, 2, 6) +
                   x4d(b, c, h + 3, w + 0) * k4d_constant(m, c, 3, 0) +
                   x4d(b, c, h + 3, w + 1) * k4d_constant(m, c, 3, 1) +
                   x4d(b, c, h + 3, w + 2) * k4d_constant(m, c, 3, 2) +
                   x4d(b, c, h + 3, w + 3) * k4d_constant(m, c, 3, 3) +
                   x4d(b, c, h + 3, w + 4) * k4d_constant(m, c, 3, 4) +
                   x4d(b, c, h + 3, w + 5) * k4d_constant(m, c, 3, 5) +
                   x4d(b, c, h + 3, w + 6) * k4d_constant(m, c, 3, 6) +
                   x4d(b, c, h + 4, w + 0) * k4d_constant(m, c, 4, 0) +
                   x4d(b, c, h + 4, w + 1) * k4d_constant(m, c, 4, 1) +
                   x4d(b, c, h + 4, w + 2) * k4d_constant(m, c, 4, 2) +
                   x4d(b, c, h + 4, w + 3) * k4d_constant(m, c, 4, 3) +
                   x4d(b, c, h + 4, w + 4) * k4d_constant(m, c, 4, 4) +
                   x4d(b, c, h + 4, w + 5) * k4d_constant(m, c, 4, 5) +
                   x4d(b, c, h + 4, w + 6) * k4d_constant(m, c, 4, 6) +
                   x4d(b, c, h + 5, w + 0) * k4d_constant(m, c, 5, 0) +
                   x4d(b, c, h + 5, w + 1) * k4d_constant(m, c, 5, 1) +
                   x4d(b, c, h + 5, w + 2) * k4d_constant(m, c, 5, 2) +
                   x4d(b, c, h + 5, w + 3) * k4d_constant(m, c, 5, 3) +
                   x4d(b, c, h + 5, w + 4) * k4d_constant(m, c, 5, 4) +
                   x4d(b, c, h + 5, w + 5) * k4d_constant(m, c, 5, 5) +
                   x4d(b, c, h + 5, w + 6) * k4d_constant(m, c, 5, 6) +
                   x4d(b, c, h + 6, w + 0) * k4d_constant(m, c, 6, 0) +
                   x4d(b, c, h + 6, w + 1) * k4d_constant(m, c, 6, 1) +
                   x4d(b, c, h + 6, w + 2) * k4d_constant(m, c, 6, 2) +
                   x4d(b, c, h + 6, w + 3) * k4d_constant(m, c, 6, 3) +
                   x4d(b, c, h + 6, w + 4) * k4d_constant(m, c, 6, 4) +
                   x4d(b, c, h + 6, w + 5) * k4d_constant(m, c, 6, 5) +
                   x4d(b, c, h + 6, w + 6) * k4d_constant(m, c, 6, 6);
        }
        y4d(b, m, h, w) = acc;
    }


#undef y4d
#undef x4d
#undef k4d_constant
}


cudaStream_t stream[STREAM_NUM];



__host__ void GPUInterface::conv_forward_gpu_prolog(const float *__restrict__ host_y,
                                                    const float *__restrict__ host_x,
                                                    const float *__restrict__ host_k,
                                                    float **device_y_ptr, float **device_x_ptr,
                                                    float **device_k_ptr, const int B, const int M,
                                                    const int C, const int H, const int W,
                                                    const int K) {
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const unsigned int xSize = B * C * H * W * sizeof(float);
    const unsigned int ySize = B * M * H_out * W_out * sizeof(float);
    const unsigned int kSize = M * C * K * K * sizeof(float);

    const int xStreamSize = B * C * H * W / STREAM_NUM;
    const int xStreamByte = xStreamSize * sizeof(float);
    const int yStreamSize = B * M * H_out * W_out / STREAM_NUM;
    const int yStreamByte = yStreamSize * sizeof(float);

    cudaMalloc((void **)device_x_ptr, xSize);
    cudaMalloc((void **)device_y_ptr, ySize);


    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamCreate(&stream[i]);
    }


    cudaMemcpyToSymbol(Kernel, host_k, kSize, 0, cudaMemcpyHostToDevice);
    for (int i = 0; i < STREAM_NUM; i++) {
        int offset = i * xStreamSize;
        cudaMemcpyAsync((void *)&(*device_x_ptr)[offset], (void *)&(host_x[offset]), xStreamByte,
                        cudaMemcpyHostToDevice, stream[i]);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *__restrict__ device_y,
                                             const float *__restrict__ device_x,
                                             const float *__restrict__ device_k, const int B,
                                             const int M, const int C, const int H, const int W,
                                             const int K) {
    // Set the kernel dimensions and call the kernel
    if (K != 7) {
        printf("ERROR: this function is designed for only K = 7\n");
    }
    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;
    // const int H_grid = ceil(H_out / float(TILE_WIDTH));
    // const int W_grid = ceil(W_out / float(TILE_WIDTH));

    const int streamSize = B / STREAM_NUM;

    // printf("B: %d\n", B);
    // printf("M: %d\n", M);
    // printf("C: %d\n", C);
    // printf("K: %d\n", K);
    // printf("M * C * K * K: %d\n", M * C * K * K);




    if (C == 1) {
        const int H_out = H - K + 1;
        const int W_out = W - K + 1;
        const int H_grid = ceil(H_out / float(TILE_WIDTH_C1));
        const int W_grid = ceil(W_out / float(TILE_WIDTH_C1));

        dim3 DimGrid(streamSize, M, H_grid * W_grid);
        dim3 DimBlock(TILE_WIDTH_C1, TILE_WIDTH_C1, 1);
        for (int i = 0; i < STREAM_NUM; i++) {
            int offset = i * streamSize;
            conv_forward_kernel_C1<<<DimGrid, DimBlock, 0, stream[i]>>>(device_y, device_x, B, M, C,
                                                                        H, W, K, offset);
        }
    } else if (C == 4) {
        const int H_out = H - K + 1;
        const int W_out = W - K + 1;
        const int H_grid = ceil(H_out / float(TILE_WIDTH_C4));
        const int W_grid = ceil(W_out / float(TILE_WIDTH_C4));

        dim3 DimGrid(streamSize, M, H_grid * W_grid);
        dim3 DimBlock(TILE_WIDTH_C4, TILE_WIDTH_C4, 1);
        for (int i = 0; i < STREAM_NUM; i++) {
            int offset = i * streamSize;
            conv_forward_kernel_C4<<<DimGrid, DimBlock, 0, stream[i]>>>(device_y, device_x, B, M, C,
                                                                        H, W, K, offset);
        }
    }

    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *__restrict__ host_y,
                                                    float *__restrict__ device_y,
                                                    float *__restrict__ device_x,
                                                    float *__restrict__ device_k, const int B,
                                                    const int M, const int C, const int H,
                                                    const int W, const int K) {
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const unsigned int ySize = B * M * H_out * W_out * sizeof(float);

    const int yStreamSize = B * M * H_out * W_out / STREAM_NUM;
    const int yStreamByte = yStreamSize * sizeof(float);


    for (int i = 0; i < STREAM_NUM; i++) {
        int offset = i * yStreamSize;
        cudaMemcpyAsync(&host_y[offset], &device_y[offset], yStreamByte, cudaMemcpyDeviceToHost,
                        stream[i]);
    }

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
}


__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor
                  << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock
                  << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, "
                  << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z"
                  << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
                  << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z"
                  << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}




/*
// Tiled version of the code, the result is slower than baseline so I didn't use it

#include <cmath>
#include <iostream>

#include "gpu-new-forward.h"

#define TILE_WIDTH 26
#define KERNEL_WIDTH 7
#define BLOCK_WIDTH (TILE_WIDTH + ((int)KERNEL_WIDTH / 2) * 2)
#define C_MAX 4

__constant__ float Kernel[4096];


__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B,
                                    const int M, const int C, const int H, const int W,
                                    const int K) {

    __shared__ float x_shared[C_MAX * BLOCK_WIDTH * BLOCK_WIDTH];


    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) \
    y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_shared_3d(i2, i1, i0) \
    x_shared[(i2) * (BLOCK_WIDTH * BLOCK_WIDTH) + (i1) * (BLOCK_WIDTH) + i0]
#define k4d_constant(i3, i2, i1, i0) Kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    // Insert your GPU convolution kernel code here
    int W_grid = ceil(W_out / float(TILE_WIDTH));
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + tx;

    if (C == 1) {
        if ((w < W) && (h < H)) {
            x_shared_3d(0, ty, tx) = x4d(b, 0, h, w);
        } else {
            x_shared_3d(0, ty, tx) = 0.0f;
        }
    } else if (C == 4) {
        if ((w < W) && (h < H)) {
            x_shared_3d(0, ty, tx) = x4d(b, 0, h, w);
            x_shared_3d(1, ty, tx) = x4d(b, 1, h, w);
            x_shared_3d(2, ty, tx) = x4d(b, 2, h, w);
            x_shared_3d(3, ty, tx) = x4d(b, 3, h, w);
        } else {
            x_shared_3d(0, ty, tx) = 0.0f;
            x_shared_3d(1, ty, tx) = 0.0f;
            x_shared_3d(2, ty, tx) = 0.0f;
            x_shared_3d(3, ty, tx) = 0.0f;
        }
    }
    __syncthreads();



    if ((tx < TILE_WIDTH) && (ty < TILE_WIDTH)) {
        float acc = 0.0f;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += x_shared_3d(c, ty + p, tx + q) * k4d_constant(m, c, p, q);
                }
            }
        }
        if ((w < W_out) && (h < H_out)) {
            y4d(b, m, h, w) = acc;
        }
    }


#undef y4d
#undef x4d
#undef k4d
#undef k4d_shared
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x,
                                                    const float *host_k, float **device_y_ptr,
                                                    float **device_x_ptr, float **device_k_ptr,
                                                    const int B, const int M, const int C,
                                                    const int H, const int W, const int K) {
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const unsigned int xSize = B * C * H * W * sizeof(float);
    const unsigned int ySize = B * M * H_out * W_out * sizeof(float);
    const unsigned int kSize = M * C * K * K * sizeof(float);

    cudaMalloc((void **)device_x_ptr, xSize);
    cudaMalloc((void **)device_y_ptr, ySize);
    cudaMalloc((void **)device_k_ptr, kSize);

    // std::cout << "Successfully allocate cuda memory" << std::endl;


    cudaMemcpy(*device_x_ptr, (void *)host_x, xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, (void *)host_k, kSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Kernel, host_k, kSize, 0, cudaMemcpyHostToDevice);

    // std::cout << "Successfully copy data to cuda memory" << std::endl;




    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x,
                                             const float *device_k, const int B, const int M,
                                             const int C, const int H, const int W, const int K) {
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int H_grid = ceil(H_out / float(TILE_WIDTH));
    const int W_grid = ceil(W_out / float(TILE_WIDTH));

    // printf("M: %d\n", M);
    // printf("C: %d\n", C);
    // printf("K: %d\n", K);
    // printf("M * C * K * K: %d\n", M * C * K * K);


    dim3 DimGrid(B, M, H_grid * W_grid);
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    conv_forward_kernel<<<DimGrid, DimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);


    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x,
                                                    float *device_k, const int B, const int M,
                                                    const int C, const int H, const int W,
                                                    const int K) {
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const unsigned int ySize = B * M * H_out * W_out * sizeof(float);
    cudaMemcpy(host_y, device_y, ySize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
}


__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor
                  << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock
                  << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, "
                  << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z"
                  << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
                  << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z"
                  << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}

*/