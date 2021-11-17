#include <cmath>
#include <iostream>

#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define STREAM_NUM 100

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cout << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
    }
    return result;
}

__constant__ float Kernel[4096];


__global__ void conv_forward_kernel(float *__restrict__ y, const float *__restrict__ x, const int B, const int M,
                                    const int C, const int H, const int W, const int K, const int offset) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

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

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d_constant(i3, i2, i1, i0) Kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(W_out / float(TILE_WIDTH));
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int b = blockIdx.x + offset;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + tx;


    if ((w < W_out) && (h < H_out)) {
        float acc = 0.0f;
        // Unroll for loop
        for (int c = 0; c < C; c++) {
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



__host__ void GPUInterface::conv_forward_gpu_prolog(const float *__restrict__ host_y, const float *__restrict__ host_x,
                                                    const float *__restrict__ host_k, float **device_y_ptr,
                                                    float **device_x_ptr, float **device_k_ptr, const int B,
                                                    const int M, const int C, const int H, const int W, const int K) {
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
    // std::cout << "Successfully allocate device memory\n";


    for (int i = 0; i < STREAM_NUM; i++) {
        checkCuda(cudaStreamCreate(&stream[i]));
    }
    // std::cout << "Successfully create cuda Streams\n";


    cudaMemcpyToSymbol(Kernel, host_k, kSize, 0, cudaMemcpyHostToDevice);
    // std::cout << "Successfully copy kernel to device\n";


    // cudaMemcpy(*device_x_ptr, (void *)host_x, xSize, cudaMemcpyHostToDevice);

    for (int i = 0; i < STREAM_NUM; i++) {
        int offset = i * xStreamSize;
        checkCuda(cudaMemcpyAsync((void *)&(*device_x_ptr)[offset], (void *)&(host_x[offset]), xStreamByte,
                                  cudaMemcpyHostToDevice, stream[i]));
    }
    // std::cout << "Successfully copy data to device\n";
}


__host__ void GPUInterface::conv_forward_gpu(float *__restrict__ device_y, const float *__restrict__ device_x,
                                             const float *__restrict__ device_k, const int B, const int M, const int C,
                                             const int H, const int W, const int K) {
    // Set the kernel dimensions and call the kernel
    if (K != 7) {
        printf("ERROR: this function is designed for only K = 7\n");
    }
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int H_grid = ceil(H_out / float(TILE_WIDTH));
    const int W_grid = ceil(W_out / float(TILE_WIDTH));

    const int streamSize = B / STREAM_NUM;

    // printf("B: %d\n", B);
    // printf("M: %d\n", M);
    // printf("C: %d\n", C);
    // printf("K: %d\n", K);
    // printf("M * C * K * K: %d\n", M * C * K * K);

    // dim3 DimGrid(B, M, H_grid * W_grid);
    // dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // conv_forward_kernel<<<DimGrid, DimBlock>>>(device_y, device_x, B, M, C, H, W, K);


    dim3 DimGrid(streamSize, M, H_grid * W_grid);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    for (int i = 0; i < STREAM_NUM; i++) {
        int offset = i * streamSize;
        conv_forward_kernel<<<DimGrid, DimBlock, 0, stream[i]>>>(device_y, device_x, B, M, C, H, W, K, offset);
    }

    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *__restrict__ host_y, float *__restrict__ device_y,
                                                    float *__restrict__ device_x, float *__restrict__ device_k,
                                                    const int B, const int M, const int C, const int H, const int W,
                                                    const int K) {
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const unsigned int ySize = B * M * H_out * W_out * sizeof(float);

    const int yStreamSize = B * M * H_out * W_out / STREAM_NUM;
    const int yStreamByte = yStreamSize * sizeof(float);


    // cudaMemcpy(host_y, device_y, ySize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < STREAM_NUM; i++) {
        int offset = i * yStreamSize;
        cudaMemcpyAsync(&host_y[offset], &device_y[offset], yStreamByte, cudaMemcpyDeviceToHost, stream[i]);
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
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1]
                  << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1]
                  << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
