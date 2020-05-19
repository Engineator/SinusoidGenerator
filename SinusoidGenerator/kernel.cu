
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FREQUENCY 1000 // Frequency in Hz
#define SAMPLE_RATE (8 * FREQUENCY)
#define N_SECONDS 4
#define N_SAMPLES (N_SECONDS * SAMPLE_RATE)

#define BLOCK_SIZE 1024

cudaError_t cudaSinusoidGenerator(double *signal, unsigned int size);

__global__ void sinusoidGeneratorKernel(double *signal)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    signal[i] = sin(2*3.14159*FREQUENCY*((double)i/SAMPLE_RATE)); 
    //signal[i] = (double)i;
}

int main()
{

    double * signal = (double *)malloc(N_SAMPLES * sizeof(double));

    for (int i = 0; i < N_SAMPLES; i++)
    {
        signal[i] = (double)i;
    }

    //// Generate Sinusoid
    cudaError_t cudaStatus = cudaSinusoidGenerator(signal,N_SAMPLES);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSinusoidGenerator failed!");
        return 1;
    }

    FILE* outFile;

    outFile = fopen("signal_out.txt", "w+");
    for (int i = 0; i < N_SAMPLES; i++)
    {
        fprintf(outFile, "%f\n", signal[i]);
    }
    fclose(outFile);

    free(signal);
    

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

 // Helper function for using CUDA to generate a sinusoid
cudaError_t cudaSinusoidGenerator(double *signal, unsigned int size)
{
    double *dev_signal = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_signal, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Launch a kernel on the GPU blocking out signal in grid
    dim3 dimBlock(BLOCK_SIZE, 1);
    dim3 dimGrid(N_SAMPLES/dimBlock.x, 1);
    sinusoidGeneratorKernel<<<dimGrid, dimBlock>>>(dev_signal);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(signal, dev_signal, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_signal);
    
    return cudaStatus;

}
