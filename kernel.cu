#include <stdio.h>
#include "wrapper.h"
#include <iostream>
#include <stdlib.h>
#include <cuda.h>

//int *hA, *hB;  // host input data
//int *hC;  // host output data

__global__ void vecAddition(int *A,int *B,int *C,int N)
{
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   C[id] = A[id] + B[id]; 
}

int demo_kernel()
{
   // host input data
   int *hA, *hB;
   // host output data
   int *hC; 

   // Vector size
   int n=10000000;
   // Vector size in bytes
   int nBytes = n*sizeof(int);
  
   int thread_block, num_blocks; 

   // Allocate memory for host input data
   hA = (int *)malloc(nBytes);
   hB = (int *)malloc(nBytes);

   // Allocate memory for host output data
   hC = (int *)malloc(nBytes);

   // Initialize device input and output data
   int *dA,*dB,*dC;

   // Number of threads per block
   thread_block=512;

   // Number of blocks
   num_blocks = n/thread_block;

   // Vector initialization
   for(int i=0;i<n;i++){
      hA[i]=i;
      hB[i]=i;
   }

   // Allocate memory on GPU
   cudaMalloc((void **)&dA, n*sizeof(int));
   cudaMalloc((void **)&dB, n*sizeof(int));
   cudaMalloc((void **)&dC, n*sizeof(int));

   // Copy host input data from host to device
   cudaMemcpy(dA, hA, n*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(dB, hB, n*sizeof(int), cudaMemcpyHostToDevice);

   // Kernel launch
   vecAddition<<<num_blocks,thread_block>>>(dA,dB,dC,n);

   // Synchronize
   cudaDeviceSynchronize()
   
   // Error checking
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess)
   {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
   }

   // Copy output data from device to host
   cudaMemcpy(hC, dC, n*sizeof(int), cudaMemcpyDeviceToHost);

   // Free device memory
   cudaFree(dA);
   cudaFree(dB);
   cudaFree(dC);

   // Free host memory
   free(hA);
   free(hB);
   free(hC);

   return 0;
}

