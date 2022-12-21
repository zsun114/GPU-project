#include <stdio.h>
#include <stdlib.h>
#include "support.h"
#define TILE_SIZE 16



__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Cvalue = 0;
    for (int p = 0; p < ((k - 1) / TILE_SIZE) + 1; ++p) {
        if ((Row < m) && (p * TILE_SIZE + tx < k)) {
            ds_A[ty][tx] = A[Row * k + p * TILE_SIZE + tx];
        } else {
            ds_A[ty][tx] = 0.0;
        }
        if ((p * TILE_SIZE + ty < k) && (Col < n)) {
            ds_B[ty][tx] = B[(p * TILE_SIZE + ty) * n + Col];
        } else {
            ds_B[ty][tx] = 0.0;
        }
        __syncthreads();
        if (Row < m && Col < n) {
            for (int i = 0; i < TILE_SIZE; ++i) {
                Cvalue += ds_A[ty][i] * ds_B[i][tx];
            }
        }
        __syncthreads();
    }
    if (Row < m && Col < n) {
        C[Row * n + Col] = Cvalue;
    }




    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C) {
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid(((n - 1) / BLOCK_SIZE) + 1, ((m - 1) / BLOCK_SIZE) + 1, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
    mysgemm<<<DimGrid, DimBlock>>>(m, n, k, A, B, C);
    /*************************************************************************/
}

int main (int argc, char *argv[]) {

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int num_stream=4;
    const int n = 4 * 64 * TILE_SIZE * num_stream;
    const int streamSize = n / num_stream;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...");
    fflush(stdout);


    cudaStream_t stream0, stream1, stream2, stream3;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream2);

    float *A_h, *B_h, *C_h;
    float *A_d0, *B_d0, *C_d0;
    float *A_d1, *B_d1, *C_d1;
    float *A_d2, *B_d2, *C_d2;
    float *A_d3, *B_d3, *C_d3;
    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    } else if (argc == 2) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    } else if (argc == 4) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
               "\n    Usage: ./sgemm-tiled                # All matrices are 1000 x 1000"
               "\n    Usage: ./sgemm-tiled <m>            # All matrices are m x m"
               "\n    Usage: ./sgemm-tiled <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
               "\n");
        exit(0);
    }
    printf("    \nA: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
           matBrow, matBcol, matArow, matBcol);

    A_sz = matArow * matAcol;
    B_sz = matBrow * matBcol;
    C_sz = matArow * matBcol;

    A_h = (float *) malloc(sizeof(float) * A_sz);
    for (unsigned int i = 0; i < A_sz; i++) { A_h[i] = (rand() % 100) / 100.00; }

    B_h = (float *) malloc(sizeof(float) * B_sz);
    for (unsigned int i = 0; i < B_sz; i++) { B_h[i] = (rand() % 100) / 100.00; }

    C_h = (float *) malloc(sizeof(float) * C_sz);


    // Allocate device variables ----------------------------------------------



    /*************************************************************************/
    //INSERT CODE HERE
    cudaMalloc((void **) &A_d0, A_sz * sizeof(float));
    cudaMalloc((void **) &B_d0, B_sz * sizeof(float));
    cudaMalloc((void **) &C_d0, C_sz * sizeof(float));
    cudaMalloc((void **) &A_d1, A_sz * sizeof(float));
    cudaMalloc((void **) &B_d1, B_sz * sizeof(float));
    cudaMalloc((void **) &C_d1, C_sz * sizeof(float));
    cudaMalloc((void **) &A_d2, A_sz * sizeof(float));
    cudaMalloc((void **) &B_d2, B_sz * sizeof(float));
    cudaMalloc((void **) &C_d2, C_sz * sizeof(float));
    cudaMalloc((void **) &A_d3, A_sz * sizeof(float));
    cudaMalloc((void **) &B_d3, B_sz * sizeof(float));
    cudaMalloc((void **) &C_d3, C_sz * sizeof(float));
    cudaHostAlloc((void**)&A_h, A_sz * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&B_h, B_sz * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&C_h, C_sz * sizeof(float), cudaHostAllocDefault);


    /*************************************************************************/


    /*************************************************************************/
    //INSERT CODE HERE
    /*cudaMemcpy(A_d, A_h, A_sz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_sz*sizeof(float), cudaMemcpyHostToDevice);*/
    for (int i = 0; i < num_stream; i++) {
        cudaMemcpyAsync(A_d0, A_h + i, A_sz * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(B_d0, B_h + i, B_sz * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(A_d1, A_h + i + n, A_sz * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(B_d1, B_h + i + n, B_sz * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(A_d2, A_h + i + n, A_sz * sizeof(float), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(B_d2, B_h + i + n, B_sz * sizeof(float), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(A_d3, A_h + i + n, A_sz * sizeof(float), cudaMemcpyHostToDevice, stream3);
        cudaMemcpyAsync(B_d3, B_h + i + n, B_sz * sizeof(float), cudaMemcpyHostToDevice, stream3);
        mysgemm<<<streamSize/TILE_SIZE, TILE_SIZE, 0, stream0>>>(matArow, matBcol, matBrow, A_d0, B_d0, C_d0);
        mysgemm<<<streamSize/TILE_SIZE, TILE_SIZE, 0, stream1>>>(matArow, matBcol, matBrow, A_d1, B_d1, C_d1);
        mysgemm<<<streamSize/TILE_SIZE, TILE_SIZE, 0, stream2>>>(matArow, matBcol, matBrow, A_d2, B_d2, C_d2);
        mysgemm<<<streamSize/TILE_SIZE, TILE_SIZE, 0, stream3>>>(matArow, matBcol, matBrow, A_d3, B_d3, C_d3);
        cudaMemcpyAsync(C_h + i, C_d0, C_sz * sizeof(float), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(C_h + i + n, C_d1, C_sz * sizeof(float), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(C_h + i + n, C_d2, C_sz * sizeof(float), cudaMemcpyDeviceToHost, stream2);
        cudaMemcpyAsync(C_h + i + n, C_d3, C_sz * sizeof(float), cudaMemcpyDeviceToHost, stream3);
}
    /*************************************************************************/

    // Launch kernel using standard sgemm interface ---------------------------
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    /*************************************************************************/
    //INSERT CODE HERE
    /*cudaMemcpy(C_h, C_d, C_sz*sizeof(float), cudaMemcpyDeviceToHost);*/
    /*************************************************************************/


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken: %f ms\n", elapsedTime);

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, matArow, matAcol, matBcol);


    // Free memory ------------------------------------------------------------

    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFreeHost(C_h);

    /*************************************************************************/
    //INSERT CODE HERE
    cudaFree(A_d0);
    cudaFree(B_d0);
    cudaFree(C_d0);
    cudaFree(A_d1);
    cudaFree(B_d1);
    cudaFree(C_d1);
    cudaFree(A_d2);
    cudaFree(B_d2);
    cudaFree(C_d2);
    cudaFree(A_d3);
    cudaFree(B_d3);
    cudaFree(C_d3);
    /*************************************************************************/

    return 0;
}
