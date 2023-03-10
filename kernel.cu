#include <stdio.h>

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


