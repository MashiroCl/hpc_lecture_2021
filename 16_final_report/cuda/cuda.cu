#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdlib.h>
using namespace std;
#define M 32

__global__ void matrix(int N, float *A, float *B, float* C){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float s1[M][M+1];
    __shared__ float s2[M][M+1];

    float sum=0;
    for(int i =0;i<N/M;i++)
{
    s1[ty][tx]=A[(by*M+ty)*N+(i*M+tx)];
    s2[ty][tx]=B[(bx*M+tx)+(i*M+ty)*N];
    __syncthreads();

   for (int k=0;k<M;k++)
      sum+=s1[ty][k]*s2[k][tx];
    __syncthreads();

}
       C[(by*M+ty)*N+bx*M+tx]=sum;
}


int main(int argc, char** argv) {
  const int N = 256;
    vector<float> A(N*N);
    vector<float> B(N*N);
    vector<float> C(N*N, 0);

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            A[N*i+j] = drand48();
            B[N*i+j] = drand48();
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * N * N);
    cudaMalloc(&d_B, sizeof(float) * N * N);
    cudaMalloc(&d_C, sizeof(float) * N * N);

    cudaMemcpy(d_A, &A[0], sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &B[0], sizeof(float) * N * N, cudaMemcpyHostToDevice);
  
  auto tic = chrono::steady_clock::now();

  int GRID_SIZE=(N+M-1)/M;
  dim3 grid(GRID_SIZE,GRID_SIZE);
  dim3 block(M,M);
  matrix<<<grid,block>>>(N,d_A,d_B,d_C);
  cudaDeviceSynchronize();

  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc-tic).count();
  
  cudaMemcpy(&C[0], d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);

  printf("N    : %d\n",N);
  printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
  printf("error: %lf\n",err/N/N);
}
