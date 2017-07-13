
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

void InitMatrxVect(float* matrx, float* vec,float* res, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			//matrx[i*cols + j] = i+j;
			matrx[i*cols + j] = i;//sqrt(cos(i*pow(i,j)+log(sin(j+i)))+rows*atan(j)/cols+atan(cos(pow(cols,i))));
			//printf(" [%.2f] ", matrx[i*cols + j]);
		}
		//printf("\n");
	}
	for (int i = 0; i < cols; i++)
	{
		vec[i] = i;
		//printf(" [%.2f] \n", vec[i]);
	}
	for (int i = 0; i < rows; i++)
	{
		res[i] = 0;
	}
}

void Mult(float *mat, float *vec, float *res, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		res[i] = 0;
		for (int j = 0; j < cols; j++)
		{
			res[i] += mat[i*cols + j] * vec[j];
		}
		//printf("[%.2f] \n", res[i]);
	}
}

__global__ void gpuMul(float *mat, float *vec, float *res, int rows,int cols)
{
	float sum = 0;
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id < rows) {
		for (int i = 0; i < cols; i++)
		{
			sum += mat[id*cols + i] * vec[i];
		}
		res[id] = sum;
	}
}
int main()
{	
	const int M = 3000, N = 3000;
	float* Matrx,*Vec,*Res,*Res1;
	float time;
	float *dev_mat, *dev_vec, *dev_res;
	int deviceCount,row = M,col = N;

	cudaEvent_t  start, stop;
	cudaDeviceProp devProp;
	cudaError_t cudaStatus;
	cudaStatus = cudaGetDeviceCount(&deviceCount);
	
	if (cudaStatus != cudaSuccess) {
		printf("Your PC hasn't any NVIDIA GPU devices");
		return 1;
	}
	printf("Found %d devices\n", deviceCount);
	for (int device = 0; device < deviceCount; device++)
	{	cudaGetDeviceProperties(&devProp, device);
		printf("Device %d\n", device);
		printf("Compute capability     : %d.%d\n", devProp.major, devProp.minor);
		printf("Name                   : %s\n", devProp.name);
		printf("Total Global Memory    : %1.0f MB\n", devProp.totalGlobalMem / (1024.f*1024.f));
		printf("Shared memory per block: %1.0f KB\n", devProp.sharedMemPerBlock / 1024.f);
		printf("Multiprocessor count    : %d\n", devProp.multiProcessorCount);
		printf("Registers per block    : %d\n", devProp.regsPerBlock);
		printf("Warp size              : %d\n", devProp.warpSize);
		printf("Max threads per block  : %d\n", devProp.maxThreadsPerBlock);
		printf("Total constant memory  : %1.0f KB\n", devProp.totalConstMem / 1024.f);
		printf("Device Overlap         : %d \n", devProp.deviceOverlap);
		printf("Max thread dimensions:  (%d, %d, %d)\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
		printf("Max grid dimensions:  (%d, %d, %d)\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
	}
	Matrx = new float[M*N];
	Vec = new float[N];
	Res = new float[M];
	Res1 = new float[M];
	
	InitMatrxVect(Matrx, Vec,Res, M, N);
	float fTimeS = clock();
	Mult(Matrx, Vec, Res, M, N);
	float fTimeF = clock();
	printf("Total time: %f \n",(fTimeF-fTimeS)/CLOCKS_PER_SEC);
	
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMalloc((void**)&dev_mat, M*N * sizeof(float));
	cudaMalloc((void**)&dev_vec, N * sizeof(float));
	cudaMalloc((void**)&dev_res, M * sizeof(float));
	//cudaMalloc((void**)row, sizeof(int));
	//cudaMalloc((void**)col, sizeof(int));
	

	cudaMemcpy(dev_mat,Matrx,M*N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vec,Vec,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_res,Res,M*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(&col, &N,sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&row, &M,sizeof(int), cudaMemcpyHostToDevice); 
	dim3 blocks(M*N, 1, 1);
	dim3 threads(32, 1, 1);
	cudaEventRecord(start,0);
	gpuMul<<<blocks, threads >> > (dev_mat, dev_vec, dev_res, M, N);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaMemcpy(Res1,dev_res, M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_mat);
	cudaFree(dev_vec);
	cudaFree(dev_res);
	cudaEventElapsedTime(&time, start, stop);
	printf("CUDA time: %f\n",time / 1000);
	printf("R[2000] = %f\n", Res[2000]);
	printf("R1[2000] = %f\n", Res1[2000]);
	getchar();
	return 0;
}