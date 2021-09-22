#include <windows.h>
#include <string>
#include <iostream>
#include <thread>
#include "Dependencies\glew\glew.h"
#include "Dependencies\freeglut\freeglut.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
using namespace std;

GLfloat r = 0.0f, g = 0.0f, b = 0.0f;
const int WIDTH = 1024, HEIGHT = 768;
int cell[WIDTH][HEIGHT];
// injection circle points x, y
int cx[9];
int cy[9];
int cx1[9];
int cy1[9];
int cx2[9];
int cy2[9];
int cx3[9];
int cy3[9];
int cx4[9];
int cy4[9];
int cx5[9];
int cy5[9];
int cx6[9]; //injectParallel() moveParallel() moveWithCuda() moveKernel()
int cy6[9];
int cx11[9];
int cy11[9];
int cx21[9];
int cy21[9];
int cx31[9];
int cy31[9];
int cx41[9];
int cy41[9];
int cx51[9];
int cy51[9];
// storage temporary data
int s0, s1, s2, s3, s4, s5, s6, s7, s8;
int s[] = { s0, s1, s2, s3, s4, s5, s6, s7, s8 };
int s10, s11, s12, s13, s14, s15, s16, s17, s18;
int si[9] = { s10, s11, s12, s13, s14, s15, s16, s17, s18 };
int sii[9];
int m[9];
int m1[9];
int m2[9];
int m3[9];
int m4[9];
int m5[9];
int m6[9]; //injectParallel() moveParallel() moveWithCuda() moveKernel()



void init() {
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(-0.5f, WIDTH - 0.5f, -0.5f, HEIGHT - 0.5f);
}

void fun(void)
{
    cout << "Exiting because of outside screen or memery overstack";
}

// receive temporary data from stored in sii[] after injectionii() 
void moveii() {
	for (int i = 1; i < 9; i++) {
		if (cx[i] > 1020 || cy[i] > 760 || cx[i] < 0 || cy[i] < 0) {
			atexit(fun);
			_Exit(10);
		}
		if (cell[cx[i]][cy[i]] == 4) {
			cell[cx[i]][cy[i]] = sii[i];
			switch (i) {
			case 1: {
				//cx[1] = x;
				cy[i] = cy[i] - 1;
			}
				  break;
			case 2: {
				cx[i] = cx[i] + 1;
				cy[i] = cy[i] - 1;
			}
				  break;
			case 3: {
				cx[i] = cx[i] + 1;
				//cy[3] = y;
			}
				  break;
			case 4: {
				cx[i] = cx[i] + 1;
				cy[i] = cy[i] + 1;
			}
				  break;
			case 5: {
				//cx[5] = x;
				cy[i] = cy[i] + 1;
			}
				  break;
			case 6: {
				cx[i] = cx[i] - 1;
				cy[i] = cy[i] + 1;
			}
				  break;
			case 7: {
				cx[i] = cx[i] - 1;
				//cy[7] = y;
			}
				  break;
			case 8: {
				cx[i] = cx[i] - 1;
				cy[i] = cy[i] - 1;
			}
				  break;
			}//end switch
			if (cx[i] > 1020 || cy[i] > 760 || cx[i] < 0 || cy[i] < 0) {
				atexit(fun);
				_Exit(10);
			}
			sii[i] = cell[cx[i]][cy[i]];
			cell[cx[i]][cy[i]] = 4;
		}// endif
	}
}

// injectionCuda() move_m5() moveWithCuda5(cx11, cy11, size, i) moveKernal_1...movekernal_5(cx5[0],cy5[0]) cx21,cy21..cx51[0]cy51[0]
__global__ void moveKernel_1(int* cx1, int* cy1) {
	int i = threadIdx.x;
	//int j = threadIdx.y;
	//cx[i] = cx[i];
	cy1[i] = cy1[i] - 1;
}

// injectionCuda() move_m5() moveWithCuda5(cx11, cy11, size, i) moveKernal_1...movekernal_5(cx5[0],cy5[0]) cx21,cy21..cx51[0]cy51[0]
__global__ void moveKernel_2(int* cx2, int* cy2) {
	int i = threadIdx.x;
	//int j = threadIdx.y;
	cx2[i] = cx2[i] + 1;
	cy2[i] = cy2[i] - 1;
}

// injectionCuda() move_m5() moveWithCuda5(cx11, cy11, size, i) moveKernal_1...movekernal_5(cx5[0],cy5[0]) cx21,cy21..cx51[0]cy51[0]
__global__ void moveKernel_3(int* cx3, int* cy3) {
	int i = threadIdx.x;
	//int j = threadIdx.y;
	cx3[i] = cx3[i] + 1;
	//cy[j] = cy[j];
}

// injectionCuda() move_m5() moveWithCuda5(cx11, cy11, size, i) moveKernal_1...movekernal_5(cx5[0],cy5[0]) cx21,cy21..cx51[0]cy51[0]
__global__ void moveKernel_4(int* cx4, int* cy4) {
	int i = threadIdx.x;
	//int j = threadIdx.y;
	cx4[i] = cx4[i] + 1;
	cy4[i] = cy4[i] + 1;
}

// injectionCuda() move_m5() moveWithCuda5(cx11, cy11, size, i) moveKernal_1...movekernal_5(cx5[0],cy5[0]) cx21,cy21..cx51[0]cy51[0]
__global__ void moveKernel_5(int* cx5, int* cy5) {
	int i = threadIdx.x;
	//int j = threadIdx.y;
	//cx[i] = cx[i];
	cy5[i] = cy5[i] + 1;
}

// moveWithCuda(cx,cy,size) moveKernel(cx,cy) injectParallel() moveParallel() cx6[i]cy6[i] m6[i]
__global__ void moveKernel(int* cx, int* cy) {
	/*extern __shared__ int bothBuffers[];
	int* ss0 = &bothBuffers[0];
	int* ss1 = &bothBuffers[1];
	int* ss2 = &bothBuffers[2];
	int* ss3 = &bothBuffers[3];
	int* ss4 = &bothBuffers[4];
	int* ss5 = &bothBuffers[5];*/
	int i = threadIdx.x;
	cx[i] = cx[i] + 1;
	cy[i] = cy[i] + 1;
}

// injectionCuda() move_m5() moveWithCuda5(cx11, cy11, size, i) moveKernal_1...movekernal_5(cx5[0],cy5[0]) cx21,cy21..cx51[0]cy51[0]
cudaError_t moveWithCuda5(int* cx, int* cy, unsigned int size, int i) {
	int* dev_cx = 0;
	int* dev_cy = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cx, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cy, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_cx, cx, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_cy, cy, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	switch (i) {
	case 1:
		moveKernel_1 << <1, size >> > (dev_cx, dev_cy);
		break;
	case 2:
		moveKernel_2 << <1, size >> > (dev_cx, dev_cy);
		break;
	case 3:
		moveKernel_3 << <1, size >> > (dev_cx, dev_cy);
		break;
	case 4:
		moveKernel_4 << <1, size >> > (dev_cx, dev_cy);
		break;
	case 5:
		moveKernel_5 << <1, size >> > (dev_cx, dev_cy);
		break;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(cx, dev_cx, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cy, dev_cy, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_cx);
	cudaFree(dev_cy);

	return cudaStatus;
}

// moveWithCuda(cx,cy,size) moveKernel(cx,cy) injectParallel() moveParallel() cx6[i]cy6[i] m6[i]
cudaError_t moveWithCuda(int* cx, int* cy, unsigned int size) {
	int* dev_cx = 0;
	int* dev_cy = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cx, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cy, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_cx, cx, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_cy, cy, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	moveKernel << <1, size >> > (dev_cx, dev_cy);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(cx, dev_cx, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cy, dev_cy, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_cx);
	cudaFree(dev_cy);

	return cudaStatus;
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
cudaError_t moveWithCuda_1(int* cx, int* cy, unsigned int size) {
	int* dev_cx = 0;
	int* dev_cy = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cx, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cy, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_cx, cx, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_cy, cy, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	moveKernel_1 << <1, size >> > (dev_cx, dev_cy);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(cx, dev_cx, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cy, dev_cy, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_cx);
	cudaFree(dev_cy);

	return cudaStatus;
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
cudaError_t moveWithCuda_2(int* cx, int* cy, unsigned int size) {
	int* dev_cx = 0;
	int* dev_cy = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cx, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cy, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_cx, cx, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_cy, cy, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	moveKernel_2 << <1, size >> > (dev_cx, dev_cy);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(cx, dev_cx, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cy, dev_cy, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_cx);
	cudaFree(dev_cy);

	return cudaStatus;
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
cudaError_t moveWithCuda_3(int* cx, int* cy, unsigned int size) {
	int* dev_cx = 0;
	int* dev_cy = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cx, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cy, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_cx, cx, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_cy, cy, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	moveKernel_3 << <1, size >> > (dev_cx, dev_cy);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(cx, dev_cx, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cy, dev_cy, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_cx);
	cudaFree(dev_cy);

	return cudaStatus;
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
cudaError_t moveWithCuda_4(int* cx, int* cy, unsigned int size) {
	int* dev_cx = 0;
	int* dev_cy = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cx, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cy, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_cx, cx, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_cy, cy, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	moveKernel_4 << <1, size >> > (dev_cx, dev_cy);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(cx, dev_cx, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cy, dev_cy, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_cx);
	cudaFree(dev_cy);

	return cudaStatus;
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
cudaError_t moveWithCuda_5(int* cx, int* cy, unsigned int size) {
	int* dev_cx = 0;
	int* dev_cy = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cx, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cy, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_cx, cx, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_cy, cy, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	moveKernel_5 << <1, size >> > (dev_cx, dev_cy);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(cx, dev_cx, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cy, dev_cy, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_cx);
	cudaFree(dev_cy);

	return cudaStatus;
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
void move_1() {
	int size = 5;
	cell[cx1[0]][cy1[0]] = m1[0];
	cudaError_t cudaStatus = moveWithCuda_1(cx1, cy1, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveWithCuda failed!");
		_Exit(9);
	}
	if (cx1[0] > 1020 || cy1[0] > 760 || cx1[0] < 0 || cy1[0] < 0) {
		atexit(fun);
		_Exit(10);
	}
	m1[0] = cell[cx1[0]][cy1[0]];
	cell[cx1[0]][cy1[0]] = 4;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		_Exit(11);
	}
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
void move_2() {
	int size = 5;
	cell[cx2[0]][cy2[0]] = m2[0];
	cudaError_t cudaStatus = moveWithCuda_2(cx2, cy2, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveWithCuda failed!");
		_Exit(9);
	}
	if (cx2[0] > 1020 || cy2[0] > 760 || cx2[0] < 0 || cy2[0] < 0) {
		atexit(fun);
		_Exit(10);
	}
	m2[0] = cell[cx2[0]][cy2[0]];
	cell[cx2[0]][cy2[0]] = 4;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		_Exit(11);
	}
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
void move_3() {
	int size = 5;
	cell[cx3[0]][cy3[0]] = m3[0];
	cudaError_t cudaStatus = moveWithCuda_3(cx3, cy3, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveWithCuda failed!");
		_Exit(9);
	}
	if (cx3[0] > 1020 || cy3[0] > 760 || cx3[0] < 0 || cy3[0] < 0) {
		atexit(fun);
		_Exit(10);
	}
	m3[0] = cell[cx3[0]][cy3[0]];
	cell[cx3[0]][cy3[0]] = 4;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		_Exit(11);
	}
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
void move_4() {
	int size = 5;
	cell[cx4[0]][cy4[0]] = m4[0];
	cudaError_t cudaStatus = moveWithCuda_4(cx4, cy4, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveWithCuda failed!");
		_Exit(9);
	}
	if (cx4[0] > 1020 || cy4[0] > 760 || cx4[0] < 0 || cy4[0] < 0) {
		atexit(fun);
		_Exit(10);
	}
	m4[0] = cell[cx4[0]][cy4[0]];
	cell[cx4[0]][cy4[0]] = 4;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		_Exit(11);
	}
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
void move_5() {
	int size = 5;
	cell[cx5[0]][cy5[0]] = m5[0];
	cudaError_t cudaStatus = moveWithCuda_5(cx5, cy5, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveWithCuda failed!");
		_Exit(9);
	}
	if (cx5[0] > 1020 || cy5[0] > 760 || cx5[0] < 0 || cy5[0] < 0) {
		atexit(fun);
		_Exit(10);
	}
	m5[0] = cell[cx5[0]][cy5[0]];
	cell[cx5[0]][cy5[0]] = 4;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		_Exit(11);
	}
}

// moveWithCuda(cx,cy,size) moveKernel(cx,cy) injectParallel() moveParallel() cx6[i]cy6[i] m6[i] i=0..4 size=5
void moveParallel() {
	int size = 5;
	cudaError_t cudaStatus;
	for (int i = 0; i < 5; i++) {
		if (cx6[i] > 1020 || cy6[i] > 760 || cx6[i] < 0 || cy6[i] < 0) {
			atexit(fun);
			_Exit(10);
		}
		cell[cx6[i]][cy6[i]] = m6[i];
		cudaStatus = moveWithCuda(cx6, cy6, size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "moveWithCuda failed!");
			_Exit(9);
		}
		if (cx6[i] > 1020 || cy6[i] > 760 || cx6[i] < 0 || cy6[i] < 0) {
			atexit(fun);
			_Exit(10);
		}
		m6[i] = cell[cx6[i]][cy6[i]];
		cell[cx6[i]][cy6[i]] = 4;
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			_Exit(11);
		}
	}
}

void move5() {
	int size = 5;
	for (int i = 1; i < 6; i++) {
		switch (i) {
		case 1: {
			cell[cx1[0]][cy1[0]] = m1[0];
			cudaError_t cudaStatus = moveWithCuda5(cx1, cy1, size, i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moveWithCuda failed!");
				_Exit(9);
			}
			if (cx1[0] > 1020 || cy1[0] > 760 || cx1[0] < 0 || cy1[0] < 0) {
				atexit(fun);
				_Exit(10);
			}
			m1[0] = cell[cx1[0]][cy1[0]];
			cell[cx1[0]][cy1[0]] = 4;
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				_Exit(11);
			}
		}
			  break;
		case 2: {
			cell[cx2[0]][cy2[0]] = m2[0];
			cudaError_t cudaStatus = moveWithCuda5(cx2, cy2, size, i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moveWithCuda failed!");
				_Exit(9);
			}
			if (cx2[0] > 1020 || cy2[0] > 760 || cx2[0] < 0 || cy2[0] < 0) {
				atexit(fun);
				_Exit(10);
			}
			m2[0] = cell[cx2[0]][cy2[0]];
			cell[cx2[0]][cy2[0]] = 4;
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				_Exit(11);
			}
		}
			  break;
		case 3: {
			cell[cx3[0]][cy3[0]] = m3[0];
			cudaError_t cudaStatus = moveWithCuda5(cx3, cy3, size, i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moveWithCuda failed!");
				_Exit(9);
			}
			if (cx3[0] > 1020 || cy3[0] > 760 || cx3[0] < 0 || cy3[0] < 0) {
				atexit(fun);
				_Exit(10);
			}
			m3[0] = cell[cx3[0]][cy3[0]];
			cell[cx3[0]][cy3[0]] = 4;
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				_Exit(11);
			}
		}
			  break;
		case 4: {
			cell[cx4[0]][cy4[0]] = m4[0];
			cudaError_t cudaStatus = moveWithCuda5(cx4, cy4, size, i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moveWithCuda failed!");
				_Exit(9);
			}
			if (cx4[0] > 1020 || cy4[0] > 760 || cx4[0] < 0 || cy4[0] < 0) {
				atexit(fun);
				_Exit(10);
			}
			m4[0] = cell[cx4[0]][cy4[0]];
			cell[cx4[0]][cy4[0]] = 4;
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				_Exit(11);
			}
		}
			  break;
		case 5: {
			cell[cx5[0]][cy5[0]] = m5[0];
			cudaError_t cudaStatus = moveWithCuda5(cx5, cy5, size, i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moveWithCuda failed!");
				_Exit(9);
			}
			if (cx5[0] > 1020 || cy5[0] > 760 || cx5[0] < 0 || cy5[0] < 0) {
				atexit(fun);
				_Exit(10);
			}
			m5[0] = cell[cx5[0]][cy5[0]];
			cell[cx5[0]][cy5[0]] = 4;
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				_Exit(11);
			}
		}
			  break;
		}//end switch
	}
}

// injectionCuda() move_m5() moveWithCuda5(cx11, cy11, size, i) moveKernal_1...movekernal_5(cx5[0],cy5[0])
void move_m5() {
	int size = 5;
	for (int i = 1; i < 6; i++) {
		switch (i) {
		case 1: {
			cell[cx11[0]][cy11[0]] = m[1];
			cudaError_t cudaStatus = moveWithCuda5(cx11, cy11, size, i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moveWithCuda failed!");
				_Exit(9);
			}
			if (cx11[0] > 1020 || cy11[0] > 760 || cx11[0] < 0 || cy11[0] < 0) {
				atexit(fun);
				_Exit(10);
			}
			m[1] = cell[cx11[0]][cy11[0]];
			cell[cx11[0]][cy11[0]] = 4;
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				_Exit(11);
			}
		}
			  break;
		case 2: {
			cell[cx21[0]][cy21[0]] = m[2];
			cudaError_t cudaStatus = moveWithCuda5(cx21, cy21, size, i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moveWithCuda failed!");
				_Exit(9);
			}
			if (cx21[0] > 1020 || cy21[0] > 760 || cx21[0] < 0 || cy21[0] < 0) {
				atexit(fun);
				_Exit(10);
			}
			m[2] = cell[cx21[0]][cy21[0]];
			cell[cx21[0]][cy21[0]] = 4;
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				_Exit(11);
			}
		}
			  break;
		case 3: {
			cell[cx31[0]][cy31[0]] = m[3];
			cudaError_t cudaStatus = moveWithCuda5(cx31, cy31, size, i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moveWithCuda failed!");
				_Exit(9);
			}
			if (cx3[0] > 1020 || cy3[0] > 760 || cx3[0] < 0 || cy3[0] < 0) {
				atexit(fun);
				_Exit(10);
			}
			m[3] = cell[cx31[0]][cy31[0]];
			cell[cx31[0]][cy31[0]] = 4;
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				_Exit(11);
			}
		}
			  break;
		case 4: {
			cell[cx41[0]][cy41[0]] = m[4];
			cudaError_t cudaStatus = moveWithCuda5(cx41, cy41, size, i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moveWithCuda failed!");
				_Exit(9);
			}
			if (cx41[0] > 1020 || cy41[0] > 760 || cx41[0] < 0 || cy41[0] < 0) {
				atexit(fun);
				_Exit(10);
			}
			m[4] = cell[cx41[0]][cy41[0]];
			cell[cx41[0]][cy41[0]] = 4;
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				_Exit(11);
			}
		}
			  break;
		case 5: {
			cell[cx51[0]][cy51[0]] = m[5];
			cudaError_t cudaStatus = moveWithCuda5(cx51, cy51, size, i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moveWithCuda failed!");
				_Exit(9);
			}
			if (cx51[0] > 1020 || cy51[0] > 760 || cx51[0] < 0 || cy51[0] < 0) {
				atexit(fun);
				_Exit(10);
			}
			m[5] = cell[cx51[0]][cy51[0]];
			cell[cx51[0]][cy51[0]] = 4;
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				_Exit(11);
			}
		}
			  break;
		}//end switch
	}
}

// store temporary data in sii[] keep from Assignment2
void injectionii(int x, int y, int m, int num) {
	// multipoints injection
	/*x = (2 * m - 1) * x / 2;
	y = y / 2;*/
	// single injection
	x = x;
	y = y;

	cx[0] = x;
	cy[0] = y;
	cx[1] = x;
	cy[1] = y - 1;
	cx[2] = x + 1;
	cy[2] = y - 1;
	cx[3] = x + 1;
	cy[3] = y;
	cx[4] = x + 1;
	cy[4] = y + 1;
	cx[5] = x;
	cy[5] = y + 1;
	cx[6] = x - 1;
	cy[6] = y + 1;
	cx[7] = x - 1;
	cy[7] = y;
	cx[8] = x - 1;
	cy[8] = y - 1;

	if (cell[cx[0]][cy[0]] == 3) {
		if (num > 5) {
			for (int i = 0; i < 9; i++) {
				if (cell[cx[i]][cy[i]] == 3) {
					cell[cx[i]][cy[i]] = 2;
				}
			}
		}
		else {
			for (int i = 1; i <= num; i++) {

				m1[i] = cell[cx[i]][cy[i]];
				cell[cx[i]][cy[i]] = 4;

			}
		}
	}
	else {
		//else if (cell[cx[0]][cy[0]] == 2) {
		for (int i = 1; i <= num; i++) {

			m1[i] = cell[cx[i]][cy[i]];
			cell[cx[i]][cy[i]] = 4;

		}
	}
	//move();
}

// moveWithCuda(cx,cy,size) moveKernel(cx,cy) injectParallel() moveParallel() cx6[i]cy6[i] m6[i] i=0..4 size=5
void injectParallel(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, int x5, int y5) {
	int i = 0;
	cx6[i] = x1;
	cy6[i] = y1;
	m6[i] = cell[cx6[i]][cy6[i]];
	i = 1;
	cx6[i] = x2;
	cy6[i] = y2;
	m6[i] = cell[cx6[i]][cy6[i]];
	i = 2;
	cx6[i] = x3;
	cy6[i] = y3;
	m6[i] = cell[cx6[i]][cy6[i]];
	i = 3;
	cx6[i] = x4;
	cy6[i] = y4;
	m6[i] = cell[cx6[i]][cy6[i]];
	i = 4;
	cx6[i] = x5;
	cy6[i] = y5;
	m6[i] = cell[cx6[i]][cy6[i]];
}

// injectionCuda() move_m5() moveWithCuda5(cx??, cy??, size, i) moveKernel_?(cx??,cy??) cx21,cy21..cx51[0]cy51[0]
void injectionCuda(int x, int y, int num) {

	cx[0] = x;
	cy[0] = y;
	cx[1] = x;
	cy[1] = y - 1;
	cx[2] = x + 1;
	cy[2] = y - 1;
	cx[3] = x + 1;
	cy[3] = y;
	cx[4] = x + 1;
	cy[4] = y + 1;
	cx[5] = x;
	cy[5] = y + 1;
	cx[6] = x - 1;
	cy[6] = y + 1;
	cx[7] = x - 1;
	cy[7] = y;
	cx[8] = x - 1;
	cy[8] = y - 1;

	cx11[0] = x;
	cy11[0] = y-1;
	cx21[0] = x+1;
	cy21[0] = y-1;
	cx31[0] = x+1;
	cy31[0] = y;
	cx41[0] = x+1;
	cy41[0] = y+1;
	cx51[0] = x;
	cy51[0] = y+1;

	if (cell[cx[0]][cy[0]] == 3) {
		if (num > 5) {
			for (int i = 0; i < 9; i++) {
				if (cell[cx[i]][cy[i]] == 3) {
					cell[cx[i]][cy[i]] = 2;
				}
			}
		}
		else {
			for (int i = 1; i <= num; i++) {

				m[i] = cell[cx[i]][cy[i]];
				cell[cx[i]][cy[i]] = 4;

			}
		}
	}
	else {
		for (int i = 1; i <= num; i++) {

			m[i] = cell[cx[i]][cy[i]];
			cell[cx[i]][cy[i]] = 4;

		}
	}
}

void injection5(int x, int y, int num) {
	cx1[0] = x;
	cy1[0] = y;
	m1[0] = cell[cx1[0]][cy1[0]];
	cx2[0] = x;
	cy2[0] = y;
	m2[0] = cell[cx2[0]][cy2[0]];
	cx3[0] = x;
	cy3[0] = y;
	m3[0] = cell[cx3[0]][cy3[0]];
	cx4[0] = x;
	cy4[0] = y;
	m4[0] = cell[cx4[0]][cy4[0]];
	cx5[0] = x;
	cy5[0] = y;
	m5[0] = cell[cx5[0]][cy5[0]];
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
void injection_1(int x, int y, int num) {
	cx1[0] = x;
	cy1[0] = y;
	m1[0] = cell[cx1[0]][cy1[0]];
	cell[cx1[0]][cy1[0]] = 4;
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
void injection_2(int x, int y, int num) {
	cx2[0] = x;
	cy2[0] = y;
	m2[0] = cell[cx2[0]][cy2[0]];
	cell[cx2[0]][cy2[0]] = 4;
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
void injection_3(int x, int y, int num) {
	cx3[0] = x;
	cy3[0] = y;
	m3[0] = cell[cx3[0]][cy3[0]];
	cell[cx3[0]][cy3[0]] = 4;
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
void injection_4(int x, int y, int num) {
	cx4[0] = x;
	cy4[0] = y;
	m4[0] = cell[cx4[0]][cy4[0]];
	cell[cx4[0]][cy4[0]] = 4;
}

// injection_?(x,y,num) move_?() moveWithCuda_?() moveKernel_?() cx?[0],cy?[0]
void injection_5(int x, int y, int num) {
	cx5[0] = x;
	cy5[0] = y;
	m5[0] = cell[cx5[0]][cy5[0]];
	cell[cx5[0]][cy5[0]] = 4;
}

void setup(int x, int y, int m) {
	int w = (m * x) + 2;

	for (int i = (w - x); i < w; i++) {
		for (int j = 2; j < y + 2; j++) {

			cell[i][j] = (rand() % 2 + 2); // 2,3
		}
	}
}

void changeColor(GLfloat red, GLfloat green, GLfloat blue) {
	r = red;
	g = green;
	b = blue;
}

//Check status of individual cell and apply the rules: 3 is cancer, 2 is health cell, 4 is medicine
static int checkStatus(int status, int x, int y) {
	int cancerNeighbours = 0;
	int medicineNeighbours = 0;

	for (int i = (x - 1); i < (x + 2); i++) {
		if (cell[i][y - 1] == 3) {
			cancerNeighbours++;
		}
		if (cell[i][y + 1] == 3) {
			cancerNeighbours++;
		}
	}
	if (cell[x - 1][y] == 3) {
		cancerNeighbours++;
	}
	if (cell[x + 1][y] == 3) {
		cancerNeighbours++;
	}

	for (int i = (x - 1); i < (x + 2); i++) {
		if (cell[i][y - 1] == 4) {
			medicineNeighbours++;
		}
		if (cell[i][y + 1] == 4) {
			medicineNeighbours++;
		}
	}
	if (cell[x - 1][y] == 4) {
		medicineNeighbours++;
	}
	if (cell[x + 1][y] == 4) {
		medicineNeighbours++;
	}

	if (status == 3 && medicineNeighbours >= 6) {
		status = 2;
	}
	else if (status == 2 && cancerNeighbours >= 6) {
		status = 3;
	}
	return status;
}

//Display individual pixels.
static void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	GLfloat red, green, blue;

	for (int i = 5; i < (WIDTH - 5); i++) {
		for (int j = 5; j < (HEIGHT - 5); j++) {
			//Check the updated status of the current cell.
			int cellV = checkStatus(cell[i][j], i, j);
			if (cellV == 0) {
				red = r;
				green = 0.0f;
				blue = 1.0;
				cell[i][j] = 0;
			}
			else if (cellV == 2) {
				red = r;
				green = 0.4f;
				blue = b;
				cell[i][j] = 2;
			}
			else if (cellV == 3) {
				red = 0.4f;
				green = g;
				blue = b;
				cell[i][j] = 3;
			}
			else if (cellV == 4) {
				red = 1.0f;
				green = 1.0f;
				blue = 0.0f;
				cell[i][j] = 4;
			}

			glPointSize(1.0f);
			glColor3f(red, green, blue);
			glBegin(GL_POINTS);
			glVertex2i(i, j);
			glEnd();
		}
	}
	glutSwapBuffers();
}

void update(int value) {
	try {
		//==test 1 ===
		moveParallel();

		//==test 2 ===
		//move_1(); //injection_?()
		//move_2(); //injection_?()
		//move_3(); //injection_?()
		//move_4(); //injection_?()
		//move_5(); //injection_?()
		//move_m5(); //injectionCuda()

		//move5(); //injection5()
	}
	catch (...) {}
	glutPostRedisplay();
	glutTimerFunc(1000 / 30, update, 0);
}

int main(int argc, char** argv)
{
	int x = 1020;
	int y = 766;
	int m = 1;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Cell Growth Simulator");
	init();

	setup(x, y, m);

	//===== test 1 =============
	injectParallel(400, 100, 400, 200, 400, 300, 400, 400, 400, 500);
	
	//======= test 3 ===========
	/*injection5(400, 400, 5);
	injectionCuda(500, 500, 5);*/

	//======= test 2 ===========
	/*injectionCuda(500, 500, 5);
	injection_1(200, 300, 1);
	injection_2(300, 300, 1);
	injection_3(400, 300, 1);
	injection_4(500, 300, 1);
	injection_5(600, 300, 1);*/

	glutDisplayFunc(display);
	glutTimerFunc(1000 / 30, update, 0);
	changeColor(r, g, b);
	glutMainLoop();

    return 0;
}
