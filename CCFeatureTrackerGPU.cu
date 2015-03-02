#include "kernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include "device_functions.h"
#include "device_functions_decls.h"

#include <stdio.h>

#include "opencv2\gpu\device\common.hpp"

using namespace cv::gpu;

namespace CCFeatureTrackerGPU_K {


	__global__ void innerProcessKernel( long cols, long rows, unsigned char* frameIn, unsigned char*  frameOut, int step )
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;


		/*int offset = x + y * blockDim.x * gridDim.x;*/
				

		if (x < cols && y < rows) {

			int offset = (x * 3) + y * step;

			frameOut[ offset ] =  frameIn[ offset + 2];
			frameOut[ offset + 1 ] = frameIn[ offset + 1];
			frameOut[ offset + 2 ] = frameIn[offset];
		}

		__syncthreads();
		
	}

	

	void innerProcess( long cols, long rows, unsigned char* frameIn, unsigned char*  frameOut, int step  )
	{
		
		cudaError_t cudaStatus;
		dim3 block(16, 16);

		
        dim3 grid;
        grid.x = divUp(cols, block.x);
        grid.y = divUp(rows, block.y);

		// Launch a kernel on the GPU with one thread for each element.
		innerProcessKernel<<< grid, block>>>(cols, rows, frameIn, frameOut, step);

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


Error:
		return;
	}

}