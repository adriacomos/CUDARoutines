#include "kernels.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include "device_functions.h"
#include "device_functions_decls.h"

#include <stdio.h>

#include "opencv2\gpu\device\common.hpp"

using namespace cv::gpu;

namespace CUDARoutines {

namespace CCFeatureTrackerGPU_K {


	__global__ void innerProcessKernel( long cols, long rows, unsigned char* frameIn, unsigned char*  frameOut, int step )
	{
	
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;

		int sizeK = 41;
		int borderk = 20;
		int sq = sizeK * sizeK;
		
		if (x < cols - borderk && y < rows-borderk && x > borderk && y > borderk) {
			int offset = (x * 3) + y * step;
			
			int R = 0,G = 0,B = 0;
			for( int iy=-borderk; iy<borderk+1; iy++)
			{
				for( int ix=-borderk; ix<borderk+1; ix++)
				{
					R += frameIn[offset + ix*3 + iy*step ];
					G += frameIn[offset+1 + ix*3 + iy*step ];
					B += frameIn[offset+2 + ix*3 + iy*step ];
				}
			}



			frameOut[ offset ] =  R/sq;
			frameOut[ offset + 1 ] = G/sq;
			frameOut[ offset + 2 ] = B/sq;
		}

		__syncthreads();
		
	}

	

	void innerProcess( long cols, long rows, unsigned char* frameIn, unsigned char*  frameOut, int step  )
	{
		
		cudaError_t cudaStatus;
		dim3 block(32, 8);

		
        dim3 grid;
        grid.x = divUp(cols, block.x);
        grid.y = divUp(rows, block.y);

		// Launch a kernel on the GPU with one thread for each element.
		innerProcessKernel<<< grid, block>>>(cols, rows, frameIn, frameOut, step);

	    // Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "innerProcessKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
    
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching innerProcessKernel!\n", cudaStatus);
			goto Error;
		}


Error:
		return;
	}

}

}