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
namespace VideoFilters {


	__global__ void deinterlaceBasicKernel( long cols, long rows,	unsigned char* frameIn, 
																unsigned char*  field1, 
																unsigned char*  field2,
																int step )
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;


		/*int offset = x + y * blockDim.x * gridDim.x;*/
				

		if (x < cols && y < rows) {

			int offset = (x * 3) + y * step;
			int offsetNextRow = offset + step;
			int offsetAntRow = offset - step;

			
			if ( y % 2 == 0 )
			{
				field1[ offset ] =  frameIn[ offset];
				field1[ offset + 1 ] = frameIn[ offset + 1];
				field1[ offset + 2 ] = frameIn[offset + 2];
								
				field2[ offset ] =  frameIn[ offsetNextRow];
				field2[ offset + 1 ] = frameIn[ offsetNextRow + 1];
				field2[ offset + 2 ] = frameIn[offsetNextRow + 2];
			}
			else
			{
				field1[ offset ] =  frameIn[ offsetAntRow];
				field1[ offset + 1 ] = frameIn[ offsetAntRow + 1];
				field1[ offset + 2 ] = frameIn[offsetAntRow + 2];

				field2[ offset ] =  frameIn[ offset];
				field2[ offset + 1 ] = frameIn[ offset + 1];
				field2[ offset + 2 ] = frameIn[offset + 2];
			}
			
		}

		__syncthreads();
		
	}

	

	void deinterlaceBasic( long cols, long rows, unsigned char* frameIn, unsigned char*  field1, unsigned char *field2, int step  )
	{
		
		cudaError_t cudaStatus;
		dim3 block(16, 16);

		
        dim3 grid;
        grid.x = divUp(cols, block.x);
        grid.y = divUp(rows, block.y);

		// Launch a kernel on the GPU with one thread for each element.
		deinterlaceBasicKernel<<< grid, block>>>(cols, rows, frameIn, field1, field2, step);

	    // Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "deinterlaceBasic launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
    
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching deinterlaceBasic!\n", cudaStatus);
			goto Error;
		}


Error:
		return;
	}

}
}