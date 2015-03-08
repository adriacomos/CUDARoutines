#include "CUDASystemInformation.h"
#include <stdio.h>
#include <iostream>

#include "cuda_runtime.h"

using namespace std;

// 1.x -> 8
// 2.0 -> 32
// 2.1 -> 48
// 3.x -> 192
// 5.x -> 128

namespace CUDARoutines {

void CUDASystemInformation::getSystemInformation()
{
	cudaError_t cudaStatus;
	int CUDADevices;
		
	cudaStatus = cudaGetDeviceCount( &CUDADevices );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching deinterlaceBasic!\n", cudaStatus);
		goto Error;
	}

	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceProperties( &prop, 0 );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching deinterlaceBasic!\n", cudaStatus);
		goto Error;
	}


	int computeMode = prop.computeMode;

	int cores = -1;
	switch (prop.major)
	{
	case 1:
		cores = 8;
		break;
	case 2:
		if (prop.minor == 0)
			cores = 32;
		else
			cores = 48;
		break;
	case 3:
		cores = 192; break;
	case 5:
		cores = 128; break;
	};

	
	cout << "Num CUDA devices: " << CUDADevices << endl;
	cout << "Device 0:" << endl;
	cout << "- Name: " << prop.name << endl;
	cout << "- Max threads dim: " << prop.maxThreadsDim[0] << "," << prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2] << endl;
	cout << "- Max grid size: " << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << endl;
	cout << "- Num multiprocessor count: " << prop.multiProcessorCount << endl;
	cout << "- Max threads per block: " << prop.maxThreadsPerBlock << endl;
	cout << "- Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << endl;
	cout << "- Compute capability: " << prop.major << "." << prop.minor << endl;
	if (cores > 0)
		cout << "- Core number: " << prop.multiProcessorCount * cores << endl;
	else
		cout << "- Core number: Unknown" << endl;

Error:
	return;
}

};