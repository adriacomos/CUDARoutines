#pragma once


#include "opencv2\core\cuda_devptrs.hpp"

void addWithCuda(int *c, const int *a, const int *b, unsigned int size);



namespace CCFeatureTrackerGPU_K {
	void innerProcess( long cols, long rows, unsigned char* frameIn, unsigned char* frameOut, int step  );
};