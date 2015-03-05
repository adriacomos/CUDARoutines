#pragma once


#include "opencv2\core\cuda_devptrs.hpp"

void addWithCuda(int *c, const int *a, const int *b, unsigned int size);

namespace CUDARoutines {

namespace CCFeatureTrackerGPU_K {
	void innerProcess( long cols, long rows, unsigned char* frameIn, unsigned char* frameOut, int step  );
};



namespace VideoFilters {
	void deinterlaceBasic( long cols, long rows, unsigned char* frameIn, unsigned char*  field1, unsigned char *field2, int step  );
	void deinterlaceInterpol( long cols, long rows, unsigned char* frameIn, unsigned char*  field1, unsigned char *field2, int step  );
}

};