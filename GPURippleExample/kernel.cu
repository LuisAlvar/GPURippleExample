#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cpu_anim.h"

#define DIM 1280

/// <summary>
/// 
/// </summary>
/// <param name="ptr">- pointer to device memory that holds the output pixels</param>
/// <param name="ticks">- the current animation time so it can generate the correct frame</param>
/// <returns></returns>
__global__ void kernel(unsigned char* ptr, int ticks)
{
	//Finding the x,y coordinates
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	//Using (x,y) to determine the linearize index 
	int linear_offset = x + y * blockDim.x * gridDim.x;

	//now caluclate the value at that position 
	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);

	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

	ptr[linear_offset * 4 + 0] = grey;
	ptr[linear_offset * 4 + 1] = grey;
	ptr[linear_offset * 4 + 2] = grey;
	ptr[linear_offset * 4 + 3] = 255;
	 
}


struct DataBlock {
	unsigned char* dev_bitmap;
	CPUAnimBitmap* bitmap;
};

// clean up memory allocated on the GPU
void cleanup(DataBlock *d){
	cudaFree(d->dev_bitmap);
}

//This function will be called by the strucutre every time it wants to generate a new frame of the animation.  
void generate_frame(DataBlock *d, int ticks) {
	dim3 blocks(DIM/16, DIM/16);
	dim3 threads(16, 16);

	kernel<<< blocks, threads >>>(d->dev_bitmap, ticks);

	cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost);
}

int main( void )
{

	DataBlock data;
	CPUAnimBitmap bitmap(DIM,DIM, &data);
	data.bitmap = &bitmap;

	cudaMalloc( (void**)&data.dev_bitmap, bitmap.image_size() );

	// We pass a function pointer to generate_frame() 
	bitmap.anim_and_exit((void (*)(void*, int))generate_frame, (void (*)(void*))cleanup);

	return 0;
}
