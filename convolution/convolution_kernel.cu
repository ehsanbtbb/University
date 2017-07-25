/*
 * Original source from nvidia cuda SDK 2.0
 * Modified by S. James Lee (sjames@evl.uic.edi)
 * 2008.12.05
 */


////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS 8
#define      KERNEL_W (2 * KERNEL_RADIUS + 1)
__device__ __constant__ float d_Kernel[KERNEL_W];

#define TILE_W 16		// active cell width
#define TILE_H 16		// active cell height
#define TILE_SIZE (TILE_W + KERNEL_RADIUS * 2) * (TILE_W + KERNEL_RADIUS * 2)

#define IMUL(a,b) __mul24(a,b)

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
)
{
    // Data cache: threadIdx.x , threadIdx.y
    __shared__ float data[ TILE_H * (TILE_W + KERNEL_RADIUS * 2) ];
	
	// global mem address of this thread
	const int gLoc = threadIdx.x + 
					 IMUL(blockIdx.x, blockDim.x) +
    				 IMUL(threadIdx.y, dataW) +
    				 IMUL(blockIdx.y, blockDim.y) * dataW;
    				     
    // load cache (32x16 shared memory, 16x16 threads blocks)
    // each threads loads two values from global memory into shared mem
    // if in image area, get value in global mem, else 0
	int x;		// image based coordinate

	// original image based coordinate
	const int x0 = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
	const int shift = threadIdx.y * (TILE_W + KERNEL_RADIUS * 2);
	
	// case1: left
	x = x0 - KERNEL_RADIUS;
	if ( x < 0 )
		data[threadIdx.x + shift] = 0;
	else	
		data[threadIdx.x + shift] = d_Data[ gLoc - KERNEL_RADIUS];

	// case2: right
	x = x0 + KERNEL_RADIUS;
	if ( x > dataW-1 )
		data[threadIdx.x + blockDim.x + shift] = 0;
	else	
		data[threadIdx.x + blockDim.x + shift] = d_Data[gLoc + KERNEL_RADIUS];
    
    __syncthreads();

	// convolution
	float sum = 0;
	x = KERNEL_RADIUS + threadIdx.x;
	for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
		sum += data[x + i + shift] * d_Kernel[KERNEL_RADIUS + i];

    d_Result[gLoc] = sum;

}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
)
{
    // Data cache: threadIdx.x , threadIdx.y
    __shared__ float data[TILE_W * (TILE_H + KERNEL_RADIUS * 2)];
	
	// global mem address of this thread
	const int gLoc = threadIdx.x + 
					 IMUL(blockIdx.x, blockDim.x) +
    				 IMUL(threadIdx.y, dataW) +
    				 IMUL(blockIdx.y, blockDim.y) * dataW;
    				     
    // load cache (32x16 shared memory, 16x16 threads blocks)
    // each threads loads two values from global memory into shared mem
    // if in image area, get value in global mem, else 0
	int y;		// image based coordinate

	// original image based coordinate
	const int y0 = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
	const int shift = threadIdx.y * (TILE_W);
	
	// case1: upper
	y = y0 - KERNEL_RADIUS;
	if ( y < 0 )
		data[threadIdx.x + shift] = 0;
	else	
		data[threadIdx.x + shift] = d_Data[ gLoc - IMUL(dataW, KERNEL_RADIUS)];

	// case2: lower
	y = y0 + KERNEL_RADIUS;
	const int shift1 = shift + IMUL(blockDim.y, TILE_W);
	if ( y > dataH-1 )
		data[threadIdx.x + shift1] = 0;
	else	
		data[threadIdx.x + shift1] = d_Data[gLoc + IMUL(dataW, KERNEL_RADIUS)];
    
    __syncthreads();

	// convolution
	float sum = 0;
	for (int i = 0; i <= KERNEL_RADIUS*2; i++)
		sum += data[threadIdx.x + (threadIdx.y + i) * TILE_W] * d_Kernel[i];

    d_Result[gLoc] = sum;

}













