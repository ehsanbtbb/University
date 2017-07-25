/*
 * Original source from nvidia cuda SDK 2.0
 * Modified by S. James Lee (sjames@evl.uic.edi)
 * 2008.12.05
 */


//Fast integer multiplication macro
#define IMUL(a, b) __mul24(a, b)



//Input data texture reference
texture<float, 2, cudaReadModeElementType> texData;

////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS 8
#define KERNEL_W      (2 * KERNEL_RADIUS + 1)
__device__ __constant__ float d_Kernel[KERNEL_W];



////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRow(float x, float y){
    return 
        tex2D(texData, x + KERNEL_RADIUS - i, y) * d_Kernel[i]
        + convolutionRow<i - 1>(x, y);
}

template<> __device__ float convolutionRow<-1>(float x, float y){
    return 0;
}

template<int i> __device__ float convolutionColumn(float x, float y){
    return 
        tex2D(texData, x, y + KERNEL_RADIUS - i) * d_Kernel[i]
        + convolutionColumn<i - 1>(x, y);
}

template<> __device__ float convolutionColumn<-1>(float x, float y){
    return 0;
}



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(
    float *d_Result,
    int dataW,
    int dataH
){
    const   int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const   int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix < dataW && iy < dataH){
        float sum = 0;

#ifdef UNROLL_INNER
        sum = convolutionRow<2 * KERNEL_RADIUS>(x, y);
#else
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += tex2D(texData, x + k, y) * d_Kernel[KERNEL_RADIUS - k];
#endif

        d_Result[IMUL(iy, dataW) + ix] = sum;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(
    float *d_Result,
    int dataW,
    int dataH
){
    const   int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const   int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix < dataW && iy < dataH){
        float sum = 0;

#ifdef UNROLL_INNER
        sum =  convolutionColumn<2 * KERNEL_RADIUS>(x, y);
#else
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += tex2D(texData, x, y + k) * d_Kernel[KERNEL_RADIUS - k];
#endif

        d_Result[IMUL(iy, dataW) + ix] = sum;
    }
}
