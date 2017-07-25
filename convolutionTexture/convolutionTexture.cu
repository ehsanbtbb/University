/*
 * Original source from nvidia cuda SDK 2.0
 * Modified by S. James Lee (sjames@evl.uic.edi)
 * 2008.12.05
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil.h>


////////////////////////////////////////////////////////////////////////////////
// Image file loader (RAW format)
////////////////////////////////////////////////////////////////////////////////
bool loadRawImage(char* filename, int w, int h, float* r, float* g, float* b)
{
	FILE *imageFile;
	imageFile = fopen(filename, "r");
    
	if (imageFile == NULL) 
	{
		printf("Cannot find texture file in data directory: %s\n", filename);
		return false;
	}
	else
	{	    
		for (int i = 0; i < h*w; i+=1)
		{
			r[i]	= fgetc(imageFile);
			g[i]	= fgetc(imageFile);
			b[i]	= fgetc(imageFile);
		}            
		
		fclose(imageFile);
		return true;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Image file writer (RAW format)
////////////////////////////////////////////////////////////////////////////////
bool writeRawImage(char* filename, int w, int h, float* r, float* g, float* b)
{
	FILE *imageFile;
	imageFile = fopen(filename, "wb");
    
	if (imageFile == NULL) 
	{
		printf("Cannot write texture file: %s\n", filename);
		return false;
	}
	else
	{	  
		for (int i = 0; i < h*w; i+=1)
		{
			fputc((int)(r[i]), imageFile);
			fputc((int)(g[i]), imageFile);
			fputc((int)(b[i]), imageFile);
		}
		            
		fclose(imageFile);
		return true;
	}
    
}

////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}


////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
#define UNROLL_INNER
#include "convolutionTexture_kernel.cu"



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//const int	dw = iAlignUp(4096, 16);
//const int	dh = 4096;

const int	KERNEL_SIZE = KERNEL_W * sizeof(float);
//const int	data_size = dw   * dh   * sizeof(float);


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    
    float *h_Kernel;
    float *h_DataR, *h_DataG, *h_DataB;
	float *h_ResultR, *h_ResultG, *h_ResultB;
	
    cudaArray *a_Data;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    float *d_Result;

    double gpuTime, runTime;

    int i, dw, dh, data_size, repeat;
    dw = dh = 1024;
    repeat = 10;

    unsigned int hTimer;


    CUT_DEVICE_INIT(argc, argv);
    CUT_SAFE_CALL( cutCreateTimer(&hTimer) );
	
	// check arg: image resolution
	char *iFilename = "../../../../hubble/hubble1kby1k.raw";
	char *oFilename = "hubble1kby1k_out.raw";
	cutGetCmdLineArgumenti(argc, (const char**) argv, "i", &dw);
	switch (dw)
    {
    case 1024:
    	iFilename = "../../../../hubble/hubble1kby1k.raw";
    	oFilename = "hubble1kby1k_out.raw";
    	dh = dw;
    	break;
    case 2048:
    	iFilename = "../../../../hubble/hubble2kby2k.raw";
    	oFilename = "hubble2kby2k_out.raw";
    	dh = dw;
    	break;
    case 4096:
    	iFilename = "../../../../hubble/hubble4kby4k.raw";
    	oFilename = "hubble4kby4k_out.raw";
    	dh = dw;
    	break;
    default:
    	dh = dw = 1024;
    	printf("use image resoluiton one of 1024, 2048, 4096...\n");
    	printf("will use 1024x1024 as default resolution this time.\n");
    	break;
    }	
	data_size = dw * dh * sizeof(int);
	
	// total iteration number for mean value
	cutGetCmdLineArgumenti(argc, (const char**) argv, "n", &repeat);
		
    printf("Initializing data...\n");
        h_Kernel    = (float *)malloc(KERNEL_SIZE);
        h_DataR     = (float *)malloc(data_size);
        h_DataG     = (float *)malloc(data_size);
        h_DataB     = (float *)malloc(data_size);
        h_ResultR = (float *)malloc(data_size);
        h_ResultG = (float *)malloc(data_size);
        h_ResultB = (float *)malloc(data_size);
        CUDA_SAFE_CALL( cudaMallocArray(&a_Data, &floatTex, dw, dh) );
        CUDA_SAFE_CALL( cudaMalloc((void **)&d_Result, data_size) );

        float kernelSum = 0;
        for(i = 0; i < KERNEL_W; i++){
            float dist = (float)(i - KERNEL_RADIUS) / (float)KERNEL_RADIUS;
            h_Kernel[i] = expf(- dist * dist / 2);
            kernelSum += h_Kernel[i];
        }
        for(i = 0; i < KERNEL_W; i++)
            h_Kernel[i] /= kernelSum;
        
    	if (!loadRawImage(iFilename, dw, dh, h_DataR, h_DataG, h_DataB) )
    	{
    		printf("File not found. random image generator will be used...\n");
        	
      		srand(2007);
       		for(i = 0; i < dw * dh; i++)
       		{
       	    	h_DataR[i] = (float)rand() / (float)RAND_MAX * 255.0f;
       	    	h_DataG[i] = (float)rand() / (float)RAND_MAX * 255.0f;
       	    	h_DataB[i] = (float)rand() / (float)RAND_MAX * 255.0f;
       		}       	
    	}

        CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_Kernel, h_Kernel, KERNEL_SIZE) );
        CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, h_DataR, data_size, cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaBindTextureToArray(texData, a_Data) );

        //Block width should be a multiple of maximum coalesced write size
        //for coalesced memory writes in convolutionRowGPU() and convolutionColumnGPU()
        dim3 threadBlock(16, 12);
        dim3 blockGrid(iDivUp(dw, threadBlock.x), iDivUp(dh, threadBlock.y));


///////////////////////
// Warm up
printf("warming up");
for (i = 0; i < 1; i++)
{
    printf(".");
    
    // red
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, h_DataR, data_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    convolutionRowGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionRowGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    
    //While CUDA kernels can't  directly write to textures, this copy is inevitable
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, d_Result, data_size, cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    convolutionColumnGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionColumnGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
	
	// green
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, h_DataG, data_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    convolutionRowGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionRowGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    
    //While CUDA kernels can't  directly write to textures, this copy is inevitable
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, d_Result, data_size, cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    convolutionColumnGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionColumnGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
	
	// blue
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, h_DataB, data_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    convolutionRowGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionRowGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    
    //While CUDA kernels can't  directly write to textures, this copy is inevitable
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, d_Result, data_size, cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    convolutionColumnGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionColumnGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
}
printf("\n");

//////////////////////////////////
	printf("=============================================================\n");
	printf(" CUDA Convolution: Image Resolution %i x %i\n", dw, dh);
	printf("=============================================================\n");
		
runTime = 0;
for (i = 0; i < repeat; i++)
{
    // red
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, h_DataR, data_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
    convolutionRowGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionRowGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStopTimer(hTimer) );
    gpuTime = cutGetTimerValue(hTimer);
    
    //While CUDA kernels can't  directly write to textures, this copy is inevitable
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, d_Result, data_size, cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
    convolutionColumnGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionColumnGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStopTimer(hTimer) );
    gpuTime += cutGetTimerValue(hTimer);
	
	if (i == (repeat-1))
		CUDA_SAFE_CALL( cudaMemcpy(h_ResultR, d_Result, data_size, cudaMemcpyDeviceToHost) );
	
	// green
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, h_DataG, data_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
    convolutionRowGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionRowGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStopTimer(hTimer) );
    gpuTime += cutGetTimerValue(hTimer);
    
    //While CUDA kernels can't  directly write to textures, this copy is inevitable
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, d_Result, data_size, cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
    convolutionColumnGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionColumnGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStopTimer(hTimer) );
    gpuTime += cutGetTimerValue(hTimer);
	
	if (i == (repeat-1))
		CUDA_SAFE_CALL( cudaMemcpy(h_ResultG, d_Result, data_size, cudaMemcpyDeviceToHost) );
	
	// blue
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, h_DataB, data_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
    convolutionRowGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionRowGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStopTimer(hTimer) );
    gpuTime += cutGetTimerValue(hTimer);
    
    //While CUDA kernels can't  directly write to textures, this copy is inevitable
    CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, d_Result, data_size, cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
    convolutionColumnGPU<<<blockGrid, threadBlock>>>( d_Result, dw, dh );
    CUT_CHECK_ERROR("convolutionColumnGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStopTimer(hTimer) );
    gpuTime += cutGetTimerValue(hTimer);
	runTime += gpuTime;
	printf("%ith GPU convolution time : %f msec\n", i, gpuTime);
	
	if (i == (repeat-1))
		CUDA_SAFE_CALL( cudaMemcpy(h_ResultB, d_Result, data_size, cudaMemcpyDeviceToHost) );

}

	printf("=============================================================\n");
	printf(" Convolution Time: %f msecs (mean of %i run)\n", runTime/ repeat, repeat);
	printf("=============================================================\n\n");
	
	// write result image
	writeRawImage(oFilename, dw, dh, h_ResultR, h_ResultG, h_ResultB);
		
    printf("Shutting down...\n");
        CUDA_SAFE_CALL( cudaUnbindTexture(texData) );
        CUDA_SAFE_CALL( cudaFree(d_Result)   );
        CUDA_SAFE_CALL( cudaFreeArray(a_Data)   );
        free(h_ResultB);
        free(h_ResultG);
        free(h_ResultR);
        free(h_DataB);
        free(h_DataG);
        free(h_DataR);
        free(h_Kernel);
        
    //CUT_EXIT(argc, argv);
}
