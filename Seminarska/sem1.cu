#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define err(msg, reason) { fprintf(stderr, "Error: %s %s\n", msg, reason); exit(1); }
#define COLOR_CHANNELS 1
#define GRAY_LEVELS 256
#define BLOCK_SIZE 1024

// uncomment me for terminal output
// #define DEBUG

// calculate histogram
__global__ void calculateHistogram(unsigned char *image, int width, int height, unsigned int *histogram) {

    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    int localX = threadIdx.x,
        localY = threadIdx.y;
    
    __shared__ unsigned int localHisto[GRAY_LEVELS];

    localHisto[blockDim.x * localY + localX] = 0;
    __syncthreads();

    if (x < width && y < height) {
        atomicAdd(&localHisto[image[y * width + x]], 1);
    }
    __syncthreads();

    atomicAdd(&(histogram[blockDim.x * localY + localX]), localHisto[blockDim.x * localY + localX]);

}

// calculate paraller prefix sum
__global__ void parallelPrefixSum(unsigned int *histogram, unsigned int *cdf) {

    __shared__ unsigned int tmp[GRAY_LEVELS * 2];
    int threadID = threadIdx.x,
        pout = 0,
        pin = 1;
    
    tmp[threadID] = histogram[threadID];
    __syncthreads();

    for (int offset = 1; offset<GRAY_LEVELS; offset<<=1) {
        pout = 1 - pout;
        pin = 1 - pout;
        if (threadID >= offset) {
            tmp[pout * GRAY_LEVELS + threadID] = tmp[pin * GRAY_LEVELS + threadID] + tmp[pin * GRAY_LEVELS + threadID - offset];
        } else {
            tmp[pout * GRAY_LEVELS + threadID] = tmp[pin * GRAY_LEVELS + threadID];
        }
        __syncthreads();
    }

    cdf[threadID] = tmp[pout * GRAY_LEVELS + threadID];

}

// get the non-zero minimum value
__global__ void calcuateMinReduction(unsigned int *cdf, unsigned int *minimum) {

    // extern __shared__ unsigned int sharedData[];
    int threadID = threadIdx.x;

    // x >> 1 is the same as x / 2
    // and because bitwise op. are faster, so
    // we are gonna use it.
    // So it is stride / 2; stride>0; stride /= 2
    // this reduces the number of threads involved in the reduction
    // by half each iteration until there is only one left that performs
    // the final combination
    for (unsigned int stride = blockDim.x >> 1; stride>0; stride >>= 1) {
        // this is reduction
        __syncthreads();
        if (threadID < stride) {
            cdf[threadID] = (cdf[threadID] != 0) ? min(cdf[threadID], cdf[threadID + stride]) : cdf[threadID + stride];
        }
    }
    // one thread remaining
    if (threadID == 0) {
        minimum[blockIdx.x] = cdf[0];
    }

}

__device__ inline unsigned char scale(unsigned int cdf, unsigned int min, unsigned int imageSize) {

    float scale;
    scale = (float)(cdf - min) / (float) (imageSize - min);
    scale = round(scale * (float) (GRAY_LEVELS - 1));

    return (int) scale;

}

__global__ void equalize(unsigned char *imageIn, unsigned char *imageOut, int width, int height, unsigned int *cdf, unsigned int *cdfMin) {

    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    int localX = threadIdx.x,
        localY = threadIdx.y;

    unsigned int imageSize = width * height;

    __shared__ unsigned int localCdf[GRAY_LEVELS];

    localCdf[blockDim.x * localY + localX] = cdf[blockDim.x * localY + localX];
    __syncthreads();

    if (x < width && y < height) {
        imageOut[y * width + x] = scale(localCdf[imageIn[y * width + x]], *cdfMin, imageSize); // cdf, cdfMin, imageSize
    }

}

void writeFile(char *filename, unsigned int *data) {

    FILE *f;
    if (!(f = fopen(filename, "w+"))) {
        err("fopen", "could not create file")
    }

    for (int i = 0; i<GRAY_LEVELS; i++) {
        if (fprintf(f, "%ld\n", data[i]) < 0) {
            err("fprintf", "could not write to file")
        }
    }

    if (fclose(f) == EOF) {
        err("fclose", "could not close file")
    }

}
void writeLineFile(char *filename, float time) {

    FILE *f;
    if (!(f = fopen(filename, "a"))) {
        err("fopen", "could not create file")
    }

    if (fprintf(f, "%f,\n", time) < 0) {
        err("fprintf", "could not write to file")
    }

    if (fclose(f) == EOF) {
        err("fclose", "could not close file")
    }

}

int main(int argc, char **argv){

    // check for arguments
    if (argc < 2) {
        err("To few arguments.", "Usage: <image_in> <image_out>")
    }

    // host variables
    int width, height, cpp = 1;
    unsigned char *imageIn, *imageOut;
    unsigned int *histogram, *cdf, *minimum;
    size_t imageSize;
    float millisecondsHisto,
          millisecondsPrefixSum,
          millisecondsMin,
          millisecondsEq,
          millisecondsFULL;

    // device variables
    unsigned char *imageInDevice, *imageOutDevice;
    unsigned int *histogramDevice, *cdfDevice, *minimumDevice;

    // load image from file
    if (!(imageIn = stbi_load(argv[1], &width, &height, &cpp, 1))) {
        err("stbi_load", "Could not load image")
    }
    printf("Loaded image %s of size %dx%d, channels %d.\n", argv[1], width, height, cpp);

    // allocate memory for histogram
    if (!(histogram = (unsigned int *) calloc(GRAY_LEVELS, sizeof(unsigned int)))) {
        err("calloc", "Could not allocate memory for histogram on host")
    }

    // allocate memory for cdf
    if (!(cdf = (unsigned int *) calloc(GRAY_LEVELS, sizeof(unsigned int)))) {
        err("calloc", "Could not allocate memory for cdf on host")
    }

    // allocate memory for minimum value
    if (!(minimum = (unsigned int *) calloc(1, sizeof(unsigned int)))) {
        err("calloc", "Could not allocate memory for minimum value on host")
    }

    // calculate needed size of memory for image and allocate it
    imageSize = height * width * sizeof(unsigned char);
    if (!(imageOut = (unsigned char *) malloc(imageSize))) {
        err("malloc", "Could not allocate memory for output image on host")
    }

    // allocate memory on GPU and set it to 0
    // image
    checkCudaErrors(cudaMalloc(&imageInDevice, imageSize));
    // histogram
    checkCudaErrors(cudaMalloc(&histogramDevice, GRAY_LEVELS * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(histogramDevice, 0, GRAY_LEVELS * sizeof(unsigned int)));
    // cdf
    checkCudaErrors(cudaMalloc(&cdfDevice, GRAY_LEVELS * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(cdfDevice, 0, GRAY_LEVELS * sizeof(unsigned int)));
    // minimum
    checkCudaErrors(cudaMalloc(&minimumDevice, 1 * sizeof(unsigned int)));
    // cudaMemset(minimumDevice, 0, 1 * sizeof(unsigned int));
    // imageOut
    checkCudaErrors(cudaMalloc(&imageOutDevice, imageSize));
    checkCudaErrors(cudaMemset(imageOutDevice, 0, imageSize));

    // copy data to device
    // image
    checkCudaErrors(cudaMemcpy(imageInDevice, imageIn, imageSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width / blockSize.x), ceil(height / blockSize.y));
    
    // dim3 blockSize(256, 1, 1); // 256 niti v bloku (1024 je max. število niti v bloku!)
    // dim3 gridSize(1024 / 256, 1, 1); // en blok v mreži

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // compute the image histogram
    calculateHistogram<<<gridSize,blockSize>>>(imageInDevice, width, height, histogramDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&millisecondsHisto, start, stop);
    millisecondsFULL += millisecondsHisto;

    // compute the cumulative distribution of the histogram
    cudaEventRecord(start);
    parallelPrefixSum<<<1,GRAY_LEVELS>>>(histogramDevice, cdfDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&millisecondsPrefixSum, start, stop);
    millisecondsFULL += millisecondsPrefixSum;

    // find non-zero minimum value
    cudaEventRecord(start);
    calcuateMinReduction<<<1,GRAY_LEVELS>>>(cdfDevice, minimumDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&millisecondsMin, start, stop);
    millisecondsFULL += millisecondsMin;

    // transform the original image using the scaled cumulative distribution as the transformation function
    cudaEventRecord(start);
    equalize<<<gridSize,blockSize>>>(imageInDevice, imageOutDevice, width, height, cdfDevice, minimumDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&millisecondsEq, start, stop);
    millisecondsFULL += millisecondsEq;

    // copy result from device to host
    // histogram
    checkCudaErrors(cudaMemcpy(histogram, histogramDevice, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // minimum
    checkCudaErrors(cudaMemcpy(minimum, minimumDevice, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // cdf
    checkCudaErrors(cudaMemcpy(cdf, cdfDevice, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // image
    checkCudaErrors(cudaMemcpy(imageOut, imageOutDevice, imageSize, cudaMemcpyDeviceToHost));

    #ifdef DEBUG
    for (int i = 0; i<GRAY_LEVELS; i++) {
        printf("Color %d has %ld pixels\n", i, histogram[i]);
    }
    #endif

    // writeFile("histogram_old.txt", histogram);

    #ifdef DEBUG
    printf("\n");
    for (int i = 0; i<GRAY_LEVELS; i++) {
        printf("How many px have color %d? %ld\n", i, cdf[i]);
    }

    printf("\n");
    printf("\nMIN IS: %ld\n\n", *minimum);
    #endif

    // write image (final step)
    stbi_write_jpg(argv[2], width, height, COLOR_CHANNELS, imageOut, 100);

    // free all allocated memory
    free(imageIn);
    free(imageOut);
    free(histogram);
    free(cdf);
    free(minimum);

    cudaFree(imageInDevice);
    cudaFree(imageOutDevice);
    cudaFree(histogramDevice);
    cudaFree(cdfDevice);
    cudaFree(minimumDevice);

    // print execution time
    printf("Kernel Execution time is: %f seconds \n", millisecondsFULL / 1000.f);

    writeLineFile("time.txt", millisecondsFULL / 1000.f);

    return 0;
}